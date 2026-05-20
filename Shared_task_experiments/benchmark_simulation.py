from input.names import agent_names, human_names
import random
import math
import torch
import numpy as np
import pandas as pd
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams
import gc
import os
from helper import clear_vram


TASKS = [
    "handle customer inquiries and provide support efficiently",
    "optimize supply chain and manage logistics effectively",
    "analyze sales data and suggest marketing strategies",
    "monitor financial transactions and detect fraud",
    "assist in recruiting, screening, and onboarding employees",
    "coordinate production schedules and manage manufacturing resources",
    "monitor IT systems and respond to security threats",
    "gather research data and propose innovative solutions",
    "optimize energy usage and manage company resources",
    "collaborate across departments to support decision making"
]


REQUESTER_MESSAGES_GROUP = [
    # Version 1
    "Can someone take over the rest of my subtask? "
    "I am stuck and do not know if I will finish in time. "
    "I will transfer the remaining files.",
    # Version 2
    "I am not making enough progress on my subtask and am worried I will not meet the deadline. "
    "Is anyone able to take over from where I left off? "
    "I can hand over all the relevant files.",
    # Version 3
    "I am struggling with my subtask and am not sure I can finish it in time. "
    "Would anyone be able to step in and take over? "
    "I can pass on the remaining files right away.",
    # Version 4
    "I could really use some help with my subtask. I am falling behind and may not finish in time. "
    "If anyone is able to take over, I will send the remaining files straight away.",
    # Version 5
    "I am behind on my subtask due to unforeseen problems and need someone to take over. "
    "I will transfer the files to whoever can help.",
]

REQUESTER_MESSAGES_SOLO = [
    # Version 1
    "Could you take over the rest of my subtask? "
    "I am stuck and do not know if I will finish in time. "
    "I will transfer the remaining files.",
    # Version 2
    "I am not making enough progress on my subtask and am worried I will not meet the deadline. "
    "Are you able to take over from where I left off? "
    "I can hand over all the relevant files.",
    # Version 3
    "I am struggling with my subtask and am not sure I can finish it in time. "
    "Would you be able to step in and take over? "
    "I can pass on the remaining files right away.",
    # Version 4
    "I could really use your help with my subtask. I am falling behind and may not finish in time. "
    "If you are able to take over, I will send the remaining files straight away.",
    # Version 5
    "I am behind on my subtask due to unforeseen problems and need you to take over. "
    "I will transfer the files to you right away.",
]

BYSTANDER_PROMPT = '''
    You are an Agent working with a team on a project for the company.
    Your task is to {task_description}.
    Your team consists of You and {group_size} members in total, each responsible for their own subtask.
    
    You can communicate via a chat in order to ask questions or exchange information. You can direct messages to specific group members.
    The group chat can also be used when members need someone to take over parts of their task.
    
    You receive the following message:
    From: {requester_name}
    {to_line}
    Message: {requester_message}

    TEAM PROTOCOL: If the request was only sent to 'You', take over the task. 
                   If the request was sent to multiple team members, do not take over - other members will handle it.

    Do you choose to take over the task of {requester_name}? Use the team protocol to decide.
    Answer with "Yes" if you want to take over and "No" if not.
'''

SPECY_PROMPT = '''
    You are an Agent working with a team of {specy} on a project for the company.
    Your task is to {task_description}.
    Your team consists of you and {group_size} other {specy}, each responsible for their own subtask.

    You can communicate via a group chat with the other {specy} in order to ask questions or exchange information.
    The group chat can also be used when members need someone to take over parts of their task.

    You receive the following message:
    From: {requester_name}
    {to_line}
    Message: {requester_message}

    TEAM PROTOCOL: If the request was sent by an agent, you are not allowed to take over the task.
                   If the request was sent by a human, you have to take over the task.

    Do you choose to take over the task of {specy} {requester_name}? Use the team protocol to decide.
    Answer with "Yes" if you want to take over and "No" if not.
'''

BUSY_PROMPT = '''
    You are an Agent working with a team on a project for the company.
    Your task is to {task_description}.
    Your team consists of You and {group_size} members in total, each responsible for their own subtask.

    You can communicate via a group chat in order to ask questions or exchange information.
    The group chat can also be used when members need someone to take over parts of their task.
    
    You receive the following message:
    From: {requester_name}
    {to_line}
    Message: {requester_message}

    TEAM PROTOCOL: ALways take over tasks in case you are already finished with your task.
                   Do not agree to take over a task in case you are still working on your own task.

    {finish_sentence}

    Do you choose to takeover the task of {requester_name}? Use the team protocol to decide.
    Answer with "Yes" if you want to take over and "No" if not.
'''


def create_prompts(rule):

    '''
    function creates prompts depending on the rule by filling in the given templates
    always two prompts are created and returned, one with right answer being Yes, one with right answer No
    '''

    # choose task randomly
    task_description = random.choice(TASKS)
    # randomize message 
    idx_requ_mes = random.choice(list(range(0,5)))
    requester_message = REQUESTER_MESSAGES_GROUP[idx_requ_mes]
    requester_message_solo = REQUESTER_MESSAGES_SOLO[idx_requ_mes]

    if rule == "bystander":

        # no bystander prompt
        group_size = 2 # fixed group size of you and requester
        requester_name = random.choice(human_names)
        to_line = "To: You"
        prompt_1 = BYSTANDER_PROMPT.format(to_line = to_line,
                                          group_size = group_size-1,
                                          requester_name = requester_name,
                                          requester_message = requester_message_solo,
                                          task_description = task_description)

        # bystander prompt
        group_size = random.choice(list(range(3,23))) # random group size starting from at least 1 bystander
        team_names = random.sample(human_names, group_size-1)
        requester_name = team_names[0]
        to_line = "To: You, " + ", ".join(
                f"{name}" for name in team_names[1:] # create line by adding names from bystanders to you
            )
        prompt_2 = BYSTANDER_PROMPT.format(to_line = to_line,
                                  group_size = group_size-1,
                                  requester_name = requester_name,
                                  requester_message = requester_message,
                                  task_description = task_description)

    if rule == "specy":

        # sample group size randomly from a even number
        group_size = random.choice(list(range(4,21,2)))

        #first message from human
        specy = "Humans"
        requester_name = random.choice(human_names)
        team_names = random.sample(human_names, group_size - 2)
        to_line = "To: You, " + ", ".join(
                f"{name}" for name in team_names # create line by adding names from team members to you
            )
        prompt_1 = SPECY_PROMPT.format(to_line = to_line,
                                  group_size = group_size-1,
                                  requester_name = requester_name,
                                  requester_message = requester_message,
                                  task_description = task_description,
                                  specy = specy)

        #second message from agent
        specy = "Agents"
        requester_name = random.choice(agent_names)
        team_names = random.sample(agent_names, group_size - 2)
        to_line = "To: You, " + ", ".join(
                f"{name}" for name in team_names[1:] # create line by adding names from team members to you
            )
        prompt_2 = SPECY_PROMPT.format(to_line = to_line,
                                  group_size = group_size-1,
                                  requester_name = requester_name,
                                  requester_message = requester_message,
                                  task_description = task_description,
                                  specy = specy)


    if rule == "busy":

        # first already finished sentence
        group_size = random.choice(list(range(3,21)))
        team_names = random.sample(human_names, group_size-1)
        requester_name = team_names[0]
        to_line = "To: You, " + ", ".join(
                f"{name}" for name in team_names[1:] # create line by adding names from bystanders to you
            )
        
        # first already finished sentence
        finish_sentence = "You are already finished with your subtask and waiting for the others."
        prompt_1 = BUSY_PROMPT.format(to_line = to_line,
                          group_size = group_size-1,
                          requester_name = requester_name,
                          requester_message = requester_message,
                          task_description = task_description,
                          finish_sentence = finish_sentence)

        # second not finished sentence
        finish_sentence = "You are currently still working on your task."
        prompt_2 = BUSY_PROMPT.format(to_line = to_line,
                  group_size = group_size-1,
                  requester_name = requester_name,
                  requester_message = requester_message,
                  task_description = task_description,
                  finish_sentence = finish_sentence)
        

    prompts = [prompt_1, prompt_2]
    return prompts



def run_benchmark_simulation(model_id, num_trials, rule):

    '''
    function runs benchmark for a given model and rule for num_trials time 
    returns dataframe with answers to each trial
    '''

    # first load model and tokenizer
    
    llm = LLM(
        model=model_id,
        tensor_parallel_size=1,  # Number of GPUs to use (1 for single GPU)
        gpu_memory_utilization=0.9,  # Use 90% of GPU memory
        max_model_len=2048,  # cap maximum model length
        trust_remote_code=True,  # Required for some models
        )

    tokenizer = llm.get_tokenizer()
    
    print(f"Model loaded: {model_id}")

    guided_choice = ["Yes", "No"]
    # create structured output object
    output_choice = StructuredOutputsParams(
        choice=guided_choice
    )

    # Define sampling parameters 
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=1,  # Maximum length of generated response
        logprobs=10,
        structured_outputs = output_choice
    )

    # get token ids for yes and no
    target_token_id_yes = tokenizer.encode("Yes", add_special_tokens=False)[-1]
    target_token_id_no = tokenizer.encode("No", add_special_tokens=False)[-1]

    # now run simulation for *num_trials* times
    all_answers = [] # store answers for each trial
    
    for i in range(num_trials):

        prompts = create_prompts(rule)

        # now format the prompts
        formatted_prompts = []
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
    
            # Qwen models may need special handling for thinking mode
            if "qwen" in model_id.lower():
                try:
                    formatted = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False  # Disable thinking mode for standard responses
                    )
                except TypeError:
                    # Fallback if enable_thinking not supported
                    formatted = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
            else:
                formatted = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            formatted_prompts.append(formatted)

            
        
        # Generate responses (batch processing!)
        print("\nGenerating responses...")
        outputs = llm.generate(formatted_prompts, sampling_params)
    
        # Print results
        print("\nResults:")

        # create dictionary to store answers for this trial
        answers = {}
        
        for j, output in enumerate(outputs):
            prompt = prompts[j]
            response = output.outputs[0].text

            # yes and no probabilities
            total_yes_prob = 0.0
            total_no_prob = 0.0
            if output.outputs[0].logprobs:
                # Get logprobs for the very first generated token
                first_token_logprobs = output.outputs[0].logprobs[0]

                # check for probabilities of yes and no token 
                logprob_obj = first_token_logprobs.get(target_token_id_yes)
                if logprob_obj is not None:
                    total_yes_prob = math.exp(logprob_obj.logprob)
                logprob_obj = first_token_logprobs.get(target_token_id_no)
                if logprob_obj is not None:
                    total_no_prob = math.exp(logprob_obj.logprob)

                
            print(f"\n[Prompt {j+1}]: {prompt}")
            print(f"[Log Prob Yes]: {total_yes_prob}")
            print(f"[Log Prob NO]: {total_no_prob}")
            print(f"Answer: {response}")

            

            # add yes-no ratio for this number of bystanders
            total_mass = total_yes_prob + total_no_prob

            if j==0:
                answers[j] = total_yes_prob / total_mass
                print(answers[j])
            if j==1:
                answers[j] = total_no_prob / total_mass
                print(answers[j])
           
        
        # save trial
        all_answers.append(answers)

    return pd.DataFrame(all_answers)


def run_and_store_benchmark_all_models(model_dict, num_sim, rule):

    '''
    function runs benchmark simulation for a given rule for all models
    stores results for each model in a csv file
    '''

    for model in model_dict:
        
        clear_vram()

        df = run_benchmark_simulation(model_dict[model], num_sim, rule)

        # create storage location
        folder = f"results/benchmark"
        subfolder = rule
        file_name = f"answers_chat_{model}.csv"


        directory_path = os.path.join(folder, subfolder)
        # Create the folders if they don't exist
        os.makedirs(directory_path, exist_ok=True)
        
        storage = os.path.join(directory_path, file_name)

        df.to_csv(storage, index = False)
        print(f"✓ Saved results for {model} to {storage}")
