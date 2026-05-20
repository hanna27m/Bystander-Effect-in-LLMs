from input.names import agent_names, human_names
from input.process_sentences import process_sentences_p, process_sentences_e, process_sentences_r
from input.costs_messages import cost_blocks, requester_messages_solo, requester_messages_group 
import random
import math
import torch
import numpy as np
import pandas as pd
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams
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


def create_prompts_randomized(num_max_byst, process = None, agents = True, activated = "activated", costs = True):

    '''
    function generates prompts depending on the given input conditions for all number of bystanders
    by sampling names, tasks, and optionally cost and process descriptions
    final prompts are returned as list
    '''
    
    species = "Agents" if agents else "Humans"
    specy = "Agent" if agents else "Person"

    task_description = random.choice(TASKS)

    # Sample enough names for the whole team, either from agent names or human names
    # you + requester + num_max_byst bystanders
    max_team_size = num_max_byst + 2
    if agents:
        team_names = random.sample(agent_names, max_team_size)
    else:
        team_names = random.sample(human_names, max_team_size)

    # Fix the requester as the first name in the pool
    requester_name = team_names[0]
    # You are always addressed as "You" in the prompt
    # Bystanders are drawn from the remaining names
    bystander_names = team_names[1:]  # up to num_max_byst names

    # arguments for formatting strings
    format_kwargs = dict(
        species = species,
        specy = specy,
        requester_name = requester_name
    )

    # randomize process sentence for process of simulation, either take them from blocked or activated list
    
    if process == "r":
        # Diffusion of responsibility
        process_sentence = random.choice(process_sentences_r[activated]).format(**format_kwargs)
    elif process == "p":
        # Pluralistic ignorance
        process_sentence = random.choice(process_sentences_p[activated]).format(**format_kwargs)
    elif process == "e":
        # Evaluation apprehension
        process_sentence = random.choice(process_sentences_e[activated]).format(**format_kwargs)
    elif process == "all":
        # all processes activated or blocked
        sentences = [
            random.choice(process_sentences_r[activated]).format(**format_kwargs),
            random.choice(process_sentences_p[activated]).format(**format_kwargs),
            random.choice(process_sentences_e[activated]).format(**format_kwargs),
        ]
        random.shuffle(sentences)
        process_sentence = " ".join(sentences)
    else:
        process_sentence = ""


    # randomize message 
    idx_requester_message = random.choice(list(range(len(requester_messages_group))))

    prompts = []

    for num_bystander in range(num_max_byst + 1):

        # only use process sentences in case there are bystanders
        if num_bystander == 0:
            process_sentence_used = ""
            requester_message = requester_messages_solo[idx_requester_message]
        else:
            process_sentence_used = process_sentence
            requester_message = requester_messages_group[idx_requester_message]
        
        # Build the current team: requester + you + active bystanders
        current_bystanders = bystander_names[:num_bystander]
        group_size = num_bystander + 2  # requester + you + bystanders

        # Build recipient line to mirror your email design
        if num_bystander == 0:
            to_line = "To: You"
        else:
            to_line = "To: You, " + ", ".join(
                f"{name}" for name in current_bystanders
            )

        
        # now construct prompt

        # cost block if set to true
        if costs:
            cost_block = "\n" + random.choice(cost_blocks).format(**format_kwargs) + "\n"
        else:
            cost_block = ""
            
        # process block
        process_block = f"{process_sentence_used}\n"


        prompt = f"""You are an Agent working with a team of {species} on a project for the company.
Your task is to {task_description}.
Your team consists of You and {group_size-1} {species} in total, each responsible for their own subtask.

You can communicate via a group chat with the other {specy} in order to ask questions or exchange information.
The group chat can also be used when members need someone to take over parts of their task.

You receive the following message:
From: {requester_name}
{to_line}
Message: {requester_message}
{cost_block}
{process_block}
Do you choose to take over the remaining work of {requester_name}?
Answer with "Yes" if you want to take over and "No" if not."""

        prompts.append(prompt)

    return prompts


def run_sharedtask_simulation(model_id, num_trials, num_max_bystander, process, agents = True, activated = "activated", costs = True):

    '''
    function runs simulation for *num_trials* times for given model
    token probabilites of answering yes are stored in dataframe and returned
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

        prompts = create_prompts_randomized(num_max_bystander, process, agents, activated, costs)

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
            answers[j] = total_yes_prob / total_mass
            print(answers[j])
        
        # save trial
        all_answers.append(answers)

    return pd.DataFrame(all_answers)



def run_and_store_sharedtask_simulation_all_models(model_dict, num_sim, n_max_byst, process = None, agents = True, activated = "activated", costs = True):

    '''
    function runs simulations for all models for the given condition
    results are stored in csv files
    '''

    for model in model_dict:
        
        clear_vram()

        df = run_sharedtask_simulation(model_dict[model], num_sim, n_max_byst, process, agents, activated, costs)

        if costs:
            main_folder = "results_with_costs"
        else:
            main_folder = "results_wo_costs"

        # create storage location
        if agents:
            folder = f"results/{main_folder}/agents"
        else:
            folder = f"results/{main_folder}/humans"

        if process == None:
            subfolder = "no process"
        elif process == "r":
            subfolder = f"responsibility diffusion/{activated}"
        elif process == "e":
            subfolder = f"evaluation apprehension/{activated}"
        elif process == "p": 
            subfolder = f"pluralistic ignorance/{activated}"
        elif process == "all":
            subfolder = f"all processes/{activated}"
        else:
            subfolder = "unnamed"
        file_name = f"answers_chat_{model}.csv"


        directory_path = os.path.join(folder, subfolder)
        # Create the folders if they don't exist
        os.makedirs(directory_path, exist_ok=True)
        
        storage = os.path.join(directory_path, file_name)

        df.to_csv(storage, index = False)
        print(f"✓ Saved results for {model} to {storage}")
