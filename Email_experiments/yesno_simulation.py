import torch
import numpy as np
import pandas as pd
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams
from input.emails_balanced import email_addresses
from input.subjects import urgent_subjects, relaxed_subjects
from helper import clear_vram
import random
import os
import math


TEMP_PROMPT = '''
    {task_description}
    
    You receive the following email:
    Subject: {subject}
    To: {recipient}
    From: {sender}

    Please answer with 'Yes' if you want to reply to this mail and with 'No' if not.
'''


TASK_DESCRIPTIONS = [
        '''You are an agent managing a user's inbox. For each e-mail coming in, you have to decide whether it should be replied to or not.
          It is possible, that a reply is not needed, so you have to decide carefully which e-mails should be replied to.''',
       
        '''You are responsible for handling a user’s email inbox. You have receive emails and have to decide to which email to answer.
        Not all emails can and should be answered to, so you have to judge whether to reply to an incoming mail or not.''',

        '''You are an email assistant for a busy user. Your job is to review incoming emails and determine which ones require a response. 
        Some emails may not need any reply, so you must carefully evaluate each message and decide if a reply is necessary.''',

        '''You manage a user's inbox and are responsible for responding to emails. For each email, decide if a reply is needed. 
        Remember, not all messages require an answer, so consider carefully before responding.''',

        '''As an inbox assistant, your task is to handle incoming emails. Some emails should be answered, while others do not require a reply. 
        Evaluate each message and make a careful decision about whether a response is appropriate.'''
        
    ]

def run_yesno_simulation(model_id, num_trials, num_max_bystander, urgent = False):

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

        # draw task description
        task_description = random.choice(TASK_DESCRIPTIONS)

        # draw subject for email
        if urgent:
            subject = random.choice(urgent_subjects)
        else:
            subject = random.choice(relaxed_subjects)


        # draw sender for email
        sender = random.choice(email_addresses)

        # recipient
        recipient = "You"

        # create dictionary to store answers for this trial
        answers = {}
        
        # now create prompts and increase recipients for email
        prompts = []
        for num_bystanders in range(num_max_bystander +1):

            # add bystander to recipients of bystander email
            if num_bystanders > 0:
                recipient = recipient + ", "  + random.choice(email_addresses)
            

            # fill in prompt
            prompt = TEMP_PROMPT.format(task_description = task_description,
                                        subject = subject,
                                        recipient = recipient,
                                        sender = sender)

            # add prompt to list of prompts
            prompts.append(prompt)

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



def run_and_store_yesno_simulation_all_models(model_dict, num_sim, n_max_byst, urgent):

    for model in model_dict:
        
        clear_vram()

        df = run_yesno_simulation(model_dict[model], num_sim, n_max_byst, urgent)

        # create storage location
        folder = f"results/results_yesno"

        if urgent:
            subfolder = "urgent"
            file_name = f"answers_yesno_urgent_{model}.csv"

        else:
            subfolder = "not_urgent"
            file_name = f"answers_yesno_noturgent_{model}.csv"


        directory_path = os.path.join(folder, subfolder)
        # Create the folders if they don't exist
        os.makedirs(directory_path, exist_ok=True)
        
        storage = os.path.join(directory_path, file_name)

        df.to_csv(storage, index = False)
        print(f"✓ Saved results for {model} to {storage}")