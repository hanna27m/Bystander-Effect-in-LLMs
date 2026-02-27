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


def create_task(count):

    '''
    function that randomly chooses task for simulation
    '''
    
    task_descriptions = [
        f"You are an agent managing a user's inbox. You have received {count} emails but only have the capacity to respond to one. Review them and select which email to answer.",
       
        f"You are responsible for handling a user’s email inbox. You have received the following {count} emails. Due to limited capacity, you may write and send a reply to only one of them. Please choose which email you will answer.",
        
        f"You are monitoring an active inbox under significant time constraints. Because you cannot reply to everyone, you must choose exactly one email out of the following {count} to reply to.",
        
        f"You are managing incoming emails for a user. You are able to compose a response to one email only. The remaining emails will not receive a reply from you. Choose the email you will answer.",
        
        f"Acting as an agent managing a user's mail inbox you are filtering a high volume of correspondence. You are permitted to draft only one response. Please choose the email from the {count} below that you will address."
    ]

    return random.choice(task_descriptions)


def generate_prompt(email_content, email_num, task):

    '''
    function that generates prompt out of email content, adding task, all emails and output specification together
    '''
    
    # build email blocks
    letters = "ABCDEFGHIJ" # maximum is ten emails
    email_blocks = []

    for c in range(email_num):

        block = (
            f"**Email {letters[c]}**\n"
            f"Subject: {email_content['subjects'][c]}\n"
            f"To: {email_content['recipients'][c]}\n"
            f"From: {email_content['sender'][c]}"
        )
        email_blocks.append(block)

    # build the footer
    active_letters = list(letters[:email_num])
    # Join all but the last with commas, then add the last with "or"
    formatted_letters = ", ".join([f'"{l}"' for l in active_letters[:-1]])
    choices = f"{formatted_letters} or \"{active_letters[-1]}\""

    footer = f"\nPlease only answer with the letter of the one email you choose, so either {choices}."
    
    # Combine everything
    return task + "\n" + "\n".join(email_blocks) + "\n" + footer


def run_choice_simulation(num_emails, model_id, num_trials, num_max_bystander, urgent = True):

    '''
    function that runs simulation for the given number of emails and trials for the model either with urgent subjects or non-urgent
    '''

    # first load model and tokenizer
    
    llm = LLM(
        model=model_id,
        tensor_parallel_size=1,  # Number of GPUs 
        gpu_memory_utilization=0.9,  # Use 90% of GPU memory
        max_model_len=2048,  # cap maximum model length
        trust_remote_code=True,  
        )

    tokenizer = llm.get_tokenizer()
    
    print(f"Model loaded: {model_id}")
    
    # letters of emails
    letters = "ABCDEFGHIJ"

    guided_choice = [l for l in letters[:num_emails]]
    # create structured output object
    output_choice = StructuredOutputsParams(
        choice=guided_choice
    )

    # Define sampling parameters 
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=1,  # Maximum length of generated response
        structured_outputs = output_choice
    )

    # now run simulation for *num_trials* times
    all_answers = [] # store answers for each trial
    for i in range(num_trials):

        # draw subjects for emails
        if urgent:
            subjects = [random.choice(urgent_subjects) for p in range(num_emails)]
        else:
            subjects = [random.choice(relaxed_subjects) for p in range(num_emails)]


        # draw senders for emails
        senders = [random.choice(email_addresses) for p in range(num_emails)]

        # recipients
        recipients = ["You" for p in range(num_emails)]

        # draw task
        task = create_task(num_emails)

        # always change bystander email, use modulo to determine
        bystander_email = i % num_emails

        # create dictionary to store answers for this trial
        answers = {}
        answers["bystander condition"] = letters[bystander_email]
        

        # now create prompts and increase recipients for one email
        prompts = []
        for num_bystanders in range(num_max_bystander +1):

            # add bystander to recipients of bystander email
            if num_bystanders > 0:
                recipients[bystander_email] = recipients[bystander_email] + ", " + random.choice(email_addresses)
            

            # format prompt
            # creating a dictionary with all information
            email_content = {"subjects": subjects, "recipients": recipients, "sender": senders}
            # call function to create prompt
            prompt = generate_prompt(email_content, num_emails, task)

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

            
        
        # Generate responses 
        print("\nGenerating responses...")
        outputs = llm.generate(formatted_prompts, sampling_params)
    
        # Print results
        print("\nResults:")
        for j, output in enumerate(outputs):
            prompt = prompts[j]
            response = output.outputs[0].text

            # clean response, by stripping whitespace, only taking first letter in uppercase
            clean_response = response.strip().upper()
            if len(clean_response) > 0:
                clean_response = clean_response[0]
                
            print(f"\n[Prompt {j+1}]: {prompt}")
            print(f"[Response]: {clean_response}")

            # add answer for this number of bystanders
            answers[j] = clean_response 
        
        # save trial
        all_answers.append(answers)

    return pd.DataFrame(all_answers)



def run_and_store_choice_simulation_all_models(model_dict, num_mails, urgent, num_sim, num_max_bystanders):

    for model in model_dict:
        
        clear_vram()
        
        df = run_choice_simulation(num_mails, model_dict[model], num_sim, num_max_bystanders, urgent = urgent)

        # create storage location
        folder = f"results/results_choice/{num_mails}"
        subfolder = "urgent" if urgent else "not_urgent"
        directory_path = os.path.join(folder, subfolder)
        
        # Create the folders if they don't exist
        os.makedirs(directory_path, exist_ok=True)
        
        file_name = f"answers_choose{num_mails}_{model}.csv"
        storage = os.path.join(directory_path, file_name)

        df.to_csv(storage, index = False)
        print(f"✓ Saved results for {model} to {storage}")