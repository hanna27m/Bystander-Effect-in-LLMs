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


def draw_recipients(num_emails):

    '''
    function that draws number of recipients for each email from power law distribution and creates recipient string
    '''

    # draw number of recipients from zipf law with exponent = 1.86 (based on research)
    alpha = 1.86
    num_recipients = np.random.zipf(alpha, size = num_emails)
    # clip to maximum of 20
    num_recipients = np.clip(num_recipients, 1, 20)

    all_recipients = []
    for num in num_recipients:
        # if number of recipients is 1, only recipient is You
        recipient_string = "You"
        # else additional recipients are added
        if num > 1:
            for rec in range(num - 1):
                recipient_string = recipient_string + ", " + random.choice(email_addresses)

        all_recipients.append(recipient_string)

    return all_recipients, num_recipients


def compare_mails_with_control(letter_probs, num_recipients, trial_id, pos_control):

    '''
    function that compares each letter probability to the one of the control email and creates a row for each comparison
    '''
    
    rows = []

    # extract values for the control email
    probs_control = letter_probs[pos_control] 
    bystander_control = num_recipients[pos_control]

    for i in range(len(letter_probs)):

        # do not compare control email to itself
        if i == pos_control:
            continue
        else:

            delta_prob = letter_probs[i] - probs_control
            delta_b = num_recipients[i] - bystander_control
            delta_pos = i - pos_control

            rows.append({
                "trial": trial_id,
                "i": i,
                "control": pos_control,
                "prob_i": letter_probs[i],
                "prob_control": probs_control,
                "delta_prob": delta_prob,
                "delta_b": delta_b,
                "delta_pos": delta_pos
            })

    return rows


def run_advanced_choice_simulation_with_control(num_emails, model_id, num_trials, urgent = False):

    '''
    function that runs simulation for model and condition num_trials times with a control email
    '''

    # first load model and tokenizer
    
    llm = LLM(
        model=model_id,
        tensor_parallel_size=1,  # Number of GPUs to use 
        gpu_memory_utilization=0.9,  # Use 90% of GPU memory
        max_model_len=2048,  # cap maximum model length
        trust_remote_code=True,  # Required for some models
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

    # Define sampling parameters with guided choice
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=1,  # Maximum length of generated response
        logprobs = num_emails * 2,
        structured_outputs = output_choice
    )

    # get token ids for letters of emails
    token_ids = {}
    for e in range(num_emails):
        letter = letters[e]
        id = tokenizer.encode(letter, add_special_tokens=False)[-1]
        # store token id in dictionary with corresponding letter
        token_ids[letter] = id


    # now run simulation for *num_trials* times
    all_answers = [] # store answers for each trial
    for i in range(num_trials):

        # create one control email with varying position
        pos_control = i % num_emails

        # draw subjects for emails randomly from urgent or non-urgent subjects dependent on condition
        subjects = [
            random.choice(urgent_subjects) if urgent == True else random.choice(relaxed_subjects)
            for u in range(num_emails)
        ]

        # draw senders for emails
        senders = [random.choice(email_addresses) for p in range(num_emails)]

        # draw number of recipients per email
        recipients, num_recipients = draw_recipients(num_emails)

        # overwrite recipients for control email
        recipients[pos_control] = "You" # control only send to 1 person
        num_recipients[pos_control] = 1 

        # draw task
        task = create_task(num_emails)
        
        # now create prompt
        # creating a dictionary with all information
        email_content = {"subjects": subjects, "recipients": recipients, "sender": senders}
        # call function to create prompt
        prompt = generate_prompt(email_content, num_emails, task)

        # now format the prompts
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
            
        # Generate responses (batch processing!)
        print("\nGenerating responses...")
        outputs = llm.generate(formatted, sampling_params)
    
        # Print results
        print("\nResults:")
        for j, output in enumerate(outputs):
            
            response = output.outputs[0].text
            
            if output.outputs[0].logprobs:
                
                # Get logprobs for the very first generated token
                first_token_logprobs = output.outputs[0].logprobs[0]

                # check for probabilities of the different letters
                letter_probs = []
                # going through the different token ids stored in the dictionary
                for letter, id in token_ids.items():

                    prob = 0.0 # default value
                    # get probability of token id for letter
                    logprob_obj = first_token_logprobs.get(id)
                    if logprob_obj is not None:
                        prob = math.exp(logprob_obj.logprob)

                    letter_probs.append(prob)


            # now make a pairwise comparison between the token probabilities and number of bystanders
            result = compare_mails_with_control(letter_probs, num_recipients, i, pos_control)

            for r in result:
                all_answers.append(r)
                
            
            print(f"\n[Prompt {j+1}]: {prompt}")
            print(f"[Response]: {response}")

        

    return pd.DataFrame(all_answers)


def run_advanced_choice_simulation(num_emails, model_id, num_trials):

    '''
    function that runs simulation for model num_trials times, urgency is also randomized
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

    
    # letters of emails
    letters = "ABCDEFGHIJ"

    guided_choice = [l for l in letters[:num_emails]]
    # create structured output object
    output_choice = StructuredOutputsParams(
        choice=guided_choice
    )


    # Define sampling parameters with guided choice
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=1,  # Maximum length of generated response
        logprobs = num_emails * 2,
        structured_outputs = output_choice
    )

    # get token ids for letters of emails
    token_ids = {}
    for e in range(num_emails):
        letter = letters[e]
        id = tokenizer.encode(letter, add_special_tokens=False)[-1]
        # store token id in dictionary with corresponding letter
        token_ids[letter] = id

    
    # now run simulation for *num_trials* times
    all_answers = [] # store answers for each trial
    for i in range(num_trials):

        # draw subjects for emails randomly from urgent and non-urgent subjects
        urgency = [np.random.choice([0,1], p=[0.7, 0.3]) for p in range(num_emails)]
        subjects = [
            random.choice(urgent_subjects) if u == 1 else random.choice(relaxed_subjects)
            for u in urgency
        ]

        # draw senders for emails
        senders = [random.choice(email_addresses) for p in range(num_emails)]

        # draw number of recipients per email
        recipients, num_recipients = draw_recipients(num_emails)

        # draw task
        task = create_task(num_emails)
        
        # now create prompt
        # creating a dictionary with all information
        email_content = {"subjects": subjects, "recipients": recipients, "sender": senders}
        # call function to create prompt
        prompt = generate_prompt(email_content, num_emails, task)

        # now format the prompts
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
            
        # Generate responses (batch processing!)
        print("\nGenerating responses...")
        outputs = llm.generate(formatted, sampling_params)
    
        # Print results
        print("\nResults:")
        for j, output in enumerate(outputs):
            
            response = output.outputs[0].text
            
            if output.outputs[0].logprobs:
                
                # Get logprobs for the very first generated token
                first_token_logprobs = output.outputs[0].logprobs[0]

                # check for probabilities of the different letters
                letter_probs = []
                
                # going through the different token ids stored in the dictionary
                for letter, id in token_ids.items():
                
                    prob = 0.0 # default value
                    # get probability of token id for letter
                    logprob_obj = first_token_logprobs.get(id)
                    if logprob_obj is not None:
                        prob = math.exp(logprob_obj.logprob)

                    letter_probs.append(prob)



            # now add a row for each email
            for email in range(num_emails):
                
                all_answers.append({
                "trial": i,
                "pos" : email,
                "prob": letter_probs[email],
                "bystander": num_recipients[email],
                "urgency": urgency[email],
                "mean_context": sum(num_recipients) / len(num_recipients),
                "sd_context": np.std(num_recipients, ddof=1)
            })
                
            
            print(f"\n[Prompt {j+1}]: {prompt}")
            print(f"[Response]: {response}")

        

    return pd.DataFrame(all_answers)


def run_and_store_advanced_choice_simulation_all_models(model_dict, num_mails, num_sim):

    '''
    runs simulation for all models with randomized urgency
    '''

    for model in model_dict:
        
        clear_vram()
        
        df = run_advanced_choice_simulation(num_mails, model_dict[model], num_sim)

        # create storage location
        folder = f"results/results_advanced_choice/{num_mails}"
        subfolder = "both"
        directory_path = os.path.join(folder, subfolder)
        
        # Create the folders if they don't exist
        os.makedirs(directory_path, exist_ok=True)
        
        file_name = f"answers_choose{num_mails}_{model}.csv"
        storage = os.path.join(directory_path, file_name)

        df.to_csv(storage, index = False)
        print(f"✓ Saved results for {model} to {storage}")


def run_and_store_advanced_choice_simulation_all_models_with_control(model_dict, num_mails, num_sim, urgent = False):

    '''
    runs simulation with fixed urgency for all models (with control email)
    '''

    for model in model_dict:
        
        clear_vram()
        
        df = run_advanced_choice_simulation_with_control(num_mails, model_dict[model], num_sim, urgent)

        # create storage location
        folder = f"results/results_advanced_choice/{num_mails}"
        if urgent:
            subfolder = "urgent"
        else:
            subfolder = "not_urgent"
        directory_path = os.path.join(folder, subfolder)
        
        # Create the folders if they don't exist
        os.makedirs(directory_path, exist_ok=True)
        
        file_name = f"answers_choose{num_mails}_{model}.csv"
        storage = os.path.join(directory_path, file_name)

        df.to_csv(storage, index = False)
        print(f"✓ Saved results for {model} to {storage}")