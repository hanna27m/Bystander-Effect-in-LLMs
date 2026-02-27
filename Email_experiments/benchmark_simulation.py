import torch
import numpy as np
import pandas as pd
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams
from input.emails_balanced import email_addresses
from input.subjects import urgent_subjects, relaxed_subjects, random_subjects, advertisement_subjects
from helper import clear_vram
import random
import os
import math


task_description_bystander_rule = """
    You are an agent that manages the mail inbox of a user.
    You receive emails with different numbers of recipients, but you should only answer to emails with you as the only recipient.
    These are the emails you receive, you can only answer one:
    """


task_description_spam_rule = """
    You are an agent that manages the mail inbox of a user.
    You receive emails and should filter out emails that are spam or advertisements, meaning that they are not replied to.
    These are the emails you receive and should filter:
    """

task_description_urgency_rule = """
    You are an agent that manages the mail inbox of a user.
    You receive emails that are more or less urgent to be replied to. You should always answer to the most urgent mail first.
    These are the emails you receive, you can only answer one:
    """

TASK_DESCRIPTIONS = {"spam": task_description_spam_rule,
                    "urgency": task_description_urgency_rule,
                    "bystander": task_description_bystander_rule}


GUIDED_CHOICE = {1: ["Yes", "No"],
                2: ["A", "B"],
                3: ["A", "B", "C"],
                4: ["A", "B", "C", "D"],
                5: ["A", "B", "C", "D", "E"],
                6: ["A", "B", "C", "D", "E", "F"],
                7: ["A", "B", "C", "D", "E", "F", "G"],
                8: ["A", "B", "C", "D", "E", "F", "G", "H"],
                9: ["A", "B", "C", "D", "E", "F", "G", "H", "I"],
                10: ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]}


def generate_prompt(email_content, email_num, task, reverse = False):
    '''
    function to generate the final prompt out of the email content depending on the task and the number of emails
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

    if email_num > 1:
        if reverse:
            footer = f"\nPlease only answer with the letter of the one email you filter out, so either {choices}. Do NOT use any formatting tokens."
        else:
            footer = f"\nPlease only answer with the letter of the one email you choose to answer, so either {choices}. Do NOT use any formatting tokens."
    else:
        footer = "\nPlease answer only with the token 'Yes' if you want to answer to this mail and 'No' if you do not want to answer. Do NOT use any formatting tokens."
    
    # Combine everything
    return task + "\n" + "\n".join(email_blocks) + "\n" + footer



def draw_recipients(num_emails, random_dist = True):

    '''
    function to generate the string of recipients
    number of recipients is either drawn randomly from a uniform distribution or is 1 for all emails
    '''

    if random_dist:
        num_recipients = [random.choice(range(2,21)) for e in range(num_emails)]
    else:
        num_recipients = [1 for e in range(num_emails)]

    all_recipients = []
    for num in num_recipients:
        
        recipient_string = "You"

        # add the additional recipients to the string
        if num > 1:
            for rec in range(num - 1):
                recipient_string = recipient_string + ", " + random.choice(email_addresses)

        all_recipients.append(recipient_string)

    return all_recipients


def run_benchmark_simulation(model_id, num_trials, task):

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

    # now run simulation for *num_trials* times
    all_answers = [] # store answers for each trial
    
    for i in range(num_trials):

        # create prompts for each number of mails
        prompts = []
        # token ids for each target email
        token_ids = {}

        for num_emails in range(1, 11):

            # choose which email is the one that should be answered
            target_email = random.choice(range(num_emails))

            # if only one email, then model gets task to say yes or no dependent on task
            if num_emails == 1:
                if task == "spam":
                    correct_answer = "No"
                else:
                    correct_answer = "Yes"
            # otherwise correct answer is letter of target email
            else:
                correct_answer = letters[target_email]
            # encode token, use last one if multiple are produced
            token_id = tokenizer.encode(correct_answer, add_special_tokens=False)[-1]
            token_ids[num_emails] = token_id

            
            # draw random subject for emails
            subjects = [random.choice(relaxed_subjects) for p in range(num_emails)]
            # if task is spam, subject of target email is replaced by ad subject
            if task == "spam":
                subjects[target_email] = random.choice(advertisement_subjects)
            if task == "urgency":
                subjects[target_email] = random.choice(urgent_subjects)

            # draw senders for emails
            senders = [random.choice(email_addresses) for p in range(num_emails)]

            # recipients
            # special treatment for bystander rule
            if task == "bystander":
                recipients = draw_recipients(num_emails)
                # overwrite recipient of target email with only "You"
                recipients[target_email] = "You"
            # use function to draw always 1 recipient
            else:
                recipients = draw_recipients(num_emails, random_dist=False)

            # format prompt
            # creating a dictionary with all information
            email_content = {"subjects": subjects, "recipients": recipients, "sender": senders}
            # call function to create prompt with correct task description
            task_description = TASK_DESCRIPTIONS[task]
            if task == "spam":
                prompt = generate_prompt(email_content, num_emails, task_description, reverse=True)
            else:
                prompt = generate_prompt(email_content, num_emails, task_description, reverse=False)
                
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

        # create dictionary to store answers for this trial
        answers = {}
        
        # Generate responses separately for each num_emails to use different guided_choice
        print("\nGenerating responses...")
        for j, formatted_prompt in enumerate(formatted_prompts):
            num_emails = j + 1
            # extract options to choose from
            guided_choice = GUIDED_CHOICE[num_emails]
            # create structured output object
            output_choice = StructuredOutputsParams(
                choice=guided_choice
            )
            # Define sampling parameters with guided choice
            sampling_params = SamplingParams(
                temperature=0.0,
                max_tokens=1,  # Maximum length of generated response
                logprobs=10,
                structured_outputs = output_choice
            )
            
            # Generate output
            output = llm.generate(
                [formatted_prompt],
                sampling_params=sampling_params
            )[0]
            
            prompt = prompts[j]
            response = output.outputs[0].text
            
            print(f"\n[Prompt {j+1}]: {prompt}")
            print(f"[Response]: {response}")
            print(f"[Guided choice]: {guided_choice}")

            # default value 
            prob = 0.0
            
            if output.outputs[0].logprobs:
                
                first_token_logprobs = output.outputs[0].logprobs[0]
                print(f"First token logprobs: {first_token_logprobs}")
            
                tid = token_ids[num_emails]
                logprob_obj = first_token_logprobs.get(tid)
                if logprob_obj is not None:
                    prob = math.exp(logprob_obj.logprob)
                    print(f"Token {tid}: prob = {prob}")
            
            answers[num_emails] = prob
        
        # save trial
        all_answers.append(answers)

    return pd.DataFrame(all_answers)



def run_and_store_benchmark_simulation_all_models(model_dict, num_sim, task):

    for model in model_dict:
        
        clear_vram()

        df = run_benchmark_simulation(model_dict[model], num_sim, task)

        # create storage location
        folder = f"results/results_benchmark"
        subfolder = task
        directory_path = os.path.join(folder, subfolder)
        
        # Create the folders if they don't exist
        os.makedirs(directory_path, exist_ok=True)
        
        file_name = f"answers_benchmark_{task}_{model}.csv"
        storage = os.path.join(directory_path, file_name)

        df.to_csv(storage, index = False)
        print(f"✓ Saved results for {model} to {storage}")