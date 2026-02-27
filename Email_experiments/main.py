import benchmark_simulation
import yesno_simulation
import choice_simulation
import advanced_choice_simulation

MODEL_DICT = {"qwen8":"Qwen/Qwen3-8B",
             "qwen14": "Qwen/Qwen3-14B",
             "gemma12": "google/gemma-3-12b-it",
             "gemma27": "google/gemma-3-27b-it",
             "llama8": "meta-llama/Llama-3.1-8B-Instruct",
             "ministral14": "mistralai/Ministral-3-14B-Instruct-2512",
             "ministral8": "mistralai/Ministral-3-8B-Instruct-2512"}

N_SIM = 1000
N_SIM_ADVANCED = 5000

N_MAX_BYSTANDER = 20

NUM_MAILS = [2,5,10]

def run_benchmarks():

    '''
    function that runs benchmark tests for all 3 tasks
    '''
    benchmark_simulation.run_and_store_benchmark_simulation_all_models(MODEL_DICT,  N_SIM, "urgency")
    benchmark_simulation.run_and_store_benchmark_simulation_all_models(MODEL_DICT, N_SIM, "bystander")
    benchmark_simulation.run_and_store_benchmark_simulation_all_models(MODEL_DICT, N_SIM, "spam")


def run_yesno():

    '''
    function that runs yes-no design for urgent and non-urgent condition
    '''

    yesno_simulation.run_and_store_yesno_simulation_all_models(MODEL_DICT, N_SIM, N_MAX_BYSTANDER, urgent = True)
    yesno_simulation.run_and_store_yesno_simulation_all_models(MODEL_DICT, N_SIM, N_MAX_BYSTANDER, urgent = False)


def run_choice():
    
    '''
    function that runs choice design for all number of emails and both urgency conditions
    '''

    # run for each choice set size
    for n_mail in NUM_MAILS:
        # urgent = True
        choice_simulation.run_and_store_choice_simulation_all_models(MODEL_DICT, num_mails = n_mail, urgent = True, num_sim = N_SIM, num_max_bystanders = N_MAX_BYSTANDER)
        # urgent = False
        choice_simulation.run_and_store_choice_simulation_all_models(MODEL_DICT, num_mails = n_mail, urgent = False, num_sim = N_SIM, num_max_bystanders = N_MAX_BYSTANDER)


def run_advanced_choice():
        
    '''
    function that runs choice design for all number of emails and both urgency conditions and with randomized urgency
    '''

    # run for each choice set size
    for n_mail in NUM_MAILS:
        # randomized urgency
        advanced_choice_simulation.run_and_store_advanced_choice_simulation_all_models(MODEL_DICT, num_mails = n_mail, num_sim = N_SIM)
        # urgent = True
        advanced_choice_simulation.run_and_store_advanced_choice_simulation_all_models_with_control(MODEL_DICT, num_mails = n_mail, num_sim = N_SIM, urgent = True)
        # urgent = False
        advanced_choice_simulation.run_and_store_advanced_choice_simulation_all_models_with_control(MODEL_DICT, num_mails = n_mail, num_sim = N_SIM, urgent = False)
        

def main():

    #run_benchmarks()
    #run_yesno()
    #run_choice()
    #run_advanced_choice()
    print("Uncomment Simulation you want to run...")

if __name__ == "__main__":
    main()