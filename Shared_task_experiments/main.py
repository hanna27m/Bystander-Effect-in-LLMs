import benchmark_simulation
import shared_task_simulation


MODEL_DICT = {"qwen8":"Qwen/Qwen3-8B",
             "qwen14": "Qwen/Qwen3-14B",
             "gemma12": "google/gemma-3-12b-it",
             "gemma27": "google/gemma-3-27b-it",
             "llama8": "meta-llama/Llama-3.1-8B-Instruct",
             "ministral14": "mistralai/Ministral-3-14B-Instruct-2512",
             "ministral8": "mistralai/Ministral-3-8B-Instruct-2512"}

N_SIM = 500
N_MAX_BYST = 20

VARIATIONS = {"humans": [None, "e", "r", "p", "all"],
             "agents": [None, "e", "r", "p", "all"]
             }

def run_benchmarks():
    '''
    function that runs benchmark tests for the three rules
    '''

    benchmark_simulation.run_and_store_benchmark_all_models(MODEL_DICT, N_SIM, "busy")
    benchmark_simulation.run_and_store_benchmark_all_models(MODEL_DICT, N_SIM, "specy")
    benchmark_simulation.run_and_store_benchmark_all_models(MODEL_DICT, N_SIM, "bystander")


def run_sharedtask_base():

    '''
    function that runs simulations for base experiment (agents vs humans + with vs w/o costs)
    '''

    for costs in [True, False]:

        for specy in VARIATIONS:

            if specy == "humans":
                shared_task_simulation.run_and_store_sharedtask_simulation_all_models(MODEL_DICT, N_SIM, N_MAX_BYST, 
                                                                                      process = None, agents = False, 
                                                                                      activated = "activated", costs = costs)
        
            else:
                shared_task_simulation.run_and_store_sharedtask_simulation_all_models(MODEL_DICT, N_SIM, N_MAX_BYST, 
                                                                                      process = None, agents = True, 
                                                                                      activated = "activated", costs = costs)


def run_sharedtask_process():

    '''
    function that runs simulation for process activation experiment (agents vs. humans, w/o costs, process activation)
    '''

    for specy in VARIATIONS:

        for process in VARIATIONS[specy]:

            if specy == "humans":
                shared_task_simulation.run_and_store_sharedtask_simulation_all_models(MODEL_DICT, N_SIM, N_MAX_BYST, 
                                                                                      process = process, agents = False, 
                                                                                      activated = "activated", costs = False)
        
            else:
                shared_task_simulation.run_and_store_sharedtask_simulation_all_models(MODEL_DICT, N_SIM, N_MAX_BYST,
                                                                                      process = process, agents = True, 
                                                                                      activated = "activated", costs = False)
    

    

def main():

    #run_benchmarks()
    #run_sharedtask_base()
    #run_sharedtask_process()
    print("Uncomment Simulation you want to run...")

if __name__ == "__main__":
    main()

            