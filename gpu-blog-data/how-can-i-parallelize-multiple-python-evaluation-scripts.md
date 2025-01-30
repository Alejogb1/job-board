---
title: "How can I parallelize multiple Python evaluation scripts with varying parameters across multiple GPUs?"
date: "2025-01-30"
id: "how-can-i-parallelize-multiple-python-evaluation-scripts"
---
Here’s my approach based on past experiences scaling machine learning workflows. The core challenge lies in efficiently distributing independent computational tasks across available GPU resources while managing the inherent limitations of Python's Global Interpreter Lock (GIL) and the need for data locality. My typical solution revolves around leveraging process-based parallelism with tools that are both GPU-aware and adept at handling diverse script executions.

The first hurdle is Python’s GIL, which prevents true multi-threading. To maximize GPU utilization, therefore, I rely on the `multiprocessing` module. This module spawns separate Python processes, each with its own GIL, allowing us to effectively parallelize our evaluation scripts. Furthermore, I must integrate this with a framework that understands GPU availability and facilitates workload assignment.

In essence, my typical workflow involves these steps: First, generating a list of parameter configurations I intend to evaluate, each one corresponding to a single evaluation run. Second, creating a processing pool from `multiprocessing`, configured to the number of available GPUs. Third, developing a wrapper function that encapsulates each evaluation script execution. This function receives a parameter configuration as an argument, performs setup if necessary (e.g., setting CUDA device visibility), executes the script, and handles return values or any necessary aggregation. Finally, using the processing pool’s `map` or `imap` functions to apply this wrapper to the list of parameter configurations and effectively distribute the evaluation jobs.

Here’s how I would implement such a pipeline.

**Code Example 1: Parameter Configuration and Basic Setup**

```python
import multiprocessing
import os
import subprocess
import json

def generate_parameter_combinations(base_params, param_ranges):
    """
    Generates a list of parameter dictionaries from given base parameters and ranges.
    """
    combinations = [base_params]
    for param, range_ in param_ranges.items():
      new_combinations = []
      for existing in combinations:
        for value in range_:
          new_combination = existing.copy()
          new_combination[param] = value
          new_combinations.append(new_combination)
      combinations = new_combinations

    return combinations


def evaluation_wrapper(params):
    """
    Executes an evaluation script with given parameters.
    """
    device_id = int(multiprocessing.current_process().name[-1]) - 1 # Extract device from process name
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device_id)

    try:
      # Assuming evaluation script accepts parameters via JSON string.
      command = ["python", "evaluation_script.py", "--params", json.dumps(params)]
      result = subprocess.run(command, capture_output=True, text=True, check=True)
      return result.stdout # Process and return result, error checking removed for brevity
    except subprocess.CalledProcessError as e:
        return f"Error: Command '{' '.join(e.cmd)}' failed with return code {e.returncode}. \n STDERR: {e.stderr} \n STDOUT: {e.stdout}"
    except Exception as e:
        return f"Exception during processing: {e}"

if __name__ == '__main__':
    base_params = {"model_name": "resnet50", "dataset": "imagenet"}
    param_ranges = {
        "learning_rate": [0.001, 0.0001],
        "batch_size": [32, 64],
        "epochs": [5, 10]
    }

    parameter_combinations = generate_parameter_combinations(base_params, param_ranges)

    num_gpus = len(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(','))  # Detect GPUs dynamically
    if num_gpus <= 1:
        num_gpus = multiprocessing.cpu_count() # Fallback to cpu cores when not running on GPU

    with multiprocessing.Pool(processes=num_gpus) as pool:
        results = pool.map(evaluation_wrapper, parameter_combinations)

    for i, result in enumerate(results):
        print(f"Results for configuration {i+1}: {parameter_combinations[i]} \n {result}")
```

**Commentary:**

This first example illustrates the fundamental structure.  `generate_parameter_combinations` builds a list of dictionaries, each dict being a distinct combination of hyperparameters. The `evaluation_wrapper` function is crucial; it sets the `CUDA_VISIBLE_DEVICES` environment variable, allowing each process to utilize a unique GPU (assuming they are available, else falls back to number of cpu cores), then calls a dummy `evaluation_script.py` with the parameter combination using `subprocess.run` and returns the standard output. The main block then defines sample parameters, dynamically retrieves the number of available GPUs, and initiates a process pool. Crucially, the `pool.map` function applies `evaluation_wrapper` to each parameter configuration, effectively distributing the workload.

**Code Example 2:  Incorporating GPU Selection Strategies and Logging**

```python
import multiprocessing
import os
import subprocess
import json
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def evaluation_wrapper_gpu_select(params, gpu_assignments):
    """
    Executes an evaluation script with given parameters, assigning specific GPU resources.
    """

    process_id = int(multiprocessing.current_process().name.split("-")[-1]) -1
    device_id = gpu_assignments[process_id % len(gpu_assignments)] # Cyclic Assignment of GPUs
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device_id)

    start_time = time.time()

    try:
        command = ["python", "evaluation_script.py", "--params", json.dumps(params)]
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        end_time = time.time()
        duration = end_time - start_time
        logging.info(f"Evaluation with params: {params} completed in {duration:.2f} seconds on GPU {device_id}.")
        return result.stdout

    except subprocess.CalledProcessError as e:
        logging.error(f"Error with command: '{' '.join(e.cmd)}' on GPU {device_id}. Error: {e}")
        return f"Error: Command '{' '.join(e.cmd)}' failed with return code {e.returncode}. \n STDERR: {e.stderr} \n STDOUT: {e.stdout}"
    except Exception as e:
        logging.exception(f"Exception during processing with params: {params} on GPU {device_id}: {e}")
        return f"Exception during processing: {e}"

if __name__ == '__main__':

    base_params = {"model_name": "transformer", "task": "translation"}
    param_ranges = {
        "learning_rate": [0.001, 0.0005, 0.0001],
        "batch_size": [16, 32, 64],
        "max_length": [128, 256]
    }
    parameter_combinations = generate_parameter_combinations(base_params, param_ranges)

    gpu_list = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(',')  # Fetch CUDA_VISIBLE_DEVICES
    gpu_ids = [int(g) for g in gpu_list if g.strip().isdigit()]

    if not gpu_ids: # if no GPU available, use a single core or fallback to cpu cores.
      logging.warning("No GPUs detected. Falling back to single CPU core processing")
      gpu_ids=[None]

    num_gpus = len(gpu_ids) if gpu_ids[0] != None else multiprocessing.cpu_count()

    with multiprocessing.Pool(processes=num_gpus) as pool:
        results = pool.starmap(evaluation_wrapper_gpu_select, [(params, gpu_ids) for params in parameter_combinations]) # Pass gpu assignments to each process

    for i, result in enumerate(results):
        print(f"Results for configuration {i+1}: {parameter_combinations[i]} \n {result}")
```

**Commentary:**

Here, I've refined GPU assignment. The `evaluation_wrapper_gpu_select` now accepts a list of `gpu_assignments` and assigns GPUs cyclically using process id. This prevents multiple processes from hammering a single GPU, ensuring more balanced usage. Further, I’ve added basic logging to monitor the start and end times of each evaluation, making it easier to identify bottlenecks or issues. I also handle the case where no GPUs are available and allow for running on CPU cores as a fallback. The usage of `starmap` in the process pool allows me to pass both the parameter combination and gpu_id to the wrapper.

**Code Example 3: Inter-Process Communication (IPC) with Return Queues**

```python
import multiprocessing
import os
import subprocess
import json
from queue import Empty
import time
import logging


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def evaluation_worker(task_queue, result_queue, gpu_assignments):
    """
    Worker process for executing evaluation scripts.
    """
    process_id = int(multiprocessing.current_process().name.split("-")[-1]) -1
    device_id = gpu_assignments[process_id % len(gpu_assignments)]
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device_id)

    while True:
        try:
            params = task_queue.get(timeout=1) # poll the queue for a task (timeout prevents workers from waiting indefinitely)

            start_time = time.time()
            command = ["python", "evaluation_script.py", "--params", json.dumps(params)]
            result = subprocess.run(command, capture_output=True, text=True, check=True)
            end_time = time.time()
            duration = end_time - start_time
            logging.info(f"Evaluation with params: {params} completed in {duration:.2f} seconds on GPU {device_id}.")
            result_queue.put((params, result.stdout))


        except Empty: # breaks out of the loop if queue is empty for a certain timeout
           break;
        except subprocess.CalledProcessError as e:
            logging.error(f"Error with command: '{' '.join(e.cmd)}' on GPU {device_id}. Error: {e}")
            result_queue.put((params, f"Error: Command '{' '.join(e.cmd)}' failed with return code {e.returncode}. \n STDERR: {e.stderr} \n STDOUT: {e.stdout}"))
        except Exception as e:
            logging.exception(f"Exception during processing with params: {params} on GPU {device_id}: {e}")
            result_queue.put((params, f"Exception during processing: {e}"))



if __name__ == '__main__':

    base_params = {"model_type": "bert", "mode": "finetuning"}
    param_ranges = {
        "learning_rate": [2e-5, 5e-5, 1e-5],
        "batch_size": [8, 16, 32],
        "gradient_accumulation_steps": [1, 2, 4]
    }
    parameter_combinations = generate_parameter_combinations(base_params, param_ranges)

    gpu_list = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(',')
    gpu_ids = [int(g) for g in gpu_list if g.strip().isdigit()]

    if not gpu_ids:
      logging.warning("No GPUs detected. Falling back to single CPU core processing")
      gpu_ids=[None]

    num_gpus = len(gpu_ids) if gpu_ids[0] != None else multiprocessing.cpu_count()

    task_queue = multiprocessing.Queue()
    result_queue = multiprocessing.Queue()

    for params in parameter_combinations:
        task_queue.put(params) # load the queue with parameter combinations.

    processes = []
    for _ in range(num_gpus):
        process = multiprocessing.Process(target=evaluation_worker, args=(task_queue, result_queue, gpu_ids))
        processes.append(process)
        process.start()

    for process in processes: # Wait for all processes to finish processing tasks.
       process.join()


    while not result_queue.empty():
        params, result = result_queue.get()
        print(f"Results for config: {params} \n {result}")
```

**Commentary:**

This final example showcases an alternative, using explicit task and result queues.  The `evaluation_worker` consumes tasks from the `task_queue` and puts results into the `result_queue`, rather than using `pool.map`. This provides more flexibility when the number of tasks outstrips the number of available workers, a scenario where the `pool.map` approach might load all tasks into memory simultaneously and delay processing. Moreover, I added a timeout for the queue getting to prevent the process from waiting indefinitely on a queue that will not receive new tasks. The main block now manages worker processes, loads the queue, and retrieves results from the queue after the process has finished. This allows for more dynamic control over workload management.

**Resource Recommendations:**

For further study, I recommend exploring resources that cover the following topics: advanced usage of Python’s `multiprocessing` module, including task queues and process pools; strategies for efficient GPU utilization in deep learning contexts; and best practices for logging and error handling within parallel processing environments. Specifically look into best practices with task queue management within multiprocessing pools. Finally review any documentation on the specific deep learning frameworks you use, as they often provide utilities for distributed and parallel training. Frameworks like PyTorch have their own `DistributedDataParallel` features, which is a common way to tackle multi-gpu training. The examples provided here serve as a baseline for a scalable evaluation pipeline, but they are not a complete alternative to framework provided utility functionality for distributed training when training and evaluation is within a single program.
