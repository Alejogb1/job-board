---
title: "In AzureML, does start_logging start asynchronous execution or synchronous execution?"
date: "2024-12-23"
id: "in-azureml-does-startlogging-start-asynchronous-execution-or-synchronous-execution"
---

Alright, let's unpack this. I remember battling this very question a few years back when migrating a sizable model training pipeline to AzureML. The 'synchronous vs. asynchronous' nature of `start_logging` – and its impact on your overall workflow – is indeed a crucial aspect to grasp, and it's not immediately obvious from the surface-level documentation.

To be precise, in the context of AzureML's `run` object and the logging mechanism, `start_logging` itself does **not** initiate asynchronous or synchronous execution of your *training code*. It primarily serves as a signal to AzureML that you're starting to log metrics and other metadata related to a specific run. The actual execution – be it synchronous or asynchronous – depends on how you've configured your training script submission via the `Experiment.submit()` or `ScriptRunConfig` methods.

Think of it like a recording device: `start_logging` activates the recording function. What you record depends entirely on what is happening concurrently, which can be synchronous or asynchronous. The 'record' button being pressed doesn’t influence whether the singer sings now or later; it just prepares to document what is happening.

In my experience, the confusion often arises because logging actions are tied to the *run object*, which *can* be associated with asynchronous job submissions. So, while `start_logging` is not itself an async operation, it often *appears* that way because of the asynchronous nature of most AzureML training jobs. Let’s break down the mechanics and see what's really happening.

The `start_logging()` function itself typically does two key things. Firstly, it creates the log file within the run’s logging directories. Secondly, it associates that log file and related metadata with the current run. It's a thin wrapper around underlying logging APIs that handle the data transportation and persistence to the Azure backend. The execution of your training code, from which these metrics are derived, happens outside the scope of `start_logging`.

Let's dive into some code examples to illustrate the point.

**Example 1: Synchronous Execution (Locally)**

This example shows a training script that is executed locally. Even though it uses start logging, the script executes in a synchronous fashion on your local machine.

```python
from azureml.core import Run
import time

if __name__ == "__main__":
    run = Run.get_context()
    run.start_logging()

    print("Starting training...")
    for epoch in range(3):
       time.sleep(2) # simulate some processing
       accuracy = 0.6 + (0.1 * epoch)
       loss = 1.0 - (0.2 * epoch)

       run.log("epoch", epoch)
       run.log("accuracy", accuracy)
       run.log("loss", loss)
       print(f"Epoch: {epoch}, Accuracy: {accuracy}, Loss: {loss}")

    print("Training completed")
    run.complete()
```

Here, `start_logging` sets up the mechanism. However, the script’s `for` loop, and thus the training simulation, happens sequentially and synchronously. The `run.log()` calls within the loop actively push data to the log file, which eventually will be picked up by Azure ML after run completion.

**Example 2: Asynchronous Execution (Remote via ScriptRunConfig)**

Now let's examine how this plays out with an asynchronous training job launched via `ScriptRunConfig`.

```python
from azureml.core import Workspace, Experiment, Environment, ScriptRunConfig
from azureml.core.compute import AmlCompute, ComputeTarget
from azureml.core.conda_dependencies import CondaDependencies
import os


ws = Workspace.from_config()
experiment = Experiment(workspace=ws, name="my-async-experiment")

compute_target_name = "my-compute-target"
if compute_target_name in ws.compute_targets:
  compute_target = ws.compute_targets[compute_target_name]
else:
  compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_DS3_V2', min_nodes=1, max_nodes=4)
  compute_target = ComputeTarget.create(ws, compute_target_name, compute_config)
  compute_target.wait_for_completion(show_output=True)

env = Environment(name='my-conda-env')
conda_dep = CondaDependencies.create(conda_packages=['scikit-learn','pandas'])
env.python.conda_dependencies = conda_dep

script_config = ScriptRunConfig(
    source_directory=".",
    script="training_script.py",  # Reference to the training_script.py from example 1
    environment=env,
    compute_target=compute_target
)

run = experiment.submit(config=script_config)
run.wait_for_completion(show_output=True)

```

The `training_script.py` is exactly the same as in Example 1. In this case, `experiment.submit()` initiates the training process on the specified compute target. The call returns immediately. The actual training happens *asynchronously* on the remote target. `start_logging` in the training script still initiates the logging mechanism, but the script now executes independently and potentially concurrently with the client that launched the job.

**Example 3: Asynchronous with a background thread in a local notebook**

Here’s an example showing a local training session using a thread to execute the training logic, which runs concurrently in a separate process within a notebook. The logging still happens on the primary context, but we can simulate an asynchronous behaviour.

```python
from azureml.core import Run
import threading
import time

def train_in_thread(run):
    print("Starting training in thread...")
    for epoch in range(3):
       time.sleep(2) # simulate some processing
       accuracy = 0.6 + (0.1 * epoch)
       loss = 1.0 - (0.2 * epoch)

       run.log("epoch", epoch)
       run.log("accuracy", accuracy)
       run.log("loss", loss)
       print(f"Thread Epoch: {epoch}, Accuracy: {accuracy}, Loss: {loss}")
    print("Training completed in thread")
    run.complete()


if __name__ == "__main__":
    run = Run.get_context()
    run.start_logging()

    training_thread = threading.Thread(target=train_in_thread, args=(run,))
    training_thread.start()

    print("Main thread continues while training runs in a separate thread")
    training_thread.join()
    print("Main thread completed")

```

Here, we created a new thread to do training simulation. `start_logging` in the context of the main thread is set up, but the training itself is done concurrently in the separate thread. In practical terms, the local notebook still is executed synchronously (top down), but now we are demonstrating concurrent execution, similar to the AzureML example.

In summary, `start_logging` is not a driver of asynchronous execution. It activates the *logging* process. The asynchronous or synchronous nature of the training workload depends entirely on where and how that workload is launched – be it locally, via remote compute, or by leveraging threads. The logging mechanism then captures data during that execution.

For a deeper dive, I strongly recommend reviewing these resources:

1.  **Microsoft Azure Machine Learning documentation**: Specifically, the sections on experiment tracking, `ScriptRunConfig`, and `Run` object. This is your primary reference for AzureML.
2.  **"Programming Concurrency on the JVM" by Venkat Subramaniam:** Although not directly AzureML related, understanding concurrency models in general helps frame your understanding of asynchronous execution. The principles are transferable.
3.  **"Operating System Concepts" by Abraham Silberschatz et al.:** A classic computer science text that explains the fundamentals of process management, threading, and concurrency. While it isn't specific to AzureML, it grounds the basic theory.

By combining practical experimentation, like the ones presented in the examples, with a good theoretical foundation from those references, you'll gain a solid grasp of how `start_logging` interacts within the larger scope of AzureML. This will enable you to build more robust and efficient ML pipelines.
