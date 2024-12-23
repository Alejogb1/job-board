---
title: "In AzureML, is `start_logging` a synchronous or asynchronous call?"
date: "2024-12-23"
id: "in-azureml-is-startlogging-a-synchronous-or-asynchronous-call"
---

, let’s get this one sorted out. Thinking back to a particularly challenging project a couple of years back, where we were orchestrating a complex series of machine learning pipelines in AzureML, the synchronous vs. asynchronous nature of `start_logging` became quite critical. We hit a snag with race conditions in our logging and debugging processes, and it forced us to really examine how AzureML handles these operations behind the scenes.

So, to address your question directly: in the AzureML SDK, the `start_logging` function, specifically when used within an `azureml.core.Run` context, is primarily *synchronous* in its execution from the perspective of your python script. This means that when you invoke `start_logging` from your script, the function will largely complete its initial setup and preparation before returning control to your main execution flow. However – and this is a critical caveat – the actual logging of metrics and artifacts isn't fully synchronous to the cloud. Let me clarify what I mean by this distinction.

The synchronous behavior mainly pertains to the initialization phase, ensuring the necessary loggers are registered and ready. This setup typically involves processes on the local compute where your python script is running. Therefore, after `start_logging` returns, your script will continue processing, and you can immediately start using `run.log()` or `run.log_list()` to record data. It's this initial setup that is synchronous, meaning the program flow halts until this phase is complete.

Now here's where the asynchronous aspect comes into play: the actual transmission of the logged metrics and artifacts to the AzureML service in the cloud happens *asynchronously*. After you've called `run.log(metric_name, metric_value)`, the SDK doesn't necessarily transmit this data instantaneously. Instead, it buffers the information. Periodically, or under specific conditions (like the completion of the run), this buffered data is uploaded to AzureML storage and becomes available for viewing within your AzureML workspace. This buffering is the core of the 'async' behavior. The point is your script will move on and perform other operations, while this data transfer is taking place in the background.

The impact of this behavior is significant. If you were, for instance, to abruptly terminate your run without giving it a chance to flush its buffers, you might lose some of the latest logging information, which was still buffered waiting to be uploaded. This was the very race condition we were battling a few years back. It also means that using logging in real-time or for immediate debugging outputs in a critical-path workflow isn’t viable; while the initial calls to `run.log()` appear to complete instantly, the data itself isn't immediately available remotely.

Here's a simplified breakdown:

1.  **`start_logging()`**: Synchronous setup of logging infrastructure within the context of your run.
2.  **`run.log()` or `run.log_list()`**: Primarily synchronous locally, adding data to an in-memory buffer.
3.  **Background Upload**: Asynchronous transfer of buffered data to the cloud.

It's crucial to bear in mind that although `start_logging()` itself is synchronous, the actual data transfer is asynchronous.

Now, let's look at a few code examples to underscore this behavior and provide some context.

**Example 1: Basic Synchronous Setup**

```python
from azureml.core import Workspace, Experiment, Run
from azureml.core.authentication import InteractiveLoginAuthentication
import os

# Load workspace config
ws = Workspace.from_config()

# Get experiment
experiment_name = 'my-logging-experiment'
exp = Experiment(workspace=ws, name=experiment_name)

# Start run
run = exp.start_logging()

# Log some metrics synchronously
run.log('accuracy', 0.85)
run.log('loss', 0.15)

# The start_logging operation is considered synchronous because once the function returns,
# subsequent commands such as logging can be called.
# These logging calls may seem instant, but their effect in the AzureML cloud happens asynchronously

# Complete the run (this is important for data to be properly uploaded)
run.complete()

print("Logging initiated and metrics logged.")
```

In this example, `run.complete()` is essential to signal the termination of the run. It triggers the upload of all buffered data. Notice how `run.log()` doesn't block your script’s execution.

**Example 2: Asynchronous Data Upload Illustration (Simulated)**

```python
from azureml.core import Workspace, Experiment, Run
from azureml.core.authentication import InteractiveLoginAuthentication
import os
import time

# Load workspace config
ws = Workspace.from_config()

# Get experiment
experiment_name = 'my-async-logging-experiment'
exp = Experiment(workspace=ws, name=experiment_name)

# Start run
run = exp.start_logging()

# Log a sequence of metrics
for i in range(5):
    run.log('iteration', i)
    run.log('progress', (i+1)*20)
    print(f"Logging data for iteration: {i}. Log command called, not immediately available remotely.")
    time.sleep(1) # simulate some processing between logging

print("Logging calls are complete. Data is buffered. Still not immediately available on cloud.")

# Complete the run
run.complete()

print("Run marked complete, data should start uploading.")

# Here we can't immediately access the results in the cloud, that's what we mean by async.
```

Here, the delay using `time.sleep()` isn’t a delay for `run.log()`. It illustrates that while `run.log()` returns quickly, the actual upload happens after a while and definitely asynchronously. You won't see the logged values immediately on the AzureML portal. It takes a few moments while they get uploaded to the storage.

**Example 3: Race Condition Avoidance with `run.complete()`**

```python
from azureml.core import Workspace, Experiment, Run
from azureml.core.authentication import InteractiveLoginAuthentication
import os

# Load workspace config
ws = Workspace.from_config()

# Get experiment
experiment_name = 'my-race-avoidance-experiment'
exp = Experiment(workspace=ws, name=experiment_name)

# Start run
run = exp.start_logging()

# Log a final metric
run.log('final_metric', 100)

# If you don't complete, some of the data in the buffer may not be uploaded.
# The below is critical in avoiding any data loss.
run.complete()

print("Run completed. Data should be uploaded shortly")
```

In this final example, it's vital to call `run.complete()` at the end. Without this, your last logged metric may be lost. If you don’t finish the run properly, the asynchronous process of uploading the log data might not have sufficient time to complete, which leads to lost metrics and frustrated debug sessions.

For further deep dives into the specifics of asynchronous behavior in Python and related patterns, I recommend consulting resources like:

*   **"Effective Python: 90 Specific Ways to Write Better Python"** by Brett Slatkin, particularly the sections related to concurrency and asynchronous programming.
*  **"Concurrent Programming in Python"** by Jason R. Briggs - for a more detailed discussion of the theory behind concurrency.
*   The official Python documentation on the `asyncio` library for a deeper understanding of asynchronous programming principles.
*   The official documentation for Azure Machine Learning SDK, under Run details, which provides some information on internal asynchronous behavior.

So, to recap, while `start_logging` is synchronous from a code execution perspective, the data transfer to AzureML cloud is asynchronous. This difference is crucial for effective logging and must be considered to avoid potential data loss and issues within your workflow.
