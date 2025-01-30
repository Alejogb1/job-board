---
title: "Why does WandB training cause kernel crashes in JupyterLab?"
date: "2025-01-30"
id: "why-does-wandb-training-cause-kernel-crashes-in"
---
The integration of Weights & Biases (WandB) within a JupyterLab environment, particularly during intensive training loops, can precipitate kernel crashes primarily due to resource contention and asynchronous operations that are not always gracefully handled by the interactive notebook environment.  I've personally encountered this issue across various projects and have spent considerable time debugging its nuances.

The core problem stems from JupyterLab's single-threaded nature with respect to executing code within a kernel. While Jupyter's frontend is designed to be responsive, each cell's execution is processed sequentially in a single Python process (the kernel). When WandB logs data, especially large datasets like images or model checkpoints, it initiates background threads to communicate with the WandB server. This asynchronous activity, which is crucial for efficient logging, introduces a delicate dance between the synchronous kernel execution and the asynchronous I/O operations. If the kernel is under heavy computation load, particularly during deep learning training on a GPU, it may not allocate sufficient resources or time to these background processes, leading to timeouts, resource deadlocks, or ultimately, the dreaded kernel crash.

Furthermore, the way WandB manages its internal buffers and files can contribute to instability. During training, WandB may accumulate substantial log data, both in memory and on disk. If the underlying filesystem or memory management within Jupyter's kernel environment is not adequately provisioned, this can result in resource exhaustion, leading to a sudden halt in kernel operation. Kernel crashes can also occur when WandB tries to serialize and log complex Python objects containing references to external resources that become invalid due to the parallel execution between the training loop and the WandB logger.

To illustrate specific instances and potential solutions, I’ll present three code examples, highlighting different scenarios and mitigation strategies.

**Example 1: Basic WandB Initialization and Logging (Problematic)**

```python
import wandb
import time
import numpy as np

wandb.init(project="crash_experiment", name="basic_crash")

for i in range(10000):
    # Simulate heavy computation with random number generation and sleep
    data = np.random.rand(100000)
    time.sleep(0.0001)  # Simulate a small amount of work
    wandb.log({"iteration": i, "random_data_sum": np.sum(data)})

wandb.finish()
```

In this first example, I demonstrate a very basic training loop that attempts to log the iteration number and sum of a randomly generated data array at each step. While seemingly benign, this code, when executed inside a JupyterLab cell and coupled with a particularly complex model training in other cells of the notebook, has the potential to cause a kernel crash, especially if large or numerous calls to `wandb.log` saturate the kernel. The synchronous training loop is potentially starved due to WandB's background I/O tasks related to data logging.  If the kernel is already heavily loaded, the background I/O calls and data serialization processes of wandb.log create enough contention to lead to the kernel crash.

**Example 2: Batch Logging with `wandb.log(step=...)` (Improved Approach)**

```python
import wandb
import time
import numpy as np

wandb.init(project="crash_experiment", name="batch_logging")

for i in range(0, 10000, 10):
    # Simulate heavy computation
    batch_data = np.random.rand(10, 100000)
    time.sleep(0.001)

    # Aggregate data before logging. This reduces the frequency of logging
    batch_sums = np.sum(batch_data, axis=1)
    avg_batch_sum = np.mean(batch_sums)

    wandb.log({"iteration": i, "avg_batch_sum": avg_batch_sum}, step=i)


wandb.finish()
```

This second example presents a significant improvement by implementing batch logging. Instead of logging at every step, I collect a batch of data, compute an aggregated value, and then log it. This drastically reduces the frequency of `wandb.log` calls, lessening the burden on the kernel’s main thread. Additionally, using `step=i` with the wandb.log function is critical here. This step parameter ensures that WandB understands that the log entries are linked to specific points within the iterative loop, preventing mis-aligned data. This practice can improve the stability of Jupyter notebooks by reducing background processing and improving the overall performance in notebook cell executions where significant data is being logged by W&B during iterative executions.

**Example 3: Using `wandb.log` with `commit=False` and `wandb.log.flush()` (Explicit Control)**

```python
import wandb
import time
import numpy as np

wandb.init(project="crash_experiment", name="explicit_flush")

for i in range(10000):
    # Simulate heavy computation
    data = np.random.rand(100000)
    time.sleep(0.0001)

    # accumulate to local cache without writing to WandB until the flush
    wandb.log({"iteration": i, "random_data_sum": np.sum(data)}, commit=False)

    if i % 100 == 0:
      wandb.log({"flush_iteration": i}) # log the flush action too.
      wandb.log.flush() # sends the data in cache to wandb

wandb.finish()
```

The third example demonstrates an approach offering more granular control over the logging process. By setting `commit=False` within `wandb.log()`, I am essentially storing the logged data in a local buffer, preventing immediate write operations with WandB's backend. I then explicitly call `wandb.log.flush()` at specified intervals (every 100 steps), which forces the data cached by W&B to be written to the server. This approach further decouples the background processes from each individual log call, allowing for better performance and stability. Adding a log entry on the flush step also helps to visually see when flushes occur while observing the W&B interface during training. This is most useful if the flush interval is not trivial.

In summary, kernel crashes in JupyterLab during WandB training are primarily due to resource contention arising from the concurrent execution of heavy computational tasks and asynchronous I/O operations by WandB's logger.  The kernel, being single-threaded, can become overwhelmed with these parallel demands. Aggregating logs into batches, minimizing the frequency of logging, and utilizing `wandb.log.flush()` with `commit=False` are strategies that I have personally found to significantly mitigate this problem.

For further information, I suggest consulting the WandB documentation, specifically the sections on logging best practices and API references. Exploring academic articles discussing resource management within interactive Python environments could also provide insights into the underlying causes of these kernel crashes. Additionally, the Jupyter Notebook community forums can offer useful practical advice and solutions from other experienced users.
