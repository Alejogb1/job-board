---
title: "How can I programmatically restart Colab runtime and rerun all cells when CUDA memory is exhausted?"
date: "2025-01-30"
id: "how-can-i-programmatically-restart-colab-runtime-and"
---
A common challenge when training large neural networks in Google Colab is encountering out-of-memory (OOM) errors related to CUDA. This typically halts execution, and manually restarting the runtime and re-executing cells is disruptive. I've found a programmatic solution, leveraging Python and Colab's environment, that automates this restart process upon a CUDA OOM. It's important to note this relies on detecting a specific error message that Colab reports.

The core idea is to wrap the primary execution logic of your notebook within a `try-except` block. This allows us to intercept the CUDA OOM error. When this specific error is caught, we then use the Colab-specific library, `google.colab`, to programmatically restart the runtime. Following the restart, we re-execute all cells in sequence. The entire process is self-contained within the notebook, requiring minimal external intervention.

The critical component for detecting the error is the traceback, specifically targeting the text pattern indicating a CUDA OOM. This error's specifics can vary slightly based on libraries and drivers. Through past debugging, the most reliable identifier I've encountered has been a traceback containing the phrase “CUDA out of memory”. Therefore, we’ll use this string as our detection trigger.

The restart and re-execution are handled by the `google.colab.runtime` module and its function `_restart_runtime()`, coupled with a Colab-specific execution command for running all cells. It is important to ensure that all necessary data and model checkpoints are saved before restarting, as the runtime environment gets completely reset.

Now, I'll provide three examples demonstrating how this mechanism can be implemented, progressing from a basic setup to a more robust, checkpoint-aware solution.

**Example 1: Basic Restart on OOM**

This first example provides the minimal code required to achieve an automated restart upon a detected CUDA OOM.

```python
import torch
import google.colab.runtime

def run_training():
    try:
        # Simulate a large operation that might cause OOM
        a = torch.rand(5000, 5000, 5000, device='cuda') # Very large tensor
        b = torch.rand(5000, 5000, 5000, device='cuda')
        c = a @ b # This might be too large to fit into GPU memory

    except RuntimeError as e:
         if "CUDA out of memory" in str(e):
            print("CUDA OOM detected, restarting runtime...")
            google.colab.runtime.restart() # Restart the runtime
            # Following this restart, Colab re-runs all cells
         else:
            raise # Re-raise if error isn't OOM

run_training()
```
**Commentary:**
Here, `run_training` encapsulates the training logic. Inside the `try` block, I’m intentionally allocating a large tensor `a` and `b`. Their subsequent matrix multiplication, `a@b`, is likely to cause an OOM on standard Colab GPUs. The `except` block catches the `RuntimeError`, and if the error message contains "CUDA out of memory," `google.colab.runtime.restart()` is called to restart the runtime.  The comment indicates that after the restart, Colab will automatically execute the notebook from the beginning. Critically, any other `RuntimeError` will still be thrown, ensuring we're not masking other problems. The `torch` library was imported to demonstrate a typical OOM error context.

**Example 2: Adding a Flag to Avoid Infinite Loops**

It's possible for the error to recur after restart, potentially leading to an infinite loop of restarts. To avoid this, we add a flag to track restarts.

```python
import torch
import google.colab.runtime
import os

restart_count = 0
MAX_RESTARTS = 2  # Prevent infinite restart loops

def run_training():
  global restart_count
  try:
      a = torch.rand(5000, 5000, 5000, device='cuda')
      b = torch.rand(5000, 5000, 5000, device='cuda')
      c = a @ b
  except RuntimeError as e:
    if "CUDA out of memory" in str(e):
      if restart_count < MAX_RESTARTS:
        restart_count += 1
        print(f"CUDA OOM detected, restarting (attempt {restart_count}/{MAX_RESTARTS})...")
        google.colab.runtime.restart()
      else:
        print("Max restarts reached. Exiting.")
        os._exit(0) # Exit gracefully to avoid repeated failures
    else:
        raise

run_training()
```

**Commentary:**

A global variable, `restart_count`, tracks how many times the runtime has restarted. `MAX_RESTARTS` sets the maximum allowed number of restarts. Each time the OOM error is caught,  `restart_count` increments. If this counter equals `MAX_RESTARTS`, the script terminates gracefully using `os._exit(0)` instead of restarting again. This provides a safeguard against potential infinite loops caused by persistent OOM errors. Note that `os` was added to the imports to allow for clean exit. This is crucial when your notebook fails to train even after a couple of attempts.

**Example 3: Saving Model Checkpoints Before Restart**

This final example demonstrates saving the model to a checkpoint before the runtime is restarted. It's imperative to save data before a restart as the environment will reset completely. Here we're simulating this save with a placeholder.

```python
import torch
import google.colab.runtime
import os
import time # Import time for sleep
import pickle # Import for saving fake models


restart_count = 0
MAX_RESTARTS = 2

def save_checkpoint(model, optimizer, epoch):
    # This would normally save the PyTorch model
    # Here we simulate a checkpoint with a dictionary
    checkpoint_data = {
        'model_state_dict': 'fake model data',
        'optimizer_state_dict': 'fake optimizer data',
        'epoch': epoch
    }

    with open(f"checkpoint_{epoch}.pkl", 'wb') as f:
      pickle.dump(checkpoint_data, f)
    print(f"Saved checkpoint at epoch {epoch}")


def run_training():
    global restart_count
    model = 'some_model_initialization' # Normally initialize
    optimizer = 'some_optimizer_initialization' # Normally init optimizer
    epoch = 0

    while True: # Loop simulates training loop
        try:
            epoch += 1
            print(f"Training Epoch {epoch}")

            a = torch.rand(5000, 5000, 5000, device='cuda')
            b = torch.rand(5000, 5000, 5000, device='cuda')
            c = a @ b
            # Normally here, you'd do training, backprop, etc.
            if epoch % 3 ==0: # Save checkpoint every three epochs
                save_checkpoint(model, optimizer, epoch)
                time.sleep(1) # Simulating saving and writing data

        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                if restart_count < MAX_RESTARTS:
                    restart_count += 1
                    print(f"CUDA OOM detected, restarting (attempt {restart_count}/{MAX_RESTARTS})...")
                    save_checkpoint(model, optimizer, epoch) # Save state before restart
                    time.sleep(1) # Simulating saving and writing data
                    google.colab.runtime.restart()
                else:
                    print("Max restarts reached. Exiting.")
                    os._exit(0)
            else:
                raise
        if epoch > 10 : # Exit criteria
          break

run_training()
```
**Commentary:**
This example includes an simulated training loop and a checkpoint saving mechanism. The `save_checkpoint` function simulates saving relevant training data, which normally involves actual `torch.save()` functions. We save a checkpoint before every restart, and every third epoch.  The `while True` loop mimics a continuous training procedure. If an OOM occurs, the checkpoint function is also called, guaranteeing the state is saved.  The simulated training loop is controlled with the `epoch` variable, and will terminate normally after 10 epochs. We've added the `time` and `pickle` modules to simulate the time taken to save a checkpoint. This version addresses the data loss aspect associated with programmatic restarts.

**Resource Recommendations:**

To deepen your understanding, I recommend consulting the official Colab documentation, which provides specific details about the `google.colab` library and its functions.  Review the PyTorch documentation to understand how models and optimizers should be saved and loaded correctly.  Furthermore, studying the documentation for your specific deep learning framework (e.g., TensorFlow, JAX) is also critical, as they may have specialized tools for checkpointing and fault tolerance. Also important is becoming proficient at understanding Python error tracebacks. Understanding error patterns will greatly assist in future problem solving.
