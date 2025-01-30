---
title: "Why did Google Colab training finish after 10 epochs despite a 50-epoch setting?"
date: "2025-01-30"
id: "why-did-google-colab-training-finish-after-10"
---
The premature termination of a Google Colab training run, despite specifying a higher epoch count, almost invariably stems from either resource exhaustion or a runtime error masked within the training loop.  My experience troubleshooting similar issues over the past five years, primarily involving large-scale image classification and natural language processing tasks, points towards these two primary causes.  Let's examine the mechanics and provide potential solutions.


**1. Resource Exhaustion:** Google Colab, while generous with its free tier, imposes limitations on RAM, disk space, and runtime.  The default runtime often proves insufficient for extensive training runs, especially when dealing with substantial datasets or complex models.  Exceeding these limits triggers automatic termination, frequently without explicit error messages beyond a generic "session crashed" notification.  The 10-epoch completion might be a coincidental point at which the available resources were fully consumed.  The memory footprint increases with each epoch as the model's internal state (gradients, optimizer parameters, etc.) grows, and the same applies to disk usage, particularly if intermediate results or checkpoints are saved.


**2. Unhandled Exceptions:** A more insidious cause lies in unhandled exceptions within the training loop.  A seemingly minor error, such as an attempt to access an invalid index in a tensor or a division by zero, might halt execution without providing a readily apparent traceback in the Colab output. This is compounded by the asynchronous nature of some Colab operations.  Errors occurring within background processes might go unnoticed, leading to silent failures.  The 10-epoch termination could then be an artifact of the error occurring at that specific point in the training process. The error could be related to data loading, model architecture, or even the optimizer itself.


**Code Examples and Commentary:**

**Example 1:  Monitoring Resource Usage**

```python
import psutil
import gc
import torch

# ... your model and data loading code ...

for epoch in range(50):
    print(f"Epoch {epoch+1}/{50}")
    # Monitor memory usage before each epoch
    process = psutil.Process()
    mem_info = process.memory_info()
    rss = mem_info.rss / (1024 ** 2)  # Resident Set Size in MB
    print(f"RSS Memory Usage: {rss:.2f} MB")

    # ... your training loop code ...

    # Explicit garbage collection to reclaim memory
    gc.collect()
    torch.cuda.empty_cache() #If using GPU

    # Check for potential issues
    if rss > 10000: # Adjust threshold as needed
        print("WARNING: Memory usage exceeding threshold. Consider reducing batch size or model complexity.")
        break

```

This example illustrates proactive resource monitoring. By using the `psutil` library, we obtain real-time memory usage information. The `gc.collect()` and `torch.cuda.empty_cache()` functions attempt to free up memory explicitly. A custom threshold is introduced to warn the user about potential resource exhaustion before it leads to automatic termination.


**Example 2:  Robust Error Handling**

```python
import torch

# ... your model and data loading code ...

try:
    for epoch in range(50):
        print(f"Epoch {epoch+1}/{50}")
        try:
            # ... your training loop code ...
        except Exception as e:
            print(f"Error during epoch {epoch+1}: {e}")
            print("Stack Trace:", traceback.format_exc()) #import traceback at the beginning
            break  #Exit the loop if an error occurs
except Exception as e:
    print(f"Critical Error: {e}")
    print("Stack Trace:", traceback.format_exc())
```

This example showcases robust error handling using `try-except` blocks.  It captures both inner loop errors (within a single epoch) and outer loop errors (affecting the entire training process).  Including `traceback.format_exc()` provides detailed information about the location and nature of the error, greatly aiding debugging. This makes it possible to identify the source of the crash without relying on Colab's limited error reporting.


**Example 3:  Checkpoint Saving and Early Stopping**

```python
import torch

# ... your model and data loading code ...

for epoch in range(50):
    print(f"Epoch {epoch+1}/{50}")
    # ... your training loop code ...
    
    # Save checkpoint every few epochs
    if (epoch + 1) % 5 == 0:
        torch.save(model.state_dict(), f'checkpoint_epoch_{epoch+1}.pth')

    #Check for early stopping conditions.
    if validation_loss > previous_validation_loss:
        print("Early stopping criteria met.")
        break
```

This example demonstrates a strategy to mitigate the impact of a crash.  By saving checkpoints regularly, you can resume training from the latest saved state even if the runtime terminates unexpectedly.  This minimizes the loss of progress.  It also introduces an early stopping condition.  This could potentially provide a plausible explanation for the 10-epoch stop, if validation loss stopped decreasing or started increasing after 10 epochs, and early stopping was not correctly implemented.  The choice of when and how often to save depends on dataset and model characteristics.


**Resource Recommendations:**

For deeper understanding of Python exception handling, I recommend consulting the official Python documentation.  For memory management in PyTorch, the PyTorch documentation on tensors and memory management is invaluable.  Finally, a good understanding of operating system processes and resource usage would be beneficial for effectively diagnosing resource-related issues. These resources provide in-depth explanations and examples to build a strong foundation in these areas.  Furthermore, becoming proficient in debugging tools will allow you to effectively identify and resolve unforeseen errors.
