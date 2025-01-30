---
title: "Can TensorFlow SavedModel GPU memory be allocated per process without a session?"
date: "2025-01-30"
id: "can-tensorflow-savedmodel-gpu-memory-be-allocated-per"
---
TensorFlow’s SavedModel format facilitates model deployment, yet the handling of GPU memory during inference outside of explicit session management presents unique challenges. Specifically, direct allocation of GPU memory on a per-process basis without an active TensorFlow session is not the typical workflow and requires careful understanding of the underlying mechanics. While SavedModels themselves don't directly manage GPU memory, the code used to load and execute them does.

The conventional approach within TensorFlow involves creating a `tf.Session` or using `tf.compat.v1.Session` in older codebases, to define the computational graph and allocate GPU resources. The session then serves as the context for model execution. However, a desire to manage GPU memory per process arises in scenarios where multiple independent inference applications are running concurrently, potentially leading to resource conflicts if all rely on a single global TensorFlow session. It's crucial to understand that TensorFlow allocates memory when the first operation is pushed to the GPU, not when the model is loaded. The mere existence of a SavedModel on disk does not directly impact GPU memory.

Attempting to circumvent sessions and directly manipulate GPU memory allocation through settings outside of the session is inherently problematic. TensorFlow relies on its own memory management mechanism, tightly integrated with the session. However, you can exert some control by manipulating the `tf.config.experimental.set_memory_growth` option or the equivalent in earlier versions using `tf.compat.v1.GPUOptions`. This setting directs TensorFlow to allocate memory incrementally, rather than claiming all available memory at once. This doesn’t allocate memory per process in the sense of discrete memory pools, but rather allows multiple processes to co-exist by not being so aggressive in memory grabbing.

The key is that per-process isolation in terms of GPU memory is not achievable directly through manipulations *around* the session; isolation must be engineered by having *each process using its own session* with appropriately configured settings. Let's examine this through a series of code examples.

**Example 1: Standard Session Allocation**

This example demonstrates the standard way to load and use a SavedModel within a session, showcasing how GPU memory is implicitly managed through a global context.

```python
import tensorflow as tf
import numpy as np

# Assuming 'path/to/saved_model' exists with a valid SavedModel
saved_model_path = 'path/to/saved_model' 

# Loading the SavedModel
loaded_model = tf.saved_model.load(saved_model_path)
infer = loaded_model.signatures['serving_default']

# Creating a session to run the model
# This is where the allocation happens on the first run
with tf.compat.v1.Session() as sess: #For TensorFlow 1.x use tf.Session()
    
    # Sample Input Data
    input_data = np.random.rand(1, 224, 224, 3).astype(np.float32) 
    
    # Running the Inference
    output = infer(tf.constant(input_data))
    print("Inference output shape: ", output['output_0'].shape)
```
In this example, the line `with tf.compat.v1.Session() as sess:` or `with tf.Session() as sess:` dictates the context for computation.  The model, upon first execution of `infer` against a GPU tensor, triggers memory allocation within the TensorFlow context. This allocation occurs as needed based on the operation graph and may not be the full GPU memory capacity. This allocation is associated with the specific session. Multiple independent processes using this approach and the same GPU would compete for memory.

**Example 2: Memory Growth Configuration**

Here, I demonstrate how `tf.config.experimental.set_memory_growth` helps manage resource usage and prevent excessive memory allocation, thus allowing co-existence with other processes. Note that you need to define which device should be used, and that multiple processes might use the same GPU.

```python
import tensorflow as tf
import numpy as np

# Assuming 'path/to/saved_model' exists with a valid SavedModel
saved_model_path = 'path/to/saved_model' 

# Loading the SavedModel
loaded_model = tf.saved_model.load(saved_model_path)
infer = loaded_model.signatures['serving_default']

# Enable Memory Growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
      # Currently, memory growth needs to be the same across GPUs
      for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
      logical_gpus = tf.config.list_logical_devices('GPU')
      print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
      # Memory growth must be enabled before GPUs have been initialized
      print(e)

# Creating a session to run the model
# This session's allocation behavior is now different
with tf.compat.v1.Session() as sess: #For TensorFlow 1.x use tf.Session()
    
    # Sample Input Data
    input_data = np.random.rand(1, 224, 224, 3).astype(np.float32) 
    
    # Running the Inference
    output = infer(tf.constant(input_data))
    print("Inference output shape: ", output['output_0'].shape)

```

By enabling memory growth via `tf.config.experimental.set_memory_growth(gpu, True)`, TensorFlow will initially allocate a minimal amount of GPU memory. As needed during the model execution, more memory will be requested in chunks. Critically, *each separate process must execute this configuration within its own session*. They would also ideally specify which logical GPU to use, if multiple exist, otherwise, all would contend to execute on the primary GPU. This strategy is key to enabling multiple processes to share the same GPU hardware without immediate memory contention. This approach is a crucial step in managing memory, and is not by itself memory allocation per process.

**Example 3: Per-Process Session & Configuration**

The most correct approach involves creating a unique session *per process*, and configuring the settings for that process within its own session configuration block, and is the closest we can get to per-process control. In practice, each program invoking this logic (e.g., a web server handling prediction requests, and a batch processing pipeline) would have *its own separate instance* of the code.

```python
import tensorflow as tf
import numpy as np
import os

# Assuming 'path/to/saved_model' exists with a valid SavedModel
saved_model_path = 'path/to/saved_model' 

def run_inference():
  # Enable Memory Growth, and configure the visible device.
  gpus = tf.config.list_physical_devices('GPU')
  if gpus:
      try:
          # Currently, memory growth needs to be the same across GPUs
          for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
          logical_gpus = tf.config.list_logical_devices('GPU')
          print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")

          # Select which GPU this process will use.
          os.environ["CUDA_VISIBLE_DEVICES"] = str(0) # For now we are using the primary GPU.
          
      except RuntimeError as e:
        # Memory growth must be enabled before GPUs have been initialized
        print(e)


  # Loading the SavedModel
  loaded_model = tf.saved_model.load(saved_model_path)
  infer = loaded_model.signatures['serving_default']


  # Creating a session to run the model * within the process, and after configuration *.
  with tf.compat.v1.Session() as sess: #For TensorFlow 1.x use tf.Session()
    
    # Sample Input Data
    input_data = np.random.rand(1, 224, 224, 3).astype(np.float32) 
    
    # Running the Inference
    output = infer(tf.constant(input_data))
    print("Process: ", os.getpid(), "Inference output shape: ", output['output_0'].shape)

if __name__ == '__main__':
  run_inference()
  # Example: Spawn another process to show the difference.
  os.system('python test.py &') #Assumes this script is `test.py`
```

In this example, if this script is run multiple times, each process will have its own Session context (because they'll be independent executions). Each process independently configures its settings before creating its session. If we assume this script is named `test.py`, we launch a second copy as a background task, showing two separate processes, each with their own session, independently using the GPU. `os.environ["CUDA_VISIBLE_DEVICES"]` is used to control which device each process will see, which is important for true memory isolation and performance.  Ideally, a more robust method of device assignment would be employed in a more complicated use case, particularly if the primary GPU was at full utilization.

**Recommendations:**

For those working with multiple TensorFlow applications, the key is to have each application utilize its own TensorFlow session, rather than sharing a single global one. This naturally separates their memory context. Moreover, memory growth must be enabled *within the context of the specific session* in each process. The `tf.config.experimental` namespace allows for low-level configurations, such as device usage settings. Understanding the interactions between these mechanisms, and how a TensorFlow Session is the root resource manager for a process is crucial for effective resource management. Consult the official TensorFlow API documentation, specifically the sections related to session management and device configuration, for details on managing the session parameters. For advanced control and in-depth understanding of device placement, delve into details surrounding logical vs. physical devices within TensorFlow's documentation. Consider exploring TensorFlow’s advanced deployment guides, focusing on model optimization and resource management.
