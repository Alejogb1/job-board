---
title: "Why is my Jupyter kernel crashing on macOS when using TensorFlow?"
date: "2025-01-30"
id: "why-is-my-jupyter-kernel-crashing-on-macos"
---
The observed kernel crashes in Jupyter Notebooks running TensorFlow on macOS are frequently tied to how macOS manages shared memory, and specifically, its interaction with TensorFlow’s memory allocation strategies. I've personally encountered this behavior across various TensorFlow versions, and the debugging process often points towards a clash between macOS’s aggressive memory management and TensorFlow’s eagerness to claim resources, particularly when employing GPUs.

Specifically, macOS, for its system-wide memory efficiency, might attempt to aggressively reclaim memory that TensorFlow has already allocated, especially for GPU computation. This is not a direct memory leak in TensorFlow but rather an external pressure that triggers TensorFlow to encounter a critical error when its allocated memory regions are altered without its consent. Consequently, the Jupyter kernel, which relies on inter-process communication with the TensorFlow backend, often crashes as a result of this sudden and unexpected change to the underlying resource landscape. The issue is often exacerbated when using large datasets or complex models, further pushing the limits of memory allocation.

The root cause generally manifests as one of a few related issues. First, there’s the issue of conflicting memory allocation strategies. TensorFlow, through its libraries, allocates significant blocks of memory in the GPU or system RAM for storing tensors and computation graphs. These allocations are managed internally. macOS also manages memory according to its own system-wide policies, including virtual memory and process-specific quotas. When these systems clash, leading to contention or unexpected memory modifications, it frequently manifests as a kernel crash. Second, the lack of explicit control over shared memory regions can lead to unforeseen conflicts, as both TensorFlow and the operating system could both be operating on similar virtual memory ranges without an adequate synchronization mechanism. Third, CUDA and Metal, the GPU frameworks, both place restrictions and require careful configuration. If TensorFlow attempts to allocate memory beyond those configured limits, particularly on resource-constrained systems, crashes can result.

To address this problem, several approaches can be taken, often requiring a mixture of changes to environment variables, configuration options, and coding practices.

The first example illustrates how to limit TensorFlow’s memory allocation. This is a proactive step to try and prevent macOS from interfering in the first place. I utilize TensorFlow's `tf.config` API to control resource consumption. I will enable GPU growth. This tells TensorFlow to only allocate memory as needed, rather than pre-allocating everything at the outset. This generally works when no other constraints are present, but when dealing with other processes, explicit settings are required, often combined with the first setting.

```python
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU memory growth enabled.")
    except RuntimeError as e:
        print(f"Error setting GPU memory growth: {e}")
else:
    print("No GPUs found.")


# Example of typical model creation
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
    tf.keras.layers.Dense(10, activation='softmax')
])
#... rest of model and training
```

In this code, I first check for the availability of GPUs. If they are found, I then iterate through each GPU and activate `set_memory_growth` for each of them. This setting informs TensorFlow to only allocate the minimum necessary GPU memory, rather than monopolizing all of it up front. Note that in a more complex setup with multiple processes, it may be necessary to set memory limits on individual processes as well. The `RuntimeError` catch is there in case the setting is invalid, or if the GPU isn’t accessible. While this can reduce crashes due to memory constraints, it doesn't eliminate all cases.

The second code example explores using TensorFlow’s environment variables for managing memory. In this case, I utilize an environment variable that limits the amount of GPU memory usable by TensorFlow, which is particularly useful on a system with a dedicated graphics card where the system’s memory isn’t shared. I've found that setting this limit can prevent unexpected allocation-related crashes, because the amount of memory is bounded.

```python
import os
import tensorflow as tf

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true' #alternative to first example
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async' #only for CUDA
os.environ['TF_MEMORY_ALLOCATION_TRACE'] = '1' #for debugging, comment in prod

#Option to limit memory
#os.environ['TF_GPU_MEM_FRAC_FOR_TEST'] = '0.5' #for testing, comment in prod
#os.environ['TF_MIN_GPU_MULTIPROCESSOR_COUNT'] = '2' #also for limiting
#If TF_FORCE_GPU_ALLOW_GROWTH=true is set, then the below two env variables are redundant

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
      print("Using GPU: ", gpus)
    except RuntimeError as e:
        print(f"Error setting GPU memory growth: {e}")
else:
    print("No GPUs found.")

#... rest of model and training
```
The environment variable `TF_FORCE_GPU_ALLOW_GROWTH` is equivalent to the `tf.config.experimental.set_memory_growth` used earlier, it’s just a different way to achieve the same result that may prove useful when running in scripts where tensorflow calls are not explicit. `TF_GPU_ALLOCATOR` helps with the management of memory in asynchronous settings. `TF_MEMORY_ALLOCATION_TRACE` is specifically for debugging purposes and will give information about where the memory allocations are occurring, but is usually left turned off because it produces verbose outputs. The commented out `TF_GPU_MEM_FRAC_FOR_TEST` sets the fractional amount of total memory allowed to be allocated. Setting it to 0.5 would allow half of the available memory. `TF_MIN_GPU_MULTIPROCESSOR_COUNT` forces the usage of multiple multiprocessors on the GPU if it has more than 2. Experimentation is key to using these environment variables effectively. For instance, when the code is not explicit in how it sets its environment, setting these environment variables at shell level or through a `.env` file might be the only effective way to adjust memory settings.

The final example focuses on best practice by explicitly clearing the model to release memory. Even with growth strategies, in longer and more complex sessions, TensorFlow might retain internal data across iterations. By explicitly clearing the model, I ensure all memory is released.

```python
import tensorflow as tf
import gc #garbage collection

# ... model definition and training (as shown before)

def train_and_clear_model(model, train_data, epochs=5, callbacks=None):
    history = model.fit(train_data, epochs=epochs, callbacks=callbacks) #run model training
    tf.keras.backend.clear_session() #clears the memory in graph, tensors, etc.
    del model  # removes pointer to model instance
    gc.collect() #force garbage collection
    return history

#Example use
#history = train_and_clear_model(model, train_dataset)
#more code after
```
I've wrapped the model fitting process into a function that explicitly calls `tf.keras.backend.clear_session()`, `del model` and `gc.collect()`. `clear_session` frees up the tensorflow-related memory, `del model` frees up the python references and `gc.collect()` forces garbage collection on the python side. This strategy can be helpful when the model is used in a larger loop and you want to reset each iteration, or when you want to perform multiple different model trainings in the same session. Even if not training in a loop, doing this at the end of model usage is a good practice to avoid memory leaks. Note that garbage collection, by nature, is often unpredictable and therefore might be delayed, but it is still effective in general.

In summary, addressing Jupyter kernel crashes on macOS with TensorFlow often necessitates careful management of memory allocation, and understanding the complex interactions between TensorFlow, macOS, and the underlying GPU frameworks. These three approaches – limiting memory growth, utilizing environment variables, and explicitly clearing the model - are steps that I have found effective in my own practice.

For further reading on advanced resource configuration, I suggest consulting the TensorFlow official documentation, specifically sections pertaining to GPU usage and memory allocation. The official CUDA and Metal documentation can provide insights into low-level details of their respective memory management. Finally, discussions in online machine learning forums often present pragmatic solutions to similar issues faced by the community. Understanding not just how to code, but how the underlying system interacts with those calls is key to preventing this kind of instability.
