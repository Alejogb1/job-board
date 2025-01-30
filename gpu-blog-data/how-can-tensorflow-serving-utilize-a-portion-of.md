---
title: "How can TensorFlow Serving utilize a portion of GPU memory per model?"
date: "2025-01-30"
id: "how-can-tensorflow-serving-utilize-a-portion-of"
---
TensorFlow Serving, by default, loads entire models into GPU memory if a GPU is available and configured. This behavior, while straightforward, can become problematic when hosting multiple models on a single GPU, particularly if these models vary significantly in size or have relatively low individual query volume. In my experience managing a distributed inference system, I’ve found that monolithic memory allocation per model leads to either underutilized resources or, more frequently, memory exhaustion. The solution lies in configuring each model's `SessionOptions` to constrain its GPU memory consumption.

Specifically, TensorFlow allows for granular control over memory allocation through the `per_process_gpu_memory_fraction` setting within the `tf.compat.v1.ConfigProto`. This configuration, when passed to a TensorFlow session, dictates the proportion of total GPU memory the session can claim. While not as precise as specifying absolute memory amounts, it provides a mechanism to enforce limits and share resources across multiple models within a single TensorFlow Serving instance. This requires that each model within the serving instance establish its own dedicated session and thus, have its own memory constraints set.

The core principle rests on the fact that each servable, be it a single model version or a complex ensemble, utilizes a distinct session to load its graph. Within this session, we can configure memory settings through the aforementioned `tf.compat.v1.ConfigProto` instance. Therefore, by manipulating this configuration prior to session initialization during the model loading phase in the TensorFlow Serving framework, we can control memory allocation. This does not directly control GPU memory usage in the traditional sense where we are requesting specific bytes, but rather the *maximum fraction* of available memory that a model can use for computations.

Let's illustrate with a series of code examples:

**Example 1: Configuration During Model Loading**

This example demonstrates a conceptual snippet of how one might modify the model loading logic within a custom TensorFlow Serving `Loader` implementation to configure GPU memory fraction:

```python
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
import grpc

def load_model_with_gpu_fraction(model_path, gpu_fraction):

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = gpu_fraction

    with tf.compat.v1.Session(config=config) as sess:
        tf.compat.v1.saved_model.loader.load(
           sess,
           [tf.compat.v1.saved_model.tag_constants.SERVING],
           model_path)

        # Perform necessary setup (e.g., loading signature definitions)
        return sess  # or wrap the session with other required structures.

# Usage:
model_a_path = "/path/to/model_a"
model_b_path = "/path/to/model_b"

session_a = load_model_with_gpu_fraction(model_a_path, 0.3)  # Model A using 30% of GPU memory.
session_b = load_model_with_gpu_fraction(model_b_path, 0.7) # Model B using 70% of GPU memory.

# At this point, session_a and session_b are ready to receive inference requests.
# In a real-world TensorFlow Serving loader, these would need to be managed appropriately.

```
In this snippet, we're encapsulating model loading and session configuration within a function. The key lies in creating a `ConfigProto` instance where `gpu_options.per_process_gpu_memory_fraction` is set to the desired value, and passing this configuration to the TensorFlow session when it is instantiated with `tf.compat.v1.Session`. Model A is configured to use 30% of available GPU memory while Model B is configured for 70%. This is a simplified example, and one would typically wrap this into a serving framework to manage model lifecycle and versioning.

**Example 2: Integration with TensorFlow Serving (Conceptual)**

This demonstrates conceptually how this configuration would be integrated within a TensorFlow Serving model loader class. This class will implement the `Loader` interface. Since this requires a very specific TensorFlow Serving environment, this is a highly simplified illustrative version.

```python

import tensorflow as tf
from tensorflow_serving.core import loader
from tensorflow_serving.core import model_version
from tensorflow_serving.core import servable_state
from tensorflow_serving.util import status

class CustomModelLoader(loader.Loader):

    def __init__(self, model_base_path, gpu_fraction_map):
        self._model_base_path = model_base_path
        self._gpu_fraction_map = gpu_fraction_map # {model_name: fraction}
        self._session_map = {}

    def load(self, load_request):

        model_version_path = os.path.join(self._model_base_path, str(load_request.version))

        model_name = load_request.model_spec.name

        if model_name not in self._gpu_fraction_map:
           return status.Status(code=status.StatusCode.NOT_FOUND, message=f"Fraction not set for model {model_name}")

        gpu_fraction = self._gpu_fraction_map[model_name]
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = gpu_fraction

        try:
          sess = tf.compat.v1.Session(config=config)
          tf.compat.v1.saved_model.loader.load(
              sess,
              [tf.compat.v1.saved_model.tag_constants.SERVING],
              model_version_path)
           self._session_map[load_request.version] = sess
        except Exception as e:
           return status.Status(code=status.StatusCode.INTERNAL, message=f"Error loading {model_name}: {e}")


        return status.Status(code=status.StatusCode.OK)


    def unload(self, unload_request):
        version = unload_request.version
        if version in self._session_map:
            self._session_map[version].close()
            del self._session_map[version]
            return status.Status(code=status.StatusCode.OK)
        else:
          return status.Status(code=status.StatusCode.NOT_FOUND, message=f"Session not found for version {version}")


    def get_servable(self, version):
         if version not in self._session_map:
            return None
         return self._session_map[version]
```

Here, we introduce a `CustomModelLoader` which implements core functionality of a loader. The `load` function now receives a `load_request` object. The core logic is the same: we construct a `ConfigProto` with the per-process GPU fraction defined in the `gpu_fraction_map` passed into the loader during its construction. When a model is unloaded, the `unload` function will terminate the TensorFlow session, releasing the memory. This loader would need additional scaffolding for real-world usage.

**Example 3: Dynamic Memory Allocation Considerations**

It is crucial to understand that `per_process_gpu_memory_fraction` does not guarantee a static allocation. TensorFlow will dynamically allocate memory as needed, up to the specified fraction.  If a model initially requires less memory than the specified limit, it will not reserve all of it upfront. This dynamic allocation can lead to situations where two models attempt to consume more memory, within their respective limits, than what is actually available on the GPU. TensorFlow will manage this within its internal memory management and potentially result in slower execution or out-of-memory errors during inference. This is a critical consideration for production use, and you may want to experiment with various fraction allocations based on profiling.

```python
import tensorflow as tf

# Simulating two models requesting memory
def simulate_model(gpu_fraction):
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = gpu_fraction
    sess = tf.compat.v1.Session(config=config)

    # Placeholder operations simulating model inference (requiring GPU memory)
    a = tf.random.normal(shape=[10000, 10000])
    b = tf.random.normal(shape=[10000, 10000])
    c = tf.matmul(a,b)

    return sess, c

# Model A configured for 0.6 of GPU memory
session_a, tensor_a = simulate_model(0.6)

# Model B configured for 0.7 of GPU memory
session_b, tensor_b = simulate_model(0.7)

# Running a computation from both models. This will either allocate the required memory if
# available, or potentially raise errors if not enough space for both sessions to expand
result_a = session_a.run(tensor_a)
result_b = session_b.run(tensor_b)

session_a.close()
session_b.close()

print("Computed outputs without memory exhaustion.")
```

In this example, the `simulate_model` function creates sessions with fractional memory limits. Even though the sum of fractional limits is over 1.0, TensorFlow will attempt to manage memory within those confines. However, if the model operations consistently require more memory than the device can provide at run time, the system may lead to slowdowns, out of memory errors, or unpredictable performance. Therefore, careful selection of memory fractions for each model is critical. Monitoring and profiling should be employed.

In summary, managing GPU memory usage with TensorFlow Serving for multiple models involves setting the `per_process_gpu_memory_fraction` within each model's `SessionOptions` during the model loading phase. This is done by manipulating the `ConfigProto` of each session. While providing a mechanism for memory control, it does not guarantee fixed allocations and can be affected by the dynamic nature of memory usage during inference. Careful profiling and adjustments are required for effective production deployments.

Regarding resource recommendations, I suggest thoroughly exploring the TensorFlow documentation on Session Configurations. Pay particular attention to the `tf.compat.v1.ConfigProto` and the `gpu_options` within it. Further, familiarity with TensorFlow Serving architecture and particularly custom model loaders is needed to implement fine-grained control over memory allocation at scale. Consulting TensorFlow's implementation guidelines for custom model servers can also provide valuable insights.
