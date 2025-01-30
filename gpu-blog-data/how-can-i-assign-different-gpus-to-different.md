---
title: "How can I assign different GPUs to different tasks within a single script?"
date: "2025-01-30"
id: "how-can-i-assign-different-gpus-to-different"
---
Modern machine learning workflows frequently demand parallel computation across multiple GPUs.  This requires careful management of resource allocation at both the software and hardware level.  I’ve encountered this challenge multiple times during my work developing distributed training pipelines for large language models, and I've found that understanding the mechanisms provided by libraries like PyTorch and TensorFlow is essential to achieve optimal performance. Effective GPU assignment involves specifying which physical GPU a particular computational graph or operation should execute on.  Without this explicit assignment, frameworks will often default to using only the first available GPU, negating the benefits of having multiple processors.

The core idea revolves around binding operations to specific device contexts, often identified by integers. These integers correspond to the physical GPU identifiers, starting from 0. The framework's runtime environment then ensures that calculations are executed on the designated GPU's computational resources and memory. Both PyTorch and TensorFlow offer mechanisms to achieve this, although the syntax and underlying implementations may differ. These mechanisms prevent resource contention and allow multiple, distinct tasks to operate concurrently, significantly reducing training times.

The underlying technical mechanisms for handling resource allocation rely on low-level API calls interacting directly with the CUDA driver. Libraries like PyTorch and TensorFlow abstract away these complexities, providing high-level APIs for developers to manage GPU device context. Essentially, when you move a tensor or a model to a specific device, the library behind the scenes is creating or accessing a memory context in the GPU's memory space and issuing instructions to the device to execute computations involving that data.  It’s akin to an operating system managing memory for various processes, but specifically focused on the GPU's architecture.

In PyTorch, the `torch.device` object and the `.to()` method are the primary tools. The `torch.device` constructor is used to specify which device (either a GPU identified by `cuda:<device_id>` or CPU). The `.to()` method moves tensors or models to the desired device.

```python
import torch

# Example: Using two GPUs for different operations.
device0 = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device1 = torch.device("cuda:1" if torch.cuda.is_available() and torch.cuda.device_count() > 1 else "cpu")

# Create tensors on different devices.
tensor0 = torch.rand(100, 100).to(device0)
tensor1 = torch.rand(200, 200).to(device1)

# Some operations are performed on tensor0, device 0
result0 = torch.matmul(tensor0, tensor0.T)
print(f"Result 0 device: {result0.device}") # Expected: cuda:0 (or cpu if no GPU)

# Perform operations on tensor1, device 1.
result1 = torch.matmul(tensor1, tensor1.T)
print(f"Result 1 device: {result1.device}") # Expected: cuda:1 (or cpu if one or no GPU)

# Example with a model: Assigning to different devices
class SimpleModel(torch.nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = torch.nn.Linear(100,10)
    def forward(self, x):
        return self.linear(x)

model0 = SimpleModel().to(device0)
model1 = SimpleModel().to(device1)

input_tensor0 = torch.rand(1, 100).to(device0)
input_tensor1 = torch.rand(1, 100).to(device1)

output_model0 = model0(input_tensor0)
output_model1 = model1(input_tensor1)
print(f"Model 0 device: {next(model0.parameters()).device}") # Expected: cuda:0 (or cpu if no GPU)
print(f"Model 1 device: {next(model1.parameters()).device}") # Expected: cuda:1 (or cpu if one or no GPU)

```

In this PyTorch example, I'm creating two `torch.device` objects, `device0` and `device1`. The code checks if CUDA is available and assigns devices accordingly, falling back to CPU if necessary. Tensors and models are then explicitly moved to specific devices using the `.to()` method. Each device can then operate concurrently, enabling distributed processing.  I frequently use this methodology to train parts of a large model with different hyperparameters on different GPUs, maximizing hardware utilization. The example also includes a basic `torch.nn.Module` to showcase how models can be assigned to devices. Note the use of `next(model0.parameters()).device` to check the actual device on which the model's parameters are stored.

TensorFlow utilizes a similar strategy using `tf.config.list_physical_devices('GPU')` to enumerate the available GPUs and `tf.device` to specify the device context.  `tf.device` is used within a Python scope to ensure that the operations within the scope are placed on the selected device, and the placement is handled by the TensorFlow runtime.

```python
import tensorflow as tf

# Example: Using two GPUs for different operations.
gpus = tf.config.list_physical_devices('GPU')
if len(gpus) >= 2:
    gpu0 = gpus[0]
    gpu1 = gpus[1]
else:
    gpu0 = None
    gpu1 = None

# Create tensors on different devices if GPUs are available.
if gpu0 and gpu1:
    with tf.device(gpu0.name):
        tensor0 = tf.random.normal((100, 100))
        result0 = tf.matmul(tensor0, tf.transpose(tensor0))
        print(f"Result 0 device: {result0.device}")

    with tf.device(gpu1.name):
        tensor1 = tf.random.normal((200, 200))
        result1 = tf.matmul(tensor1, tf.transpose(tensor1))
        print(f"Result 1 device: {result1.device}")

    class SimpleModel(tf.keras.Model):
        def __init__(self):
            super(SimpleModel, self).__init__()
            self.linear = tf.keras.layers.Dense(10)
        def call(self, x):
            return self.linear(x)

    with tf.device(gpu0.name):
        model0 = SimpleModel()
        input_tensor0 = tf.random.normal((1, 100))
        output_model0 = model0(input_tensor0)
        print(f"Model 0 device: {model0.trainable_variables[0].device}")


    with tf.device(gpu1.name):
        model1 = SimpleModel()
        input_tensor1 = tf.random.normal((1, 100))
        output_model1 = model1(input_tensor1)
        print(f"Model 1 device: {model1.trainable_variables[0].device}")
else:
    print("Less than 2 GPUs available, running on CPU")
```

Here,  I am retrieving the list of physical GPU devices.  If two or more GPUs are present, I select the first two. The `tf.device` context manager ensures that the tensor operations are executed on the specified device. Similarly, `tf.keras.Model` instances are created within the device contexts.  This is how I often separate different branches of a computational graph to optimize training time. Note the use of `model0.trainable_variables[0].device` to verify the device of the model's weights. This approach is essential when different parts of a model are trained in separate physical devices.

TensorFlow also provides a slightly different approach by enabling explicit placement of the operations to different devices in a single function without the `tf.device` context. This is done by using the `tf.config.experimental.set_visible_devices` configuration and subsequently declaring the device context of an operation directly within the call.

```python
import tensorflow as tf
# Example of explicit device placement without device context manager
gpus = tf.config.list_physical_devices('GPU')
if len(gpus) >= 2:
  gpu0 = gpus[0]
  gpu1 = gpus[1]
  tf.config.experimental.set_visible_devices([gpu0, gpu1], 'GPU')
  def my_function():
    # Define tensor on gpu0
      tensor0 = tf.random.normal((100, 100))
      #  operation on gpu0
      with tf.device(gpu0.name):
        result0 = tf.matmul(tensor0, tf.transpose(tensor0))
      # Define tensor on gpu1
      tensor1 = tf.random.normal((200, 200))
      #  operation on gpu1
      with tf.device(gpu1.name):
         result1 = tf.matmul(tensor1, tf.transpose(tensor1))
      print(f"Result 0 device: {result0.device}")
      print(f"Result 1 device: {result1.device}")
      return
  my_function()
else:
    print("Less than 2 GPUs available, running on CPU")
```

In this example, `tf.config.experimental.set_visible_devices` configures which devices are visible to the session, allowing subsequent device selection within operations. Note the explicit device selection using `with tf.device(gpu0.name):` inside the function. This enables a flexible and fine-grained control of execution. I've used this method to implement custom distributed training loops where operation placement is defined in a central function, offering fine-grained control.

For further study on GPU management within deep learning frameworks, I recommend exploring the official documentation for both PyTorch and TensorFlow.  Specifically look for topics around distributed training, device management and using multiple GPUs. Additionally, academic papers that address distributed training methodologies and resource allocation in large-scale machine learning environments are helpful for advanced use cases. Books specializing in deep learning with these frameworks often contain dedicated chapters on optimizing resource utilization.
