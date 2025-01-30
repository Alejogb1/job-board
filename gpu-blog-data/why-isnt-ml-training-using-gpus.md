---
title: "Why isn't ML training using GPUs?"
date: "2025-01-30"
id: "why-isnt-ml-training-using-gpus"
---
The statement “ML training isn't using GPUs” is fundamentally incorrect; the reality is that modern machine learning overwhelmingly leverages GPUs for training, not the inverse. My experience developing models for large-scale image recognition and natural language processing has consistently highlighted the performance gulf between CPU-based and GPU-based training. To understand why the proposition is inaccurate, and to clarify the dominant role GPUs play, it's essential to dissect the computational characteristics of machine learning and how they map onto GPU architecture.

Machine learning training, especially deep learning, is inherently data-parallel. This means the same set of mathematical operations, predominantly matrix multiplications and additions, are performed on vast amounts of data simultaneously. Traditional CPUs are designed for serial processing, excelling at complex logic and instruction flow, but their limited core count makes them inefficient for this type of computation. A CPU might have 8 or 16 cores capable of complex tasks, whereas a modern GPU has thousands of simpler processing units. These units, structured in a parallel architecture, are exceptionally well-suited to execute the identical operations required for each data point in a batch during training. The massive parallelism afforded by GPUs translates directly into drastically reduced training times, often by factors of 10x or more, compared to CPU-based approaches.

The core operations of ML training, such as forward propagation, backpropagation, and gradient descent, are all matrix-based and highly parallelizable. Consider a simple neural network. During forward propagation, each layer involves matrix multiplication between the input data and weight matrices. For backpropagation, the gradients are calculated using similar matrix operations. These calculations form the bottleneck for the training process, especially as the models become more complex with deeper networks and larger input data. GPUs, with their many-core architecture and high memory bandwidth, are naturally designed to execute these matrix operations concurrently and efficiently. Specialised hardware units like Tensor Cores, found on NVIDIA GPUs, accelerate these matrix operations even further through mixed-precision computing, allowing for reduced memory footprint and faster computation, while maintaining acceptable precision in most training scenarios. CPUs lack the architecture to exploit this level of parallelism or to implement hardware level support for operations central to ML training.

To illustrate, consider a simple matrix multiplication implemented on both a CPU using NumPy and a GPU using PyTorch. The first example uses Python’s NumPy, which executes on the CPU:

```python
import numpy as np
import time

# Define matrix size
n = 4096
matrix_a = np.random.rand(n, n)
matrix_b = np.random.rand(n, n)

# Perform matrix multiplication and time it
start_time = time.time()
matrix_c_cpu = np.dot(matrix_a, matrix_b)
end_time = time.time()

print(f"CPU matrix multiplication time: {end_time - start_time:.4f} seconds")
```

This code snippet demonstrates a standard matrix multiplication using NumPy. While functional, this relies entirely on the CPU. For a matrix of this size, the calculation takes a noticeable amount of time.

Now consider the GPU counterpart, implemented using PyTorch:

```python
import torch
import time

# Define matrix size
n = 4096
matrix_a = torch.rand(n, n).cuda() # Move matrix to GPU
matrix_b = torch.rand(n, n).cuda() # Move matrix to GPU

# Perform matrix multiplication and time it
start_time = time.time()
matrix_c_gpu = torch.matmul(matrix_a, matrix_b)
torch.cuda.synchronize() # Ensure GPU operations complete
end_time = time.time()

print(f"GPU matrix multiplication time: {end_time - start_time:.4f} seconds")
```

This snippet does the same matrix multiplication, but first, the matrices are explicitly moved to the GPU using `.cuda()`. Note the `torch.cuda.synchronize()` call, which is important to ensure that we only measure the time it takes for the matrix operation to run on the GPU. The time difference between the CPU and GPU execution for this single operation will be significant. The GPU's parallel processing capabilities result in much faster execution of the identical calculation.

A third example, showing a more realistic scenario when training a simple neural network, emphasizes the difference more dramatically:

```python
import torch
import torch.nn as nn
import torch.optim as optim
import time

# Define a simple neural network
class SimpleNet(nn.Module):
  def __init__(self):
    super(SimpleNet, self).__init__()
    self.fc1 = nn.Linear(100, 50)
    self.relu = nn.ReLU()
    self.fc2 = nn.Linear(50, 10)

  def forward(self, x):
    x = self.relu(self.fc1(x))
    x = self.fc2(x)
    return x

# Generate random data
input_size = 100
output_size = 10
num_samples = 10000
inputs = torch.rand(num_samples, input_size)
targets = torch.randint(0, output_size, (num_samples,))

# Initialize the network, loss, and optimizer
model_cpu = SimpleNet()
model_gpu = SimpleNet().cuda() # Move model to GPU

criterion = nn.CrossEntropyLoss()
optimizer_cpu = optim.SGD(model_cpu.parameters(), lr=0.01)
optimizer_gpu = optim.SGD(model_gpu.parameters(), lr=0.01)

# Train on CPU
start_time_cpu = time.time()
for _ in range(100):
    optimizer_cpu.zero_grad()
    outputs = model_cpu(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer_cpu.step()
end_time_cpu = time.time()

# Train on GPU
start_time_gpu = time.time()
inputs = inputs.cuda() # Move data to GPU
targets = targets.cuda() # Move data to GPU
for _ in range(100):
    optimizer_gpu.zero_grad()
    outputs = model_gpu(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer_gpu.step()
torch.cuda.synchronize() # Ensure GPU operations complete
end_time_gpu = time.time()

print(f"CPU training time: {end_time_cpu - start_time_cpu:.4f} seconds")
print(f"GPU training time: {end_time_gpu - start_time_gpu:.4f} seconds")

```

This example demonstrates the training of a very small neural network for a short period, both using the CPU and the GPU. Observe that the `model_gpu` as well as the training data is moved to the GPU. The results will again demonstrate a drastic time difference. This discrepancy will grow dramatically as the dataset size, model complexity, and training duration increases.

Beyond the hardware advantages, the development of libraries like PyTorch and TensorFlow have further cemented the reliance on GPUs for training. These libraries are explicitly designed to leverage GPU resources efficiently and provide abstraction layers, simplifying the process of developing and training ML models on GPUs. Moreover, specialized software tools and frameworks like CUDA and cuDNN (for NVIDIA GPUs) provide low-level optimizations and high-performance implementations for matrix operations, deep learning layers, and other building blocks of machine learning algorithms.

In summary, the premise that “ML training isn't using GPUs” is inaccurate. The vast majority of modern machine learning training leverages GPUs because their parallel architecture maps directly to the data-parallel nature of ML computations. This leads to significant improvements in training times compared to CPU-based alternatives. While CPUs are still vital for the overall computing environment, their roles are primarily in tasks such as pre-processing of data, post-processing of model outputs, or controlling program flow, and not for the core training process. Any discussion of the feasibility of ML training today must assume utilization of GPUs for its most computationally demanding components.

For resources on this topic, I'd recommend delving into the documentation for PyTorch and TensorFlow. Both platforms offer extensive guides on leveraging GPUs for model training. I’d also advise reviewing academic literature and blog posts on parallel computing architectures, especially those focused on GPU computing, to gain a deeper understanding of the hardware underpinnings. Finally, a survey of resources on high-performance computing will highlight further the essential role that GPU’s have assumed in the field of machine learning.
