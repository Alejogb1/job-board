---
title: "Do CPU, GPU, and TPU implementations exhibit equal execution time in Colab?"
date: "2025-01-30"
id: "do-cpu-gpu-and-tpu-implementations-exhibit-equal"
---
No, CPU, GPU, and TPU implementations do not exhibit equal execution times in Google Colab, and the variance is substantial, primarily dictated by the nature of the computation and the architecture of each processing unit. I’ve consistently observed this disparity across multiple projects involving diverse workloads. The key factor isn’t just raw processing power, but how effectively the given task’s characteristics align with the strengths of each hardware type.

A central aspect is the different design philosophies underlying these processors. CPUs, or Central Processing Units, are general-purpose processors optimized for a wide range of tasks. They excel in sequential processing and branch prediction, which involves handling complex instructions and decision-making logic. GPUs, or Graphics Processing Units, are designed for parallel processing, excelling in calculations that can be performed simultaneously across numerous independent data points. TPUs, or Tensor Processing Units, are specialized hardware accelerators explicitly built for neural network workloads, further optimizing the matrix multiplications and tensor operations at the heart of deep learning.

When executing code in Colab, selecting the correct hardware accelerator is crucial. If a code snippet predominantly utilizes sequential logic, the CPU will often outperform the other options, especially when the scale of the calculations is relatively small. This advantage stems from the CPU's superior single-core performance and efficient instruction processing. However, when moving to operations that can be parallelized, like large-scale matrix computations or image processing, the GPU starts to demonstrate significant improvements in execution time. Finally, for deep learning model training and inference, TPUs offer a distinct advantage. They have an architecture that is fine-tuned for the specific tensor operations present in neural network workflows.

To provide a more concrete perspective, consider the following code examples.

**Example 1: Sequential Calculation**

This example demonstrates a scenario where the CPU excels due to its efficient handling of sequential iterations.

```python
import time

def sequential_calculation(n):
    result = 0
    for i in range(n):
        result += i * i
    return result

start_time = time.time()
sequential_calculation(10**7)  #Large number of iterations
end_time = time.time()
print(f"CPU time: {end_time - start_time:.4f} seconds")
```

In this scenario, creating a sequential loop to calculate the sum of squares is performed most effectively on the CPU. If I were to execute this code on Colab across different hardware accelerators, the CPU would consistently complete it in the least amount of time compared to both the GPU and TPU. The GPU’s parallel nature is rendered useless in this context as each calculation is contingent on the prior result, meaning we cannot parallelize the calculation. The TPU, being optimized for neural network operations, is ill-suited to this type of general sequential processing.

**Example 2: Parallelizable Matrix Multiplication**

In contrast, consider the following matrix multiplication example.

```python
import time
import numpy as np
import torch

def matrix_multiply(size):
  a = np.random.rand(size, size).astype(np.float32)
  b = np.random.rand(size, size).astype(np.float32)
  start_time = time.time()
  result = np.dot(a,b)
  end_time = time.time()
  print(f"Numpy time: {end_time - start_time:.4f} seconds")

  a = torch.rand(size, size).float()
  b = torch.rand(size, size).float()
  a = a.cuda()
  b = b.cuda()
  start_time = time.time()
  result = torch.matmul(a,b)
  torch.cuda.synchronize()
  end_time = time.time()
  print(f"GPU time: {end_time - start_time:.4f} seconds")

matrix_multiply(1000)
```
In this scenario, the operation can be effectively parallelized across cores or streaming multiprocessors. When run on Colab with both a CPU execution and then forced onto the GPU by the use of pytorch’s CUDA support, you will see the GPU complete significantly faster than the CPU. The use of the torch.cuda.synchronize() command is important, as the GPU will offload execution of the operations, and this forces the main thread to wait until the GPU has finished and is available. The first test of NumPy demonstrates how the computation is completed on the CPU. I added a simple random matrix generation to ensure the matrix multiplication takes a significant amount of time to complete.

**Example 3: Deep Learning Model Training**

Finally, we have a straightforward deep learning example.

```python
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Model
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(10, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 2)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def training_loop(device, epochs=100):
    model = SimpleNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    inputs = torch.randn(1000, 10).to(device)
    labels = torch.randint(0, 2, (1000,)).to(device)
    dataset = TensorDataset(inputs, labels)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    start_time = time.time()
    for epoch in range(epochs):
        for batch_inputs, batch_labels in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_inputs)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

    end_time = time.time()
    print(f"{device} time: {end_time - start_time:.4f} seconds")

training_loop('cpu')
if torch.cuda.is_available():
  training_loop('cuda')
# TPU is not directly accessible via simple string argument. See documentation for TPU specific instructions for its execution
```

This code defines a simple neural network, generates random training data, and performs backpropagation in a loop. I’ve parameterized the device used so I can directly control which execution unit I am targetting. For each epoch, the optimizer steps using the generated training data, which is pushed to the selected device. If this training procedure was to be completed using a TPU backend, a further increase in the speed of model training can be achieved, provided the code has been written correctly. The TPU is optimized specifically for these tensor operations and will yield faster training times than the CPU, and for most deep learning workloads, it will also out-perform a GPU. Note that TPUs can be complicated to configure and use, which is why it is not included directly in the test.

It is crucial to be aware that TPUs also introduce a specific set of considerations, including a different memory management model and a potentially more intricate setup compared to GPUs, typically requiring additional code scaffolding and specific libraries to be used effectively.

Based on my experience, several resources can provide a deeper understanding of these hardware accelerators. Textbooks on computer architecture and parallel computing can offer detailed explanations of the underlying design principles of CPUs, GPUs, and TPUs. Research papers focused on hardware acceleration and specific technologies, such as CUDA and TensorFlow, can provide further insight. Finally, the documentation and tutorials for deep learning frameworks like TensorFlow and PyTorch provide detailed guides on leveraging different hardware accelerators effectively. The use of these resources can help guide future model creation and reduce the time necessary to obtain a final result.
