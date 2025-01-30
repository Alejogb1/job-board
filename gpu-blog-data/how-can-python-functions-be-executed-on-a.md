---
title: "How can Python functions be executed on a GPU using Ray?"
date: "2025-01-30"
id: "how-can-python-functions-be-executed-on-a"
---
Leveraging the computational power of GPUs for Python functions requires careful orchestration, particularly when dealing with distributed workloads. Ray offers a compelling solution to this challenge by abstracting away much of the complexity inherent in GPU utilization, allowing developers to focus on their core algorithms. The crucial concept is to designate specific functions or tasks to execute on a GPU-enabled worker within the Ray cluster, rather than forcing entire programs to run on these specialized units.

To effectively utilize GPUs with Ray, I've found it's paramount to first understand how Ray handles resource management. A Ray cluster comprises several nodes, some of which may be equipped with GPUs. Each node maintains a resource table, indicating the number of CPUs, GPUs, and other resources available. When a function is submitted for execution, Ray's scheduler examines this resource table and dispatches the task to an appropriate worker that possesses the requested resources. This resource declaration is critical; without it, the function will default to CPU execution.

The primary mechanism for GPU assignment in Ray is through the `@ray.remote` decorator, combined with explicit resource requests. This decorator transforms a regular Python function into a remote task that can be executed asynchronously. When defining the decorated function, you can include a `resources` argument, specifying the required number of GPUs. The scheduler will then prioritize workers that meet the requested criteria. For optimal performance, this often involves a small amount of data transfer between the CPU-bound Ray controller process and the GPU-bound workers, but Ray's efficient object store minimizes such overhead.

Here’s a basic example demonstrating GPU execution using Ray:

```python
import ray
import time
import numpy as np

@ray.remote(num_gpus=1)
def gpu_intensive_task(matrix_size):
    start_time = time.time()
    matrix = np.random.rand(matrix_size, matrix_size)
    result = np.dot(matrix, matrix)
    end_time = time.time()
    duration = end_time - start_time
    return duration

if __name__ == "__main__":
    ray.init()
    matrix_sizes = [1000, 2000, 3000, 4000, 5000]
    futures = [gpu_intensive_task.remote(size) for size in matrix_sizes]
    durations = ray.get(futures)
    ray.shutdown()
    for size, duration in zip(matrix_sizes, durations):
      print(f"Matrix size: {size}, Computation Time (s): {duration:.4f}")
```

In this example, `gpu_intensive_task` is decorated with `@ray.remote(num_gpus=1)`, indicating that each invocation requires one GPU resource. The function simulates a computationally intensive operation, matrix multiplication, to showcase the potential speedup offered by GPU acceleration. Notice that the function uses `numpy`, a library that can optionally utilize GPUs if available. In order to observe a significant speedup from GPU acceleration in this example, it is essential that the `numpy` library you are using is backed by either CUDA or similar GPU acceleration methods. Ray schedules this task on a worker possessing a GPU based on the provided resource constraints.

Following this, let’s look at a case of managing multiple GPUs for parallel processing:

```python
import ray
import torch

@ray.remote(num_gpus=1)
def train_model_on_gpu(data_size, epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.nn.Linear(data_size, 1).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    data = torch.randn(100, data_size).to(device)
    target = torch.randn(100, 1).to(device)
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(data)
        loss = torch.nn.functional.mse_loss(output, target)
        loss.backward()
        optimizer.step()
    return loss.item()

if __name__ == "__main__":
    ray.init()
    data_sizes = [100, 200, 300]
    epochs = [10, 10, 10]
    futures = [train_model_on_gpu.remote(size, ep) for size, ep in zip(data_sizes, epochs)]
    losses = ray.get(futures)
    ray.shutdown()
    for size, loss in zip(data_sizes, losses):
        print(f"Data size: {size}, Loss: {loss:.4f}")
```

Here, I’ve incorporated `pytorch` to create a simple linear regression model. Each `train_model_on_gpu` instance also receives `num_gpus=1`. Crucially, within the function, we explicitly select the "cuda" device if it's available. This ensures that tensor operations utilize the allocated GPU. In a real world deep learning training scenario, the `train_model_on_gpu` function would implement a much more sophisticated neural network. The key point here, however, remains the same; Ray provides the necessary framework for running computations on GPUs and handling the distribution of training jobs.

Finally, resource management is not restricted to just single GPUs; we can also request a fraction of a GPU if the GPU allows for such fine-grained usage. Let's illustrate with an example:

```python
import ray
import time
import numpy as np

@ray.remote(resources={"gpu": 0.5})
def partial_gpu_task(matrix_size):
    start_time = time.time()
    matrix = np.random.rand(matrix_size, matrix_size)
    result = np.dot(matrix, matrix)
    end_time = time.time()
    duration = end_time - start_time
    return duration

if __name__ == "__main__":
    ray.init()
    matrix_sizes = [1000, 2000, 3000]
    futures = [partial_gpu_task.remote(size) for size in matrix_sizes]
    durations = ray.get(futures)
    ray.shutdown()
    for size, duration in zip(matrix_sizes, durations):
      print(f"Matrix size: {size}, Computation Time (s): {duration:.4f}")
```

In this case, instead of requesting a full GPU with `num_gpus=1`, the `@ray.remote` decorator uses `resources={"gpu": 0.5}`. This implies each task can utilise only a part of a single GPU, allowing for more efficient utilization when individual tasks do not require the full resources of a single GPU. This feature is important in cases where there are multiple small or medium sized computations where sharing a GPU across them can improve the overall throughput.

When implementing GPU tasks, you will find the documentation for the NVIDIA CUDA toolkit helpful in understanding the capabilities of your hardware and how they interact with Python libraries like `pytorch` or `tensorflow`. Further, investigation into GPU monitoring tools will be beneficial in confirming that code is indeed executing as intended and making use of the GPU. Moreover, reading through case studies and tutorials related to parallel processing on GPU hardware will provide a deeper understanding of the potential benefits and challenges of using this approach. Understanding the underlying architecture of your GPUs and how they interact with other hardware components can also help to pinpoint any bottlenecks in your parallel computing pipeline.
