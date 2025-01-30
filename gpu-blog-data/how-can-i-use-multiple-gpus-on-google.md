---
title: "How can I use multiple GPUs on Google Colab?"
date: "2025-01-30"
id: "how-can-i-use-multiple-gpus-on-google"
---
Utilizing multiple GPUs within a Google Colab environment requires strategic configuration because, by default, Colab instances typically allocate only one GPU. My experience with scaling deep learning models highlighted the necessity of distributed training and its practical application within Colab's constraints. The process centers around leveraging libraries designed for parallel computation, specifically focusing on TensorFlow or PyTorch, and recognizing that Colab instances do not provide a traditional multi-GPU server environment. We’re essentially managing distributed processes on the single physical machine backing a Colab notebook, and thus the ‘multiple GPUs’ are accessible via software abstractions.

**Explanation**

Fundamentally, achieving multi-GPU utilization in Colab isn't about directly accessing several physically separate GPUs as one might find in a dedicated server. Instead, it involves structuring your code to distribute computational load across the available cores that the assigned Colab GPU offers via TensorFlow or PyTorch libraries, essentially making it behave as a parallel computing environment. The core principle behind this is *data parallelism*. In data parallelism, the same model is replicated across all available ‘devices’ (which in our case, represent the cores within the single Colab-provided GPU), and the training dataset is divided into batches. Each ‘device’ processes its allocated batch of data, computes gradients, and these gradients are then combined or averaged to update the model parameters. This mechanism allows for substantial speedups in training, especially with larger models and datasets, by performing similar operations concurrently across these device representations of the GPU’s cores.

The crucial factor to address is that while a single Colab instance may not expose multiple distinct GPUs, the CUDA enabled GPU it provides, along with the software libraries, is capable of processing data in parallel across its various core compute units. Libraries such as TensorFlow and PyTorch provide abstractions, such as `tf.distribute.MirroredStrategy` in TensorFlow or `torch.nn.DataParallel` or `torch.distributed.launch` in PyTorch, that facilitate this distribution. These tools manage the complexities of data partitioning, gradient communication, and model parameter synchronization, freeing you to focus on your model architecture and training logic. The performance increase hinges on proper implementation and leveraging these abstractions effectively. These tools can provide different trade-offs between ease of use and performance depending on training scenario and can sometimes become performance bottlenecks when the communication cost becomes too high compared to the computation cost.

For TensorFlow, I frequently utilized `MirroredStrategy` when initial model prototyping and evaluation. This strategy synchronizes gradient updates across devices. It is generally a straightforward solution for most cases but can sometimes become limiting at large scales. For more complex scenarios or for customized distributed training, I have relied upon a combination of `tf.distribute.MultiWorkerMirroredStrategy` (for multi machine scenarios - where each Colab notebook can be considered a node in a cluster) and `tf.distribute.ParameterServerStrategy` when model weights are too large for each device to replicate. This is, however, beyond the practical limitation of a single Colab instance, which limits us to a single machine. Within a Colab, and similar constraints, Mirrored strategy has always proven sufficient.

In contrast, PyTorch provides `torch.nn.DataParallel` which is quite easy to implement for a basic multi-device setup, and utilizes a single-process, multi-threaded, approach which while very straightforward to code, can limit performance for larger models. For more control over the distribution and scaling of training, `torch.distributed.launch` coupled with `torch.nn.parallel.DistributedDataParallel` is better suited and allows more customization of different parallelization strategies.

The primary takeaway is that the single GPU exposed in Colab becomes your *logical multi-GPU environment*. The efficient use of this environment requires careful attention to batch sizes, memory management, and understanding the nuances of the chosen library’s distributed training capabilities.

**Code Examples and Commentary**

*Example 1: TensorFlow with `MirroredStrategy`*

```python
import tensorflow as tf

# Define the distribution strategy
strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    # Model definition (simplified example)
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    optimizer = tf.keras.optimizers.Adam(0.001)
    loss_fn = tf.keras.losses.CategoricalCrossentropy()

    # Loss function definition outside the strategy scope for better efficiency
    def compute_loss(labels, predictions):
       return loss_fn(labels, predictions)

    # Function to calculate gradients
    def gradient_calculation(x, y):
        with tf.GradientTape() as tape:
            predictions = model(x)
            loss = compute_loss(y, predictions)

        gradients = tape.gradient(loss, model.trainable_variables)
        return gradients

    # Function for training step
    @tf.function
    def train_step(inputs, labels):
        gradients = gradient_calculation(inputs, labels)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return compute_loss(labels, model(inputs))


# Create dummy data
X = tf.random.normal((1000, 10))
Y = tf.one_hot(tf.random.uniform(shape=(1000,), minval=0, maxval=10, dtype=tf.int32), depth=10)
dataset = tf.data.Dataset.from_tensor_slices((X,Y)).batch(64).shuffle(100)


# Training Loop
epochs = 5
for epoch in range(epochs):
    for step, (inputs, labels) in enumerate(dataset):
        loss_value = strategy.run(train_step, args=(inputs, labels))
        print(f'Epoch:{epoch} Step:{step} Loss:{loss_value.numpy()}')


```

*Commentary:* This code snippet demonstrates how to wrap the model definition and optimization logic within a `MirroredStrategy` scope.  The `strategy.run` call handles the distribution of the training step across the available devices. The `@tf.function` decorator compiles the training step, further optimizing the operations for efficient execution on the GPU. In my experience, explicitly handling functions within this kind of strategy scope is beneficial when debugging and ensuring that code executes as intended on available cores. In this case, using a dummy dataset allows to focus on how the multi-device strategy works while still running the training operation.

*Example 2: PyTorch with `torch.nn.DataParallel`*

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Model Definition (simplified)
class SimpleModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleModel, self).__init__()
        self.linear1 = nn.Linear(input_size, 64)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(64, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return x


# Instantiate Model and move it to available devices
input_size = 10
output_size = 10
model = SimpleModel(input_size, output_size)
model = nn.DataParallel(model)
model.to('cuda')


# Loss function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Create dummy data
X = torch.randn(1000, input_size).to('cuda')
Y = torch.randint(0, output_size, (1000,)).to('cuda')
dataset = torch.utils.data.TensorDataset(X,Y)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

epochs = 5
for epoch in range(epochs):
    for step, (inputs, labels) in enumerate(dataloader):
      optimizer.zero_grad()
      outputs = model(inputs)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()
      print(f'Epoch: {epoch}, Step:{step}, Loss: {loss.item()}')

```

*Commentary:* The `nn.DataParallel` module in PyTorch wraps the model, and then, like in the TensorFlow example, we move the model to the available CUDA devices. PyTorch’s approach is simpler, however, the underlying implementation implies that model operations will be performed on the multiple devices, which might lead to some bottlenecks when dealing with large models.  The rest of the training loop remains largely the same as the typical PyTorch setup. As an important note, the `batch_size` is the effective batch size across *all* the used devices. The individual per device batch size will thus be the overall batch size divided by the number of devices. This is a key consideration when tuning hyperparameters. Also, when using DataParallel the model has to be passed to CUDA *before* it is wrapped within the DataParallel.

*Example 3: PyTorch with `torch.distributed` (conceptual within Colab limitations, as it needs a 'launch' command on the command line)*

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import os

# Initialize distributed process group
def init_process_group(rank, world_size, backend='nccl'):
   os.environ['MASTER_ADDR'] = 'localhost'
   os.environ['MASTER_PORT'] = '12355'
   dist.init_process_group(backend, rank=rank, world_size=world_size)


# Model definition (same as before)
class SimpleModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleModel, self).__init__()
        self.linear1 = nn.Linear(input_size, 64)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(64, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return x

def main(rank, world_size):
    init_process_group(rank, world_size)
    input_size = 10
    output_size = 10
    model = SimpleModel(input_size, output_size)
    model = nn.parallel.DistributedDataParallel(model.to(rank), device_ids=[rank])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)


    X = torch.randn(1000, input_size)
    Y = torch.randint(0, output_size, (1000,))
    dataset = torch.utils.data.TensorDataset(X,Y)

    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas = world_size, rank=rank)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False, sampler=sampler)

    epochs = 5
    for epoch in range(epochs):
      for step, (inputs, labels) in enumerate(dataloader):
          optimizer.zero_grad()
          outputs = model(inputs.to(rank))
          loss = criterion(outputs, labels.to(rank))
          loss.backward()
          optimizer.step()
          if rank==0:
           print(f'Epoch: {epoch}, Step:{step}, Loss: {loss.item()}')


if __name__=='__main__':
    world_size = torch.cuda.device_count() # Number of GPUs available.
    torch.multiprocessing.spawn(main, args=(world_size,), nprocs = world_size, join=True)

```

*Commentary:* This example showcases the usage of `DistributedDataParallel` in PyTorch. While it is difficult to fully execute within a single Google Colab notebook because the launch needs to be done outside of the python interpreter, it demonstrates the general logic of creating a distributed environment. The `torch.multiprocessing.spawn` function emulates a `torch.distributed.launch`, starting multiple processes (as many as there are available GPUs in the colab runtime). It is important to initialize the process group using `init_process_group` and use the `DistributedSampler` to load the data in the different processes. This is also the recommended method for using multi device strategies, even if each process corresponds to the same GPU. The `main` function will be run on every single device. A notable improvement over `DataParallel` is the fact that every single device will perform the computations and only collect gradients in the aggregation phase.

**Resource Recommendations**

To further expand your understanding, I recommend exploring the official documentation for both TensorFlow and PyTorch. Both libraries offer comprehensive tutorials and guides specifically covering distributed training methodologies. Moreover, reading research papers on parallel and distributed computing within the context of deep learning can offer insight into the theoretical underpinnings of these concepts. Also, focusing on the CUDA toolkit documentation is valuable to understand how the GPU hardware and CUDA runtime support the parallel operations. Finally, investigating other available training techniques that can optimize performance while respecting Colab’s resource limitations such as gradient accumulation and mixed precision training should also be performed.
