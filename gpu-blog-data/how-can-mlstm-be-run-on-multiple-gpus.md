---
title: "How can mLSTM be run on multiple GPUs?"
date: "2025-01-30"
id: "how-can-mlstm-be-run-on-multiple-gpus"
---
Multilayer Long Short-Term Memory (mLSTM) networks, essential for modeling sequential data, often demand substantial computational power. Distributing the training process across multiple Graphics Processing Units (GPUs) is a necessity for handling large datasets and complex models. I've personally scaled several mLSTM implementations across multiple GPUs over the past few years, encountering various challenges and solutions, and can elaborate on this process.

The core idea is to leverage parallel processing. However, unlike simpler feedforward networks, the sequential nature of LSTMs presents unique challenges. Directly replicating the model on each GPU and simply averaging gradients won't work due to the hidden state dependencies across timesteps. The correct approach involves either data parallelism or model parallelism, or in some cases, a hybrid of both, each with its own considerations for mLSTMs. I predominantly focus here on data parallelism as this is typically more applicable and more straightforward to implement when scaling mLSTMs.

Data parallelism distributes the training dataset across multiple GPUs. Each GPU receives a distinct subset of the training data and computes gradients on its local batch. These local gradients are then aggregated, and the model parameters are updated using these aggregated gradients. This means each GPU processes the entire model, but operates on different batches of input sequences. The crux of the matter then becomes ensuring efficient communication of gradients and maintaining consistent states of the hidden layers across GPUs.

For a synchronous update approach, which is generally more efficient and reliable for mLSTM training, you typically use a communication strategy to aggregate gradients computed by different devices. This can be implemented using libraries such as TensorFlow's `tf.distribute.Strategy` or PyTorch's `torch.nn.DataParallel` or `torch.nn.parallel.DistributedDataParallel` (DDP). DDP is the preferred approach in PyTorch, specifically designed for distributed training with optimized communication. TensorFlow’s `MirroredStrategy` or `MultiWorkerMirroredStrategy` would be the analogue.

I'll demonstrate using PyTorch, as that’s where I've invested most of my distributed training time. The following examples will illustrate the use of `DistributedDataParallel` and assume you have multiple GPUs available.

**Example 1: Basic Distributed Setup with DDP**

The initial step involves setting up the distributed environment and wrapping the model. Here's a simplified example of how it might look:

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os

def setup(rank, world_size):
  os.environ['MASTER_ADDR'] = 'localhost'
  os.environ['MASTER_PORT'] = '12355'
  dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
  dist.destroy_process_group()

class SimpleLSTM(nn.Module):
  def __init__(self, input_size, hidden_size, num_layers, output_size):
      super(SimpleLSTM, self).__init__()
      self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
      self.fc = nn.Linear(hidden_size, output_size)

  def forward(self, x):
      out, _ = self.lstm(x)
      out = self.fc(out[:, -1, :]) # Use the last time step output
      return out

def train(rank, world_size, num_epochs=5):
    setup(rank, world_size)
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    model = SimpleLSTM(input_size=10, hidden_size=20, num_layers=2, output_size=5).to(device)
    model = DDP(model, device_ids=[rank]) # Wrap model
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    #Dummy Data
    input_data = torch.randn(128, 50, 10).to(device)
    target = torch.randint(0, 5, (128,)).to(device)

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(input_data)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        if rank == 0: # Only print from rank 0 process
          print(f"Epoch: {epoch+1}, Loss: {loss.item()}")

    cleanup()

if __name__ == "__main__":
  world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
  if world_size > 1:
    import torch.multiprocessing as mp
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)
  else:
      train(0, 1)
```

*   **`setup(rank, world_size)`:** Initializes the distributed environment using `torch.distributed.init_process_group`, providing the communication backend (`nccl` is preferred for GPU training) and process group details.
*   **`DDP(model, device_ids=[rank])`:** Wraps the model using `DistributedDataParallel`, distributing gradients and ensuring each process’s weights remain synchronized. The `device_ids` are used to select the specific GPU for each process.
*   **`mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)`:** Launches multiple processes, one per available GPU, running the training loop in parallel.
*   **Note:** I've used dummy input data here for brevity, you would of course, implement a dataset and dataloader.

**Example 2: Handling Variable Sequence Lengths**

mLSTMs are frequently used for tasks involving sequences of variable lengths, such as text or time-series data. To correctly handle this with DDP, we must employ masking or padding effectively. Here, I'll demonstrate how to pad variable-length sequences and how to modify the loss computation.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
from torch.nn.utils.rnn import pad_sequence

# Assuming setup and cleanup functions are the same as in Example 1
def setup(rank, world_size):
  os.environ['MASTER_ADDR'] = 'localhost'
  os.environ['MASTER_PORT'] = '12355'
  dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
  dist.destroy_process_group()

class SimpleLSTM(nn.Module):
  def __init__(self, input_size, hidden_size, num_layers, output_size):
    super(SimpleLSTM, self).__init__()
    self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
    self.fc = nn.Linear(hidden_size, output_size)
  def forward(self, x):
      out, _ = self.lstm(x)
      out = self.fc(out[:, -1, :])
      return out

def train(rank, world_size, num_epochs=5):
    setup(rank, world_size)
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    model = SimpleLSTM(input_size=10, hidden_size=20, num_layers=2, output_size=5).to(device)
    model = DDP(model, device_ids=[rank])
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # Dummy Data with variable lengths
    seq_lengths = [30, 40, 20, 50, 25, 35, 45, 30] * 16 #8 sequences/gpu
    input_sequences = [torch.randn(length, 10) for length in seq_lengths]
    padded_input = pad_sequence(input_sequences, batch_first=True).to(device)
    target = torch.randint(0, 5, (len(seq_lengths),)).to(device)
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        outputs = model(padded_input)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        if rank == 0:
            print(f"Epoch: {epoch+1}, Loss: {loss.item()}")
    cleanup()

if __name__ == "__main__":
  world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
  if world_size > 1:
    import torch.multiprocessing as mp
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)
  else:
    train(0,1)
```

*   **`pad_sequence`:** Utilized to pad sequences to a uniform length. The data remains padded throughout training.
*   **Note:** Real-world scenarios often require more sophisticated handling of padding, including masking to exclude padded values from loss and gradient calculations. This is particularly critical for performance and accuracy in sequence modeling.

**Example 3:  Gradient Accumulation with DDP**

When the dataset size or memory constraints prevent large batch sizes per GPU, gradient accumulation is a technique to simulate larger batch sizes by accumulating gradients over multiple iterations before a parameter update.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
def cleanup():
    dist.destroy_process_group()

class SimpleLSTM(nn.Module):
  def __init__(self, input_size, hidden_size, num_layers, output_size):
      super(SimpleLSTM, self).__init__()
      self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
      self.fc = nn.Linear(hidden_size, output_size)

  def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

def train(rank, world_size, num_epochs=5, accumulation_steps=4):
  setup(rank, world_size)
  device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
  model = SimpleLSTM(input_size=10, hidden_size=20, num_layers=2, output_size=5).to(device)
  model = DDP(model, device_ids=[rank])
  optimizer = optim.Adam(model.parameters(), lr=0.001)
  criterion = nn.CrossEntropyLoss()

  input_data = torch.randn(128, 50, 10).to(device)
  target = torch.randint(0, 5, (128,)).to(device)

  for epoch in range(num_epochs):
    optimizer.zero_grad()  # Initialize accumulated gradients to zero
    for i in range(0, input_data.size(0), 16):  # Simulate smaller batches within larger dataset
      batch_inputs = input_data[i:i+16]
      batch_targets = target[i:i+16]

      outputs = model(batch_inputs)
      loss = criterion(outputs, batch_targets)
      loss = loss/accumulation_steps  # Scale loss by accumulation steps
      loss.backward()
      if (i + 16) % (16 * accumulation_steps) == 0:
        optimizer.step()       # Update parameters
        optimizer.zero_grad()  # Reset accumulated gradients

    if rank == 0:
      print(f"Epoch: {epoch+1}, Loss: {loss.item() * accumulation_steps}")
  cleanup()

if __name__ == "__main__":
  world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
  if world_size > 1:
    import torch.multiprocessing as mp
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)
  else:
    train(0,1)
```

*   **`accumulation_steps`:** The number of forward and backward passes before a parameter update occurs. We divide the loss by the number of accumulation steps to properly scale gradient contributions, which is a requirement for the simulated large batch size.

In addition to these examples, optimizing data loading and ensuring data synchronization across GPUs are critical aspects of efficient distributed training. Techniques such as using distributed samplers and avoiding unnecessary I/O operations can greatly improve training performance.  For more detailed guidance, the documentation for the `torch.nn.parallel.DistributedDataParallel` and related packages provides comprehensive information and further optimization strategies.  Research papers and tutorials focusing on scaling deep learning models across GPUs, available on platforms like Arxiv and via educational resources from leading deep learning providers, often address common challenges and offer solutions in more depth.  Additionally, consulting PyTorch's official documentation, especially on the distributed training section, provides indispensable practical advice.
