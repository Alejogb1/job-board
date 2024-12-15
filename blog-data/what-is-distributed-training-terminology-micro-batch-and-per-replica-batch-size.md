---
title: "What is Distributed Training Terminology: Micro-batch and Per-Replica batch size?"
date: "2024-12-15"
id: "what-is-distributed-training-terminology-micro-batch-and-per-replica-batch-size"
---

alright, let's get down to brass tacks with distributed training, specifically micro-batch and per-replica batch sizes. i've been elbow-deep in this stuff for years, and trust me, it can get confusing fast if you don't nail the basics. it's not rocket science, but it’s close, and the devil is often in the detail when dealing with multiple machines.

so, first off, when we talk about "batch size" in machine learning, we're generally referring to the number of training examples processed before updating the model's parameters. standard single-gpu training deals with one batch size at a time. you feed the model, compute gradients, adjust weights, and move on to the next batch. simple enough. but then you go into the land of distributed training, and things change. dramatically.

now, distributed training splits the workload across multiple devices (gpus or even machines), and here's where micro-batches and per-replica batches enter the fray. think of it as a factory line: each machine is an individual station doing its job, but collectively all the stations work to produce a final product faster. in our case, the 'product' is a trained machine learning model.

let's start with the per-replica batch size. imagine each of the machines is a 'replica' of our training process, all executing the same code but on different parts of the data. the *per-replica batch size* is the number of training examples each individual replica processes in each forward and backward pass. it's just like your standard batch size in single gpu training except now each worker has its own and the batches may be different between workers.

for example, if you set a per-replica batch size of 32 and you're using 4 gpus, each gpu will process 32 training examples. independently, mind you.

the micro-batch is a more recent concept that usually emerges when you are dealing with larger models or limited memory on each of the workers. it breaks down the per-replica batch size into even smaller chunks. you do a forward and backward pass for each micro-batch, accumulating gradients, and only update the weights *after* you've gone through all of the micro-batches that comprise one per-replica batch. think of it as a form of gradient accumulation but done explicitly at a framework level (like pytorch or tensorflow) at the level of each worker. it enables larger per-replica batch sizes than your limited gpu memory might otherwise allow.

for instance, if you have a per-replica batch size of 64 and a micro-batch size of 16, each replica will actually do four forward and backward passes before updating the model parameters for this per-replica batch. it will process 16 elements, accumulate the gradients, then 16 more, and so on, until you reach 64 and now the weights are updated on each worker. this process is a way to simulate a larger per-replica batch on memory limited gpus.

it's crucial to understand that the effective batch size (sometimes called the *global batch size*) you are effectively using is the *per-replica batch size* times the number of replicas, not the micro-batch. continuing our previous example, you have a per-replica batch size of 64 and 4 gpus. this results in a global batch size of 64 x 4 = 256. regardless of the micro-batch size. the micro-batch is just an implementation detail and allows for larger per-replica batch sizes on each worker to work.

it can be confusing at first but once you understand each element you will get it.

now, why bother with all this complexity? it’s about scaling. by using multiple gpus or even multiple machines, we can reduce the overall training time quite a bit. especially for large deep learning models. it allows us to use more computational power and handle larger datasets efficiently. micro-batches allow for using larger per-replica batch sizes to achieve this scaling.

let me give you a simplified example in pseudocode, keeping it as close as i can to actual python-like behavior:

```python
def train_one_replica(data_shard, per_replica_batch_size, micro_batch_size, model, optimizer):
    """trains a single replica with mini-batches within a per_replica batch."""
    accumulated_gradients = {} # initialize gradient accumulator for all params of model.
    num_micro_batches = per_replica_batch_size // micro_batch_size # how many forward/backward to do before updating params

    for i in range(num_micro_batches):
        start = i * micro_batch_size
        end = start + micro_batch_size
        micro_batch = data_shard[start:end]
        
        # forward pass
        output = model(micro_batch)
        loss = compute_loss(output, labels)

        # backward pass
        gradients = compute_gradients(loss, model)

        # accumulate gradients
        for param_name, gradient in gradients.items():
            if param_name not in accumulated_gradients:
                accumulated_gradients[param_name] = torch.zeros_like(gradient) # create if it doesnt exist
            accumulated_gradients[param_name] += gradient
    
    #update parameters
    optimizer.apply_gradients(accumulated_gradients)
    # clear accumulated gradients after param update on the worker
    for param_name in accumulated_gradients:
        accumulated_gradients[param_name] = torch.zeros_like(accumulated_gradients[param_name])
    
```

in this example, we have the `train_one_replica` which takes a data shard, batch sizes and the model and does the training process in one replica. you can see the explicit micro-batch implementation, gradient accumulation and then the optimizer is updated at the end of processing all the micro-batches within the per-replica batch.

now, this is just one replica's perspective. in a distributed setup, you would have multiple of these running in parallel (assuming data parallel method) with each one working on a different part of the dataset. you would synchronize the gradients between each replica before applying them, usually. that's where the real magic of distributed training happens, but it is outside the scope of this problem. but it gives you an idea of how the per-replica and micro-batch are intertwined.

some things to keep in mind. choosing the right batch sizes is crucial for optimal performance. too small, and training can be slow. too large, and you might run out of memory or get into convergence problems. so some experimentation is always needed. if you're starting, the per-replica batch size should be as high as your memory allows, and then if needed, reduce the micro-batch size. but never reduce the per-replica batch size.

i remember this one project i worked on years back, oh man it was a mess. we were training a transformer model for natural language processing, and the initial per-replica batch size we chose was way too high. gpus started throwing out of memory errors like it was a game. we had to go back and forth with micro-batch sizes and per-replica batch sizes to get to a stable training process. it took me a day or two to get it right and then another day to figure out the optimal sizes. one of those days when you think you know, and then you realize you know nothing at all. it's always an humbling experience. it wasn't easy, but we managed to scale the training and get the results needed at that time.

it's all about the details in distributed training.

here’s another example but a bit more practical using tensorflow:

```python
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import numpy as np

# Define a simple model
def create_model():
    return models.Sequential([
        layers.Dense(64, activation='relu', input_shape=(784,)),
        layers.Dense(10, activation='softmax')
    ])

# Dummy data for the sake of this example
num_samples = 1000
input_shape = (784,)
X = np.random.rand(num_samples, *input_shape).astype(np.float32)
y = np.random.randint(0, 10, size=(num_samples,)).astype(np.int32)

# convert to tensors
X = tf.convert_to_tensor(X)
y = tf.convert_to_tensor(y)

# Distribution Strategy
strategy = tf.distribute.MirroredStrategy()
per_replica_batch_size = 64
micro_batch_size = 32 # this is just an example
num_replicas = strategy.num_replicas_in_sync
global_batch_size = per_replica_batch_size * num_replicas
print(f"global batch: {global_batch_size} and each replica batch {per_replica_batch_size}")

# make dataset
dataset = tf.data.Dataset.from_tensor_slices((X, y))
dataset = dataset.batch(per_replica_batch_size)
dataset = dataset.repeat()

# optimizer and model within strategy scope
with strategy.scope():
    model = create_model()
    optimizer = optimizers.Adam()

# training step function
@tf.function
def train_step(inputs, labels):
    with tf.GradientTape() as tape:
        predictions = model(inputs)
        loss = tf.keras.losses.sparse_categorical_crossentropy(labels, predictions)
        loss = tf.reduce_sum(loss) / global_batch_size
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

# Distributed training loop
def distributed_training(dataset, num_steps=10):
    #dataset iterator
    dist_dataset = strategy.experimental_distribute_dataset(dataset)
    iterator = iter(dist_dataset)
    for step in range(num_steps):
        data = next(iterator)
        replica_loss = strategy.run(train_step, args=(data[0], data[1]))
        print(f"step: {step} and loss {replica_loss}")

# kick training loop
distributed_training(dataset)
```

in this example, the `tf.distribute.MirroredStrategy` handles the distribution aspect, and you set the `per_replica_batch_size`. there is no explicit micro-batch, but is using batching at dataset level. the gradient aggregation is handled behind the scenes by the distributed strategy. this makes the code easier to write but more complex under the hood.

here's a very simple example with pytorch:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os

# Define a simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(784, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 10)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class DummyDataset(Dataset):
    def __init__(self, num_samples=1000, input_shape=(784,)):
        self.X = np.random.rand(num_samples, *input_shape).astype(np.float32)
        self.y = np.random.randint(0, 10, size=(num_samples,)).astype(np.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def train_one_epoch(dataloader, model, optimizer, rank, micro_batch_size):
        # Set model to training mode
    model.train()

    for batch_idx, (inputs, labels) in enumerate(dataloader):
        per_replica_batch_size = inputs.shape[0]
        num_micro_batches = per_replica_batch_size // micro_batch_size # how many forward/backward to do before updating params
        
        accumulated_gradients = {} # initialize gradient accumulator for all params of model.
        
        for i in range(num_micro_batches):
            start = i * micro_batch_size
            end = start + micro_batch_size
            micro_batch_x = inputs[start:end].to(rank)
            micro_batch_y = labels[start:end].to(rank)

            # forward pass
            output = model(micro_batch_x)
            loss = nn.CrossEntropyLoss()(output, micro_batch_y)

            # backward pass
            loss.backward()

            # accumulate gradients
            for param_name, param in model.named_parameters():
                if param.grad is not None:
                    if param_name not in accumulated_gradients:
                        accumulated_gradients[param_name] = torch.zeros_like(param.grad)
                    accumulated_gradients[param_name] += param.grad
                    
            # zero grads
            optimizer.zero_grad()

        # update params
        for param_name, param in model.named_parameters():
             if param_name in accumulated_gradients:
                param.grad = accumulated_gradients[param_name]
        optimizer.step()

        if rank == 0 and batch_idx % 5 == 0:
            print(f"batch {batch_idx} loss: {loss.item()}")

def setup_distributed(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def train_distributed(rank, world_size):
    setup_distributed(rank, world_size)

    # Data and Model
    dataset = DummyDataset()
    per_replica_batch_size = 64
    micro_batch_size = 32 # example micro-batch size
    dataloader = DataLoader(dataset, batch_size=per_replica_batch_size, shuffle=True) # should be sharded/distributed in real life
    model = SimpleModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    optimizer = optim.Adam(ddp_model.parameters())
    
    # Train for 3 epochs.
    for epoch in range(3):
        train_one_epoch(dataloader, ddp_model, optimizer, rank, micro_batch_size)

    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = 2 # set this as needed
    torch.multiprocessing.spawn(
        train_distributed,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )
```

in this last example, you can see how the micro-batch training process is implemented as well as the use of distributed data parallel.

i’d suggest diving into some solid resources for a deeper understanding. the pytorch documentation on distributed training is quite good as well as the tensorflow tutorials. some academic papers that have been foundational in this are “distributed sgd with mini-batches” by dean, jeffrey; corrado, greg; mongiovì, rafael; chen, kai; mathieu, matthieu; ranzato, marc'aurelio; and "imagenet classification with deep convolutional neural networks" by krizhevsky, alex; sutskever, ilya; and hinton, geoffrey e. if you want to get even more academic. also "deep learning" by goodfellow, ian; bengio, yoshua; and courville, aaron, has excellent chapters about parallelization and optimization methods for deep learning. reading these books and papers will definitely clarify the topics we discussed today.

so, to summarize, the *per-replica batch size* is how much data each individual device processes before updating its local weights. the *micro-batch size* is an implementation detail, a way to further subdivide this work into mini-batches. it allows us to work around the memory limitation of our devices and still achieve faster training via larger *global batch sizes*.

that's my take on the topic, hope this helps, and happy training.
