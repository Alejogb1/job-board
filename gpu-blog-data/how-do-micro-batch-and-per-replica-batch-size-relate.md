---
title: "How do micro-batch and per-replica batch size relate in distributed training?"
date: "2025-01-30"
id: "how-do-micro-batch-and-per-replica-batch-size-relate"
---
The core distinction between micro-batch size and per-replica batch size in distributed training lies in their scope: micro-batch size governs the frequency of gradient updates within a single replica, while per-replica batch size determines the amount of data processed before a gradient update is computed *on that replica*.  This subtle but crucial difference significantly impacts training efficiency, memory consumption, and overall convergence behavior.  My experience optimizing large language model training across multiple GPU clusters highlights the importance of understanding this relationship.


**1. Clear Explanation:**

In distributed training, we aim to parallelize the training process across multiple computing units (replicas, often GPUs).  A standard batch gradient descent approach involves processing a batch of data, calculating the gradient, and updating the model's weights.  In distributed scenarios, we introduce two key size parameters:

* **Per-replica batch size:** This represents the number of samples processed by a single replica before it computes and sends its local gradient.  It's chosen to balance memory constraints on each replica with the desired computational efficiency.  Larger per-replica batch sizes lead to more computationally stable gradient updates (reduced variance) but increase memory demands.

* **Micro-batch size:** This is a subset of the per-replica batch size.  The per-replica batch is divided into multiple smaller micro-batches.  After processing each micro-batch, the replica computes its gradient and *immediately* applies it to a local copy of the model. This local update is then synchronized with other replicas, typically using techniques like all-reduce.  The frequency of these micro-updates – determined by the micro-batch size – introduces a form of asynchronous updating which can improve training speed and potentially help avoid deadlocks.

The global batch size (also referred to as the total batch size), is the product of the per-replica batch size and the number of replicas.  For instance, with 8 replicas and a per-replica batch size of 32, the global batch size is 256.  The choice of these parameters requires careful consideration based on the hardware, the dataset, and the training algorithm.


**2. Code Examples with Commentary:**

These examples illustrate the concept using a simplified TensorFlow-like syntax, focusing on the core logic.  Error handling and advanced optimization techniques are omitted for brevity.

**Example 1: Single-Replica Training (for comparison):**

```python
import numpy as np

def train_step(model, data, batch_size):
  """Single-replica training step."""
  batch = data[0:batch_size]
  loss = model.compute_loss(batch)
  gradients = model.compute_gradients(loss)
  model.apply_gradients(gradients)

# ... Data loading and model initialization ...
for epoch in range(num_epochs):
  for i in range(0, len(data), batch_size):
    train_step(model, data, batch_size)
```

This example shows a standard batch gradient descent approach without distribution. The `batch_size` here directly corresponds to the global batch size in distributed scenarios.

**Example 2: Distributed Training with Micro-batching:**

```python
import numpy as np

def replica_train_step(model, data, per_replica_batch_size, micro_batch_size):
  """Training step for a single replica."""
  for i in range(0, per_replica_batch_size, micro_batch_size):
    microbatch = data[i:i + micro_batch_size]
    loss = model.compute_loss(microbatch)
    gradients = model.compute_gradients(loss)
    model.apply_gradients(gradients) # Local update

  # ... All-reduce operation to synchronize gradients across replicas ...

# ... Data loading and model initialization across replicas ...
for epoch in range(num_epochs):
  for i in range(0, len(data), per_replica_batch_size):
    replica_train_step(model, data[i:i + per_replica_batch_size], per_replica_batch_size, micro_batch_size)
```

This example demonstrates distributed training with micro-batching.  The per-replica batch is processed in smaller micro-batches, with a local gradient update after each micro-batch.  The `all-reduce` operation (represented by comments) synchronizes the updates across all replicas.

**Example 3:  Illustrating Gradient Accumulation (alternative to micro-batching):**

```python
import numpy as np

def replica_train_step(model, data, per_replica_batch_size):
  """Training step using gradient accumulation."""
  accumulated_gradients = None
  for i in range(0, per_replica_batch_size, micro_batch_size):  # micro_batch_size only affects the gradient accumulation strategy
      microbatch = data[i:i + micro_batch_size]
      loss = model.compute_loss(microbatch)
      gradients = model.compute_gradients(loss)
      if accumulated_gradients is None:
          accumulated_gradients = gradients
      else:
          accumulated_gradients = [a + b for a, b in zip(accumulated_gradients, gradients)]

  model.apply_gradients(accumulated_gradients) # local update after processing the entire per-replica batch
  # ... All-reduce operation to synchronize gradients across replicas ...

# ... Data loading and model initialization across replicas ...
for epoch in range(num_epochs):
    for i in range(0, len(data), per_replica_batch_size):
        replica_train_step(model, data[i:i + per_replica_batch_size], per_replica_batch_size)
```

Example 3 showcases gradient accumulation as an alternative approach.  While using micro-batches, the gradients are accumulated before applying the updates. This avoids the frequent synchronization inherent to micro-batching in Example 2,  potentially improving efficiency in certain communication scenarios. The choice between frequent synchronization (Example 2) or accumulation before synchronization (Example 3) depends greatly on the hardware and communication overhead.


**3. Resource Recommendations:**

For a deeper understanding, I recommend studying the distributed training documentation for major deep learning frameworks (TensorFlow, PyTorch).  Further, explore research papers on the convergence properties of asynchronous and synchronous gradient descent methods.  Finally, a comprehensive text on parallel and distributed computing would provide necessary background on the underlying concepts.  Pay close attention to discussions of all-reduce algorithms and their impact on performance.  Consider investigating the efficiency of different communication backends (e.g., NCCL, Gloo) for your specific hardware setup.  Analyzing the impact of different batching strategies on memory footprint and training time using profiling tools is critical for practical optimization.
