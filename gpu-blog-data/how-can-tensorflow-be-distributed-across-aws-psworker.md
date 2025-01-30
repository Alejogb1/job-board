---
title: "How can TensorFlow be distributed across AWS ps/worker hosts?"
date: "2025-01-30"
id: "how-can-tensorflow-be-distributed-across-aws-psworker"
---
TensorFlow's distributed training on AWS using parameter servers (ps) and workers necessitates a deep understanding of the underlying communication mechanisms and infrastructure configuration. I’ve managed several large-scale training runs on AWS, and a misconfiguration can lead to significant performance bottlenecks or even outright failure. To achieve effective distribution, one must configure TensorFlow to properly handle the responsibilities of each type of node.

At the core of distributed TensorFlow is the concept of a `tf.distribute.cluster_resolver.TFConfigClusterResolver`. This resolver is crucial for providing the information needed to instantiate the distributed runtime. It extracts the cluster configuration, which includes the addresses and task types (ps or worker) of all involved machines, from either environment variables or a provided configuration dictionary. Incorrect configuration of this resolver can prevent workers from connecting to parameter servers, stalling training. This system uses gRPC as its primary protocol for data transfer between ps and workers, which has to be well managed to avoid network congestions.

The standard approach involves launching separate TensorFlow processes on each host, with each process configured as either a parameter server or a worker. Workers perform the heavy computation of gradient calculations, while parameter servers maintain the model's parameters. This separation of duties enables parallel processing and avoids memory contention, provided that the data feeding mechanism is properly distributed.

First, the `TFConfigClusterResolver` needs to be set up correctly: this includes defining the cluster information with the IP addresses of the parameter servers and worker nodes. This information is often passed through environment variables like `TF_CONFIG` or `AWS_BATCH_JOB_ARRAY_INDEX` when using AWS Batch. For example, the following python code sets up the cluster resolver for 3 worker hosts and 2 parameter servers.

```python
import os
import tensorflow as tf

def get_cluster_resolver_from_env():
    tf_config = os.environ.get('TF_CONFIG')
    if not tf_config:
        # Fallback in case of local testing. Not advisable for prod
        cluster_def = {
            'cluster': {
                'worker': ["10.0.0.10:2222", "10.0.0.11:2222", "10.0.0.12:2222"],
                'ps': ["10.0.0.13:2222", "10.0.0.14:2222"],
            },
            'task': {'type': 'worker', 'index': 0} # this will be replaced programmatically per host
        }
    else:
         import json
         cluster_def = json.loads(tf_config)
    resolver = tf.distribute.cluster_resolver.TFConfigClusterResolver(cluster_def)
    return resolver

resolver = get_cluster_resolver_from_env()
```

Here, `os.environ.get('TF_CONFIG')` attempts to retrieve the cluster configuration from the environment. If `TF_CONFIG` is not present (during local development), a static cluster definition is used for testing purposes. In a proper AWS setup, this `TF_CONFIG` environment variable will contain the necessary JSON configuration specifying the roles and addresses for all hosts. The `cluster_def` dictionary is then used to initialise `TFConfigClusterResolver`, which is used to create a distributed strategy. In a large deployment on AWS, this value is typically automatically populated by the job management system, and the `task` attribute in the cluster definition is set on each host to match its task index. It is important to note that this sample assumes that the hosts listed have gRPC ports opened and are reachable from one another. The hosts are on their own respective virtual private cloud subnets and are not publicly accessible.

Next, one can use this resolver with `tf.distribute.MultiWorkerMirroredStrategy`. This strategy allows workers to compute gradients simultaneously on their local copy of the model, and averages the computed gradients across all workers and parameter servers before model updates. It simplifies the distributed training logic and is suitable for most common use-cases.

```python
strategy = tf.distribute.MultiWorkerMirroredStrategy(resolver=resolver)
with strategy.scope():
  model = tf.keras.Sequential([tf.keras.layers.Dense(10)])
  optimizer = tf.keras.optimizers.SGD(0.01)
  loss_fn = tf.keras.losses.MeanSquaredError()

@tf.function
def train_step(inputs, labels):
    with tf.GradientTape() as tape:
        predictions = model(inputs)
        loss = loss_fn(labels, predictions)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

dataset = tf.data.Dataset.from_tensor_slices(([1.0], [1.0]))
dataset = dataset.batch(10).repeat()

for inputs, labels in dataset.take(100):
  loss = train_step(inputs, labels)
  tf.print("Loss:", loss)
```

In this example, the model, optimizer, and loss function are created within the `strategy.scope()`. This ensures that all variables are created on each worker and parameter server. A `tf.function`-decorated `train_step` function, computes the gradients, applies the gradients and returns the loss value. This is executed as a distributed graph. The `dataset` is initialized using the `from_tensor_slices` method and the training is done with `take(100)`. In practice this code should be used inside of a distributed training loop, with the `task` attributes in the `TF_CONFIG` environment variable set up per host, typically in an AWS Batch or SageMaker setup.

Finally, managing the dataset input pipeline is crucial for efficient distributed training. It is highly advisable to ensure each worker receives a distinct subset of the data. This can be achieved by utilizing `tf.data.Dataset` API with `tf.data.Dataset.shard`. Sharding the dataset ensures that each worker receives a unique slice of the dataset, preventing duplication and over-training on the same data across workers.

```python
global_batch_size = 30 # total across all workers
batch_size_per_worker = global_batch_size // strategy.num_replicas_in_sync

dataset = tf.data.Dataset.from_tensor_slices(([1.0]*1000, [1.0]*1000))

def get_dataset_for_worker(dataset, num_replicas, rank):
    dataset = dataset.shard(num_replicas, rank)
    dataset = dataset.batch(batch_size_per_worker)
    dataset = dataset.repeat()
    return dataset

distributed_dataset = strategy.distribute_datasets_from_function(lambda input_context: get_dataset_for_worker(dataset, strategy.num_replicas_in_sync, input_context.input_pipeline_id))


for inputs, labels in distributed_dataset.take(100):
  loss = train_step(inputs, labels)
  tf.print("Loss:", loss)

```

Here, `dataset.shard(num_replicas, rank)` splits the dataset such that `rank` worker only gets 1/`num_replicas` of the data. The `strategy.distribute_datasets_from_function` call then makes the sharded dataset available to each of the worker. The batch size is determined by dividing the overall batch size by the number of workers and this `batch_size_per_worker` value is then used.

Effective distributed training also involves monitoring the training process. The standard way to do this in tensorflow is to utilize `tf.summary` functionality to record training metrics. These logs can be then examined using TensorBoard to diagnose training performance. It is also advisable to use AWS CloudWatch to monitor system performance.

In terms of resource recommendations, I suggest investigating the official TensorFlow documentation, specifically the guides on distributed training. Additionally, publications on scaling deep learning models with TensorFlow can provide valuable insights into best practices. I would also recommend experimenting with different strategies provided by TensorFlow and reading through the associated code. These resources, in combination with the understanding of the cluster setup and effective data sharding, are critical for successfully implementing distributed TensorFlow on AWS.
