---
title: "How can I execute multiple TensorFlow code instances on a single GPU?"
date: "2025-01-30"
id: "how-can-i-execute-multiple-tensorflow-code-instances"
---
The efficient utilization of a single GPU for multiple TensorFlow model training or inference processes is crucial when resource constraints are a factor, especially in environments lacking access to multi-GPU systems. The primary challenge stems from TensorFlow's default behavior of attempting to monopolize the available GPU memory, causing conflicts when multiple scripts attempt to initialize their respective sessions. The solutions involve carefully managing GPU memory allocation and leveraging TensorFlow's graph execution mechanisms to avoid memory conflicts and ensure proper resource utilization. I've successfully implemented several strategies in the past while managing machine learning pipelines on compute-constrained edge devices, and these experiences have shaped the following approaches.

A fundamental concept is to prevent each TensorFlow instance from assuming it has exclusive access to the entire GPU. Two main techniques facilitate this: limiting GPU memory visibility per process and employing graph-level parallelism through multiple queues.

**1. Limiting GPU Memory Visibility per Process**

By configuring the GPU memory allocation for each process, we prevent one script from over-consuming resources that another script might require. This doesn’t directly execute code concurrently but allows for a controlled sharing of the GPU’s memory. This is achieved through `tf.config.experimental.set_memory_growth` or `tf.config.experimental.set_virtual_device_configuration`. The former allows TensorFlow to allocate memory as needed, instead of reserving the whole GPU initially, which leads to the most effective usage and prevents memory exhaustion errors when multiple models are loaded on the same device. This option is preferred since it is dynamic.

The following example demonstrates how to configure memory growth for a single TensorFlow script that will run in this multiple process context.

```python
import tensorflow as tf

# Configure GPU options to only grow memory as needed, and don't lock the GPU memory at first.
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU memory growth enabled.")
    except RuntimeError as e:
        print(f"Error enabling GPU memory growth: {e}")
else:
    print("No GPUs found.")


# Dummy computation to illustrate that memory growth is enabled. In actual scenario this is
# replaced by loading the model and performing training/inference.
a = tf.random.normal((10000, 10000))
b = tf.random.normal((10000, 10000))
c = tf.matmul(a, b)

print("Computation done")
```

This script checks for available GPUs, iterates over them, and enables memory growth using `tf.config.experimental.set_memory_growth(gpu, True)`. Instead of allocating memory, this will now allocate as it’s needed for a given process. When multiple scripts are launched with this setup, each will independently manage its own memory usage instead of one hogging up all the available memory, preventing the often observed out-of-memory errors. The matrix multiplication serves as a placeholder for any TensorFlow operations. In my experience, this is most effective when loading different models across separate processes.

**2. Utilizing Graph-Level Parallelism with Multiple Queues**

While the prior method primarily focuses on memory management, true concurrent execution of multiple TensorFlow models on the same GPU needs a more involved approach. This typically requires dividing computation into subgraphs, placing them in queues, and processing these queues using a shared set of GPU resources, allowing for concurrency. This requires careful design and is more complex to manage. It is the most effective method if multiple models need to be run concurrently rather than sequentially.

The core idea revolves around splitting the workload and using queues to pass the data and the operations to the shared resources. This is particularly useful when the goal isn’t to run different models, but instead, different parts of the same model. In the context of multiple independent TensorFlow instances, this approach still requires splitting each model into subgraphs which are passed to different queues that execute on the same GPU. Here’s a simplified structure of how that could work using the Queue functionality available in TensorFlow:

```python
import tensorflow as tf
import threading
import time

# Define a function for processing elements from the queue
def process_queue(queue, device_name, index):
    with tf.device(device_name):
        while True:
            try:
               op_name = queue.dequeue()
               print(f"Device: {device_name} executing {op_name} in thread {index}")
               time.sleep(2) # Simulate processing
            except tf.errors.OutOfRangeError:
                print(f"Thread {index} done.")
                break

# Example graph operations
with tf.Graph().as_default():
    q = tf.queue.FIFOQueue(10, dtypes=['string'])

    enq_ops = [q.enqueue("Model 1 Operation 1"), q.enqueue("Model 1 Operation 2"),
               q.enqueue("Model 2 Operation 1"), q.enqueue("Model 2 Operation 2"),
               q.enqueue("Model 1 Operation 3")]

    qr = tf.queue.QueueRunner(q, [q.dequeue() for i in range(1)])

    # Assuming we have a GPU available, we pick the first one.
    gpus = tf.config.list_physical_devices('GPU')
    device_name = '/device:GPU:0' if gpus else '/device:CPU:0'

    with tf.compat.v1.Session() as sess:
        if gpus:
            tf.config.experimental.set_memory_growth(gpus[0], True)

        coord = tf.train.Coordinator()
        enqueue_threads = qr.create_threads(sess, coord=coord, start=True)
        thread1 = threading.Thread(target=process_queue, args=(q, device_name, 1))
        thread2 = threading.Thread(target=process_queue, args=(q, device_name, 2))

        thread1.start()
        thread2.start()
        for enq_op in enq_ops:
            sess.run(enq_op)


        coord.request_stop()
        coord.join(enqueue_threads)
        thread1.join()
        thread2.join()
```

In this more complex scenario, the key is that rather than different models attempting to occupy the GPU’s resources, independent models are decomposed into operations that can be sent to a queue to be processed. In the provided example the operations are simply a string, but this would be a subgraph of a model in an actual scenario. As a result, multiple models can be processed, and if there is sufficient capacity, they can be done so in parallel using the same GPU. In my prior experience, this required a lot more engineering, but yielded the best performance if concurrent execution is the key objective. This is especially true when the same model needs to process inputs in parallel and at scale.

**3. Using `tf.distribute.Strategy`**

TensorFlow’s distribution strategies offer another way to control execution and, indirectly, can manage multiple instances on a single GPU. It is particularly useful for data parallelism. If you have data that is distributed across multiple input queues, and each should execute the same model, this method is the most efficient.

Here is an example that uses the mirror strategy:

```python
import tensorflow as tf
import time

# Define a dummy model
def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(10,)),
        tf.keras.layers.Dense(1)
    ])
    return model

# Generate some dummy data
def generate_data():
    x = tf.random.normal((100, 10))
    y = tf.random.normal((100, 1))
    return tf.data.Dataset.from_tensor_slices((x,y)).batch(10).repeat()


# Use MirroredStrategy for data parallelism
strategy = tf.distribute.MirroredStrategy()

with strategy.scope():

    model = create_model()
    optimizer = tf.keras.optimizers.Adam()
    loss_fn = tf.keras.losses.MeanSquaredError()

    def train_step(inputs):
        x, y = inputs
        with tf.GradientTape() as tape:
            predictions = model(x)
            loss = loss_fn(y, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    # Compile the training step function as a graph
    distributed_train_step = strategy.run(train_step)

    # Distribute the data set
    dataset = generate_data()
    distributed_dataset = strategy.experimental_distribute_dataset(dataset)


    num_epochs = 3
    for epoch in range(num_epochs):
        print(f"Starting Epoch {epoch + 1}")
        i = 0
        for inputs in distributed_dataset:
          loss = distributed_train_step(inputs)
          i = i + 1
          print(f"Step {i}: Loss: {loss}")
          if i>10:
              break
```

In this example, `tf.distribute.MirroredStrategy` duplicates the model weights and variables across the GPU (or multiple GPUs if available), effectively running it multiple times in parallel on different data batches. The strategy takes care of data distribution, gradient aggregation, and model updates under the hood, which makes it easier to manage multiple model execution within the same process. This provides data parallelism in a way that is much more manageable than the queue example, at the expense of potentially having the same model running for all threads. In my prior work this method was the most effective when I needed to train the same model on different subsets of data at the same time, using distributed training.

These three methods, `set_memory_growth`, graph parallelism with queues, and `tf.distribute.Strategy`, offer viable paths for executing multiple TensorFlow code instances on a single GPU. The appropriate approach depends heavily on the specific requirements of the application, with the choice between them depending on the type of concurrency, whether memory management is the key objective, or whether data parallelism is more important.

For further study of TensorFlow’s resource management, I recommend reviewing the official TensorFlow documentation on GPU configuration and distribution strategies. Books detailing advanced TensorFlow programming patterns and multi-threading techniques are useful as well. Furthermore, examining the source code of frameworks built upon TensorFlow, such as TensorFlow Serving and TensorFlow Distributed, can yield significant insights into these more advanced practices.
