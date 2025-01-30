---
title: "How can TensorFlow experiments run evaluations on worker nodes instead of the master?"
date: "2025-01-30"
id: "how-can-tensorflow-experiments-run-evaluations-on-worker"
---
Running TensorFlow evaluations on worker nodes, instead of the chief or master, is crucial for scaling distributed training effectively and preventing resource bottlenecks. In my experience building large-scale deep learning models, relying on the chief node for both training *and* evaluation often led to significant performance degradation. The chief node, already burdened with coordinating training tasks and managing parameter updates, simply couldn't handle the additional load of running evaluation loops. Moving evaluations to worker nodes is a strategy I’ve found essential to maintaining optimal throughput and resource utilization.

The core concept hinges on distributing the computation graph across multiple devices. In a standard distributed TensorFlow setup, a single 'master' or 'chief' node coordinates the training process. Worker nodes perform the actual computation, pulling gradient updates and pushing updated model weights. The evaluation process, traditionally initiated and performed on the chief node, can become a bottleneck as the model scales. To distribute evaluation, we must carefully structure the computation graph such that evaluation logic is defined and executed on the worker nodes. This requires modifying the training script to explicitly specify device placement for evaluation ops and distributing the evaluation datasets. We do not run evaluation *during* the training process on worker nodes. Instead we set up separate evaluation execution for worker nodes *after* the model is fully trained.

Consider this simplified training process which, while illustrative, does not include all the complexities of real-world distributed TensorFlow:

```python
import tensorflow as tf

# Dummy data and model
def create_model():
    return tf.keras.models.Sequential([
        tf.keras.layers.Dense(10, activation='relu', input_shape=(10,)),
        tf.keras.layers.Dense(2, activation='softmax')
    ])

def create_data(num_samples):
    X = tf.random.normal(shape=(num_samples, 10))
    y = tf.random.uniform(shape=(num_samples,), minval=0, maxval=2, dtype=tf.int32)
    return X, y


strategy = tf.distribute.MultiWorkerMirroredStrategy()

with strategy.scope():
    model = create_model()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    metrics = ['accuracy']


@tf.function
def train_step(x, y):
  with tf.GradientTape() as tape:
      logits = model(x, training=True)
      loss_value = loss_fn(y, logits)
  grads = tape.gradient(loss_value, model.trainable_variables)
  optimizer.apply_gradients(zip(grads, model.trainable_variables))
  return loss_value

def train_loop(epochs, train_data):
   for epoch in range(epochs):
      total_loss = 0
      for x_batch, y_batch in train_data:
        loss = strategy.run(train_step, args=(x_batch, y_batch))
        total_loss += loss.numpy()
      print(f'Epoch {epoch}, Average Loss: {total_loss / len(train_data)}')
   return model

train_samples = 1000
epochs = 20
X_train, y_train = create_data(train_samples)
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(32).shuffle(train_samples)
distributed_train_dataset = strategy.distribute_datasets_from_function(lambda _: train_dataset)

trained_model = train_loop(epochs, distributed_train_dataset)
```

In this first example, we have a basic training loop. The `tf.distribute.MultiWorkerMirroredStrategy` handles the data and model distribution, and the training loop runs across the worker nodes. This code, however, doesn't yet incorporate evaluation distribution. The core issue is that, by default, if we add an evaluation loop after training, it will execute on the chief node. We will not go into the details of setting up a proper multi-worker environment, which can vary based on the type of cluster we are operating in.

The subsequent step is to explicitly create a separate evaluation execution. We can do that by defining an eval function that runs after the training.

```python
@tf.function
def eval_step(x, y):
  logits = model(x, training=False)
  loss_value = loss_fn(y, logits)
  accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, axis=1, output_type=tf.int32), tf.cast(y, tf.int32)), tf.float32))
  return loss_value, accuracy


def eval_loop(test_data):
  total_loss = 0
  total_accuracy = 0
  for x_batch, y_batch in test_data:
      loss, accuracy = strategy.run(eval_step, args=(x_batch, y_batch))
      total_loss += loss.numpy()
      total_accuracy += accuracy.numpy()
  print(f'Evaluation Loss: {total_loss/len(test_data)}, Accuracy: {total_accuracy/len(test_data)}')


eval_samples = 200
X_test, y_test = create_data(eval_samples)
eval_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(32).shuffle(eval_samples)

distributed_eval_dataset = strategy.distribute_datasets_from_function(lambda _: eval_dataset)

eval_loop(distributed_eval_dataset)
```

In this second code segment, we define `eval_step` which performs the prediction and calculate the loss and accuracy metrics. The `eval_loop` then goes through the test dataset and calculates the average loss and accuracy. Importantly, the evaluation happens after the training is done and through `strategy.run`, the code is executed in the workers and the aggregated results are returned. In my projects, I have had to be very careful about data pipelining and to distribute the data properly across the workers.

The core distinction here is the explicit creation of an evaluation function, and then passing the test dataset and running it through `strategy.run` after the training phase. This ensures that evaluation computation happens across workers instead of on the chief. The primary difference here is the separation of training from evaluation, and running evaluation on the workers. This can be further expanded using custom metrics and callbacks, which need to be properly distributed as well. Furthermore, a complex task involves coordinating the checkpointing and loading models for evaluation by the workers.

Finally, let's explore distributing the *data loading* itself, since all of the above code assumes that the data is somehow accessible to all workers, perhaps through memory. When reading from disk, distributing this operation is crucial for large scale datasets. We will be showing one of the many possible approaches to do that using the `tf.data.Dataset.from_tensor_slices`. This approach requires us to read in data using a different method.

```python
import numpy as np
import os

def create_data_files(num_files, samples_per_file, output_dir):
  if not os.path.exists(output_dir):
     os.makedirs(output_dir)
  for i in range(num_files):
     X = np.random.normal(size=(samples_per_file, 10))
     y = np.random.randint(0, 2, size=samples_per_file)
     np.savez(os.path.join(output_dir, f'data_{i}.npz'), X=X, y=y)

def load_data(file_pattern):
    files = tf.io.gfile.glob(file_pattern)
    def _load_function(filename):
        data = np.load(filename.decode('utf-8'))
        X = data['X']
        y = data['y']
        return X.astype(np.float32), y.astype(np.int32)

    dataset = tf.data.Dataset.from_tensor_slices(files)
    dataset = dataset.map(lambda filename: tf.py_function(_load_function, [filename], [tf.float32, tf.int32]),
                                      num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.unbatch()
    return dataset

num_files = 4
samples_per_file = 100
data_dir = "data_files"

create_data_files(num_files, samples_per_file, data_dir)

train_file_pattern = os.path.join(data_dir, 'data_*.npz')

train_dataset = load_data(train_file_pattern)
train_dataset = train_dataset.batch(32)
distributed_train_dataset = strategy.distribute_datasets_from_function(lambda _: train_dataset)

test_dataset = load_data(train_file_pattern) # For demonstration, using the same dataset for eval
test_dataset = test_dataset.batch(32)
distributed_test_dataset = strategy.distribute_datasets_from_function(lambda _: test_dataset)


trained_model = train_loop(epochs, distributed_train_dataset)

eval_loop(distributed_test_dataset)
```

Here we create some dummy data files that are read in. The `load_data` function uses a `file_pattern` and loads the corresponding files and also converts them to `tf.data.Dataset`, which now allows for better data pipelining and distribution. `tf.py_function` is necessary here to work with `numpy`, inside of the tensorflow graph. In practice, if the data is stored in TFRecord files, that would be a better alternative. It's important to also distribute these files across different disks so that all the workers are not reading from the same location which would create a bottleneck.

The key takeaway is to avoid running evaluation tasks on the chief or master worker node. Instead, distribute the evaluation computation across available worker nodes. This requires careful design of the TensorFlow graph, explicit device placement, appropriate data distribution strategies, and, if necessary, checkpoint handling and custom metric implementations. When implemented correctly, this approach will result in more efficient resource utilization and more scalable training and evaluation.

Regarding resources, I recommend focusing on official TensorFlow documentation regarding distributed strategies, particularly the `MultiWorkerMirroredStrategy`, and the `tf.distribute.Strategy` API in general. The `tf.data` documentation, especially around performance optimization, will be invaluable when loading and distributing data from a variety of sources. Moreover, explore examples related to distributed training on platforms like Google Cloud TPUs or GPUs, as these often present the most complex scenarios requiring distribution. Finally, understanding the `tf.function` decorator and its implications for graph creation is important in the context of distributed training.
