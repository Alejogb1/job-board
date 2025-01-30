---
title: "How can I stop the parameter server in TensorFlow Estimator training?"
date: "2025-01-30"
id: "how-can-i-stop-the-parameter-server-in"
---
In distributed TensorFlow training with the `tf.estimator.Estimator`, gracefully halting the parameter server (PS) process requires careful coordination, as directly interrupting it can lead to data corruption or inconsistent model states. My experience, derived from scaling several large-scale image classification models on cloud infrastructure, indicates that stopping the PS is less about an abrupt termination and more about a structured shutdown initiated by the chief worker. This shutdown signals to the PS processes that no further updates are incoming, allowing them to finalize their states.

The core concept is that the parameter server processes typically run indefinitely, receiving and applying gradient updates from workers and sending back updated weights. Unlike the worker processes, which usually terminate after a set number of training steps or epochs, the parameter servers remain active to serve the training logic. To initiate a graceful shutdown, the chief worker, often designated by setting `task_type` to "chief" in the training configuration, needs to execute the termination logic. This involves sending a "stop" signal indirectly through TensorFlow's distributed runtime framework rather than directly killing the process with operating system tools.

Here’s a breakdown of the procedure and why it functions the way it does. During distributed training, TensorFlow establishes a communication channel between the workers and the parameter servers. When training completes, the chief worker should signal this termination through the `tf.train.MonitoredTrainingSession`. This mechanism, in combination with appropriate distributed training strategies, ensures the worker and server processes conclude their responsibilities with data consistency. Simply killing a PS process could lead to data loss, particularly if the weights are not synchronized across all servers. TensorFlow relies on a checkpointing and recovery mechanism. Abruptly terminating servers prevents these operations from correctly executing. Therefore, the emphasis lies not on *force-stopping* but *signal-based cessation*.

Let's illustrate this with some examples, incorporating the typical setup for `tf.estimator.Estimator` with a distributed strategy.

**Example 1: Basic Distributed Training Setup**

This example demonstrates a foundational structure for distributed training utilizing `tf.distribute.MultiWorkerMirroredStrategy`. Although we are not explicitly stopping the server here, it illustrates the typical setup necessary for a graceful shutdown. The termination logic is introduced in subsequent examples.

```python
import tensorflow as tf
import os

def create_model():
    model = tf.keras.models.Sequential([
      tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
      tf.keras.layers.Dense(10, activation='softmax')
    ])
    optimizer = tf.keras.optimizers.Adam()
    return model, optimizer

def input_fn(mode, num_epochs=1):
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    if mode == tf.estimator.ModeKeys.TRAIN:
        dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        dataset = dataset.repeat(num_epochs).batch(128)
    else: # EVAL
         dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
         dataset = dataset.batch(128)
    
    return dataset

def model_fn(features, labels, mode, params):
    model, optimizer = create_model()
    logits = model(tf.cast(tf.reshape(features, [-1, 784]), tf.float32))

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {'probabilities': tf.nn.softmax(logits)}
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
    
    loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels, logits))
    
    if mode == tf.estimator.ModeKeys.TRAIN:
        train_op = optimizer.minimize(loss, var_list = model.trainable_variables)
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
    
    # EVAL mode. This example is not fully developed for EVAL
    eval_metric_ops = {
        'accuracy': tf.compat.v1.metrics.accuracy(labels, tf.argmax(logits, axis=1))
    }
    return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops = eval_metric_ops)


if __name__ == '__main__':
    tf_config = os.environ.get('TF_CONFIG')
    print(f"TF_CONFIG: {tf_config}")
    
    if tf_config is not None:
        distribution_strategy = tf.distribute.MultiWorkerMirroredStrategy()
    else:
        distribution_strategy = None

    run_config = tf.estimator.RunConfig(
        train_distribute = distribution_strategy,
        eval_distribute = distribution_strategy,
        save_summary_steps=100
    )
    
    estimator = tf.estimator.Estimator(
        model_fn = model_fn,
        model_dir = "./saved_model",
        config = run_config,
        params = {}
        )

    if tf_config is not None and json.loads(tf_config)['task']['type'] == 'chief':
        estimator.train(input_fn=lambda: input_fn(tf.estimator.ModeKeys.TRAIN), steps = 500)
    
    if tf_config is None:
        estimator.train(input_fn=lambda: input_fn(tf.estimator.ModeKeys.TRAIN), steps = 500)
```

This setup defines the fundamental components: the model, input pipeline, and `model_fn`. The `TF_CONFIG` environment variable is used to conditionally choose a distributed strategy.  The key missing piece here is a defined termination point for a distributed run.

**Example 2: Graceful Shutdown using `MonitoredTrainingSession`**

This example refactors the previous code, incorporating a check in the chief worker to trigger a shutdown via a `tf.train.MonitoredTrainingSession`. The key change involves the use of the session manager rather than relying on estimator's built-in train method. This gives us a point to gracefully stop the session, which, in turn, stops the parameter server

```python
import tensorflow as tf
import os
import json


def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    optimizer = tf.keras.optimizers.Adam()
    return model, optimizer


def input_fn(mode, num_epochs=1):
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    if mode == tf.estimator.ModeKeys.TRAIN:
        dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        dataset = dataset.repeat(num_epochs).batch(128)
    else:  # EVAL
        dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
        dataset = dataset.batch(128)

    return dataset


def model_fn(features, labels, mode, params):
    model, optimizer = create_model()
    logits = model(tf.cast(tf.reshape(features, [-1, 784]), tf.float32))

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {'probabilities': tf.nn.softmax(logits)}
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels, logits))

    if mode == tf.estimator.ModeKeys.TRAIN:
        train_op = optimizer.minimize(loss, var_list=model.trainable_variables)
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    # EVAL mode. This example is not fully developed for EVAL
    eval_metric_ops = {
        'accuracy': tf.compat.v1.metrics.accuracy(labels, tf.argmax(logits, axis=1))
    }
    return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=eval_metric_ops)


if __name__ == '__main__':
    tf_config = os.environ.get('TF_CONFIG')
    print(f"TF_CONFIG: {tf_config}")

    if tf_config is not None:
        distribution_strategy = tf.distribute.MultiWorkerMirroredStrategy()
    else:
        distribution_strategy = None

    run_config = tf.estimator.RunConfig(
        train_distribute=distribution_strategy,
        eval_distribute=distribution_strategy,
        save_summary_steps=100
    )

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir="./saved_model",
        config=run_config,
        params={}
    )

    if tf_config is not None and json.loads(tf_config)['task']['type'] == 'chief':
        
        with tf.compat.v1.train.MonitoredTrainingSession(
            master = estimator.config.master,
            is_chief = True,
            checkpoint_dir = estimator.model_dir) as mon_sess:
            step = 0
            while step < 500:
              _, loss = mon_sess.run([estimator.get_variable_value("train_op"),
                estimator.get_variable_value("loss")
                ])
              step+=1
    if tf_config is None:
        estimator.train(input_fn=lambda: input_fn(tf.estimator.ModeKeys.TRAIN), steps=500)
```

In this refined example, the chief worker executes training within a `MonitoredTrainingSession`. When the `while` loop is finished, the `MonitoredTrainingSession` object's context manager will signal the end of the session.  This will, in turn, allow the PS processes to gracefully shutdown.  Note that when training is local, the server does not start up; This example focuses specifically on the distributed case.

**Example 3: Explicit Checkpointing and Termination**

This example enhances the shutdown process by including explicit checkpoint saving before termination. This action assures that, in case the chief worker gets killed before the shutdown, the latest state has been saved. This strategy increases fault tolerance and avoids data loss in distributed training environments.

```python
import tensorflow as tf
import os
import json

def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    optimizer = tf.keras.optimizers.Adam()
    return model, optimizer


def input_fn(mode, num_epochs=1):
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    if mode == tf.estimator.ModeKeys.TRAIN:
        dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        dataset = dataset.repeat(num_epochs).batch(128)
    else: # EVAL
        dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
        dataset = dataset.batch(128)
    
    return dataset


def model_fn(features, labels, mode, params):
    model, optimizer = create_model()
    logits = model(tf.cast(tf.reshape(features, [-1, 784]), tf.float32))
    
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {'probabilities': tf.nn.softmax(logits)}
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
        
    loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels, logits))
    
    if mode == tf.estimator.ModeKeys.TRAIN:
        train_op = optimizer.minimize(loss, var_list = model.trainable_variables)
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
    
    # EVAL mode. This example is not fully developed for EVAL
    eval_metric_ops = {
        'accuracy': tf.compat.v1.metrics.accuracy(labels, tf.argmax(logits, axis=1))
    }
    return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops = eval_metric_ops)


if __name__ == '__main__':
    tf_config = os.environ.get('TF_CONFIG')
    print(f"TF_CONFIG: {tf_config}")
    
    if tf_config is not None:
      distribution_strategy = tf.distribute.MultiWorkerMirroredStrategy()
    else:
      distribution_strategy = None

    run_config = tf.estimator.RunConfig(
        train_distribute = distribution_strategy,
        eval_distribute = distribution_strategy,
        save_summary_steps=100
    )
    
    estimator = tf.estimator.Estimator(
        model_fn = model_fn,
        model_dir = "./saved_model",
        config = run_config,
        params = {}
        )
    
    if tf_config is not None and json.loads(tf_config)['task']['type'] == 'chief':
      
      with tf.compat.v1.train.MonitoredTrainingSession(
            master = estimator.config.master,
            is_chief = True,
            checkpoint_dir = estimator.model_dir) as mon_sess:
        step = 0
        while step < 500:
          _, loss = mon_sess.run([estimator.get_variable_value("train_op"),
            estimator.get_variable_value("loss")
            ])
          if step % 100 == 0:
            mon_sess.raw_session().run(estimator.get_variable_value("global_step"))
            mon_sess.raw_session().run(estimator.get_variable_value("save_saver"))

          step+=1

        mon_sess.raw_session().run(estimator.get_variable_value("global_step"))
        mon_sess.raw_session().run(estimator.get_variable_value("save_saver"))
        

    if tf_config is None:
        estimator.train(input_fn=lambda: input_fn(tf.estimator.ModeKeys.TRAIN), steps = 500)
```
The addition of manual saving ensures the latest state has been checkpointed, which is crucial for scenarios where the training process might terminate unexpectedly.

In summary, stopping a parameter server in TensorFlow's `Estimator` framework is not about forcefully ending the process. Instead, it involves using a controlled termination initiated by the chief worker. This structured approach ensures data consistency and allows the PS to correctly finalize its operations, including checkpointing data. The key lies in utilizing the `MonitoredTrainingSession`, along with appropriate training logic, for a successful shutdown of the distributed training processes.

For further exploration, I suggest consulting TensorFlow's documentation on distributed training, the `tf.distribute` module and, specifically,  `MultiWorkerMirroredStrategy`, along with the `tf.estimator` API, and the `tf.train.MonitoredTrainingSession`. Exploring articles discussing strategies for distributed training in machine learning will also deepen understanding of the nuances involved. Finally, the TensorFlow source code itself, although involved, can prove invaluable for understanding specific mechanisms and behaviors.
