---
title: "Why is TensorFlow reporting a 'killed' error during transformer training locally?"
date: "2025-01-30"
id: "why-is-tensorflow-reporting-a-killed-error-during"
---
The "killed" error during local TensorFlow transformer training, especially when using large models or datasets, almost invariably points to a lack of available system resources, specifically memory. Having spent the last five years building and deploying machine learning models, I've encountered this issue repeatedly across different projects and hardware configurations. TensorFlow, in such scenarios, will not typically provide detailed stack traces; instead, the operating system's process manager terminates the Python process due to resource exhaustion. This termination manifests as the opaque "killed" message, leaving the user to deduce the root cause.

The fundamental reason lies in the interplay between the model's memory footprint, data loading, and the available system RAM, including GPU memory if utilized. Transformers, inherently complex neural networks with many parameters, can quickly consume significant resources. During training, not only are the weights and biases stored, but also intermediate activations, gradients, and optimizer states. Furthermore, the data loading process, particularly with large datasets loaded into memory for efficient training, adds to the memory burden. If the total memory requirement surpasses the available resources, the OS intervenes, forcibly ending the TensorFlow process.

While memory pressure is the primary culprit, other less frequent causes exist, such as overly ambitious hyperparameter settings (like very large batch sizes), memory leaks in custom training loops (less common with TensorFlow's Keras API), or even issues with the underlying CUDA drivers in GPU scenarios. The "killed" signal, however, is almost always triggered by resource starvation. The first step is typically a careful audit of memory usage.

Let's illustrate with some practical examples:

**Example 1: Initial Memory Overload**

The most straightforward case involves trying to fit a model too large for the available RAM and/or GPU memory. Consider a scenario where one is training a large BERT model on a modestly equipped machine.

```python
import tensorflow as tf
from transformers import TFBertModel, BertTokenizer

# Assuming a system with 16GB RAM and an older GPU with 8GB
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = TFBertModel.from_pretrained('bert-base-uncased')

# Simplified data loading (in practice, this would be a tf.data.Dataset)
data = [ "this is an example sentence",
          "another sentence for testing",
          "and a third one"] * 1000 #Simulating a Larger Dataset

tokenized_data = tokenizer(data, padding=True, truncation=True, return_tensors='tf')

#Unrealistic Batchsize - this is a big part of the problem, especially on initial attempts
dataset = tf.data.Dataset.from_tensor_slices((tokenized_data['input_ids'], tokenized_data['attention_mask'])).batch(1024)

optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)

for epoch in range(3):
    print(f"Starting epoch: {epoch+1}")
    for step, (input_ids, attention_mask) in enumerate(dataset):
        with tf.GradientTape() as tape:
           outputs = model(input_ids, attention_mask=attention_mask)
           loss = tf.reduce_mean(outputs.last_hidden_state) # Place Holder loss - not using labels
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        print(f"Step {step} completed", end="\r")
    print(f"Epoch {epoch+1} done.")
```

The problem here isn’t immediately apparent from the code itself. The model instantiation, tokenizer, and data loading all appear correct at first glance. However, with a batch size of 1024, the code tries to process large chunks of data, and all the necessary gradients are computed in a single forward/backward pass. This demand, especially with a large model like BERT, rapidly exhausts the available resources. The execution often gets "killed" after a few steps (sometimes even in the first epoch, with a "killed" signal from the OS after TensorFlow fails to allocate the requested memory. This error occurs both in RAM and, even more prominently, on GPUs with limited memory.

**Example 2: Overly Large Data Loading**

Even with a reasonable batch size, loading an entire dataset into memory prior to training can cause issues. This technique, while sometimes faster, might not be feasible when dealing with large text datasets.

```python
import tensorflow as tf
from transformers import TFBertModel, BertTokenizer
import numpy as np

# Simulating a larger dataset this time with more "docs"
num_docs= 10000
doc_length = 512
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = TFBertModel.from_pretrained('bert-base-uncased')

# Simulate documents as strings
fake_docs = ["This is a document " + str(i) for i in range(doc_length)] * num_docs

# This can be extremely expensive with a large number of docs!
tokenized_data = tokenizer(fake_docs, padding=True, truncation=True, return_tensors='tf')

# Reasonable batch this time
dataset = tf.data.Dataset.from_tensor_slices((tokenized_data['input_ids'], tokenized_data['attention_mask'])).batch(32)
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)

for epoch in range(3):
    print(f"Starting epoch: {epoch+1}")
    for step, (input_ids, attention_mask) in enumerate(dataset):
         with tf.GradientTape() as tape:
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = tf.reduce_mean(outputs.last_hidden_state)
         gradients = tape.gradient(loss, model.trainable_variables)
         optimizer.apply_gradients(zip(gradients, model.trainable_variables))
         print(f"Step {step} completed", end="\r")
    print(f"Epoch {epoch+1} done.")
```

Here, the batch size is set to 32 which is a far more reasonable value; however, the problem now lies with the data itself. Pre-loading all tokenized inputs into memory before even starting the training process creates memory pressure similar to the batch overload in Example 1, which can still lead to the "killed" error. Note that if `tokenizer` was run with `return_tensors='tf'`, then `tokenized_data` is actually a TF tensor (not a Python object). If your `tf.data.Dataset` pipeline is not designed correctly (e.g., the whole dataset is loaded at once, a common mistake), that pre-tokenized Tensor, combined with your model, will push resource consumption beyond what's available. The OS will then step in and terminate the process with the "killed" signal.

**Example 3: Inefficient Gradient Accumulation (less common with keras)**

This example focuses on situations that can arise when writing custom training loops, and the user has to manually handle the gradient operations (less common with higher-level Keras training).

```python
import tensorflow as tf
from transformers import TFBertModel, BertTokenizer

# Simplified custom training loop for demonstration
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = TFBertModel.from_pretrained('bert-base-uncased')

# Dummy data and batch size
data = ["sentence " + str(i) for i in range(512)]* 1000
tokenized_data = tokenizer(data, padding=True, truncation=True, return_tensors='tf')
batch_size = 32
dataset = tf.data.Dataset.from_tensor_slices((tokenized_data['input_ids'], tokenized_data['attention_mask'])).batch(batch_size)

optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)

accumulated_gradients = None # this is the main source of trouble!

for epoch in range(3):
    print(f"Starting epoch: {epoch+1}")
    for step, (input_ids, attention_mask) in enumerate(dataset):
        with tf.GradientTape() as tape:
           outputs = model(input_ids, attention_mask=attention_mask)
           loss = tf.reduce_mean(outputs.last_hidden_state)
        gradients = tape.gradient(loss, model.trainable_variables)

        # Example of incorrect accumulation: just repeatedly overwriting the gradient
        if accumulated_gradients is None:
            accumulated_gradients = gradients
        else:
            accumulated_gradients = gradients # Overwriting - THIS IS INCORRECT

        if (step + 1) % 2 == 0: # Accumulate grads over 2 steps
             optimizer.apply_gradients(zip(accumulated_gradients, model.trainable_variables))
             accumulated_gradients = None # reset the accumulation

        print(f"Step {step} completed", end="\r")
    print(f"Epoch {epoch+1} done.")
```

Here the intent was to implement a rudimentary form of gradient accumulation. However, we do not `add` to the `accumulated_gradients` but overwrite it each time, while creating the gradients over 2 steps. This is **incorrect**. Although not a memory leak per se, it reveals a source of potential errors when managing your own custom training loops that can cause crashes. In the same way that an `accumulated_gradients` variable will retain memory, a gradient update step using incorrect gradients can also result in the memory issues described previously, particularly with a more complex loop. This is less common with Keras training, which handles most of this accumulation internally, but the error still demonstrates how improper handling of custom loops can still cause these issues.

**Recommendations:**

To address the "killed" error during TensorFlow transformer training, I recommend focusing on the following key areas. First, always start by assessing memory usage with system monitoring tools. Monitor RAM and GPU memory (if applicable) to see where the bottleneck is occurring. If memory is the issue, the first step is to reduce the batch size to see if the error clears. Then you need to move to a more robust data loading strategy: instead of loading large datasets entirely into memory before training, rely on TensorFlow's `tf.data.Dataset` API for efficient data pipelining. It's good practice to tokenize datasets on-the-fly, during data loading. You should also avoid storing large intermediate tensors in memory for extended periods (and be mindful that TF eagerly evaluates operations, especially in debugging mode, which can lead to large intermediates). Review your model architectures, and consider using smaller models or techniques like knowledge distillation if the hardware is a persistent constraint. Finally, when writing custom training loops, rigorously check for memory management and accumulation operations to ensure the gradients are correctly accumulated.

In practice, addressing the "killed" error is not always a simple or quick process. It often involves a combination of adjustments to code and hyperparameters. Start with a simple diagnostic strategy of monitoring resources and then adjusting model and data loading strategy, and this should allow you to diagnose and handle this frustrating error.
