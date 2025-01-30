---
title: "Why am I getting a 'KeyError: 'see'' when using fit_generator() in TensorFlow 2.7.0?"
date: "2025-01-30"
id: "why-am-i-getting-a-keyerror-see-when"
---
The `KeyError: 'see'` when using `fit_generator()` in TensorFlow 2.7.0 arises from a mismatch between the data structure returned by your generator and the expectations of the `fit_generator()` function, specifically concerning the expected keys within the returned dictionaries. In my experience debugging similar issues, particularly during a project involving custom image augmentation, the problem typically centers on how the generator yields data.

Let's delve into the root cause. The `fit_generator()` method, although deprecated in favor of `model.fit()` utilizing data generators, expected a specific format. Before TensorFlow 2.1, it primarily accepted a tuple of (inputs, targets). However, with TensorFlow 2.x, including version 2.7.0, when used with dictionary-based input/output, the generator needs to yield dictionaries for both inputs and targets, matching the model's input and output layers. The 'see' key, or any unexpected key, indicates the generator is not consistently delivering these dictionaries, or the keys do not align with the model definition and compiled training procedure. If the generator yields a tuple or dictionary without the keys expected by `fit_generator()` or the model, a `KeyError` arises. The specific error `'see'` suggests, the code is trying to access a key that doesn't exist in the dictionary the generator is producing.

I've encountered three variations of this scenario commonly and will illustrate how to address each of them.

**Example 1: Incorrect Key Structure**

Consider the scenario where a generator attempts to yield a simple tuple:

```python
import numpy as np
import tensorflow as tf

def sample_generator_tuple(batch_size):
  while True:
    images = np.random.rand(batch_size, 28, 28, 3)
    labels = np.random.randint(0, 10, batch_size)
    yield (images, labels)

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 3)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

batch_size = 32
gen_tuple = sample_generator_tuple(batch_size)

try:
  model.fit_generator(gen_tuple, steps_per_epoch=10, epochs=1) # KeyError
except KeyError as e:
    print(f"Caught KeyError: {e}")
```

In this case, the generator yields a tuple `(images, labels)`. However, if the model is designed to accept and return dictionaries with keys such as 'input_1' (or equivalent) and 'output_1', and the `compile` or `fit` methods are expecting such named inputs/targets,  `fit_generator` will throw a `KeyError` if the generator doesn't produce the dictionary. If one inspects the stack trace of a typical `KeyError` here, they’d often see the traceback point to the Keras internals attempting to access a key like `input_1` or `labels` within what was received from the generator and not being able to find them because a simple tuple was given, not a dictionary.

**Resolution:**

Modify the generator to return a dictionary structure:

```python
def sample_generator_dict(batch_size):
  while True:
    images = np.random.rand(batch_size, 28, 28, 3)
    labels = np.random.randint(0, 10, batch_size)
    yield {'input_1': images}, {'output_1': labels}

gen_dict = sample_generator_dict(batch_size)

model.fit_generator(gen_dict, steps_per_epoch=10, epochs=1) # Now works correctly.
```
Here, the generator yields dictionaries with keys `'input_1'` and `'output_1'`. Ensure these keys match the input and output layer names or the expected names within model compilation. If you have a more complex model with multiple inputs/outputs, the dictionary must reflect all named keys.

**Example 2: Inconsistent Key Usage**

Another common scenario is a generator that sometimes returns the expected dictionary, and other times returns data in a different format, for example, during error handling or data pre-processing steps.

```python
def sample_generator_inconsistent(batch_size):
  i = 0
  while True:
     images = np.random.rand(batch_size, 28, 28, 3)
     labels = np.random.randint(0, 10, batch_size)

     if i % 5 == 0:
       yield (images, labels)
     else:
       yield {'input_1': images}, {'output_1': labels}
     i += 1

gen_inconsistent = sample_generator_inconsistent(batch_size)
try:
    model.fit_generator(gen_inconsistent, steps_per_epoch=10, epochs=1) # KeyError
except KeyError as e:
    print(f"Caught KeyError: {e}")
```

This generator returns a tuple every fifth iteration and a dictionary otherwise. While it returns the correct structure at times, that inconsistency causes the `KeyError` because `fit_generator` expects every batch to have an identical structure.

**Resolution:**

Ensure consistent dictionary structure in every iteration.
```python
def sample_generator_consistent(batch_size):
    while True:
      images = np.random.rand(batch_size, 28, 28, 3)
      labels = np.random.randint(0, 10, batch_size)
      yield {'input_1': images}, {'output_1': labels}

gen_consistent = sample_generator_consistent(batch_size)
model.fit_generator(gen_consistent, steps_per_epoch=10, epochs=1)
```
Here we ensure every yield returns the same structure: input and output data in a dictionary with the proper keys. This solves the inconsistency.

**Example 3: Key Mismatch with Model Layers**

Sometimes, the generator returns dictionaries, but the keys do not align with the expected input and output names, particularly for named layers within a more complex model.
```python
model_named_layers = tf.keras.models.Model(inputs=tf.keras.Input(shape=(28, 28, 3), name='input_layer'),
    outputs = tf.keras.layers.Dense(10, activation='softmax', name='output_layer')(tf.keras.layers.Flatten()(tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(tf.keras.Input(shape=(28, 28,3))))))

model_named_layers.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


def sample_generator_wrong_keys(batch_size):
  while True:
    images = np.random.rand(batch_size, 28, 28, 3)
    labels = np.random.randint(0, 10, batch_size)
    yield {'input_1': images}, {'output_1': labels}  # Wrong keys

gen_wrong_keys = sample_generator_wrong_keys(batch_size)

try:
    model_named_layers.fit_generator(gen_wrong_keys, steps_per_epoch=10, epochs=1)  # KeyError
except KeyError as e:
    print(f"Caught KeyError: {e}")
```
The model is set up with input named 'input_layer' and output named 'output_layer', but the generator is using 'input_1' and 'output_1' keys.

**Resolution:**
The keys in the yielded dictionaries must align with the defined model layer names and the `compile` method.

```python
def sample_generator_correct_keys(batch_size):
    while True:
        images = np.random.rand(batch_size, 28, 28, 3)
        labels = np.random.randint(0, 10, batch_size)
        yield {'input_layer': images}, {'output_layer': labels} # Correct Keys

gen_correct_keys = sample_generator_correct_keys(batch_size)
model_named_layers.fit_generator(gen_correct_keys, steps_per_epoch=10, epochs=1) # Now works.
```

By using the layer names, `input_layer` and `output_layer` in this case, as keys, `fit_generator` now successfully passes the data to the model.

**Resource Recommendations**

For further understanding, consult the TensorFlow documentation focusing on the Keras API, particularly for custom data loading with `tf.data`. Also, review information on model compilation and the structure of input data for training. The Keras documentation on data generators is also helpful, although it's best to migrate away from `fit_generator()` to modern approaches utilizing the `tf.data` API for more flexibility and efficiency. Specific books and courses on deep learning and TensorFlow will often have sections on data pipelines.

In conclusion, the `KeyError: 'see'` stems from inconsistent or incorrect dictionary structure in your generator, and often due to a mismatch between the keys in the data dictionary yielded by the generator, with the keys the `fit_generator` method, or more specifically the underlying Keras Model expects to be present based on the model's layer input and output names or how it was compiled. Careful review and adherence to these data format requirements, especially when working with dictionary-based input, will eliminate the `KeyError`. The examples provided should help clarify potential causes and the needed adjustments.
