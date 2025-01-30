---
title: "Why can't a Keras model run on GPU using PlaidML?"
date: "2025-01-30"
id: "why-cant-a-keras-model-run-on-gpu"
---
The inability of a Keras model to run on a GPU via PlaidML, despite the promise of hardware acceleration, stems primarily from PlaidML’s architecture and its interaction with the TensorFlow backend, often coupled with limitations in device driver support and PlaidML's own developmental stage. Specifically, PlaidML operates as a translation layer, not as a native low-level GPU driver, which introduces complexities that directly impact the expected seamless integration with Keras.

The core of the issue lies in how Keras interfaces with its backend. Keras, as a high-level API, abstracts away the lower-level details of tensor operations. It relies on either TensorFlow, Theano, or CNTK (although CNTK support is now deprecated) to execute the actual numerical computations. When TensorFlow is selected as the backend, Keras constructs a TensorFlow computational graph, which then is handed over to TensorFlow's execution engine. This is where the incompatibility often emerges. TensorFlow’s core functionality has been extensively optimized for NVIDIA’s CUDA architecture, relying on cuDNN for highly efficient GPU acceleration. PlaidML, in contrast, attempts to intercept the TensorFlow operations and translate them into its own Intermediate Representation (IR). This IR is designed to be portable across different hardware backends including AMD, Intel and Apple GPUs, as well as CPUs.

The challenge arises from the fact that this translation process is not perfect, and several factors can contribute to a breakdown in GPU execution. First, PlaidML’s coverage of TensorFlow’s extensive operator set is not exhaustive. Some less common or highly optimized TensorFlow ops may not have a direct equivalent within PlaidML’s IR, leading to either a fallback to CPU execution, a partial execution on GPU, or outright failure. This incomplete translation can effectively negate the intended performance gains associated with GPU hardware. Second, the optimization pipeline within PlaidML’s framework is fundamentally different from the highly tuned CUDA-centric execution paths within TensorFlow itself. PlaidML may not perform necessary tensor layout transpositions or memory allocation strategies which are optimal for specific hardware leading to severely degraded performance. Finally, PlaidML may exhibit limited support for specific driver versions or specific GPU architectures, as the library itself is in a stage of active development. I've experienced first-hand, projects stalled due to this, where specific hardware worked on some older driver versions but not on newer ones with PlaidML. This highlights one of the core limitations: that its ecosystem is often less mature, and has a faster rate of changes in comparison to the CUDA ecosystem.

To clarify, it's not that Keras models cannot run at all when PlaidML is in place; they will often execute on the CPU or on the GPU in a sub-optimal way, but achieving efficient GPU acceleration for computationally intensive workloads, which is the key advantage of using GPUs in Deep Learning, proves to be problematic. It is also important to remember that TensorFlow's inherent support for a limited range of GPUs is not the cause, because PlaidML can interface with hardware other than NVIDIA GPUs. It is the translation layer that becomes problematic and not the hardware or backend selection.

Here are three code examples to illustrate where failures can occur.

**Example 1: A simple CNN model without advanced layers**

```python
import tensorflow as tf
from tensorflow import keras
import plaidml.keras

plaidml.keras.install_backend()

model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Assume x_train, y_train are properly formatted data
# model.fit(x_train, y_train, epochs=2)

```

In this case, the model, comprising common layers like `Conv2D`, `MaxPooling2D`, `Flatten`, and `Dense` may run, but you will typically find that PlaidML defaults to CPU execution or performance is significantly below expectations. If you set up `plaidml.settings` to allow printing of the operations, you will see that PlaidML does not correctly utilize your GPU. Some of the common reasons can be that the kernels for the specific convolution or pooling ops might not have been perfectly translated. Even if they have, sub-optimal memory movement between GPU memory, and host RAM can become a bottleneck. This has been a fairly consistent experience in my usage of PlaidML.

**Example 2: Model with Batch Normalization**

```python
import tensorflow as tf
from tensorflow import keras
import plaidml.keras
plaidml.keras.install_backend()

model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Assume x_train, y_train are properly formatted data
# model.fit(x_train, y_train, epochs=2)

```

Batch Normalization layers can create problems because of the way they operate and how they need to be translated into the PlaidML IR. PlaidML may not correctly handle the statistical calculations performed during Batch Normalization, leading to inconsistencies or fallback to CPU calculations. This is in particular the case when batch sizes are small, or when using recurrent neural networks.

**Example 3: Model using an LSTM layer**

```python
import tensorflow as tf
from tensorflow import keras
import plaidml.keras
plaidml.keras.install_backend()

model = keras.Sequential([
    keras.layers.LSTM(128, input_shape=(None, 10)),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Assume x_train, y_train are properly formatted data
# model.fit(x_train, y_train, epochs=2)
```

Long Short-Term Memory (LSTM) layers involve complex iterative computations. PlaidML often does not fully optimize the execution of such recurrent operations, especially with variable length sequence input. PlaidML might not be able to utilize the full parallelism offered by the GPU architecture, leading to performance which does not achieve the levels seen when running on optimized NVIDIA cuDNN implementations. Based on my experience, more complex RNN cells such as GRU, are also likely to trigger such failures. In some cases, PlaidML may exhibit memory allocation issues when dealing with LSTMs, causing crashes or incorrect results.

To improve the situation and move towards viable GPU usage, several resources can be useful. I recommend consulting the documentation for your specific GPU manufacturer; the latest AMD documentation, Intel OpenCL resources and Apple's Metal API guides are crucial to ensuring that your device drivers are properly configured. The PlaidML repository on GitHub should be checked for recent updates and known issues. Also, thorough testing of PlaidML with simpler models and a systematic increase in complexity may allow you to pinpoint the particular op or configuration that prevents your model from running on the GPU. Looking at community forums related to PlaidML, and Tensorflow issues relating to different GPU setups can be helpful. These are usually good sources of insights into issues specific to different hardware.
