---
title: "How do I install TensorFlow GPU 1.15 on Ubuntu?"
date: "2025-01-30"
id: "how-do-i-install-tensorflow-gpu-115-on"
---
TensorFlow 1.15's GPU support hinges critically on CUDA toolkit version compatibility.  My experience troubleshooting installations across diverse hardware configurations highlighted this repeatedly.  Ignoring the precise CUDA version required often results in cryptic errors during TensorFlow's import or runtime, frequently masked as broader system issues. Therefore, meticulously verifying CUDA compatibility is paramount before attempting any TensorFlow installation.

**1.  Clear Explanation of the Installation Process:**

Installing TensorFlow 1.15 with GPU support on Ubuntu requires a systematic approach involving several distinct steps.  Failure to properly execute each step sequentially will almost certainly lead to installation failure. This process can be broken down into:

a) **System Preparation:** Ensuring your Ubuntu system meets the minimum requirements is fundamental. This includes verifying your GPU's CUDA capability,  sufficient RAM (at least 8GB is recommended, though 16GB is preferable for larger models), and a compatible NVIDIA driver installation.  I've seen numerous instances where outdated or conflicting drivers caused unforeseen problems.  Confirming your driver version via `nvidia-smi` is essential.

b) **CUDA Toolkit Installation:** This is arguably the most crucial step.  TensorFlow 1.15 demands a specific CUDA toolkit version; deviating from this will cause immediate incompatibility. Consulting the official TensorFlow 1.15 release notes for the correct CUDA version is imperative.  This often involves downloading the appropriate `.run` file from the NVIDIA website, executing it, and verifying the installation with the `nvcc --version` command. The installation will likely require you to add the CUDA binaries to your system's PATH environment variable.

c) **cuDNN Installation:** The CUDA Deep Neural Network library (cuDNN) accelerates deep learning computations.  Again, a specific cuDNN version matching your chosen CUDA toolkit is mandatory.  Download the correct cuDNN library (as a `.tar.gz` file) from the NVIDIA website, extract the contents, and copy the necessary files to the appropriate CUDA directories.  The NVIDIA documentation provides clear instructions on this, which I have followed countless times with success.

d) **TensorFlow Installation:** With CUDA and cuDNN correctly configured, installing TensorFlow 1.15 itself becomes relatively straightforward.  Using pip is generally the preferred method.  I've encountered issues with other methods, primarily due to dependency conflicts.  The pip command will typically resemble: `pip install tensorflow-gpu==1.15.0`.  Pay close attention to the `==1.15.0` to ensure the correct version is installed.  This command leverages the pre-built TensorFlow wheel file for GPU support.

e) **Verification:** Finally, verify your installation.  Open a Python interpreter and attempt to import TensorFlow.  Then, execute code utilizing GPU operations to confirm hardware acceleration is functioning.  Failing this stage usually indicates problems in the preceding steps, particularly regarding CUDA and cuDNN configurations.


**2. Code Examples with Commentary:**

**Example 1:  Verifying CUDA and cuDNN Installation:**

```python
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
try:
    from tensorflow.python.client import device_lib
    print(device_lib.list_local_devices())
except ImportError:
    print("Failed to import device_lib.  CUDA/cuDNN may be improperly configured.")

```

This code snippet first checks the number of GPUs TensorFlow can detect.  If zero is returned, a crucial error exists. Next, it attempts to list available devices. The output should list your NVIDIA GPU.  If it fails to import `device_lib` or lists no GPUs, revisit your CUDA and cuDNN installations. This section is crucial for diagnostics.

**Example 2:  Simple TensorFlow GPU computation:**

```python
import tensorflow as tf

with tf.device('/GPU:0'):  # Specify GPU device
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0], shape=[5,1])
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0], shape=[1,5])
    c = tf.matmul(a, b)
    print(c)

```

This example performs a simple matrix multiplication on the GPU.  If the GPU is working correctly, you'll see the result printed rapidly.  A slow computation or an error suggests the GPU is not being utilized.  Observe the `tf.device('/GPU:0')` line; adjust the index if you have multiple GPUs.  I have found this type of minimal test highly effective in isolating GPU-related issues.

**Example 3:  More Complex GPU Usage with Keras (Illustrative):**

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense

model = keras.Sequential([
    Dense(128, activation='relu', input_shape=(784,)),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Assuming you have your data loaded into x_train and y_train
model.fit(x_train, y_train, epochs=10)

```

This example showcases a basic Keras model, utilizing TensorFlow's GPU backend for training. It's important to understand that the speed advantage here is dependent on data size and model complexity.  I've encountered situations where this benefits are limited for smaller datasets.  This exemplifies a more realistic scenario where GPU acceleration is significant. However, remember to replace placeholder data (`x_train`, `y_train`) with your actual training data.


**3. Resource Recommendations:**

The official TensorFlow documentation.  NVIDIA's CUDA and cuDNN documentation.  Consult reputable tutorials and guides focused specifically on TensorFlow 1.15 GPU installation on Ubuntu.  Pay close attention to version compatibility; this is the most common source of error. Review any error messages meticulously, as they often pinpoint the problem's root cause.  Understanding the interplay between the NVIDIA driver, CUDA, cuDNN, and TensorFlow is key to successful installation.  Thorough testing after each step is highly recommended. Remember to use appropriate system monitoring tools to observe GPU utilization during TensorFlow operation.
