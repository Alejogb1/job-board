---
title: "How do I install TensorFlow on macOS 2020 M1 silicon for deep learning?"
date: "2025-01-30"
id: "how-do-i-install-tensorflow-on-macos-2020"
---
The fundamental shift from x86-64 to ARM-based M1 silicon in macOS requires a different approach to TensorFlow installation compared to older Intel-based Macs. Direct pip installation of the standard TensorFlow package often fails due to architecture incompatibilities, necessitating the use of Apple-optimized TensorFlow builds. My experience involves migrating several legacy deep learning projects to M1 machines, and I’ve encountered this specific issue repeatedly. The process is not inherently complex but demands adherence to particular steps.

First, it’s crucial to understand that TensorFlow, at its core, is a highly optimized library written primarily in C++ and Python. The performance of TensorFlow depends heavily on low-level numerical computation libraries, like the BLAS library, which interact directly with the processor. These libraries are compiled specifically for the instruction set of the target processor. Standard TensorFlow builds are typically compiled for x86-64 architecture, leaving M1 users with a compatibility hurdle. While emulation layers like Rosetta 2 exist, they impose performance penalties that negate a significant portion of the M1 chip's inherent speed advantages for computationally demanding tasks like deep learning. Consequently, the best approach is to utilize a version of TensorFlow compiled directly for the ARM architecture.

Apple's solution to this issue was to create a custom-built TensorFlow package, optimized for M1 chips and its Metal API, which accelerates computations via the M1 GPU. This package is typically found within Apple's developer ecosystem and is not directly available through `pip` like the standard package. Instead, it’s often distributed as part of a specific environment or a dedicated installation process.

The initial step involves setting up a Python environment designed for use with TensorFlow. This is best done using a virtual environment tool like `venv` or Conda. I prefer `venv` for its lightweight nature, and because it comes standard with Python. I avoid using the system Python installation to prevent potential version conflicts and ensure a clean working space. To create a virtual environment named "tf-m1," one can use the following command within the desired project directory:

```bash
python3 -m venv tf-m1
```

After creation, the virtual environment must be activated:

```bash
source tf-m1/bin/activate
```

With the virtual environment activated, the next crucial step is installing Apple's optimized TensorFlow package. The specific package name and method of installation may vary slightly depending on the current version and recommendations. The most common approach involves leveraging the `tensorflow-macos` and `tensorflow-metal` packages. These packages can be installed using `pip`, but it is imperative to install them in the correct order. It's essential to use the latest versions supported to benefit from performance improvements and bug fixes. This installation process should be considered a specific requirement for macOS M1 systems. This process has changed slightly over time, so it’s a good idea to check the latest Apple developer documentation for the most up to date instructions. I usually perform the following command sequence within the activated environment:

```bash
pip install tensorflow-macos
pip install tensorflow-metal
```
The `tensorflow-macos` package provides the base TensorFlow implementation for macOS M1, while `tensorflow-metal` enables the use of the M1's GPU acceleration capabilities. These are typically the only dependencies needed for basic deep learning tasks, though a full project might demand further packages. For example, image processing can be implemented with a package like Pillow or scikit-image; however these packages are not strictly required for a base TensorFlow setup.

To verify the installation and confirm that TensorFlow is indeed using the M1’s GPU, a simple test script can be employed. I usually check the device assignment in a basic TensorFlow calculation, creating a tensor, and checking what device it is assigned to. This also acts as a good simple check that the framework is functional and I haven’t run into any installation issues.

```python
import tensorflow as tf

# Check available devices
print(tf.config.list_physical_devices())

# Create a simple tensor
a = tf.constant([1.0, 2.0, 3.0], dtype=tf.float32)

# Print the device it is assigned to
print(a.device)
```

Executing this script will first display the list of devices available for TensorFlow. If the GPU, identified as a "Metal" device, is present, this indicates successful usage of the `tensorflow-metal` package. Subsequent output showing a tensor assigned to the Metal GPU device confirms the accelerated operation. If the tensor is assigned to the CPU, the installation may have issues and further troubleshooting will be required. For a complete, working code example, here’s a more realistic simple test that utilizes the GPU with the Metal backend.

```python
import tensorflow as tf
import numpy as np

# Generate some random data
x = np.random.rand(1000, 10).astype(np.float32)
y = np.random.rand(1000, 1).astype(np.float32)

# Simple linear model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=(10,))
])

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(x, y, epochs=10, verbose=0)

# Make a prediction
prediction = model.predict(np.random.rand(1, 10).astype(np.float32))

print("Model prediction:", prediction)
print(f"Device used: {model.layers[0].kernel.device}")
```
This code snippet first generates some random training data for a simple linear regression model. It then compiles the model using the Adam optimizer and Mean Squared Error loss and proceeds to train the model for a short period. Finally, the script prints a prediction using a random input and also prints the device that the model layer was assigned to. If the device output shows a Metal device, the GPU is being correctly utilized.

It is important to note that the installation process is subject to changes as Apple and the TensorFlow team release new updates. Consequently, relying solely on outdated online tutorials can lead to compatibility issues. Consulting the official Apple documentation regarding TensorFlow installation for M1 Macs is paramount. Likewise, referring to TensorFlow’s own documentation for platform specific requirements is essential. The `tensorflow-macos` and `tensorflow-metal` packages are generally the preferred starting point. However, depending on the specific use case, alternative options may exist.

For comprehensive learning and support, I generally recommend exploring the official TensorFlow website for in-depth documentation. For macOS-specific considerations, the Apple Developer website provides specific notes and examples. Furthermore, the TensorFlow GitHub repository is a valuable resource for examining the source code and understanding the underlying mechanisms. There are also several high-quality online courses that often contain dedicated modules for setting up environments, including M1 systems. These are often platform agnostic, so I’d advise checking the specifics of their teaching environment against the current version of TensorFlow and macOS. Finally, the TensorFlow community itself is a vast resource; searching through previous forum posts and asking new questions when needed is an essential skill when debugging complex dependency issues.
