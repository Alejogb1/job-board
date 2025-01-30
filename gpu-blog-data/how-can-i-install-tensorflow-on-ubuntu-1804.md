---
title: "How can I install TensorFlow on Ubuntu 18.04 AWS?"
date: "2025-01-30"
id: "how-can-i-install-tensorflow-on-ubuntu-1804"
---
Successfully deploying TensorFlow on an Ubuntu 18.04 instance within Amazon Web Services (AWS) requires careful consideration of environment variables, Python package management, and hardware acceleration capabilities. I've navigated this process numerous times, encountering various pitfalls from mismatched CUDA versions to improperly configured virtual environments. Based on these experiences, a reliable approach combines a virtual environment for dependency isolation, a carefully chosen TensorFlow installation method (CPU-only or GPU-enabled), and verification of correct functioning through a basic test script.

First, establishing a virtual environment is non-negotiable. This isolates TensorFlow’s dependencies from system-wide packages, preventing conflicts and ensuring reproducibility. I’ve seen cases where system-level Python packages, especially older versions, interfere with TensorFlow operations, leading to difficult-to-diagnose errors. Therefore, I consistently use `virtualenv` or `venv` for this purpose.

Second, the installation itself depends on whether GPU acceleration is necessary. For purely CPU-based workloads, `pip` can directly install the TensorFlow package. For GPU-accelerated applications, the situation is more complex. NVIDIA drivers and the CUDA toolkit must be installed correctly, along with the corresponding `cudnn` library. The correct versions must be aligned with the chosen TensorFlow version. Misalignment here is the most common cause of GPU acceleration failures in my experience.

Finally, a simple test should always confirm the installation. This involves importing the TensorFlow library and performing a basic operation, like multiplying two tensors. This guarantees that the installation is functioning correctly, that all dependencies are met and that acceleration is available if needed.

Here are three code examples with explanations that showcase this process:

**Example 1: Setting up a Virtual Environment and Installing CPU TensorFlow**

```bash
# 1. Navigate to a directory for your project
cd ~/projects

# 2. Create a virtual environment named 'tf_env'
python3 -m venv tf_env

# 3. Activate the virtual environment
source tf_env/bin/activate

# 4. Upgrade pip (recommended)
pip install --upgrade pip

# 5. Install CPU-only TensorFlow
pip install tensorflow

# 6. Check the installation
python3 -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices('CPU'))"

# 7. Deactivate virtual environment
deactivate
```

*Explanation:* This example illustrates the foundational steps for a basic TensorFlow CPU installation. First, a virtual environment named `tf_env` is created within the user's `projects` directory using Python's built-in `venv` module. `virtualenv` would be another suitable option. The environment is then activated using the `source` command. Crucially, `pip` is upgraded to its latest version, ensuring package installations work smoothly. Subsequently, `tensorflow` is installed via `pip`. I’ve observed that neglecting to update `pip` can lead to unexpected dependency conflicts.  The Python command following the `pip install` verifies the installation by printing the TensorFlow version and listing available CPUs. This is crucial to confirm proper installation.  Finally, the virtual environment is deactivated, releasing its control over the terminal session. This process prepares a clean and isolated environment for running TensorFlow experiments without influencing system packages.

**Example 2: Installing TensorFlow with GPU Support**

```bash
# Assumes NVIDIA drivers, CUDA toolkit, and cuDNN are installed correctly

# 1. Navigate to a directory for your project
cd ~/projects

# 2. Create a virtual environment named 'tf_gpu_env'
python3 -m venv tf_gpu_env

# 3. Activate the virtual environment
source tf_gpu_env/bin/activate

# 4. Upgrade pip (recommended)
pip install --upgrade pip

# 5. Install GPU-enabled TensorFlow
pip install tensorflow-gpu

# 6. Check GPU availability and installation
python3 -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"

# 7. Deactivate virtual environment
deactivate
```

*Explanation:* This example expands upon the previous one by installing the GPU-enabled version of TensorFlow. **Crucially, this example assumes that the required NVIDIA drivers, the correct CUDA toolkit version, and `cudnn` are already installed and configured on the Ubuntu instance.**  This prior installation is not covered here but is essential. I’ve often found that compatibility issues between these components are the most common source of problems. The initial steps mirror the CPU setup: creating and activating a virtual environment (`tf_gpu_env`), upgrading `pip`, and then installing `tensorflow-gpu`. The key difference lies in installing the GPU package using `pip install tensorflow-gpu`.  The Python command verifies GPU availability by listing available GPUs after printing the version.  Without this confirmation,  one may believe that the installation is working, but it might be only running on the CPU, which would invalidate any performance advantage of a GPU. Similarly, the virtual environment is deactivated.

**Example 3:  A Basic TensorFlow Operation (CPU or GPU)**

```python
import tensorflow as tf

# Create two tensors
a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
b = tf.constant([[5.0, 6.0], [7.0, 8.0]])

# Multiply the two tensors
c = tf.matmul(a, b)

# Print the result
print(c)


#Verify device usage

if tf.config.list_physical_devices('GPU'):
    print("Using GPU")
else:
    print("Using CPU")
```

*Explanation:* This Python script performs a basic matrix multiplication using TensorFlow, acting as a final check. The script imports the library and defines two constant tensors, `a` and `b`. The core functionality lies in `tf.matmul(a, b)`, which performs the matrix multiplication. I regularly employ simple checks such as this to make sure core TensorFlow features are operational. The result is then printed to the console. Importantly, the script now includes a check for GPU usage. This is vital, as the previous step may indicate that a GPU device is detected, but this can be misconstrued. The script now checks if the TensorFlow is using the GPU or CPU, confirming hardware acceleration is being used if available. It’s vital to run this after the install because it can fail in multiple ways. The lack of output can indicate that the import failed (most likely due to an incorrect environment), the matrix multiplication itself can fail (most likely due to a misconfigured library or hardware issue), and finally the incorrect printout can indicate a misunderstanding of the actual hardware in use. A passing case provides high confidence in the TensorFlow setup.

**Resource Recommendations**

For comprehensive information, I suggest consulting resources provided by TensorFlow itself.  The official TensorFlow website provides extensive documentation covering installation, troubleshooting, and detailed tutorials for various use cases. NVIDIA also has documentation that is critical for installing drivers, CUDA, and cuDNN. Reading their documentation carefully is critical to avoiding common installation pitfalls. Finally, I would consult online communities focused on TensorFlow. These communities can be extremely valuable because other users have likely encountered the same challenges, and their solutions are often freely available.
These three sources can assist with nearly all issues encountered during installation and subsequent development.

In conclusion, successfully installing TensorFlow on Ubuntu 18.04 within AWS requires a well-structured approach based on the use of virtual environments, carefully selected installation methods based on intended hardware usage, and thorough testing to verify that dependencies are met and acceleration is correctly configured. I’ve found this iterative process to be effective and reliable in my previous endeavors.
