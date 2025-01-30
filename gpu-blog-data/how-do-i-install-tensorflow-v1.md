---
title: "How do I install TensorFlow v1?"
date: "2025-01-30"
id: "how-do-i-install-tensorflow-v1"
---
TensorFlow v1, while predating the more streamlined v2, remains relevant for maintaining legacy codebases and specific research contexts, necessitating a clear understanding of its installation process. The installation path deviates significantly from v2 due to differences in dependency management and the reliance on Session-based execution. I've personally navigated this process numerous times during model deployments for various clients who, for a range of reasons, have not transitioned to newer versions.

**Understanding the Installation Process for TensorFlow v1**

The primary challenge in installing TensorFlow v1 stems from its tight coupling with specific versions of CUDA, cuDNN, and Python, along with a complex dependency graph for supporting packages. Unlike v2, which offers greater flexibility and backward compatibility, v1 often requires a specific alignment of these components to avoid runtime errors. Moreover, v1's installation relies primarily on `pip` for package management, demanding careful attention to the installation order and environment setup.

Specifically, the installation typically involves the following steps, which I'll explain in more detail:

1.  **Environment Setup:** A clean, isolated Python environment is paramount, achieved via tools like `virtualenv` or `conda`. This prevents conflicts with other Python projects and ensures a controlled space for v1's dependencies. The Python version itself must be carefully selected, as v1 versions often specify support for particular Python releases (e.g., Python 3.5, 3.6, 3.7). Choosing the wrong version will typically result in pip installation errors or runtime issues.
2.  **CUDA and cuDNN Installation (GPU Support):** If GPU acceleration is needed, installing the correct versions of CUDA (NVIDIA's parallel computing platform) and cuDNN (CUDA Deep Neural Network library) is critical. Compatibility matrices are provided by NVIDIA and TensorFlow to match the driver and library versions precisely to the target TensorFlow release. Mismatches here will lead to the inability to use the GPU or, worse, runtime crashes when TensorFlow attempts to utilize GPU resources. These NVIDIA installations are typically done outside of `pip` via package managers such as `apt` on Debian-based Linux distributions, via `yum` on Red Hat-based systems, or by direct download and installer executions on Windows.
3.  **TensorFlow Installation via `pip`:** The actual installation of the TensorFlow package is accomplished using the Python package installer `pip`. Given the version-specific nature of v1, specific wheel files (`.whl`) corresponding to Python version, operating system, and CPU or GPU support are required. One will either use pre-built wheels or build a custom wheel from source if the desired combination is not readily available. The prebuilt wheels are often found on the TensorFlow project’s release page on GitHub or PyPi, but they may also be found on third party resources. 
4.  **Verification:** After the install, a simple Python test script can verify that TensorFlow is correctly installed and can access either the CPU or GPU as configured. Issues at this stage often indicate problems with package versions or underlying CUDA drivers.

**Code Examples and Commentary**

These examples demonstrate the described procedure and are illustrative; precise versions would need to be chosen based on current NVIDIA driver availability.

**Example 1: Creating a Virtual Environment**

This example outlines setting up a virtual environment for TensorFlow v1 using `virtualenv` on a Linux-based system.  I choose virtual environments over system-level installations to avoid library conflicts.

```bash
# Create a directory for our project and enter it
mkdir tf1_project && cd tf1_project

# Create a virtual environment named 'tf1_env' using Python 3.6
virtualenv -p python3.6 tf1_env

# Activate the virtual environment
source tf1_env/bin/activate

# Verify the active python path
which python
# should return something like /path/to/tf1_project/tf1_env/bin/python
```

**Commentary:**
*   The `virtualenv -p python3.6 tf1_env` creates an isolated environment with a specific version of Python (3.6 in this case). It's critical that the python version be chosen to match tensorflow's supported releases.
*   The `source tf1_env/bin/activate` command activates the virtual environment. All subsequent `pip install` commands will install into this isolated environment. I’ve experienced multiple failed installs by forgetting this crucial step.
*   `which python` ensures you are using the python executable inside the isolated environment.

**Example 2: Installing a CPU-Based TensorFlow v1**

This illustrates installing a specific TensorFlow v1 package using pip for CPU only operation. I will assume an appropriate wheel has been identified and downloaded into the current directory.

```bash
# Install a tensorflow v1.15.0 CPU package
pip install ./tensorflow-1.15.0-cp36-cp36m-linux_x86_64.whl

# Verify tensorflow is installed correctly
python -c "import tensorflow as tf; print(tf.__version__)"

# Exit the virtual environment
deactivate
```

**Commentary:**
*   The `pip install ./tensorflow-1.15.0-cp36-cp36m-linux_x86_64.whl` command installs the specific tensorflow wheel. The actual file name will vary based on the OS and desired version. I have frequently had to search through the TensorFlow release page for the correct version for the hardware.
*   The `python -c ...` snippet attempts to import TensorFlow and print the version, confirming the installation was successful. Errors at this point typically mean version mismatches or installation failures.
*   `deactivate` exits the virtual environment, returning you to the system default python executable.

**Example 3: Testing GPU support (Post-CUDA/cuDNN Install)**

Assuming that correct GPU drivers, CUDA, and cuDNN has been properly configured, this tests whether GPU usage is correctly available in TensorFlow.

```python
import tensorflow as tf

# Check if TensorFlow can find any GPUs
physical_devices = tf.config.list_physical_devices('GPU')

if physical_devices:
  print("GPUs are available:")
  for device in physical_devices:
    print(f"- {device}")

  # attempt to run an operation on the gpu
  with tf.device('/GPU:0'): # Explicitly place operation on GPU 0
      a = tf.constant([1.0, 2.0, 3.0], name='a')
      b = tf.constant([4.0, 5.0, 6.0], name='b')
      c = tf.add(a,b,name='add_c')

  with tf.compat.v1.Session() as sess:
    result = sess.run(c)
    print("GPU compute result:",result)
else:
  print("No GPUs detected. TensorFlow will use CPU.")

```

**Commentary:**
*   `tf.config.list_physical_devices('GPU')` lists any GPUs detected. An empty list indicates that CUDA or cuDNN is likely configured incorrectly. I've personally lost hours to faulty CUDA configurations.
*   The code tests whether a simple tensor addition can be executed on a designated GPU device with the `tf.device('/GPU:0')` context manager. Failure here could indicate issues with cuDNN or with insufficient access permissions to GPU devices on the system.
*   The result is computed and printed, confirming basic GPU operation if no exception occurs. `tf.compat.v1.Session()` is used since TensorFlow v1 requires a Session context for evaluating tensors.

**Resource Recommendations**

For information and troubleshooting, I recommend consulting the following types of resources:

1.  **The official TensorFlow v1 documentation:** While officially deprecated, the original documentation often still provides the most direct insights into specific version quirks and installation details.
2.  **NVIDIA's website and developer forums:** These resources are crucial for identifying the specific CUDA and cuDNN versions compatible with the target TensorFlow build. Troubleshooting GPU-related issues usually requires close attention to these resources.
3.  **Stack Overflow:** Although I have responded to this question, countless relevant discussions can be found on Stack Overflow. The sheer breadth of community knowledge is useful for diagnosing specific error messages. When searching here, be sure to qualify your search with terms like “TensorFlow v1” or “TensorFlow 1.x.”
4.  **Online guides and tutorials specific to v1:** Many technical blogs and university websites still maintain content about v1. They can supplement the official documentation and provide additional context.

Remember that installing TensorFlow v1 often requires patience and meticulous version tracking. Careful attention to the specific steps and resources outlined above will increase the likelihood of a successful installation.
