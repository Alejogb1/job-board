---
title: "How can I install TensorFlow on Ubuntu 16.04?"
date: "2025-01-30"
id: "how-can-i-install-tensorflow-on-ubuntu-1604"
---
TensorFlow, while fundamentally a library, requires meticulous handling of its underlying dependencies and the user's execution environment for smooth operation, particularly on older operating systems like Ubuntu 16.04.  My experience maintaining legacy deep learning systems has highlighted the common pitfalls encountered during TensorFlow installations, especially concerning Python versions and CUDA compatibility. This response will detail the installation process for TensorFlow on Ubuntu 16.04, addressing potential compatibility issues and providing practical examples.

First, it's critical to understand that Ubuntu 16.04 is no longer supported by Canonical. This impacts the availability of readily compatible pre-built TensorFlow packages. Consequently, a more hands-on approach is often necessary.  The primary choice lies between a CPU-only installation and one leveraging NVIDIA GPUs.  GPU acceleration significantly enhances performance for training deep learning models, but it involves an additional layer of complexity in terms of CUDA and cuDNN configuration. This response will cover both scenarios, beginning with the simpler CPU-only installation.

**CPU-Only Installation**

The most straightforward method for CPU-only TensorFlow installation involves `pip`, Python’s package installer. Given that Ubuntu 16.04 often ships with older versions of Python, it is advisable to install a modern Python interpreter and virtual environment manager. I typically employ `pyenv` and `virtualenv` for this purpose.  The advantage of a virtual environment is that it isolates your TensorFlow installation from other Python projects, preventing potential dependency conflicts.  Here's a typical installation sequence:

1.  **Python Installation:** Download a Python 3.7 or 3.8 distribution using `pyenv` and then select it for use:

    ```bash
    pyenv install 3.8.10
    pyenv local 3.8.10
    ```
    *Commentary:* This installs a specific Python version (3.8.10 in this case) and makes it active within the current directory and its subdirectories. Choose a compatible Python version based on TensorFlow's compatibility matrix available in their official documentation.  Versions above Python 3.8 are less reliable on older Linux distributions without extensive package management workarounds.

2.  **Virtual Environment Creation:** Set up a virtual environment. I usually name mine `tf_env`.

    ```bash
    python3 -m venv tf_env
    source tf_env/bin/activate
    ```
    *Commentary:* This creates a virtual environment within the folder 'tf_env' and activates it. All subsequent `pip install` operations will be contained within this isolated environment.

3.  **TensorFlow Installation:** Install the CPU version of TensorFlow.

    ```bash
    pip install tensorflow
    ```
    *Commentary:* This installs the latest available version of TensorFlow that's compatible with Python 3.8 and is configured for CPU processing.  You can pin a specific TensorFlow version via `pip install tensorflow==2.8.0` for example if needed.

4. **Verification:** Confirm successful installation by running a simple script.
    ```python
    import tensorflow as tf
    print(tf.__version__)
    ```
    *Commentary:* This snippet should output the installed TensorFlow version if successful. If an error occurs, review the terminal messages for missing dependencies or other installation issues.

**GPU-Enabled Installation**

If GPU acceleration is desired, the installation process becomes more involved. This requires an NVIDIA GPU, the correct drivers, CUDA toolkit, and cuDNN library versions. Compatibility matrices provided by NVIDIA and TensorFlow are paramount for successful setup.  My experiences have shown that meticulous version matching between TensorFlow, CUDA, and cuDNN is essential.  The installation process breaks down as follows:

1.  **NVIDIA Drivers:** Install appropriate NVIDIA drivers for your graphics card model. You should use drivers recommended by NVIDIA for your GPU, noting that newer drivers may not be compatible with older cards.  Official NVIDIA documentation provides installation instructions.

2.  **CUDA Toolkit:** Download and install a specific version of the CUDA Toolkit that’s compatible with your chosen TensorFlow version. Usually, older TensorFlow versions require older CUDA toolkits. My experience shows that CUDA 11.2 or 10.1 are often reasonable starting points for older cards on Ubuntu 16.04. The correct version should be confirmed by consulting the TensorFlow installation guide for your particular version.

    ```bash
    # Example (replace with your desired CUDA install)
    wget https://developer.download.nvidia.com/compute/cuda/11.2.0/local_installers/cuda_11.2.0_460.27.04_linux.run
    sudo sh cuda_11.2.0_460.27.04_linux.run
    ```
   *Commentary:* This command downloads the CUDA toolkit and executes the installer.  Follow the installer prompts. Make sure to add CUDA libraries to the system path, usually via the `.bashrc` or similar initialization scripts.  The specific command to download will depend on your specific CUDA toolkit selection, found on NVIDIA's developer site.

3.  **cuDNN Library:** Download and install a corresponding version of the cuDNN library, matching the CUDA toolkit.  cuDNN is typically provided as a set of files (e.g., `.so` files on Linux) that you must extract into the CUDA installation directory. You’ll need an NVIDIA developer account for this.

    ```bash
    # Example (this assumes that you have extracted the cuDNN files into /path/to/cudnn/extract/location
    sudo cp /path/to/cudnn/extract/location/include/cudnn*.h /usr/local/cuda/include
    sudo cp /path/to/cudnn/extract/location/lib64/libcudnn* /usr/local/cuda/lib64
    sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
    ```
    *Commentary:* This command set is an example, and the actual steps will depend on the exact location of the files. The key is copying the `include` files to the CUDA include path and the library files to the CUDA library path and giving the appropriate permissions.

4. **Python Environment and GPU-Enabled TensorFlow Installation:** Create the virtual environment as before, but install the GPU-enabled TensorFlow. This is important – there are different packages for CPU and GPU functionality.

    ```bash
    python3 -m venv tf_gpu_env
    source tf_gpu_env/bin/activate
    pip install tensorflow-gpu
    ```
   *Commentary:* This sets up a new virtual environment and installs the GPU-enabled version of TensorFlow.  The `tensorflow-gpu` package name signals that TensorFlow should be built and optimized to use CUDA and cuDNN.

5.  **Verification:** Verify GPU usage using the following python code:

    ```python
    import tensorflow as tf
    print(tf.config.list_physical_devices('GPU'))
    ```
    *Commentary:* This should output a list of available GPUs detected by TensorFlow if your install is successful.  If the list is empty, recheck the CUDA and cuDNN installation steps and paths to ensure TensorFlow can locate the required libraries.

**Example Code for GPU check:**

```python
import tensorflow as tf

try:
  gpus = tf.config.list_physical_devices('GPU')
  if gpus:
    for gpu in gpus:
       print("GPU available:",gpu)
    print("TensorFlow is using GPU acceleration")
    a = tf.constant([1.0, 2.0, 3.0], dtype=tf.float32)
    b = tf.constant([4.0, 5.0, 6.0], dtype=tf.float32)
    c = tf.add(a, b)
    print("Result of the addition:", c)
  else:
    print("No GPU detected. TensorFlow is using CPU")
except Exception as e:
    print("An error occurred during GPU check:", e)

```

*Commentary:* This code block not only verifies the presence of the GPU but also performs a small addition operation using TensorFlow. This further validates that TensorFlow is actively leveraging the GPU. If everything is correctly installed, you'll see GPU information, confirmation that TensorFlow is using the GPU, and the result of the vector addition. It has an exception block to handle any errors related to initialization gracefully.

**Resource Recommendations**

For a comprehensive understanding of the TensorFlow installation process, I recommend consulting the official TensorFlow documentation. Additionally, NVIDIA provides detailed guides for CUDA and cuDNN installations. The online community forums are also valuable for troubleshooting specific installation issues, often containing workarounds for known conflicts. You can also find well-maintained resources on python packaging and virtual environment management, which are important topics when building complex machine learning environments. Checking NVIDIA’s compatibility matrices is extremely important to avoid any conflict. These matrices provide lists of supported versions that will work together.
