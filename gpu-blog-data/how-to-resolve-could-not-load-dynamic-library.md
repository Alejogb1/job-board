---
title: "How to resolve 'Could not load dynamic library 'libcudnn.so.8'' error when using TensorFlow on Ubuntu 20.04?"
date: "2025-01-30"
id: "how-to-resolve-could-not-load-dynamic-library"
---
The "Could not load dynamic library 'libcudnn.so.8'" error, commonly encountered when using TensorFlow with GPU support on Ubuntu 20.04, indicates a mismatch or misconfiguration in the CUDA Deep Neural Network library (cuDNN) installation. This library provides highly optimized implementations of standard deep learning routines and is essential for GPU acceleration within TensorFlow. Its absence or incorrect version linkage prevents TensorFlow from utilizing the GPU, resulting in the aforementioned error and fallback to CPU-based computation.

The core issue stems from the fact that cuDNN is a separate package from the CUDA toolkit and is not automatically installed or updated alongside it. I’ve encountered this several times over the years while setting up development environments and production systems. TensorFlow, specifically, expects cuDNN to be present in a location where the dynamic loader can find it, and the installed version must be compatible with the version of CUDA and TensorFlow itself. Furthermore, the specific error “libcudnn.so.8” indicates that TensorFlow is looking for version 8 of the cuDNN library.

To resolve this issue, a methodical approach involving identifying and addressing the underlying discrepancies is necessary. Firstly, confirming the CUDA Toolkit version currently installed is critical. If you have upgraded your CUDA installation, ensure that the compatible cuDNN library is also present on the machine. The compatibility between CUDA, cuDNN, and TensorFlow is extremely precise, and deviating from the correct pairings often results in this library-loading issue. Secondly, one should verify that the paths where the dynamic loader searches for shared libraries include the directory where the cuDNN library is located. If the library is present but not on the correct path, the error will persist. Finally, verifying the integrity of the downloaded cuDNN library itself through checksum verification is also a good practice.

Here’s how I typically address this issue:

**1. Confirm CUDA Installation and Version:**

Start by ensuring that the correct CUDA toolkit is installed and its environment variables are set up. This involves checking the CUDA installation directory (often `/usr/local/cuda`) and verifying the version using commands like `nvcc --version`. For instance, if `nvcc` shows CUDA 11.0, one needs cuDNN compatible with CUDA 11.0. If the NVIDIA drivers and CUDA aren't correctly set up, no amount of cuDNN will fix this issue. This command reveals the installed CUDA toolkit version, which forms the foundation for the other libraries we will use.

**2. Download and Install Correct cuDNN Version:**

Once the CUDA version is identified, the corresponding cuDNN library must be obtained from the NVIDIA developer website. Note, a NVIDIA developer account is required. Ensure that you select the specific cuDNN version that is compatible with the installed CUDA toolkit version. NVIDIA provides pre-compiled versions of cuDNN for various CUDA versions and operating systems.  After downloading the appropriate version, usually compressed into a `.tgz` archive, the next step is extraction. Assuming the archive is named `cudnn-11.0-linux-x64-v8.0.4.30.tgz` (please replace with your downloaded file), the following would extract its content:

```bash
tar -xzvf cudnn-11.0-linux-x64-v8.0.4.30.tgz
```

This command will create a new folder named `cuda`, which will contain the include and library files. Note that in my experience, cuDNN versioning changes regularly, so the filenames and directory names will need to be adjusted according to the NVIDIA download.

**3. Copy cuDNN Files to the CUDA Toolkit Directory:**

The extracted files must then be copied into the corresponding directories within the CUDA toolkit installation path. These usually reside under `/usr/local/cuda/`.  Here, we copy the header files into the `include` directory and the libraries into the `lib64` directory. Specifically, the `.h` files are copied into `/usr/local/cuda/include`, and the `.so` files (e.g., `libcudnn.so.8`) into `/usr/local/cuda/lib64`.  These files represent the APIs that TensorFlow will use to utilize the GPU, and they must be placed correctly for the program to find them. This is usually done by following sequence of copy commands. Example (replace paths and library names as appropriate for your downloaded file):

```bash
sudo cp cuda/include/cudnn*.h /usr/local/cuda/include
sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/lib64/libcudnn*
```

The first command copies header files to include directory, second copies library files to lib directory, and third command gives read permissions. It's important to execute these commands with `sudo` to obtain the necessary administrative privileges for writing into these system directories. Finally, verify the copied libraries are present using the `ls` command to see them listed in the directory.

**4. Verify System Library Paths:**

After copying the files, one must ensure that the operating system is aware of the location of cuDNN. This is typically achieved by adding the relevant directory to the `LD_LIBRARY_PATH` environment variable. The `LD_LIBRARY_PATH` is a colon-separated list of directories that the dynamic loader searches when resolving shared library dependencies. This path modification, however, is temporary. To persistently set it, one needs to add a line to a shell startup script, such as `~/.bashrc` or `~/.zshrc`. This can be achieved using command like:

```bash
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

The first command appends the required export statement to the `.bashrc` file. The second command then immediately applies this change to the current shell session. The `source` command re-reads the `.bashrc` file, activating the modifications without requiring you to start a new terminal session. This will permanently add the library path in all new terminal sessions.

**5. Test the TensorFlow with GPU:**

With these adjustments made, one should now be able to execute TensorFlow code utilizing GPU support without encountering the “Could not load dynamic library ‘libcudnn.so.8’ ” error. The following simple python code test will verify if the configuration is working correctly:

```python
import tensorflow as tf

if tf.test.is_gpu_available():
    print("GPU is available.")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)
else:
    print("GPU is not available.")
```

This code first checks if TensorFlow has access to the GPU. If a GPU is detected, it attempts to enable memory growth on the GPU and list available GPU devices, both physical and logical. If the GPU is available, the print statement should indicate "GPU is available", and the number of detected physical and logical devices will be printed. Otherwise, the output will show "GPU is not available”. In practice, you will want to run more exhaustive tests, especially if dealing with specific models.

Should the problem persist after these steps, meticulously re-checking the version compatibility and file paths is crucial. Occasionally, there may be conflicts with other installed libraries that require more in-depth debugging. As a general practice, I advise installing TensorFlow and CUDA within a dedicated virtual environment to isolate these library dependencies from system-wide installations, preventing other system libraries from conflicting with the libraries used by TensorFlow.

For further guidance, the official NVIDIA documentation for CUDA and cuDNN installation is a great starting point, specifically their installation and getting started guides. TensorFlow’s official documentation is also invaluable, especially their guide on using GPUs, which covers specific compatibility requirements between TensorFlow, CUDA, and cuDNN versions. Many community forums dedicated to machine learning provide helpful troubleshooting tips and shared experiences. Consulting these sources frequently will provide clarity, especially since the ecosystem changes so rapidly. Finally, the Stack Overflow community remains a solid place to search for similar issues and their solutions.
