---
title: "How can Tensorflow GPU/CUDA dependencies be installed on a machine without internet access?"
date: "2025-01-30"
id: "how-can-tensorflow-gpucuda-dependencies-be-installed-on"
---
The installation of TensorFlow with GPU support on an air-gapped machine presents a considerable challenge, primarily due to the need to acquire CUDA toolkit and cuDNN library binaries, both of which are typically fetched through online package managers or direct downloads. I've encountered this scenario multiple times in secure data environments, and the process requires meticulous planning and a staged transfer of pre-downloaded dependencies.

The core issue isn’t TensorFlow itself; rather, it is the dependency management for its hardware acceleration. TensorFlow’s GPU functionality leverages NVIDIA’s CUDA framework for parallel computation and cuDNN for optimized deep learning primitives. Without internet access, the standard `pip install tensorflow` or similar commands won't work out of the box, as these assume the availability of a network to pull necessary dependencies. My experience indicates that the most reliable approach involves a multi-step process: (1) Download dependencies on a machine with internet access, (2) transfer the downloaded files to the target machine, and (3) install the dependencies in the correct order and configuration.

The process begins by precisely identifying the required versions of CUDA toolkit, cuDNN, and the corresponding TensorFlow package. These versions must be compatible. For example, TensorFlow 2.10 might require CUDA 11.2 and cuDNN 8.1. Inconsistent versions will result in runtime errors, such as "CUDA driver version is insufficient for CUDA runtime version," or segmentation faults. This version matrix compatibility is critical and can be confirmed from the official TensorFlow documentation. Once compatibility is confirmed, all the required packages can be downloaded using a machine that is connected to the internet.

I prefer to download the installer packages for CUDA and cuDNN as opposed to just library files. This ensures that the entire CUDA environment gets installed correctly. For CUDA, NVIDIA offers installer packages suitable for different operating systems. On Linux, this usually comes as a `.run` or `.deb` file, and on Windows, it's an `.exe` file. CuDNN, often distributed as an archive containing several libraries (`.so` or `.dll` files), should be specifically downloaded for the target CUDA version. Furthermore, the correct version of the TensorFlow wheel (`.whl`) file must be downloaded using pip on the internet-enabled machine. This can be done using a specific command line flag, such as:

```bash
pip download tensorflow==2.10 -d /path/to/download/location
```

This downloads only the whl file to the specified directory rather than installing it. The download directory will now contain the TensorFlow `.whl` file as well as all of the whl files it depends on. These dependencies can also be transferred.

Once the CUDA, cuDNN, TensorFlow wheel file, and its dependencies are all downloaded, they must be transferred to the target machine via a mechanism like USB drive. The process on the target machine then involves installing the CUDA toolkit first, followed by placing cuDNN files in the right locations, and finally installing TensorFlow with the downloaded `whl` file. The order matters considerably; CUDA needs to be set up before cuDNN and TensorFlow.

On Linux, after transferring the CUDA `.run` file, it typically involves running the following from the command line (make the script executable, first, if required):

```bash
chmod +x cuda_installer.run
sudo ./cuda_installer.run
```

The installation process is interactive and it is very important to select the installation options carefully, particularly if there are different GPUs installed. After installation, the environment variables such as `LD_LIBRARY_PATH`, `CUDA_HOME`, and other paths might need to be adjusted. I’ve found that manually adding these entries to `~/.bashrc` or `/etc/profile` often helps to avoid future issues. For cuDNN, the files should be extracted and placed into the appropriate CUDA directories. It typically involves copying `libcublas.so*`, `libcudnn.so*`, `libcudnn_*.so*`, `include/cudnn.h` into locations like `/usr/local/cuda/lib64` and `/usr/local/cuda/include` (adjust to match the installation). This can be done via the following commands:

```bash
sudo cp libcublas.so* /usr/local/cuda/lib64
sudo cp libcudnn.so* /usr/local/cuda/lib64
sudo cp include/cudnn.h /usr/local/cuda/include
```

Similarly, on windows, the NVIDIA installers must be run with admin rights. This will unpack files to the specified locations. The cuDNN files are extracted and copied to the CUDA installation folder. The `bin`, `include`, and `lib` folders inside of the extracted cuDNN folder should be placed inside the CUDA installation folder, such as `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.2`.

Finally, after ensuring that CUDA and cuDNN are correctly installed, one can install TensorFlow from the transferred `.whl` file. Using the same `pip` utility on the target machine:

```bash
pip install /path/to/download/location/tensorflow-2.10-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl --no-index --find-links /path/to/download/location
```

The `--no-index` flag prevents `pip` from connecting to the internet, while `--find-links` points to the location of the wheel file and its dependencies. The path to the directory containing the downloaded TensorFlow whl file must be specified. After this step, TensorFlow can be imported with GPU support on the target machine.

Troubleshooting is an unavoidable part of the process. Common problems involve incorrect CUDA installation, improperly placed cuDNN files, or incompatible driver versions. Checking error logs provided by TensorFlow or during CUDA driver initialization is essential to diagnose these issues. The command `nvidia-smi` can also be very useful to check if CUDA drivers are installed and have initialized the GPU. The command `tf.config.list_physical_devices('GPU')` inside a Python environment will check whether TensorFlow detects the GPU hardware correctly. If no GPUs are detected it implies a misconfiguration in the steps taken previously.

For in-depth understanding and updates, I recommend consulting the official NVIDIA CUDA documentation for installation instructions specific to the operating system. The TensorFlow website also provides a dedicated section on GPU setup, which, though assuming internet availability, provides critical insights on version compatibility. While there are many online resources that discuss these issues, a primary focus should be placed on the NVIDIA and TensorFlow documentation. For specific CUDA version compatibility to a specific tensorflow version, consulting tensorflow's documentation is vital. Finally, a dedicated reference on Linux file and directory paths for installations could prove useful.
