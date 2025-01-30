---
title: "How can I run Theano on a GPU using the terminal, instead of PyCharm?"
date: "2025-01-30"
id: "how-can-i-run-theano-on-a-gpu"
---
Theano's utilization of GPUs for accelerated computation, although superseded by newer frameworks, remains a relevant exercise in understanding how low-level GPU configurations interact with high-level deep learning libraries.  Direct terminal execution, bypassing an IDE like PyCharm, offers greater insight into this process and its underlying mechanisms. I've spent years debugging Theano on various Linux systems, and the core challenge often boils down to correctly configuring environment variables, device selection, and necessary CUDA/cuDNN libraries.

To run Theano on a GPU via the terminal, the primary hurdle involves ensuring Theano can identify and utilize your GPU effectively. This involves configuring Theano's backend to use CUDA, which typically requires setting environmental variables before launching your Python scripts. The default behavior of Theano assumes usage of the CPU, necessitating a manual specification to shift computation to the GPU. In essence, we are orchestrating communication between our code, the Theano library, the CUDA toolkit, and the GPU hardware itself, all from the command line. This process begins with establishing the appropriate environment.

The foundational step involves confirming your CUDA installation and verifying that the NVIDIA drivers are properly installed and functioning. The `nvidia-smi` command (available after driver installation) provides a clear status of your GPU, including its model, driver version, and current usage. It's crucial to have a compatible CUDA toolkit and cuDNN library installed, matching the driver version and your chosen Theano setup. The precise installation and configuration process are very system and distribution dependent. Generally, you'd download the appropriate packages from NVIDIA's developer portal and follow their respective installation guides.

After completing the installation, the crucial aspect involves modifying environment variables that Theano reads when it initiates. These settings directly influence whether computations are directed to the GPU or remain on the CPU. Specifically, we will use `THEANO_FLAGS` variable, where we specify the device, CUDA and cuDNN flags. Incorrect or missing configurations will result in Theano reverting to the CPU, undermining the objective of utilizing GPU acceleration. Setting `device=cuda` and the relevant `cuda.*` flags within `THEANO_FLAGS` variable triggers the correct usage.

Let’s consider a scenario where you have a simple Theano script that performs matrix multiplication. The script, named `theano_matmul.py`, is as follows:

```python
import theano
import theano.tensor as T
import numpy as np

# Define symbolic variables
a = T.matrix('a')
b = T.matrix('b')

# Perform matrix multiplication
c = T.dot(a, b)

# Compile Theano function
f = theano.function([a, b], c)

# Generate some test data
A = np.random.rand(1000, 1000).astype(theano.config.floatX)
B = np.random.rand(1000, 1000).astype(theano.config.floatX)

# Execute the Theano function
C = f(A, B)

# Print the first five elements
print(C.flatten()[0:5])
```

To run this code on a GPU, you would precede the Python execution command with the appropriate `THEANO_FLAGS` variable assignment directly in the terminal. Here's the first example of how we accomplish this:

```bash
export THEANO_FLAGS="device=cuda,floatX=float32,optimizer=fast_compile,cuda.root=/usr/local/cuda,cuda.cxx=/usr/bin/g++"
python theano_matmul.py
```
In this example, I assume that the CUDA toolkit is installed at `/usr/local/cuda` and the GNU C++ compiler, `g++`, is located at `/usr/bin/g++`. Also I specify the data type to be float32.  If, for instance, your CUDA toolkit is located in a different directory, you’d adjust the `cuda.root` parameter accordingly.  The other flags, `floatX=float32` and `optimizer=fast_compile` influence numerical precision and Theano's compilation process. The crucial flag here is `device=cuda`, which instructs Theano to use a GPU as the primary computation unit. Failing to set this, or setting it incorrectly, would force computation to occur on the CPU.

Next, consider a slightly more advanced scenario involving multiple GPUs. Theano allows you to select which specific GPU to use. To specify the usage of GPU with index ‘1’, the flag has to be modified in `THEANO_FLAGS`. This can be done by setting device value as `cuda1`.

```bash
export THEANO_FLAGS="device=cuda1,floatX=float32,optimizer=fast_compile,cuda.root=/usr/local/cuda,cuda.cxx=/usr/bin/g++"
python theano_matmul.py
```

This second example assumes you have multiple GPUs and explicitly chooses the second available GPU for Theano computation. The first GPU would be `cuda0`, the third `cuda2`, and so on. You would replace `cuda1` with whatever GPU index you need. If the system has only a single GPU, typically `cuda0` would be the most suitable option. Additionally, verify your setup with `nvidia-smi` before running to ensure the card is functional. Using the wrong GPU index or specifying a device that doesn't exist could result in errors or Theano reverting to using the CPU.

Finally, let’s examine a case where cuDNN is present, as this significantly enhances Theano’s performance on a GPU, particularly for deep learning models. You would then add some additional cuDNN flags to `THEANO_FLAGS`. Here's the third example:

```bash
export THEANO_FLAGS="device=cuda,floatX=float32,optimizer=fast_compile,cuda.root=/usr/local/cuda,cuda.cxx=/usr/bin/g++,dnn.enabled=True,dnn.include_path=/usr/local/cuda/include,dnn.library_path=/usr/local/cuda/lib64"
python theano_matmul.py
```

In this example,  I explicitly enabled cuDNN by setting `dnn.enabled=True`.  Crucially, `dnn.include_path` and `dnn.library_path` must reflect the exact locations of cuDNN include and library files, which are typically placed within the CUDA toolkit directory. If cuDNN is installed separately, ensure the paths point to their appropriate locations. Failure to set these paths will lead to Theano not utilizing cuDNN, and hence underperforming on GPU. The correct installation of cuDNN often requires additional manual steps and may involve copying files to the correct locations according to NVIDIA's guide. The `nvidia-smi` command can also provide insights into the versions of the NVIDIA driver, CUDA and cuDNN, allowing you to verify compatibility with Theano.

Several considerations are pertinent when debugging. Theano provides useful error messages, but deciphering them requires some experience, especially with CUDA related issues. Issues usually stem from incorrect path configurations, wrong CUDA/cuDNN versions,  or missing libraries. Checking the console output for these error messages, inspecting the versions of drivers, CUDA toolkit, and cuDNN installation are often the key to identify root cause.

To further deepen your understanding, explore the official documentation for Theano, CUDA, and cuDNN. These resources often contain comprehensive details and troubleshooting guides. Various online forums and communities dedicated to scientific computing with Python often address recurring configuration issues. Furthermore, review the official NVIDIA documentation for CUDA and cuDNN, providing specific guidelines on their installation and usage with different versions of Theano. The official Theano source code, even though the project has become inactive, can provide valuable insights into the mechanisms involved. By referencing these resources and debugging systematically, it’s possible to successfully configure Theano to run on a GPU from the terminal.
