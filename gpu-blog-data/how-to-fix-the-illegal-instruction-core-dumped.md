---
title: "How to fix the 'Illegal instruction (core dumped)' error when installing TensorFlow on Ubuntu 20.04 Python 2.7 in a VirtualBox environment?"
date: "2025-01-30"
id: "how-to-fix-the-illegal-instruction-core-dumped"
---
The "Illegal instruction (core dumped)" error during TensorFlow installation, especially on older Python versions within a virtualized environment like VirtualBox, typically stems from CPU instruction set incompatibility. Specifically, TensorFlow binaries pre-compiled for optimized performance often leverage Advanced Vector Extensions (AVX), AVX2, or similar instruction sets. If the host CPU supports these but the virtual machine’s configuration masks them or the guest operating system does not expose them correctly to the virtualized processor, the TensorFlow library, upon executing these instructions, triggers the illegal instruction fault, leading to the core dump. I've encountered this scenario multiple times while building legacy deployment environments on virtual machines.

The core issue is not a fault of TensorFlow itself but rather a mismatch in expectations between the pre-built library and the available CPU features within the guest OS environment. When TensorFlow attempts to execute an AVX instruction, for instance, and that instruction isn't actually supported by the emulated processor presented to the guest OS, the error is produced. This error manifests more frequently in older versions of Python (2.7 in this case) due to the lack of wheel package flexibility and the reliance on pre-compiled binaries tailored for broad hardware compatibility, often at the expense of excluding very old hardware. Modern `pip` installs of TensorFlow typically have more options for custom building with more limited feature sets.

To rectify this problem, there are generally two primary approaches: compiling TensorFlow from source or forcing CPU feature restrictions when installing prebuilt binaries. Compiling from source allows targeted feature enablement; however, it is time-consuming and requires substantial resources. For quick deployments or working in constrained virtual environments, the pre-compiled binary solution, along with CPU feature restriction, is a more practical and faster solution. With the latter method, specific environment variables need to be set before executing the TensorFlow libraries. These environment variables essentially inform TensorFlow that the executing CPU does not possess the full feature set assumed by the pre-compiled binary, causing it to take execution paths that avoid unsupported operations.

The first example demonstrates how to constrain TensorFlow to use only the basic SSE4.1 and SSE4.2 instruction sets, which are significantly more widely supported, by setting the `TF_XLA_FLAGS` and `TF_DISABLE_MKL` environment variables within the virtual environment shell. This avoids invoking AVX and similar vector instructions.

```bash
export TF_XLA_FLAGS="--tf_cpu_no_sse --tf_cpu_no_sse4_1 --tf_cpu_no_sse4_2 --tf_cpu_no_avx --tf_cpu_no_avx2 --tf_cpu_no_avx512f"
export TF_DISABLE_MKL=1
python -c "import tensorflow as tf; print(tf.__version__)" # Example to test the setting
```

In the preceding snippet, `TF_XLA_FLAGS` is populated with flags to explicitly disable all but base instruction sets that are generally available on legacy systems. The `--tf_cpu_no_*` family of flags instructs the TensorFlow XLA (Accelerated Linear Algebra) component to avoid the associated CPU instruction sets. `TF_DISABLE_MKL=1` disables usage of the Intel Math Kernel Library, an optimized numerical computation library that heavily relies on advanced instructions like AVX. This step is critical to avoid libraries that use AVX even if AVX is technically not exposed via the XLA flags.

The second example uses a different approach, by manipulating the dynamic linker behavior. By setting the `LD_PRELOAD` environment variable and pointing it to a specialized library called `libx86-noavx.so`, we're forcing a library to be pre-loaded before the actual TensorFlow dynamic libraries, and this library intercepts AVX instruction calls and throws dummy exceptions. This approach does not prevent the TensorFlow code from attempting to use these instructions, but it handles the resulting crash gracefully. Note, this is a more advanced technique and might need compilation on your specific guest OS. I've relied on this when the XLA flags themselves were insufficient to bypass AVX related errors for legacy Tensorboard versions.

```bash
# Assumes libx86-noavx.so is compiled and available.
export LD_PRELOAD=/path/to/libx86-noavx.so
python -c "import tensorflow as tf; print(tf.__version__)" # Example to test the setting
unset LD_PRELOAD # Unset this later if not required
```

This technique depends on a custom-built `libx86-noavx.so` library, which must be compiled explicitly for the target environment. This pre-loading approach, while more involved, serves as a last resort if disabling instructions through XLA flags proves insufficient. I often find it necessary with older builds of TensorFlow with tight coupling to AVX via static C++ libraries.

The third example addresses a specific instance of the problem within virtualized environments. When the virtualization layer, such as the VirtualBox configuration, does not properly forward all required CPU features to the guest, even the above methods may not work. Here I have found it useful to specify a limited set of processor instructions when starting VirtualBox, forcing a different instruction set profile. When I used this I usually needed to install a new image in VirtualBox.

```bash
VBoxManage modifyvm "YourVMName" --cpu-profile "Intel Core i7-4770"
# Replace YourVMName with the actual name of your virtual machine.
# This configures the virtual machine to emulate a Haswell series i7
# that has limited AVX support. After this, I would reload the virtual machine.
```

This command, executed from the host operating system command line (not the guest), modifies the virtual machine configuration, specifically limiting the instruction sets exposed to the guest. By choosing a CPU profile that does not have AVX2, we prevent the virtual processor from pretending to support instructions it will not implement. After this change, the guest OS is effectively using an emulated processor without the advanced extensions that were causing the “illegal instruction” error. You would need to shut down the VM and restart it for these changes to take effect.

In summary, the "Illegal instruction" error arises when there’s a mismatch between TensorFlow’s assumed CPU capabilities and the actual features available within a virtualized environment. Environment variables (`TF_XLA_FLAGS`, `TF_DISABLE_MKL`), a custom `LD_PRELOAD` library, or even adjustments to virtual machine configurations are used to circumvent this issue. These approaches, by either avoiding the usage of problematic instructions or by altering the environment, allow the TensorFlow library to function without crashing. The choice of solution often depends on the complexity and specifics of your environment and the flexibility to manipulate your environment, often choosing the lowest effort solution that will work.

For additional resources, I suggest researching documentation related to CPU instruction sets, particularly SSE, AVX, AVX2, and AVX512. Intel's Software Developer Manuals provide a deep dive into these architectural details, although understanding all of it is not always necessary for a working environment. Documentation for VirtualBox's command-line tools will explain how to adjust virtual machine configurations and CPU profiles which can often be the root cause of these problems. Finally, reading the TensorFlow installation documentation in the context of virtualized environments or specific hardware limitations will often give insights into the root causes of many of these install problems.
