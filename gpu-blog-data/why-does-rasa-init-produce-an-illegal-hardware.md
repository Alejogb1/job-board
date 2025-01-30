---
title: "Why does `rasa init` produce an 'illegal hardware instruction' error on Apple M1 Macs using zsh?"
date: "2025-01-30"
id: "why-does-rasa-init-produce-an-illegal-hardware"
---
The “illegal hardware instruction” error encountered when executing `rasa init` on Apple M1 Macs using zsh arises primarily from architectural mismatches in precompiled binary dependencies within the Rasa framework’s installation process, specifically related to its Python environment and associated machine learning libraries. These libraries often include components compiled for the x86_64 architecture, which the ARM-based M1 processor cannot directly execute.

My direct experience with this involved setting up a new Rasa development environment after receiving an M1 MacBook Pro. Initially, I meticulously followed the standard installation guide, creating a virtual environment and installing Rasa using pip. However, the `rasa init` command consistently crashed with the “illegal hardware instruction” message, even after re-installing Python, Rasa and its dependencies. The problem, as I eventually discovered after considerable debugging, wasn’t in Rasa’s core code itself but stemmed from binary dependencies not compiled for the ARM64 architecture present in the M1 series chips.

The underlying issue revolves around the way Python packages, especially those involved in numerical computations or machine learning, often rely on pre-compiled C/C++ libraries. These libraries, when included as wheels (.whl files) in the Python package, are specific to the architecture for which they were built. Most libraries were, until recently, primarily distributed as x86_64 binaries. When a package with such an x86_64 wheel is installed on an ARM64 system, it will try to execute instructions that the ARM64 processor does not recognize, leading to the fatal “illegal hardware instruction” error. While Rosetta 2 enables translation of x86_64 instructions, this is not universally effective, particularly when the binaries are deeply integrated into the system and have dependencies that themselves have not been compiled for ARM64.

The problem manifests specifically with `rasa init` because it triggers the setup of a full Rasa project, including installation of various dependencies such as TensorFlow, PyTorch, or scikit-learn, which are heavily reliant on these compiled libraries. Even if the installation appears to proceed without error using pip, the incompatibility only surfaces when these libraries attempt execution, a crucial part of project initialization.

To mitigate the issue, it's essential to ensure that the installation process favors ARM64 (also known as `arm64`, `aarch64`, or `darwin-arm64`) binaries or, in situations where no prebuilt binaries are available, compiles them specifically for the M1 chip. I have identified three primary approaches that, in my practical experience, have proven effective.

**Code Example 1: Using miniforge for an ARM64-Specific Environment**

The first successful strategy I employed was to use miniforge, a minimal installer for conda, a package, dependency, and environment management system. Miniconda or Anaconda distributions typically offer a broader range of packages, some of which might still fall back to x86_64 even on an ARM64 system. Miniforge, explicitly designed for ARM64, guarantees that the created environment will strongly prioritize the correct architecture for library installs.

```bash
# Download and install miniforge for ARM64
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh
bash Miniforge3-MacOSX-arm64.sh

# Create a new environment using miniforge
conda create -n rasaenv python=3.9 -y
conda activate rasaenv

# Install rasa using pip, within the correctly architected environment
pip install rasa
# Run rasa init, should now proceed without illegal instruction
rasa init
```

*Commentary:* This example highlights the most robust solution. Using miniforge for ARM64 ensures that conda will retrieve and install correct libraries compiled for the Apple M1 processors. Specifying a target Python version (3.9 in this case) aligns with Rasa’s requirements. The crucial aspect is creating the environment using `conda create`, not a standard virtual environment, which, even on a M1 chip, might install x86_64 binaries by default.

**Code Example 2: Installing TensorFlow-Metal (if necessary)**

While Rasa itself may not directly require TensorFlow, some configurations, particularly those involving machine learning components, may depend on it or trigger the installation of x86_64 versions by dependency resolution. To explicitly use the Metal accelerated version, which is optimized for Apple silicon, I have manually installed it, and sometimes, found it necessary. This is especially useful if other parts of the projects need optimized GPU computation on the M1 chip.

```bash
# Ensure pip is updated
python -m pip install --upgrade pip

# install tensorflow-metal explicitly, after the initial `pip install rasa`
pip install tensorflow-macos
pip install tensorflow-metal

# This may require installing apple's ml compute framework, if not already present
# pip install "apple-ml-compute==0.1.0"

# Retest with rasa init
rasa init
```

*Commentary:* This example addresses an auxiliary but related problem. If TensorFlow gets installed as part of Rasa's installation using pip, it might not choose the optimal, metal accelerated version. Explicitly specifying tensorflow-macos and tensorflow-metal ensures that GPU acceleration works well on M1 chips if TensorFlow is a necessary component for a given project. Note that the `apple-ml-compute` library may also be needed based on the tensorflow-metal installation requirements. This should be added as required.

**Code Example 3: Reinstalling All Dependencies with pip flags (as a last resort)**

As a final troubleshooting step, and often less ideal, I have forced pip to ignore cached wheels and install from source, thereby ensuring compilation for the target ARM64 architecture if necessary. This is generally slower, but can be beneficial if no precompiled binaries are found and source distribution is available. Note that this approach requires a functioning development toolchain like Xcode and potentially other C/C++ build dependencies.

```bash
# clean the pip cache
rm -rf ~/.cache/pip

# reinstall all with --no-cache-dir, to force retrieval
pip install --no-cache-dir rasa --force-reinstall --no-binary :all:

# Test rasa init
rasa init
```

*Commentary:* This approach forces `pip` to disregard any cached wheels and reinstall all packages from their respective source distributions. The `--no-binary :all:` part ensures that binary packages aren't used and compilation will be done locally on the system. While this can be effective, it's also slower, and requires more pre-setup and a properly configured development environment (e.g., with Xcode Command Line Tools). It also can cause issues if building from source fails for individual dependencies. This should be used only if the previous two approaches fail.

For resource recommendations, consult the official documentation for Rasa, which includes sections for installing and setting up development environments. The documentation for miniforge and conda are also very useful, especially concerning environment management. Additionally, the Python Packaging Authority’s (PyPA) websites, focusing on pip and wheels, offers deeper insight into package distribution and resolution processes. These, combined with community forums and bug reports, provide a comprehensive understanding of dependency issues on ARM64 architectures.

The “illegal hardware instruction” error with `rasa init` on M1 Macs using zsh is a consequence of architectural mismatches in precompiled binary dependencies. Using miniforge for an ARM64 environment is a direct and effective strategy. Supplemented by targeted dependency installations or forced source compilation where necessary, it resolves the core problem and ensures successful Rasa project initialization on these platforms. My experiences have shown that meticulous attention to the architecture of the development environment, especially in the context of machine learning and numerical Python, is paramount to a stable setup.
