---
title: "Why can't pipenv on an M1 Mac locate TensorFlow?"
date: "2025-01-26"
id: "why-cant-pipenv-on-an-m1-mac-locate-tensorflow"
---

The inability of `pipenv` to locate TensorFlow on an M1 Mac, despite its apparent installation, typically arises from the architecture mismatch between the pre-built wheels available on PyPI and the M1's ARM64 architecture. I've personally encountered this numerous times while migrating development environments from Intel-based Macs. The core issue stems from how Python packages, particularly those with compiled C/C++ extensions like TensorFlow, are distributed. These packages are often pre-compiled into wheels targeting specific operating systems and CPU architectures.

Here's a more detailed explanation:

Pre-compiled Python wheels, the format `pipenv` primarily relies on, are optimized for specific processor instruction sets. While PyPI, the Python Package Index, hosts a wide array of packages, availability of wheels for the ARM64 architecture, used in M1 Macs, has been a work in progress. Many popular packages, TensorFlow included, historically prioritized x86-64 (Intel/AMD) architectures. When `pipenv` attempts to install a package like TensorFlow, it preferentially searches for a wheel matching the host architecture. If a suitable ARM64 wheel is not present on PyPI (or, if present, it might not be compatible with the current Python version in the virtual environment), `pipenv` will either fail the installation or silently install an x86-64 wheel. In the latter case, that wheel simply cannot operate on an M1 processor. Crucially, the command `pipenv install tensorflow` will report that tensorflow is installed without error because that's what pip was able to download. This then generates a great deal of confusion when trying to `pipenv shell` and `import tensorflow` because it fails.

Furthermore, Python on macOS, as a result of Rosetta, can be run in an x86-64 emulation layer. If pipenv attempts to use a version of python that runs under Rosetta emulation then the problem is compounded because this emulated architecture cannot use native M1 packages. The resulting environment can be a mixture of x86-64 and ARM64, leading to unpredictable behavior and further confusion.

The problem is exacerbated by how TensorFlow handles its dependencies. It relies on specific versions of libraries such as `absl-py` or `numpy`, which may also have the same architecture compatibility issue. In effect, a successful installation depends on not only having TensorFlow wheels available, but also ensuring compatibility of the entire dependency tree. Using `pip`, which directly installs a package, can sometimes bypass the issue by automatically building from source, but this is not the default behavior for `pipenv` which prefers to use pre-built wheels.

Here are some strategies with corresponding code examples that can mitigate this issue:

**Code Example 1: Explicitly specifying the platform architecture via `pipenv` arguments**

While `pipenv` lacks an explicit command for architecture specification, `pip` can be invoked with `pipenv run` to do so. We can override the default behaviour using the `PIP_PLATFORM` environment variable.
```bash
# Attempt to force the installation of ARM64 architecture package using `pipenv run`
PIP_PLATFORM="macosx_11_0_arm64" pipenv run pip install --no-cache-dir --force-reinstall tensorflow
```
This code sets the `PIP_PLATFORM` environment variable for the duration of the `pip` command. This forces pip to search for a version of the package using this architecture target. The `--no-cache-dir` flag helps ensure a fresh install and the `--force-reinstall` flag can deal with situations where an x86 version is already partially installed.

This approach is a bit of a workaround, directly engaging with `pip` through the `pipenv run` interface.  It is not native to `pipenv` itself, so it doesn’t modify `pipenv` behaviour directly. The key is that it can often find and install the right wheel.

**Code Example 2: Using Apple's `tensorflow-macos` package**

TensorFlow provides its own specialized package for macOS M1 called `tensorflow-macos`. This is typically accompanied by `tensorflow-metal` which provides GPU acceleration.
```bash
# Use tensorflow-macos and tensorflow-metal for native M1 support
pipenv install tensorflow-macos
pipenv install tensorflow-metal
```
This example illustrates the simplest way to install the M1-optimized package. This is now the best method for most situations, rather than the first example's workaround. The `tensorflow-macos` package includes all the core functionality, while `tensorflow-metal` enhances performance for macOS's Metal API. This is, in my experience, the most reliable way to get TensorFlow working correctly. It's also worth noting that if a tensorflow installation exists from other attempts then it might be worth removing with `pipenv uninstall tensorflow` first.

**Code Example 3: Creating a new virtual environment with the correct Python version**

It's also important to ensure that pipenv is using the correct python binary as described in the introduction. It is possible for `pipenv` to create environments with `x86` python which would require the x86 build of `tensorflow` and fail silently when an M1-native version is used.
```bash
# Create a new environment with a specific Python version.
pipenv --python 3.11

# then proceed with installations as normal
pipenv install tensorflow-macos
pipenv install tensorflow-metal

```
This first command instructs `pipenv` to create a new virtual environment using python 3.11 (or whatever the user specified). This often makes it clear if `pipenv` is attempting to use an emulated `x86` python by explicitly choosing the target python executable. It avoids the issue of the user not understanding how `pipenv` is creating environments. The subsequent commands install the tensorflow packages, assuming that a native version of python was chosen.

**Resource Recommendations:**

*   **Python Documentation:** The official Python documentation provides detailed explanations of packaging and distribution, including wheels and platform tags. Understanding how packages are built and identified is fundamental for resolving this issue.
*   **PyPI website:** The Python Package Index website (pypi.org) allows you to search for specific package versions and their respective platform tags. This is useful to identify whether a suitable wheel exists for the target architecture.
*   **TensorFlow documentation:**  The official TensorFlow documentation contains comprehensive information on installation, including specific instructions for macOS, especially now that Apple M1 compatibility is becoming ubiquitous. This documentation also describes the differences between tensorflow, tensorflow-macos, and tensorflow-metal.
*   **macOS System Information:** System Information in macOS can confirm the chip type (Intel or Apple Silicon) which helps in diagnosing architecture mismatches. It is useful for ensuring that a python environment is not running under emulation.
*   **Homebrew:** The Homebrew package manager often has versions of `python` compiled for M1 which are compatible with native M1 packages. If a specific version of python is needed and the existing installation is causing problems, then reinstallation or a fresh install via Homebrew may be appropriate.

Resolving the `pipenv` and TensorFlow issue on M1 Macs involves a careful examination of architecture compatibility, particularly for compiled packages. The shift to ARM64 is relatively recent and many existing python packages do not work with M1 hardware without explicit versions. Focusing on M1 compatible packages, using the correct python version, and using the `tensorflow-macos` distribution is the recommended path to resolving these types of dependency issues. Using `pipenv run pip` with specific options is often helpful for resolving older issues but should not be necessary anymore.
