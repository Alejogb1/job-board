---
title: "Why is python36.dll not found when installing Python 3.6 libraries via pip?"
date: "2025-01-30"
id: "why-is-python36dll-not-found-when-installing-python"
---
The frequent "python36.dll not found" error encountered during `pip` installations of Python 3.6 libraries, particularly within virtual environments or custom installation setups, typically arises from a discrepancy between the Python distribution `pip` is interacting with and the actual location of the core Python runtime libraries. Specifically, `pip` relies on metadata and environmental variables pointing towards the correct Python installation path and its associated `.dll` files; when these pointers are misaligned or incomplete, the loader fails to locate `python36.dll`, a critical dynamic link library for Python 3.6.

My experience supporting diverse Python development environments, including legacy systems and custom embedded solutions, has shown this issue can manifest in several ways, all related to the system's ability to resolve the Python runtime's location. Incorrectly set `PYTHONPATH` environmental variables, a fragmented Python installation directory, or `pip` mistakenly referencing a different Python distribution entirely can all contribute to this error. Essentially, when `pip` attempts to execute part of the install process, it needs to dynamically load elements of the Python runtime using `python36.dll`. This happens when `pip` or some packages during their installation require a live python environment to run python scripts.

One primary cause stems from virtual environments. Virtual environments isolate Python installations, which is their purpose; however, if not set up or activated properly prior to running `pip`, the system’s path may not be pointing to the location of the required `.dll`. Suppose the user has multiple Python installations on their machine, such as a global Python 3.9 installation and a virtual environment with Python 3.6. If the virtual environment is not activated (and thus its specific bin or script directories not prepended to the system path), a `pip` command will utilize the global installation of Python. It will then be unable to locate `python36.dll` as that specific file only exists within the 3.6 install directory.

A further complication arises from the dynamic linking behavior of Windows systems. The `python36.dll` needs to be discoverable by the operating system's dynamic linker. When dealing with virtual environments, or even installing a package locally from a custom source, these libraries are often not present in the standard Windows library paths. This absence causes the runtime to be unable to load when the package attempts to make use of python code via the system calls.

Let’s examine three practical examples that illustrate the issue along with possible resolutions:

**Example 1: Virtual Environment Misconfiguration**

In this scenario, a user creates a virtual environment named `myenv` for a legacy project requiring Python 3.6. However, they install a package, `requests`, without activating the environment first.

```bash
# Without virtual environment activation
pip install requests

# Expected output showing error
# ...
# ERROR: Could not install packages due to an OSError:
#        [WinError 126] The specified module could not be found
#        python36.dll could not be found
```

This occurs because the global `pip` is being used (and therefore the global Python interpreter), which doesn't have access to `python36.dll` associated with the virtual environment’s installation. The fix involves properly activating the virtual environment before using `pip`:

```bash
# Activate virtual environment (on Windows)
myenv\Scripts\activate
# or
#.\myenv\Scripts\activate

# Install the package after activation
pip install requests

# Expected Output:
# Successfully installed requests-2.28.1
```

The activation process correctly sets necessary environment variables, enabling `pip` to locate the correct Python distribution and its associated libraries, resolving the `python36.dll` issue.

**Example 2: Inconsistent Path Variables**

Another common scenario involves an environment where the `PATH` variable is not correctly configured after installing a custom Python distribution, specifically the install location of the python3.6 dll. Assume the user has installed Python 3.6 at `C:\Python36` and has added `C:\Python36;C:\Python36\Scripts` to the system `PATH`. The user is installing a local package from a directory containing a `setup.py` using pip:

```bash
cd C:\local_pkg_dir

pip install .
# or
python setup.py install

# Output shows error due to python36.dll not being found
# ...
# ERROR: Could not install packages due to an OSError:
#        [WinError 126] The specified module could not be found
#        python36.dll could not be found
```
Even if the user has a global python installation, or has an activated virtual environment, when trying to use `pip install .` to run the setup script, the system will look at the path specified by the operating system for any python executables or dlls. If the `C:\Python36` is not a recognized installation location for an operating system, it will not be in the system wide dynamic linker list. This will then cause the error as the system is searching for this particular dll, which is in the specified path, but is not in the OS defined list of possible dll load locations.

The solution is to either explicitly add `C:\Python36` to the system path, or if you wish to avoid system wide changes, you can use a fully specified call to python. So rather than using the system path `pip`, you would use the executable installed at `C:\Python36\python.exe`.

```bash
C:\Python36\python.exe -m pip install .
```

By explicitly calling the python installation related to the libraries you will use, you avoid the incorrect `python36.dll` lookup and the package installs correctly. This explicitly tells pip to use the `python.exe` in the custom install directory, and it's associated libraries and dlls, rather than some generic path version of python.

**Example 3: Conflicting Python Installations**

This example highlights a more complex scenario involving multiple installed Python versions, with the system path pointing to the incorrect one. Assume both Python 3.6 and Python 3.9 are installed, and `pip` is configured for Python 3.9, but we wish to install into the 3.6 environment using `pip` in the system path. The `PATH` environment variable could be set to use Python 3.9's `pip` by default and not the activated 3.6 virtual environment's pip.

```bash
# With virtual environment activated with 3.6 Python
# but system path pip version is 3.9
pip install some-package
# ...
# ERROR: Could not install packages due to an OSError:
#        [WinError 126] The specified module could not be found
#        python36.dll could not be found
```

Even with an activated virtual environment, the system's default `pip` may be associated with the wrong Python version. Resolving this requires either manually specifying the pip.exe that comes with the virtual environment by specifying it's install location:

```bash
# Correctly specifying pip in virtual environment
.\myenv\Scripts\pip.exe install some-package

# Or running with the virtual environment python installation:
.\myenv\Scripts\python.exe -m pip install some-package
```

This explicitly invokes the correct instance of `pip` associated with Python 3.6 located in the virtual environment's install directory, thereby resolving the `python36.dll` error.

To aid in resolving these issues, several resources can be consulted. Start with the official Python documentation, which details installation procedures and the behavior of virtual environments. The `pip` documentation also offers insight into usage and troubleshooting. Additionally, numerous online forums and community Q&A sites (such as Stack Overflow) contain discussions and solutions contributed by other Python developers who have encountered similar problems. Finally, review articles covering Windows dynamic linking behavior, which can provide fundamental knowledge about how dynamic libraries load on windows systems. Understanding the system behaviors as well as python's implementation helps diagnose and resolve this issue in a way that is specific and robust for each developer environment.
