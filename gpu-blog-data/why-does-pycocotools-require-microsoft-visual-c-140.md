---
title: "Why does pycocotools require Microsoft Visual C++ 14.0 or greater, even though it's installed?"
date: "2025-01-30"
id: "why-does-pycocotools-require-microsoft-visual-c-140"
---
The `pycocotools` library, commonly used for working with the COCO (Common Objects in Context) dataset, frequently exhibits installation difficulties specifically tied to the underlying Microsoft Visual C++ runtime requirements, even when a seemingly compatible version is present on the system. This stems from a crucial aspect of compiled Python extensions: they are often linked against specific versions of runtime libraries, and mismatches can lead to load-time failures.

The issue is not solely about having *a* Visual C++ redistributable installed, but rather about having the precise *version* that the `pycocotools` C extension was compiled against. When a Python package incorporates C or C++ code, as is the case with `pycocotools`, the resulting binary extension (.pyd on Windows) will be dynamically linked to specific versions of runtime libraries on the target platform.  `pycocotools` relies on the Microsoft Visual C++ Redistributable because it utilizes C++ under the hood for performance critical tasks, especially within its image processing and evaluation code. During compilation, the specific Visual C++ compiler, for instance Visual Studio 2015, 2017, 2019 or 2022, introduces a dependency on its specific redistributable package. If a Python environment picks up a binary compiled with a different compiler it will attempt to load the relevant redistributable. If it's not found or the wrong version is present, Python cannot load the binary correctly leading to the errors users encounter.

From my experience debugging numerous package installations and deployments, I've found that the key problems revolve around this: the pre-built `pycocotools` wheels available on PyPI are often compiled using specific versions of Visual Studio. While users might possess a seemingly "newer" version of the redistributable, it does not guarantee backward compatibility for runtime linking.  The older redistributables aren’t replaced when newer versions are installed, so both may reside in the environment and in general the correct one will be selected. However, this is not always the case and the correct redistributable may be missing.

Consider the scenario where the pre-built `pycocotools` wheel was generated using Visual Studio 2015 (VC++ 14.0). If you only have the Visual Studio 2019 redistributable (VC++ 14.2) or higher installed, Python will not be able to load the `pycocotools` extension. This is because the extension’s import procedure will fail to find a matching runtime library. Although newer versions include compatibility features, the runtime linker in Windows still requires the specific versions of dependencies to match. This is distinct from the system's ability to run applications compiled with newer compiler versions, which can often link against older runtime libraries.

Further complicating matters, if you encounter an error stating that a particular .dll file is missing, for example ‘vcruntime140.dll’ or similar, even when the correct redistributable is installed, it can often be due to system path issues or the .dll’s version in the system directories not matching the version that `pycocotools` was linked against. This issue is further exacerbated by the fact that Microsoft has released multiple variants of the redistributable with incremental version updates, each requiring specific patches or version support.

Let's explore this with a few scenarios and corresponding code examples, illustrating common errors and diagnostic steps.

**Code Example 1: Import Error After Installation**

Assume `pycocotools` was installed through `pip install pycocotools` and results in the following import error during execution:

```python
# scenario_1.py
import pycocotools.coco as coco

print("Successfully imported pycocotools")

```

The error traceback might indicate something like: `ImportError: DLL load failed while importing coco: The specified module could not be found`. This doesn't always explicitly say Visual Studio redistributable is at fault, however this is the most probable cause. The code itself is perfectly correct. It's the underlying compiled extension that cannot be loaded. This indicates a problem with binary dependencies, most often the C++ runtime dependencies.

The solution isn't to simply reinstall the package. The correct resolution in this situation would be to explicitly install the Visual C++ 14.0 redistributable from Microsoft's website.

**Code Example 2: Diagnosing Missing Dependencies**

To delve deeper, we can utilize the `depends.exe` tool (available within Visual Studio development tools) to inspect the dependencies of `pycocotools`. Given a directory with a potentially problematic `pycocotools` installation, the command, in a Visual Studio developer command prompt, `depends <python_env_path>\lib\site-packages\pycocotools\_mask.cp3x-win_amd64.pyd`, will display the .dll dependencies.

```console
# Example Output of dependency walker showing missing 'VCRUNTIME140.dll'

Dependency Walker:
---
   Module: C:\Program Files\Python39\lib\site-packages\pycocotools\_mask.cp39-win_amd64.pyd
   ---
   ...
   Missing Dependencies:
       VCRUNTIME140.dll
   ...

```
Here the `VCRUNTIME140.dll` is missing, indicating the need for the Visual C++ 2015 redistributable. This output confirms our previous assessment that the core of the issue is a missing dependency. This method allows direct diagnosis of what's missing and eliminates guesswork.

**Code Example 3: Custom Compile**

In advanced scenarios, where the pre-built wheels do not work, or there are version mismatches, it becomes beneficial to build `pycocotools` from source.

First, clone the repository: `git clone https://github.com/cocodataset/cocoapi.git`
navigate into the Python API directory: `cd cocoapi/PythonAPI`
Then, ensure you have the correct Visual Studio compiler, and then, issue the command `python setup.py build_ext --inplace`.

```python
# setup.py (from cocoapi/PythonAPI)
from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        'pycocotools._mask',
        sources=['pycocotools/_mask.pyx'],
        include_dirs=[numpy.get_include()]
    ),
]


setup(
    name='pycocotools',
    packages=['pycocotools'],
    ext_modules=cythonize(extensions),
    install_requires=['numpy'],
    zip_safe=False
)
```

This example demonstrates the core setup script for compiling the `_mask` extension of `pycocotools`. This forces the extension to be linked against the runtime libraries of the current Visual Studio installation. This custom build process ensures that the extension is linked against the available Visual C++ runtime. Post compilation a similar command `python scenario_1.py` should now succeed if the correct compiler, redistributable and all other dependencies are present.

**Resource Recommendations:**

To gain a deeper understanding of compiled Python extensions, I would recommend consulting the official Python documentation on C extensions, as well as exploring literature about Windows DLL loading mechanisms. For a detailed breakdown of Visual Studio and its associated runtimes, researching Microsoft’s documentation is advisable, including how each version of Visual Studio influences which specific Visual C++ redistributables must be present. Moreover, materials on the workings of the Windows dynamic linker would be pertinent to grasp the subtle nuances of runtime loading and versioning constraints. Lastly, consulting Python packaging guidelines on distributing compiled extensions provides a practical perspective on the challenges faced by package maintainers and developers using these types of packages. These resources provide a comprehensive overview and enable better debugging practices.
