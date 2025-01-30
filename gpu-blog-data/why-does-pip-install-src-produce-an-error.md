---
title: "Why does pip install src produce an error?"
date: "2025-01-30"
id: "why-does-pip-install-src-produce-an-error"
---
The core issue with `pip install .` (or `pip install src`, assuming `src` is the project directory) frequently stems from a mismatch between the project's setup and the expectations of `pip`.  In my years working on large-scale Python projects and contributing to open-source libraries, I've encountered this problem countless times.  The error isn't inherently about the `src` directory itself, but rather how your project's metadata (found primarily in `setup.py` or `pyproject.toml`) interacts with the `pip` installation process.

**1.  Explanation:**

`pip install .` initiates a local installation.  `pip` searches for a setup script (either `setup.py` or a build-system declaration in `pyproject.toml`) to determine how to build and install your project.  Errors arise when this setup script is missing, improperly configured, or incompatible with your project's structure.  Common causes include:

* **Missing or Incorrect `setup.py`:**  Older projects rely on `setup.py`, a script that defines metadata such as package name, version, dependencies, and the location of source code.  Errors often occur when this file is absent, contains syntax errors, or fails to correctly specify the project's structure.  Specifically, the `packages` argument in `setup()` must accurately reflect the directory containing your Python modules.  Incorrectly specifying this can lead to `ModuleNotFoundError` exceptions during the installation process.

* **Improper `pyproject.toml` Configuration:** Newer projects often use `pyproject.toml` with a build-backend such as `setuptools` or `poetry`.  Errors arise when the `[build-system]` section is misconfigured, missing required fields, or if the specified build-backend is not installed. This file is more flexible than `setup.py` allowing for better separation of concerns.  However, incorrect specification of the build-system or the `[tool.setuptools]` or `[tool.poetry]` sections can lead to build failures.

* **Incorrect Package Structure:**  The location of your Python modules is crucial.  `pip` expects a standard package layout. If your Python modules are not correctly organized within the project directory (typically under a directory with the same name as the package, e.g., a `mypackage` directory containing `__init__.py` and other modules), `pip` will fail to locate and install them.

* **Missing Dependencies:**  Your project might depend on external libraries.  If these dependencies are not listed in the `install_requires` section of `setup.py` or the `dependencies` section of `pyproject.toml`, `pip` will attempt to install your project without them, leading to errors during the later stages of the installation or when trying to use the installed package.

* **Conflicting Dependencies:**  Version conflicts between your project's dependencies and the existing packages in your environment can also cause installation failures. `pip`'s dependency resolution mechanism might struggle to find a compatible combination, resulting in errors.

**2. Code Examples with Commentary:**

**Example 1: Incorrect `setup.py`**

```python
# setup.py (incorrect)
from setuptools import setup

setup(
    name='mypackage',
    version='0.1.0',
    # INCORRECT:  packages should list the package directory
    packages=['src'], #Should be ['mypackage']
    )
```

This `setup.py` incorrectly lists `src` as a package.  If `mypackage` is actually a subdirectory of `src` containing `__init__.py`, it should specify `packages=['mypackage']` instead.  `pip` would fail because it wouldn't find the expected package structure.  This leads to cryptic error messages related to module imports.


**Example 2: Correct `setup.py`**

```python
# setup.py (correct)
from setuptools import setup, find_packages

setup(
    name='mypackage',
    version='0.1.0',
    packages=find_packages(), # Automatically finds packages
    install_requires=['requests>=2.28.0'],
    )
```

This version uses `find_packages()`, simplifying the process.  It automatically detects packages based on the project's structure.  Including `install_requires` ensures that the `requests` library (version 2.28.0 or higher) is installed alongside `mypackage`. This is crucial for dependency management.


**Example 3: Using `pyproject.toml` with Poetry**

```toml
# pyproject.toml
[build-system]
requires = ["poetry-core>=1.0.0"]
build-backend = "poetry.core.masonry.api"

[tool.poetry]
name = "mypackage"
version = "0.1.0"
description = ""
authors = ["Your Name <you@example.com>"]
packages = [{include = "mypackage"}]
dependencies = [
    "requests >= 2.28.0",
]
```

This uses Poetry, a popular dependency management and build system. It clearly specifies the build-backend, the package to include (`mypackage`) and the project dependencies.  Poetry handles many aspects of package creation and installation automatically, reducing the chances of errors associated with manual configuration.  This is often preferred for its robustness and ease of use.


**3. Resource Recommendations:**

The official Python Packaging User Guide.  It thoroughly covers various aspects of packaging, from basic concepts to advanced techniques.  Consult this document for detailed information on `setup.py`, `pyproject.toml`, build backends, and dependency management.  The documentation for `setuptools`, `poetry`, and other build systems provide in-depth instructions and examples.  Furthermore, reviewing the official `pip` documentation is crucial to understand how `pip` interacts with your project's configuration files.  Understanding the intricacies of these tools is essential for effective package management.
