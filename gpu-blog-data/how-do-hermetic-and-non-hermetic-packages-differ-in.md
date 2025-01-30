---
title: "How do Hermetic and non-Hermetic packages differ in Python?"
date: "2025-01-30"
id: "how-do-hermetic-and-non-hermetic-packages-differ-in"
---
The primary distinction between hermetic and non-hermetic Python packages lies in their dependency management and reproducibility properties. Hermetic packages, ideally, encapsulate all necessary code and dependencies within their distribution, thereby minimizing reliance on external factors during installation and execution. Non-hermetic packages, conversely, frequently declare dependencies that are resolved at install time, often pulling components from various sources. This difference has profound implications for build repeatability and deployment consistency. My experience maintaining complex scientific software environments has vividly illustrated these trade-offs.

Let's break down the specifics. A hermetic package, in principle, functions almost as a self-contained unit. Its installation should not require network access (beyond the initial download of the package archive) because all required code, including third-party libraries, is bundled within it. This approach significantly reduces the risk of environment-related errors and ensures that the application behaves the same regardless of the host system’s pre-existing library landscape. This is achieved by distributing copies of all necessary dependency packages alongside the primary project code. The package’s installation mechanism (e.g., a setup script or wheel archive) is designed to integrate these internal dependencies, bypassing any system-level or virtual environment-level resolution. In practical scenarios, creating truly hermetic packages from scratch can be extremely difficult due to the sheer volume of libraries that may be required, along with ongoing dependency upgrades.

On the other hand, non-hermetic packages usually specify their dependencies through metadata, such as a `requirements.txt` file or in the `setup.py` file. When the package is installed via `pip`, these dependencies are fetched and installed from external repositories like PyPI or specified locations. This approach offers flexibility and minimizes package size because the dependency code is not duplicated in every package that needs it. However, it also introduces vulnerabilities. Changes to upstream repositories, such as package upgrades or removal, can affect the behavior of your installed package, leading to the dreaded "it works on my machine" problem. Furthermore, version conflicts between the package’s dependency requirements and the user's existing environment are common and require careful management. This often involves extensive use of virtual environments or containerization to mitigate these issues. My experience with version-related dependency conflicts across different build agents underscores this issue’s prevalence and the effort required to resolve these discrepancies.

Now, let's look at some simplified code examples illustrating these concepts, keeping in mind the practical limitations of achieving perfect hermeticity.

**Example 1: Non-Hermetic Package (Conceptual)**

```python
# setup.py for a non-hermetic package
from setuptools import setup

setup(
    name='my_nonhermetic_package',
    version='1.0.0',
    install_requires=[
        'requests>=2.28.0',
        'numpy>=1.23.0'
    ],
    packages=['my_nonhermetic_package'],
)
```

```python
# my_nonhermetic_package/__init__.py
import requests
import numpy as np

def fetch_data(url):
    response = requests.get(url)
    return np.array(response.json())
```

**Commentary:** This package specifies dependencies on `requests` and `numpy` in its `setup.py` metadata. When a user installs this package using `pip`, the installer will download compatible versions of these dependencies from PyPI (or another specified repository). The key element here is the reliance on external sources to satisfy dependencies. This external reliance means potential problems if the specified or compatible version changes or becomes unavailable. While convenient for distribution and development, it isn't conducive to precise, repeatable deployment.

**Example 2: A Hermetic Package (Simplified - Illustrative)**

```python
# setup.py for a simplified hermetic package
from setuptools import setup, find_packages
import shutil
import os
from pathlib import Path

# Simplified example - in reality this is more complex
def copy_dependencies(package_dir, dep_dir):
    os.makedirs(dep_dir, exist_ok=True)
    
    # Assume we have a pre-downloaded numpy package in 'dependencies/numpy'
    source_numpy = Path("dependencies/numpy") 
    dest_numpy = Path(dep_dir) / "numpy"
    
    if source_numpy.exists():
      shutil.copytree(source_numpy, dest_numpy)

    # Assume we have a pre-downloaded requests package in 'dependencies/requests'
    source_requests = Path("dependencies/requests")
    dest_requests = Path(dep_dir) / "requests"
    if source_requests.exists():
      shutil.copytree(source_requests, dest_requests)

package_name = 'my_hermetic_package'
package_dir = package_name

copy_dependencies(package_dir, f"{package_dir}/vendored_deps")


setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(),
    package_data={
      package_name: ["vendored_deps/*/*"] #Include vendored dependencies in the package
    }

)

```

```python
# my_hermetic_package/__init__.py
import sys
import os
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'vendored_deps'))

import numpy as np
import requests

def fetch_data(url):
    response = requests.get(url)
    return np.array(response.json())
```

**Commentary:** In this illustrative (and greatly simplified) example, we *manually* copy pre-downloaded versions of `numpy` and `requests` into a `vendored_deps` directory inside the package. During setup, this directory is included as package data, thus embedding these dependency packages within the project itself. The package's `__init__.py` modifies the Python module search path to include the internal `vendored_deps` folder before importing the dependencies.  This approach removes the dependency on external repositories *after* the initial bundling. It's also crucial to note this method is highly simplified; handling namespaces, binary extensions, platform differences, and ensuring compatibility across many versions of bundled dependencies is a complex issue. Real world hermetic packaging involves significantly more sophisticated techniques and is not a task to be taken lightly. It does, however, demonstrate the fundamental concept.

**Example 3: Non-Hermetic Package with Dependency Pinning**

```python
# requirements.txt for a non-hermetic package with pinned dependencies
requests==2.28.1
numpy==1.23.3
```

```python
# setup.py for non-hermetic package, using requirements.txt
from setuptools import setup

with open("requirements.txt", "r") as f:
    required = f.read().splitlines()

setup(
    name='my_nonhermetic_pinned',
    version='1.0.0',
    install_requires=required,
    packages=['my_nonhermetic_pinned'],
)

```

```python
# my_nonhermetic_pinned/__init__.py
import requests
import numpy as np

def fetch_data(url):
    response = requests.get(url)
    return np.array(response.json())
```

**Commentary:** This approach improves dependency management compared to our first non-hermetic example. By specifying exact versions in the `requirements.txt` file, we significantly reduce the variability introduced by updates to the dependency packages. However, it does *not* create a hermetic package, as `pip` still needs to access external sources to download the pinned versions. The advantage here is a more reproducible build process, as it ensures that the same versions are retrieved as long as those external repositories remain consistent. This method attempts to mitigate some of the risks associated with non-hermetic packaging but does not fully eliminate them. In my experience, the meticulous maintenance of pinned dependencies is essential for stable deployments but remains vulnerable to repository changes.

**Resource Recommendations (No Links):**

For in-depth understanding of Python packaging, consult the official documentation of setuptools and pip. The Python Packaging Authority (PyPA) also provides excellent guides and tutorials covering best practices for dependency management. Furthermore, several articles and presentations on the challenges of reproducible builds and deployment are available from different communities within the Python ecosystem. Reviewing discussions around virtual environments, containerization, and specific package management tools like Poetry and Conda will also significantly enhance your understanding. Exploring how build tools such as Bazel address hermeticity principles in software development in general will also provide broader context. Pay specific attention to the tooling around bundling dependencies as well as solutions proposed around caching dependency archives. This will offer a full perspective of the practical difficulties in implementing real world solutions. A clear grasp of these resources will provide the foundation to make informed decisions when developing, deploying, and maintaining Python applications.
