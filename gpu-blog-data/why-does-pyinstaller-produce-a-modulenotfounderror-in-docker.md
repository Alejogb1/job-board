---
title: "Why does PyInstaller produce a ModuleNotFoundError in Docker?"
date: "2025-01-26"
id: "why-does-pyinstaller-produce-a-modulenotfounderror-in-docker"
---

Dockerizing Python applications packaged with PyInstaller often exposes a subtle but common `ModuleNotFoundError` that doesn’t manifest during local development. This discrepancy arises primarily from how PyInstaller bundles dependencies and how Docker images manage their environment, particularly concerning shared libraries and implicit paths. During my tenure building embedded Python services at a previous firm, I encountered this issue repeatedly when deploying to containerized edge devices. The root cause isn’t a bug in either PyInstaller or Docker but a mismatch in their expected contexts.

Specifically, PyInstaller attempts to bundle all necessary modules into a single executable, often using a temporary or relative path to access those dependencies during runtime. This process involves analyzing the script, identifying imported modules, and then copying or embedding these modules into the output directory, or in the case of a single executable, embedding the modules within the executable itself. The bundled application expects the relative path or bundled location of the dependencies to be present at runtime. However, when this bundled executable is placed inside a Docker container, it executes within a confined file system, often a minimalistic Linux distribution. The paths and environments that PyInstaller relied upon during the bundling process, are not necessarily replicated in the Docker container's environment. The crucial point is that PyInstaller's bundled path resolution can break if the internal layout it created is moved without proper adjustment of related paths.

The most frequent scenario for this error is when PyInstaller uses relative paths to identify dependencies. Inside the Docker container, these paths may be interpreted with respect to different root directories than during the bundling process, causing imported modules to become inaccessible. For example, if a dependency is bundled relative to the application directory, such as `my_app/lib/my_module.py`, when this app runs in Docker, the relative path from which PyInstaller expects the application to start may not match how the directory structure within the Docker image is set up. Thus, if PyInstaller packaged the path `/path/to/my_app/lib/my_module.py`, but within Docker the application is located at `/app` the path is no longer valid and will result in a module error, even if the module does exist within the container.

To illustrate, consider a simplified Python application using an external library:

```python
# my_app.py
import requests

def fetch_data(url):
    response = requests.get(url)
    response.raise_for_status()
    return response.json()

if __name__ == "__main__":
    data = fetch_data("https://api.example.com/data")
    print(data)
```

Now, we can use PyInstaller to bundle it: `pyinstaller --onefile my_app.py`. Without any adjustments, the resulting `my_app` executable might run without error when executed from the system where it was generated. However, if we place the compiled `my_app` into a Docker image without careful consideration, a `ModuleNotFoundError` might occur related to the 'requests' module.

Here’s a Dockerfile demonstrating the potential issue:

```dockerfile
# Dockerfile
FROM python:3.10-slim-buster

WORKDIR /app

COPY my_app .

CMD ["./my_app"]
```
This seemingly simple Dockerfile copies the `my_app` executable into the container’s `/app` directory and attempts to execute it. However, because the `requests` module path was not correctly resolved within the context of the Docker container, the compiled application will often fail to run with a `ModuleNotFoundError`.

Another scenario for encountering this error stems from dynamically loaded libraries (shared objects, `.so` files on Linux or `.dll` on Windows). These are commonly associated with modules using C extensions. PyInstaller attempts to include these in the bundle, but the system libraries expected by the application might not exist in the container image. This is most often visible when packages like `numpy`, or scientific computing packages are used that rely on system level libraries that are not usually bundled with PyInstaller.

For instance:

```python
# numpy_app.py
import numpy as np

def create_matrix(rows, cols):
    return np.random.rand(rows, cols)

if __name__ == "__main__":
    matrix = create_matrix(5, 5)
    print(matrix)
```

If we bundle `numpy_app.py` using PyInstaller and then deploy this to a minimalist container, we may get an error related to missing shared libraries that `numpy` or other dependencies may require.

Here's an example Dockerfile highlighting the issue:

```dockerfile
# Dockerfile
FROM python:3.10-slim-buster

WORKDIR /app

COPY numpy_app .

CMD ["./numpy_app"]
```
Even if the compiled `numpy_app` executable is deployed successfully to the Docker image, it will not run if the libraries needed for `numpy` are not available, resulting in `ModuleNotFoundError` related to missing shared objects, although the module `numpy` itself may have been bundled.

A third scenario is often encountered with implicit or relative paths that are present in your system, but not present in the Docker image. For instance, in a local development environment if the directory structure is assumed by a specific package, that assumption will not be true in the Docker image. For example:

```python
# my_complex_app.py

import sys
import os

my_library_path = os.path.join(os.path.dirname(__file__), "mylib")
sys.path.append(my_library_path)

from my_custom_module import my_function

if __name__ == "__main__":
    my_function()
```
Here, the python script appends a subdirectory to the path to be searched for modules.

In this example, within the directory `mylib`, there is a file `my_custom_module.py` with a function defined as:

```python
# mylib/my_custom_module.py
def my_function():
    print("This is my custom function")
```

If we try to bundle the `my_complex_app.py` and its associated directory and files using PyInstaller we might get issues when running this within a Docker container.

Here is a Dockerfile example of such:

```dockerfile
# Dockerfile
FROM python:3.10-slim-buster

WORKDIR /app

COPY my_complex_app.py .
COPY mylib ./mylib

CMD ["./my_complex_app"]
```

In this case the code is copied to the directory `/app` and then the `mylib` folder is copied to `/app/mylib`. Even though all the files exist inside the Docker container, the path resolution will not work once bundled into an executable using PyInstaller. The path that the PyInstaller package will be searching for will be from the context of where it was generated, and not from the context of the Docker container.

To address these issues, several strategies are helpful. First, explicitly set the path in PyInstaller using the `--paths` option, ensuring that PyInstaller includes necessary directories beyond the default scope. When using `numpy` and other packages relying on system libraries, ensure those libraries are present in your Docker image. This can be done by including the appropriate base packages in the Dockerfile, or using a Docker image that already provides such libraries. Finally, avoid using relative paths within the application itself and when using PyInstaller's `--paths` directive, make sure that the path resolves within the container.

Recommended resources for further understanding include: documentation for PyInstaller’s command-line options, specifically concerning paths, hooks, and the differences between `--onefile` and `--onedir` packaging; Docker’s official documentation focusing on creating Dockerfiles, understanding image layering, and working with file systems; and resources on Python’s module importing system for understanding how Python locates and loads modules. Additionally, several informative blog posts and forum threads from experienced developers discuss strategies on dealing with this specific `ModuleNotFoundError` within containerized deployments using PyInstaller.
