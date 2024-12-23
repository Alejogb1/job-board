---
title: "Why am I getting `Azure ML ExecutableNotFound` when running dot?"
date: "2024-12-23"
id: "why-am-i-getting-azure-ml-executablenotfound-when-running-dot"
---

Okay, let's tackle this Azure ML `ExecutableNotFound` situation when trying to use `dot`. It’s a frustrating error, I know, and I've certainly spent my share of time debugging similar issues in past projects. It usually boils down to a few core reasons, and it's rarely a problem with azure ml itself, but rather with how the environment is set up.

First, let’s clarify what `dot` refers to here. It's part of Graphviz, an open-source graph visualization software. Azure ML environments, being sandboxed and curated, don't inherently come pre-loaded with tools like Graphviz. When your pipeline tries to execute a python script that uses the `graphviz` library (which, under the hood, needs the `dot` executable), you can get this error if `dot` isn't present or accessible in the execution environment. This was exactly what hit me about two years ago when deploying a complex model that required visualizing decision trees, for example.

The underlying cause isn’t simply about graphviz not being installed, but specifically about the `dot` executable being absent or not within the execution PATH. Libraries like the python `graphviz` wrapper don't inherently include the `dot` executable; they rely on it being somewhere the operating system can find it via the environment variables. In the case of cloud-based execution environments like Azure ML, this is usually not the default configuration.

The solution, then, is centered on ensuring that `dot` is installed and that its directory is added to the `PATH` for the execution environment used by your Azure ML pipeline. There are several approaches, and I will show you three common patterns I've employed over the years.

**Method 1: Direct Install in the Environment Definition (for compute clusters)**

This approach is straightforward and involves directly installing `graphviz` and ensuring its binary (`dot`) is accessible in the environment used by your compute cluster. I prefer this for compute clusters, because the environment stays persistent with the compute.

```yaml
# environment.yml for Azure ML environment
name: graphviz-env
channels:
  - conda-forge
dependencies:
  - python=3.8
  - pip
  - graphviz
  - pip:
    - azureml-defaults
    - <your-other-dependencies> # Add your model's other dependency
```

```python
# example using the environment.yml
from azureml.core import Environment
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.core import Workspace
from azureml.core.conda_dependencies import CondaDependencies

ws = Workspace.from_config()

compute_name = "my-compute-cluster"

# If cluster doesn't exist, create it
try:
    compute_target = ComputeTarget(workspace=ws, name=compute_name)
    print("Found existing compute target.")
except ComputeTargetException:
    print("Creating a new compute target...")
    compute_config = AmlCompute.provisioning_configuration(vm_size="STANDARD_DS3_V2",
                                                        min_nodes=0,
                                                        max_nodes=4)

    compute_target = ComputeTarget.create(ws, compute_name, compute_config)

    compute_target.wait_for_completion(show_output=True)

env_name = "graphviz-env"
env_file_path = "environment.yml"
myenv = Environment.from_conda_specification(name = env_name, file_path = env_file_path)

print(f"Environment '{env_name}' successfully loaded from '{env_file_path}'.")

myenv.register(workspace=ws)
print(f"Environment '{env_name}' successfully registered to the workspace.")
```

Here, the `environment.yml` specifies `graphviz` as a direct dependency.  This ensures that when your azure ml job runs, the `dot` executable will be installed and be in the path of the process.

**Method 2: Post-Install Script (for custom containers or greater control)**

In cases where I needed fine-grained control over the environment setup, like when using custom Docker images or needing specific versions of Graphviz, I've employed a post-install script.

```dockerfile
# Dockerfile (example)
FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest
COPY post_install.sh /tmp/
RUN chmod +x /tmp/post_install.sh && /tmp/post_install.sh
```

```shell
#!/bin/bash
# post_install.sh
apt-get update && apt-get install -y graphviz
```

```python
# example using a dockerfile environment
from azureml.core import Environment
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.core import Workspace
from azureml.core.conda_dependencies import CondaDependencies

ws = Workspace.from_config()

compute_name = "my-compute-cluster"

# If cluster doesn't exist, create it
try:
    compute_target = ComputeTarget(workspace=ws, name=compute_name)
    print("Found existing compute target.")
except ComputeTargetException:
    print("Creating a new compute target...")
    compute_config = AmlCompute.provisioning_configuration(vm_size="STANDARD_DS3_V2",
                                                        min_nodes=0,
                                                        max_nodes=4)

    compute_target = ComputeTarget.create(ws, compute_name, compute_config)

    compute_target.wait_for_completion(show_output=True)

env_name = "docker-graphviz-env"
dockerfile_path = "Dockerfile"
myenv = Environment.from_dockerfile(name = env_name, dockerfile= dockerfile_path)

print(f"Environment '{env_name}' successfully loaded from '{dockerfile_path}'.")
myenv.register(workspace=ws)
print(f"Environment '{env_name}' successfully registered to the workspace.")
```

Here, the Dockerfile includes a `post_install.sh` script that runs `apt-get install graphviz`. This is more flexible than the conda approach, but you are now responsible for managing the base OS, packages, etc. The flexibility here is that you can customize anything you want.

**Method 3:  Python-level Path Handling (least ideal)**

While not preferred, you could technically attempt to handle path issues directly in your python script using something like:

```python
# your_script.py
import os
import graphviz
import shutil

def find_executable(executable):
    if shutil.which(executable):
        return shutil.which(executable)

    #attempt to find dot using other common locations
    potential_paths = [
        "/usr/bin", "/usr/local/bin", "/opt/bin"
    ]
    for path in potential_paths:
        full_path = os.path.join(path, executable)
        if os.path.exists(full_path) and os.access(full_path, os.X_OK):
           return full_path

    return None

dot_path = find_executable("dot")


if dot_path:
    os.environ["PATH"] = os.pathsep.join([os.path.dirname(dot_path), os.environ.get("PATH","")])
    # your graphviz code
    try:
        dot = graphviz.Digraph(comment='The Round Table', format='png')
        dot.node('A', 'King Arthur')
        dot.node('B', 'Sir Bedevere the Wise')
        dot.node('L', 'Sir Lancelot the Brave')

        dot.edge('A', 'B')
        dot.edge('A', 'L')

        dot.render("round_table", view=False)
        print("Graph rendered successfully!")

    except Exception as e:
        print(f"Exception rendering graph: {e}")


else:
    print("Warning: 'dot' executable not found in path. Graphviz will not function correctly.")
```

This approach tries to locate the `dot` executable and add its directory to the `PATH`, and is generally not recommended since it's fragile. It’s prone to failing in various environments and makes the codebase less clean. The above will at least attempt to render the graph while making you aware of the problem.

**Why these solutions work**

The key to resolving this issue is acknowledging that libraries rely on system utilities, often external executables, like `dot`. Azure ML pipelines run in isolated containers, meaning these executables must be explicitly included in the environment. Methods 1 and 2 manage this at the environment level, which is cleaner and more consistent. Method 3 should be a last resort.

**Further reading and resources**

For a deeper understanding of this and related topics, I highly recommend the following:

*   **"Operating System Concepts" by Abraham Silberschatz, Peter Baer Galvin, and Greg Gagne:** This book covers fundamental concepts related to operating systems, process execution, and environment variables. Understanding the mechanics behind these will aid in grasping the problem.
*  **"Effective Python: 90 Specific Ways to Write Better Python" by Brett Slatkin:** While not directly about environment configurations, the principles of clean, maintainable code and the avoidance of fragile solutions are crucial here. The book offers excellent guidance on good python practices.
*   **Azure Machine Learning Documentation:** The official documentation is always the most authoritative source for specific configuration details and environment management practices related to Azure ML. Pay close attention to environment creation, custom images, and dependency management.

In summary, the `ExecutableNotFound` for `dot` in Azure ML pipelines is usually caused by an improperly configured environment. Using the appropriate approach, such as those outlined in examples 1 and 2, to include the necessary components and paths, will solve this common issue, leading to a more reliable Azure ML process. Remember to choose the method that best fits your overall project scope and complexity. I hope this explanation and these practical examples prove helpful.
