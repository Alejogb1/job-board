---
title: "How do I identify nodes with GPUs?"
date: "2025-01-30"
id: "how-do-i-identify-nodes-with-gpus"
---
Identifying nodes with GPUs within a distributed computing environment often relies on a combination of system inspection and resource management tools. Direct access to a hardware manifest isn't typically available; instead, we query the available resources programmatically. In my experience managing large-scale machine learning infrastructure, this requires understanding the underlying operating system, the cluster management software, and the programming language being used.

The core challenge resides in abstracting the physical hardware details into a usable software interface. The operating system detects the presence of GPUs and exposes them as devices. Within a cluster environment, schedulers like Kubernetes or Slurm then become the central point for resource requests and allocation. Therefore, identifying GPU-equipped nodes generally involves querying these layers.

**Explanation:**

At the most fundamental level, on Linux systems, information about GPUs can be obtained by inspecting the `/sys/` filesystem. Specifically, the directory `/sys/class/drm/` contains information about Direct Rendering Manager devices, which often correspond to graphics cards, including GPUs. Each entry in this directory usually represents a distinct GPU device. However, interpreting this information directly is usually not suitable for production-level resource management due to its low-level nature.

A more practical approach involves using dedicated tools that provide a higher-level abstraction. For NVIDIA GPUs, the `nvidia-smi` command-line utility is invaluable. This tool directly queries the NVIDIA driver and provides detailed information about installed GPUs, including their names, memory usage, and driver versions. The command's output, however, is text-based and requires parsing to be used programmatically.

Within containerized environments, such as those orchestrated by Kubernetes, the resource discovery process changes. Kubernetes relies on a node's label and annotation system. Specifically, nodes with GPUs are commonly labeled with identifiers such as `nvidia.com/gpu.present=true`. Pods requiring GPUs then use these labels in their resource requests. Kubernetes itself interrogates the underlying system for the available hardware and then makes that information available through the node labels. Thus, to find GPU nodes, one needs to filter by these labels.

Cluster schedulers like Slurm, which are often employed in high-performance computing (HPC) environments, maintain a central database of resources. Queries to the Slurm scheduler through the `scontrol` command line tool or the Slurm API can identify nodes possessing GPUs. Slurm nodes are configured with features indicating the presence and type of accelerators.

The strategy, therefore, isn't to directly inspect the physical node's hardware in every scenario. Rather, it is to query the relevant abstraction layer—the OS, the cluster manager, or specific libraries—and parse the results to determine which nodes contain GPUs.

**Code Examples:**

Here are three examples demonstrating how GPU nodes might be identified using common scenarios:

1. **Python using `nvidia-smi` (Direct System Query):** This example directly executes `nvidia-smi` and parses the output to identify the number of GPUs on a local system. This works for nodes where direct command execution is possible.

```python
import subprocess
import re

def count_gpus():
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=gpu_name', '--format=csv,noheader'], capture_output=True, text=True, check=True)
        output = result.stdout.strip()
        gpu_names = output.split('\n')
        num_gpus = len(gpu_names)
        return num_gpus, gpu_names

    except (subprocess.CalledProcessError, FileNotFoundError):
         return 0, []

num_gpus, gpu_names = count_gpus()
print(f"Number of GPUs: {num_gpus}")
if num_gpus > 0:
    print(f"GPU Names: {gpu_names}")

# Commentary: This code leverages `subprocess` to interact with `nvidia-smi`. The `--query-gpu=gpu_name` specifies to retrieve the name of each GPU, and `--format=csv,noheader` ensures a machine-readable output. We count the lines in the standard output and return the count and names as a result. If `nvidia-smi` is not found or an error occurs, it returns 0 and an empty list.
```
2.  **Python using Kubernetes Client Library (Kubernetes Cluster):** This example uses the Kubernetes Python client library to list nodes with the `nvidia.com/gpu.present` label set to `true`. This is commonly used in Kubernetes environments.

```python
from kubernetes import client, config

def get_gpu_nodes():
    config.load_kube_config()
    v1 = client.CoreV1Api()
    gpu_nodes = []

    nodes = v1.list_node().items
    for node in nodes:
       if node.metadata.labels and node.metadata.labels.get("nvidia.com/gpu.present") == "true":
            gpu_nodes.append(node.metadata.name)
    return gpu_nodes

gpu_node_names = get_gpu_nodes()
print(f"GPU Nodes: {gpu_node_names}")

# Commentary: This code first loads the kubeconfig, establishing the connection to the Kubernetes cluster. It then retrieves all nodes. For each node, it checks if the 'nvidia.com/gpu.present' label exists and is set to 'true'. If so, it adds the node name to the list of GPU nodes. The `metadata.labels` check helps prevent exceptions in the case of a missing label. This is designed for code that runs within the cluster or that has proper kubeconfig setup.
```

3. **Python using Slurm API (Slurm Cluster):**  This example uses the `pyslurm` library to query Slurm nodes for their features.  Nodes with GPU accelerators are identified based on specified feature names (e.g., "gpu" or "nvidia"). This is typical for HPC clusters using Slurm.

```python
from pyslurm import node

def get_slurm_gpu_nodes(gpu_feature_names = ["gpu", "nvidia"]):
  all_nodes = node()
  gpu_nodes = []

  for node_name, node_info in all_nodes.items():
      if node_info.get('features'):
        for feature in gpu_feature_names:
            if feature in node_info.get('features'):
               gpu_nodes.append(node_name)
               break
  return gpu_nodes

gpu_node_names = get_slurm_gpu_nodes()
print(f"GPU Nodes: {gpu_node_names}")

# Commentary: This code uses the pyslurm library to query Slurm for node information. The function iterates over the nodes returned by `node()`. If a node's "features" contain one of the `gpu_feature_names` (e.g., "gpu"), it appends the node name to the `gpu_nodes` list. The loop breaks once a matching feature name is found, ensuring nodes are not added multiple times. The `get` method prevents errors in case a node lacks the `features` key. This requires the `pyslurm` library to be installed and is only valid for Slurm managed environments.
```

**Resource Recommendations:**

For further exploration of this topic, consider these resources:

*   **NVIDIA Documentation:** The official documentation for the `nvidia-smi` utility offers comprehensive details on its usage and output format. These resources frequently have sections specifically devoted to querying GPUs programmatically.

*   **Kubernetes Documentation:** The Kubernetes documentation provides information on node labels, resource management, and device plugins.  Specific sections detail how to use labels for node selection and how the scheduler interacts with hardware resources.

*   **Slurm Documentation:** The Slurm documentation describes how to use the `scontrol` command and the API for querying cluster resources. The documentation also covers the Slurm features framework, essential for identifying specialized hardware like GPUs.

*   **Programming Language Specific Libraries:** Libraries like `kubernetes` for Python, or similar tools in other programming languages offer convenient ways to access cluster information and simplify the interactions. Research specific libraries relevant to your environment and language.

*   **Operating System Documentation:** Consult your operating system documentation for information on the `/sys` filesystem and how it exposes hardware details. Although not usually used for direct programmatic access, it provides a deeper understanding of hardware representation at the OS level.

Using a combination of these resources, I have consistently been able to implement robust, automated mechanisms for identifying GPU-equipped nodes in a variety of computing environments. The examples above are a starting point; the complexity and specific method will necessarily vary based on your specific context.
