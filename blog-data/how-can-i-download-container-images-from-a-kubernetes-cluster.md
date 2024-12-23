---
title: "How can I download container images from a Kubernetes cluster?"
date: "2024-12-23"
id: "how-can-i-download-container-images-from-a-kubernetes-cluster"
---

Right, let’s talk about pulling container images from a Kubernetes cluster—not a trivial task, and something I've tackled more than a few times in my career. It's not as straightforward as you might first think, mainly because kubernetes doesn't *store* images, it schedules pods that *use* images pulled from a registry. So, extracting them isn't about retrieving them from the cluster itself, but rather identifying which images are being used and then pulling those from their origin.

The core issue here is this: a kubernetes cluster manages containers, not their underlying images. When a pod is created, the kubelet on the node pulls the necessary image from a container registry. This registry, often something like docker hub, gcr, or a private registry, is where the image is actually stored. Kubernetes just references them via image name and tags. Therefore, ‘downloading’ an image from a cluster really means identifying the deployed images and then retrieving them directly from the relevant registry. It's a subtle but crucial distinction.

My first encounter with this problem was during a rather large migration project a few years ago. We were moving a legacy application to kubernetes, and, for auditing purposes, had to pull copies of all images that were running in production. Simple, I thought initially – boy, was I wrong. That’s when I realized there’s no single command, no magic ‘download all’ button on a kubernetes cluster. You have to be a bit more methodical.

The approach involves these core steps: first, you need to query kubernetes to find which images are currently in use. Second, you need to use a tool like `docker pull` (or a suitable equivalent) to fetch those images from the respective registry. Third, you might need to deal with authentication if the images are stored in private registries. Let's illustrate these steps with some practical examples using `kubectl` and then using a Python script.

**Example 1: Using `kubectl` to Identify Images**

The easiest way to get the currently running images is using `kubectl`. This command will fetch all pods across all namespaces and extract the image information from each pod specification:

```bash
kubectl get pods --all-namespaces -o jsonpath='{range .items[*].spec.containers[*]}{.image}{"\n"}{end}'
```

This single line performs a few tricks. `kubectl get pods --all-namespaces` gets all the pod definitions across all namespaces. The `-o jsonpath='...` part processes the output of that command. This specific jsonpath query, `{range .items[*].spec.containers[*]}{.image}{"\n"}{end}`, iterates over all the items (which are pods), then for each pod it iterates over its containers, and then it extracts the `.image` property, printing it, followed by a newline.

The output is a list of the image names currently used in your cluster. However, this might include duplicates, so you might want to pipe that output to `sort | uniq` to clean it up. I often find myself using a variation of this, often filtering by specific namespaces or using more complicated `jsonpath` queries when the pod specs get complex – those can get nested pretty quickly when init containers or sidecars are included.

**Example 2: Using a Script for Automated Image Extraction**

Sometimes you need more than just a list – you might need to script the whole process to pull all images automatically, especially in large environments or during complex migrations. Here’s an example of how to do that using Python, relying on the Kubernetes Python client library:

```python
from kubernetes import client, config
import subprocess

def get_images_from_cluster():
    config.load_kube_config()
    v1 = client.CoreV1Api()
    images = set()
    pods = v1.list_pod_for_all_namespaces(watch=False)
    for pod in pods.items:
        for container in pod.spec.containers:
            images.add(container.image)
    return list(images)

def pull_images(images):
    for image in images:
      try:
        print(f"Pulling image: {image}")
        subprocess.run(['docker', 'pull', image], check=True, capture_output=True)
        print(f"Successfully pulled: {image}")
      except subprocess.CalledProcessError as e:
          print(f"Failed to pull {image}: {e.stderr.decode()}")


if __name__ == "__main__":
    images = get_images_from_cluster()
    print("Found the following images:")
    for image in images:
      print(f"- {image}")
    pull_images(images)

```

In this snippet, `kubernetes.config.load_kube_config()` loads the kubernetes configuration (usually from `~/.kube/config`). The `kubernetes.client.CoreV1Api()` object is used to talk to the Kubernetes API. The `v1.list_pod_for_all_namespaces()` method retrieves all pods and its subsequent loops extract unique image names. Then, the `pull_images` function uses the `subprocess` module to execute `docker pull` for each image. The `check=True` makes sure that an exception is thrown if the pull operation fails.

This is a basic example, but it can be easily extended to handle private registries, perhaps by incorporating the docker CLI's authentication helpers or by providing a pull secret. You’d need to handle potential errors, such as rate limiting from registries, or images that might have been deleted. Also, notice the use of a `set` to store the images for automatic de-duplication. These little details make a big difference in real world usage.

**Example 3: Handling Private Registries**

Now, let's address the trickier case of private registries, which is extremely common. Let's assume our images are in a private registry that requires credentials. We can modify the previous script to use a Docker configuration:

```python
import subprocess
from kubernetes import client, config
import json
import os

def get_images_from_cluster():
    config.load_kube_config()
    v1 = client.CoreV1Api()
    images = set()
    pods = v1.list_pod_for_all_namespaces(watch=False)
    for pod in pods.items:
        for container in pod.spec.containers:
            images.add(container.image)
    return list(images)

def configure_docker_auth(registry_url, username, password):
    config_dir = os.path.expanduser("~/.docker")
    os.makedirs(config_dir, exist_ok=True)
    config_file = os.path.join(config_dir, "config.json")
    auth_string = f"{username}:{password}".encode('ascii').hex()
    config_data = {
        "auths": {
            registry_url: {
                "auth": auth_string
            }
        }
    }

    with open(config_file, "w") as f:
        json.dump(config_data, f, indent=2)


def pull_images(images, registry_url, username, password):
    configure_docker_auth(registry_url, username, password)

    for image in images:
      try:
        print(f"Pulling image: {image}")
        subprocess.run(['docker', 'pull', image], check=True, capture_output=True)
        print(f"Successfully pulled: {image}")
      except subprocess.CalledProcessError as e:
          print(f"Failed to pull {image}: {e.stderr.decode()}")

if __name__ == "__main__":
    # Replace with your actual registry details
    registry_url = "your-registry.example.com"
    username = "your_username"
    password = "your_password"

    images = get_images_from_cluster()
    print("Found the following images:")
    for image in images:
      print(f"- {image}")
    pull_images(images, registry_url, username, password)
```

This expanded version now has a `configure_docker_auth` function. It programmatically builds a docker config file with the credentials to login to the private registry. Please be aware that hardcoding credentials like this is generally not advised, and in production code you would ideally retrieve these from secure secret stores (e.g., Hashicorp Vault, AWS Secrets Manager), or by using the docker login command, however, for simplicity I kept it inline for demonstration. The `pull_images` function now calls this authentication step before pulling any images.

In a production environment, you should avoid creating the docker config on the host like in this script as it is insecure and can compromise the host environment. Instead, one might use a docker login command in a separate step or retrieve an authentication token using the docker API and not store it on disk at all.

For further in-depth study, I strongly recommend reviewing "Kubernetes in Action" by Marko Lukša, a deep dive into kubernetes concepts, and for deeper understanding of the Kubernetes API client libraries, the official Kubernetes documentation (kubernetes.io) is your best resource. Another good source is "Programming Kubernetes" by Michael Hausenblas and Stefan Schimanski, which goes much deeper into writing code that interacts with Kubernetes, similar to what I've shown in the Python examples above. Understanding these concepts allows you to move beyond simple kubectl commands and build powerful, automated solutions for managing Kubernetes resources.

Hopefully this gives you a solid understanding of how to tackle this task, and shows how seemingly simple questions can involve complex underlying systems.
