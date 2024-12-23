---
title: "How can I get a configfile to access minikube from outside the cluster to create pods, delete pods, etc?"
date: "2024-12-23"
id: "how-can-i-get-a-configfile-to-access-minikube-from-outside-the-cluster-to-create-pods-delete-pods-etc"
---

, let’s tackle this. It's a common scenario, and one I’ve personally encountered a few times, usually when setting up local development environments or CI pipelines. The core challenge is ensuring your `kubectl` client, running outside the minikube VM, has the correct configuration to interact with the cluster's api server. Simply put, we need to provide it with the appropriate authentication and endpoint details.

Now, when you initially install minikube, it configures `kubectl` on your host machine to communicate with the newly created cluster, generating a kubeconfig file for this purpose. However, this file is usually specific to your local user context and may not be immediately accessible or usable from other environments, including those running within different terminal sessions or, crucially, from other applications outside the confines of your shell.

The crucial element here is the `kubeconfig` file. This is a yaml file that contains all the necessary information for `kubectl` to communicate with a kubernetes cluster. Specifically, it contains cluster details such as the api server address, the certificate authority data for verification, and authentication credentials, like user keys or tokens. You typically find the minikube-generated config located in your home directory, usually within `.kube/config`. But the specific location may vary based on OS.

So, how do we access this information and use it reliably from any context outside of minikube itself? There are several ways to accomplish this, and while the exact method depends on your setup, the principles remain the same. I’ve seen different development teams use various approaches.

Here’s the breakdown of the process and some key techniques:

First, it's helpful to understand *why* a simple copy-paste of the kubeconfig might fail. The main reason is that the kubeconfig stores *local* paths to certificates and keys. For example, a certificate might point to `/home/youruser/.minikube/certs/apiserver.crt` inside the minikube vm. When you use this from the host, the path is likely invalid. Thus, what needs to be done is adjusting for these paths or providing the data directly. We don't need to necessarily extract from the minikube vm, it is perfectly sufficient to get the data from our local host environment.

Let's examine some concrete examples, along with associated code.

**Approach 1: Explicit Kubeconfig Path**

This is the most straightforward solution for situations where you want to access minikube from a script or application running on the *same* machine as where minikube is running. It involves explicitly telling `kubectl` where to find the kubeconfig file.

```bash
#!/bin/bash

# Get the path to the kubeconfig file. Assuming default location.
KUBECONFIG="$HOME/.kube/config"

# Set the KUBECONFIG environment variable.
export KUBECONFIG="$KUBECONFIG"

# Example command to create a pod
kubectl create deployment nginx --image=nginx

# Example command to list pods
kubectl get pods

# Example command to delete pod
kubectl delete deployment nginx
```

This snippet uses the `KUBECONFIG` environment variable. `kubectl` automatically looks for this variable and loads the specified config file. Setting it this way is also useful for running commands from within scripts, provided the script is run in the same environment (user context) as the original minikube setup.

**Approach 2: Using a Dedicated Context**

Sometimes, your `kubeconfig` may contain configurations for multiple clusters (e.g., minikube, dev, prod). In such cases, you need to specify the correct context for minikube. You can achieve this using the `--context` flag in `kubectl`. This approach is great for multi-cluster setups.

First, find the name of the minikube context:

```bash
kubectl config get-contexts
```
This command will show you available contexts; usually, one will be named `minikube`.

Now, use the context name in your kubectl commands:

```python
import subprocess

def create_pod(context_name):
    try:
        subprocess.run(["kubectl", "--context", context_name, "create", "deployment", "nginx", "--image=nginx"], check=True)
        print("Nginx pod created successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error creating pod: {e}")

def list_pods(context_name):
    try:
       result = subprocess.run(["kubectl", "--context", context_name, "get", "pods"], capture_output=True, text=True, check=True)
       print("Pods:")
       print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error listing pods: {e}")

def delete_pod(context_name):
    try:
      subprocess.run(["kubectl", "--context", context_name, "delete", "deployment", "nginx"], check=True)
      print("Nginx pod deleted.")
    except subprocess.CalledProcessError as e:
        print(f"Error deleting pod: {e}")

if __name__ == "__main__":
   minikube_context = "minikube"  #Replace with actual context name if needed
   create_pod(minikube_context)
   list_pods(minikube_context)
   delete_pod(minikube_context)

```
Here, I've used Python to execute kubectl commands. The key is passing `--context` followed by the minikube context name with every invocation. This helps to avoid confusion when you're working with multiple clusters or configurations. You can easily adapt this approach to other scripting or application languages.

**Approach 3: Embedding the Config Data**

In more advanced scenarios, where you may not want to rely on the environment or external files, you can use the *in-memory* method. This approach involves directly embedding the crucial parts of the `kubeconfig` file as strings within your application. While this could be a convenient method, it's worth noting security considerations as this method stores tokens or certs directly.

```python
from kubernetes import config, client

def main():
    kube_config = {
        "apiVersion": "v1",
        "clusters": [
          {
             "cluster": {
                "certificate-authority-data": "BASE64_ENCODED_CERTIFICATE_AUTHORITY_DATA",
                "server": "MINIKUBE_API_SERVER_URL"
             },
            "name": "minikube"
         }
        ],
        "contexts": [
            {
                "context": {
                    "cluster": "minikube",
                    "user": "minikube"
                },
                "name": "minikube"
            }
        ],
        "current-context": "minikube",
        "kind": "Config",
        "preferences": {},
        "users": [
            {
                "name": "minikube",
                "user": {
                    "client-certificate-data": "BASE64_ENCODED_CLIENT_CERTIFICATE_DATA",
                    "client-key-data": "BASE64_ENCODED_CLIENT_KEY_DATA"
                }
            }
        ]
    }

    #Replace the data here with the values from your local .kube/config file
    # Do not commit this to git with actual sensitive data!

    configuration = client.Configuration()
    configuration.host = kube_config['clusters'][0]['cluster']['server']
    configuration.ssl_ca_cert = None # Do not use cert file, instead base64 cert
    configuration.verify_ssl = False #Do not verify
    configuration.api_key['authorization'] = "Bearer some_token_key" # If using tokens
    configuration.api_key_prefix['authorization'] = 'Bearer' # If using tokens

    api_client = client.ApiClient(configuration=configuration)
    v1 = client.CoreV1Api(api_client)

    # Example of creating a pod using k8s client api
    pod_manifest = {
        'apiVersion': 'v1',
        'kind': 'Pod',
        'metadata': {
            'name': 'example-pod'
        },
        'spec': {
            'containers': [{
                'name': 'example-container',
                'image': 'nginx'
             }]
          }
       }

    try:
      v1.create_namespaced_pod(namespace='default', body=pod_manifest)
      print("Pod created successfully")

      # Example getting a list of pods
      pod_list = v1.list_namespaced_pod(namespace='default')
      for pod in pod_list.items:
        print(f"Pod Name:{pod.metadata.name}")

      # Example deleting a pod
      v1.delete_namespaced_pod(name="example-pod", namespace='default')
      print("Pod deleted successfully")


    except client.ApiException as e:
      print(f"Error communicating with kubernetes: {e}")


if __name__ == "__main__":
   main()
```

This snippet utilizes the Kubernetes python client library to directly access the api server, using inline `kubeconfig` data. You'd need to extract the certificate and key data from your local kubeconfig and base64 encode them, replacing the placeholder strings. This allows for a completely independent client configuration, useful when you don't want to depend on external files.

For further reading, I highly recommend exploring the official Kubernetes documentation (kubernetes.io). There is comprehensive coverage of kubeconfig details and authentication mechanisms. Also, "Kubernetes in Action" by Marko Luksa provides a detailed exploration of kubernetes concepts, including the underlying configuration principles. Finally, "Programming Kubernetes" by Michael Hausenblas and Stefan Schimanski is great resource for those wanting to understand deeper programmatic aspects.

These approaches, while having different use cases, are useful for getting `kubectl` to communicate with the minikube cluster from outside. Remember, always prioritize security, and if storing certificates inline be extremely cautious. Choose the method that best fits your security policies and setup.
