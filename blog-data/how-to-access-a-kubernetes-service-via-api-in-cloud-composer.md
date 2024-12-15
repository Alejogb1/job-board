---
title: "How to access a Kubernetes Service via API in Cloud Composer?"
date: "2024-12-15"
id: "how-to-access-a-kubernetes-service-via-api-in-cloud-composer"
---

alright, so accessing a kubernetes service from within a cloud composer environment, i've been there, done that, got the t-shirt, and probably even debugged the dag late into the night. it's not exactly straightforward, but definitely doable once you grasp a few core concepts.

the crux of the matter is that composer, while running in google kubernetes engine (gke), doesn't automatically expose the kubernetes cluster’s apiserver to your dags. it's all about security and isolation, which is honestly a good thing but it does make our lives a little more complicated.

the first hurdle is authentication. your dag needs to prove it’s allowed to talk to the kubernetes api. the most common way to do this, and the way i've used countless times, involves using a kubernetes service account. the cool thing is, google cloud manages service account keys and provides a really nice way to get them. instead of manually generating and handling keys, we use a technique called workload identity federation which is far more secure.

when you create a composer environment, it automatically creates a google service account associated with the gke cluster where your workflows run. then, your dag, running as pods in the cluster, can assume the permissions of that gke service account. this means the pods have the same gke permissions. this gke service account will not have direct access to kubernetes cluster. to achieve this we need to create a kubernetes service account, then bind it to google service account. it’s a bit of dance but makes sure access is granted in a secure manner.

here's the basic process: first you create a kubernetes service account. second you create a role and rolebinding that links that service account to the kubernetes resources you intend to access. finally, you create the workload identity configuration. you will need to create a role with the necessary permissions to access the kubernetes resources. after that it is necessary to bind the kubernetes service account with the google service account.

let's look at a python example. we'll use the official kubernetes python client, which is what i always reach for in this situation. i recommend having a look at the official documentation, it has really good examples.
```python
from kubernetes import client, config
from google.auth import credentials
import google.auth
import os
from google.oauth2 import service_account

def get_k8s_client():
    """
    creates and returns a kubernetes client object
    """

    try:

        _, project_id = google.auth.default()
        
        # load kubernetes configuration, this will leverage workload identity
        config.load_incluster_config()
    
        k8s_client = client.CoreV1Api()
        return k8s_client
    except Exception as e:
        print(f"error creating k8s client: {e}")
        return None

def list_pods(namespace):
    k8s_client = get_k8s_client()

    if not k8s_client:
        print("failed to create kubernetes client")
        return []

    try:
        pod_list = k8s_client.list_namespaced_pod(namespace)
        return pod_list.items
    except Exception as e:
        print(f"error listing pods: {e}")
        return []

if __name__ == "__main__":
    namespace = "default"
    pods = list_pods(namespace)
    for pod in pods:
        print(f"pod name:{pod.metadata.name}, pod phase: {pod.status.phase}")
```

this code assumes that your pod already has workload identity configured correctly. this is the part that most often trips people up. in the past, when i had some issues with workload identity setup, i spent hours going through the google cloud documentation on workload identity federation, it was a learning experience. it turned out the service account binding was missing a necessary annotation. so double checking that part is key.

the `config.load_incluster_config()` function is important, it automatically detects it is running inside a kubernetes cluster and loads the correct configuration using the service account credentials. this works by picking up the service account token volume mounted by the kubernetes pod. when you use `google.auth.default()` it also figures out from the environment that it is running on gcp and it will provide the google project id.

another thing i've frequently needed to do is to access service exposed via kubernetes services. suppose you have an app behind a service called `my-app-service` exposed on port 8080 and it is on the `my-namespace` namespace. for that we can use the python library `requests`.
```python
import requests
from kubernetes import client, config
import google.auth


def get_service_ip(service_name, namespace):
    """
    retrieves the cluster ip of a kubernetes service
    """
    try:
        config.load_incluster_config()
        k8s_client = client.CoreV1Api()
        service = k8s_client.read_namespaced_service(service_name, namespace)
        return service.spec.cluster_ip
    except Exception as e:
        print(f"error getting service ip: {e}")
        return None

def call_my_app(service_name, namespace):
    """
    calls a service using cluster ip
    """
    service_ip = get_service_ip(service_name,namespace)
    if not service_ip:
        print("failed to get service ip")
        return
    
    try:
        url = f"http://{service_ip}:8080/health"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"request error: {e}")
        return None

if __name__ == "__main__":
    namespace = "my-namespace"
    service_name = "my-app-service"
    response_data = call_my_app(service_name, namespace)
    if response_data:
        print("successful request")
        print(response_data)
```

in the example above, we first retrieve the service cluster ip, then use it to call the service. this will work as long as you are on the same cluster, since we are talking to the internal service ip, which is not available from outside the cluster.

there are situations where accessing services by their internal cluster ip is not enough. for instance, when the service exposed is a load balancer, you might need to use the external ip. but this goes outside the scope of the question. usually for accessing external services, it is recommended to go via a dedicated reverse proxy, so your application does not need to handle the authentication/authorization.

sometimes you might need to interact with custom resource definitions, crds. the process for this is similar to the previous ones. the first step is making sure that the service account you are using has permissions to read or write to the crd. after that you can interact with the crds, you just need to use the `CustomObjectsApi` client and provide the group, version and plural name of the crd, which are normally defined on the yaml definition of crds.
```python
from kubernetes import client, config

def get_custom_objects():
    """
    gets custom objects from a group, version and plural name
    """
    try:
        config.load_incluster_config()
        k8s_client = client.CustomObjectsApi()
        group = 'mygroup.example.com'
        version = 'v1'
        plural = 'mycustomobjects'
        objects = k8s_client.list_cluster_custom_object(group, version, plural)
        return objects
    except Exception as e:
        print(f"error getting custom object list: {e}")
        return None

if __name__ == "__main__":
    objects = get_custom_objects()
    if objects:
        print("custom objects:")
        for obj in objects['items']:
             print(f"name: {obj['metadata']['name']}")
```
the important thing here is to make sure that the service account that the pod is running under has sufficient rbac access. otherwise you will get an `unauthorized` exception.

i cannot say this enough. double and triple check that the kubernetes service account has the rbac permissions to do whatever operation you need on the kubernetes api. it is common for people to make a small mistake on the role or rolebindings, which makes the process look broken. another potential issue is network connectivity. make sure the service you are trying to access is actually available and exposed on the expected namespace. and also double check that you are using the correct namespaces. i once spent a good hour debugging a problem that turned out i was accessing the wrong namespace… i felt like i needed a vacation after that one.

if you're getting stuck, i would seriously recommend checking out kubernetes documentation on rbac and authentication. also, google cloud's official docs on workload identity federation are a must. there is also a great book about kubernetes, "kubernetes in action" if you have the time to dig deeper on the architecture of the platform.

in short, accessing a kubernetes service via api in cloud composer is all about correctly configuring workload identity, rbac and using the kubernetes python client. it can seem complex at first, but once you've got the basics down, it becomes quite manageable. it’s really more of a configuration and permission issue than some deep code wizardry, which is great, that makes it easier to solve with enough practice. i once heard someone say that debugging kubernetes is just like trying to read tea leaves, i’m not sure about that, but sometimes it feels like it.
