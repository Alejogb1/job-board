---
title: "What REST API POST commands are supported by Azure Container Instances?"
date: "2025-01-30"
id: "what-rest-api-post-commands-are-supported-by"
---
Azure Container Instances (ACI) primarily uses a specific subset of the REST API for its POST commands, focusing on operations related to container group lifecycle management. Specifically, ACI doesn't directly support fine-grained POST commands for actions within a running container instance like injecting commands or sending signals. Instead, interactions are geared towards creating, updating, and deleting the container group as a whole, reflecting its nature as a serverless container execution service. Over the past few years working with various orchestration technologies, I've learned that ACI’s API surface emphasizes declarative configuration over imperative actions on running instances, differing from systems like Kubernetes.

The core REST API endpoint for interacting with container groups is located under the Azure Resource Manager (ARM) API. For POST requests, we're concerned with a few key operations, most importantly, the creation of a container group. Let's break down the typical use cases:

1.  **Creating a Container Group (Deployment):** This is by far the most common POST operation. The payload specifies the full definition of the container group, including container images, resource requirements (CPU, memory), environment variables, restart policies, and networking configurations. The HTTP method is `PUT` (not POST technically, but the effect is deployment), and the endpoint conforms to the pattern `/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.ContainerInstance/containerGroups/{containerGroupName}?api-version={apiVersion}`. The body of this request will be a JSON document detailing all aspects of the desired container group. The successful response results in a `201 Created` HTTP status code, along with the created resource information.

2. **Updating a Container Group:** Once a container group has been deployed, some aspects can be updated, although the changes are limited. Typically, the update endpoint is the same as the creation endpoint: `/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.ContainerInstance/containerGroups/{containerGroupName}?api-version={apiVersion}`. When updating, the full definition of the container group must again be provided, even if certain sections aren't changing, with the only exception being the `id`, `name`, and `type` of the container group. Modifying, for example, the container image or the allocated resources will result in a rolling update of the container group, potentially causing a temporary downtime depending on your application’s resilience. Changes that will not cause an immediate restart are limited to container group properties such as environment variables in the container's `environmentVariables` array or container group tags. Updating container group properties that would cause a restart, such as the container image name, will lead to a new deployment of the container group, and the old one will be terminated after the new one is successfully deployed.

3.  **Restarting a Container Group:** A specific `POST` operation allows to restart containers of a container group. The endpoint uses the same URI for creation/update but with a different final path segment `/restart`. Therefore, the endpoint will be `/subscriptions/{subscriptionId}/resourceGroups/{resourceGroupName}/providers/Microsoft.ContainerInstance/containerGroups/{containerGroupName}/restart?api-version={apiVersion}`. The request body does not require content. The successful request returns a `204 No Content` status code. Note that this restarts *all* containers within the container group and, in that sense, can be a disruptive operation if not managed properly.

Let’s move onto a few code examples using `curl`. These represent how such API requests would be formed, keeping in mind that in practice you'd likely use an SDK.

**Example 1: Creating a Container Group**

```bash
curl -X PUT \
  -H "Authorization: Bearer <YOUR_ACCESS_TOKEN>" \
  -H "Content-Type: application/json" \
  -d '{
    "location": "eastus",
    "properties": {
        "containers": [
            {
                "name": "mycontainer",
                "properties": {
                    "image": "mcr.microsoft.com/azuredocs/aci-helloworld",
                    "resources": {
                        "requests": {
                            "cpu": 1,
                            "memoryInGB": 1.5
                        }
                    },
                    "ports": [
                      {
                        "port": 80,
                        "protocol": "TCP"
                      }
                    ]
                }
            }
        ],
        "osType": "Linux",
        "ipAddress": {
            "type": "Public",
            "ports": [
                {
                    "protocol": "TCP",
                    "port": 80
                }
             ]
         }
      }
    }' \
    "https://management.azure.com/subscriptions/<YOUR_SUBSCRIPTION_ID>/resourceGroups/myResourceGroup/providers/Microsoft.ContainerInstance/containerGroups/mycontainergroup?api-version=2023-05-01"
```

**Commentary:** This `curl` command demonstrates creating a container group named "mycontainergroup" within the "myResourceGroup" resource group. It specifies a single container based on the `mcr.microsoft.com/azuredocs/aci-helloworld` image, requesting 1 CPU and 1.5GB of memory. Public IP is enabled with port 80 exposed. Note that `<YOUR_ACCESS_TOKEN>` should be replaced with a valid Azure Active Directory access token, and `<YOUR_SUBSCRIPTION_ID>` with your subscription ID.  This full body JSON definition is what drives the container group’s deployment. The `api-version` query parameter is crucial as it specifies which version of the ACI API to use.

**Example 2: Updating a Container Group**

```bash
curl -X PUT \
  -H "Authorization: Bearer <YOUR_ACCESS_TOKEN>" \
  -H "Content-Type: application/json" \
  -d '{
    "location": "eastus",
    "properties": {
      "containers": [
          {
              "name": "mycontainer",
              "properties": {
                "image": "nginx:latest",
                "resources": {
                  "requests": {
                    "cpu": 1,
                    "memoryInGB": 2
                  }
                }
              }
          }
      ],
        "osType": "Linux",
        "ipAddress": {
            "type": "Public",
            "ports": [
                {
                    "protocol": "TCP",
                    "port": 80
                }
             ]
         }
    }
  }' \
    "https://management.azure.com/subscriptions/<YOUR_SUBSCRIPTION_ID>/resourceGroups/myResourceGroup/providers/Microsoft.ContainerInstance/containerGroups/mycontainergroup?api-version=2023-05-01"
```

**Commentary:** This `curl` command shows an update to the container group "mycontainergroup". Crucially, it replaces the existing container image with `nginx:latest` and also updates the memory request to 2 GB. ACI will interpret this as a full redeployment, terminating the existing container and launching a new one with the updated configuration. Even though a single attribute is changed the entire container group definition must be re-submitted, as ACI updates follow the pattern of "replace by full definition."

**Example 3: Restarting a Container Group**

```bash
curl -X POST \
  -H "Authorization: Bearer <YOUR_ACCESS_TOKEN>" \
  -H "Content-Type: application/json" \
    "https://management.azure.com/subscriptions/<YOUR_SUBSCRIPTION_ID>/resourceGroups/myResourceGroup/providers/Microsoft.ContainerInstance/containerGroups/mycontainergroup/restart?api-version=2023-05-01"
```

**Commentary:** This example demonstrates a restart of the container group. As visible in the endpoint definition, the `/restart` path segment specifies to ACI that this should be handled as a container group restart operation. The request body is empty, and after successful execution all the containers within the container group will be stopped and started once more, using their original image and container configuration.

Regarding resource recommendations for further exploration, it is beneficial to review the official Microsoft Azure documentation, specifically the sections covering Azure Container Instances and the REST API for ACI. Additionally, examining the OpenAPI definition for the ACI REST API is crucial for a comprehensive understanding of all operations, request bodies, response structures, and required parameters. The Azure CLI's documentation can also be very useful, as the underlying command calls the same API; the command structure maps quite well to the API structures. Finally, the Azure SDK for your chosen language contains abstractions around these API calls and can serve as good examples of code working with the APIs without having to form raw REST requests.
