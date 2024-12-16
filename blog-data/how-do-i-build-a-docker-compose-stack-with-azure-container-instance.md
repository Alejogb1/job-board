---
title: "How do I build a Docker Compose stack with Azure Container Instance?"
date: "2024-12-16"
id: "how-do-i-build-a-docker-compose-stack-with-azure-container-instance"
---

Alright, let's dive into this. I've tackled this particular scenario a fair few times, particularly back in my early days implementing cloud-native solutions, and it’s a surprisingly common challenge. Directly translating a docker compose setup to Azure Container Instances (ACI) isn't a straightforward, one-to-one process. Docker Compose, inherently, manages multiple containers on a single host (or a swarm). ACI, on the other hand, focuses more on individual containers or groups of tightly coupled containers, often with more limited networking capabilities compared to a full docker environment. So, the key here is to adapt your compose logic, not directly copy it.

Essentially, you’ll need to rethink your compose file in terms of ACI's architectural constructs. Think of it less about orchestrating a whole system on one virtual machine and more about deploying and connecting independent container groups.

The primary hurdle often lies in how Docker Compose handles networking and inter-container communication. Compose typically establishes an internal network for containers to communicate using service names. ACI, while offering virtual network integration, doesn't automatically map service names in the same way. Instead, it generally expects containers within the same group to communicate via localhost or, if spread across multiple container groups within the same virtual network, via fully qualified domain names or private IPs.

Furthermore, ACI doesn't natively support the same level of persistent storage and volume management you might be accustomed to in Docker. You need to leverage Azure storage services and mount them as volumes in ACI. This often implies restructuring the data persistence strategy in your application.

Let’s break this down into concrete steps, and I’ll include code snippets to illustrate these points.

**Step 1: Analyzing Your Docker Compose File**

Before we proceed, meticulously examine your existing `docker-compose.yml` file. Pay close attention to these sections:

*   **`services`:** Identify how many distinct services you have. Each of these will likely translate to an ACI container group or, in some instances, a single container within a group.
*   **`networks`:** Note how your services communicate. If relying heavily on custom networks, be prepared to adjust communication using ACI's virtual network capabilities.
*   **`volumes`:** Understand what data needs persistence. You might need to adopt Azure File Share, Azure Blob Storage, or other relevant Azure storage solutions.
*   **`ports`:** Identify the port mappings, particularly external port mappings. Note that ACI exposes ports at the container group level, not per container within a group.
*   **`depends_on`:** If you have inter-service dependencies, ensure you account for this through appropriate application-level retries or orchestration capabilities.

**Step 2: Translating Compose Services to ACI Container Groups**

For most setups, you'll translate each of your significant services into an ACI container group. ACI container groups are the primary deployment unit and essentially group containers that will share the same network and local resources. The important thing is the container group shares the host name - each container inside uses `localhost` to communicate with other containers in the same group.

Here’s a simple example. Suppose your `docker-compose.yml` file has the following structure:

```yaml
version: "3.9"
services:
  web:
    image: my-web-app:latest
    ports:
      - "80:80"
    depends_on:
      - api
  api:
    image: my-api:latest
    ports:
      - "5000:5000"
    environment:
      DATABASE_URL: "db-url"
```

We would translate that into two separate ACI deployment resources (usually via an ARM template or Azure Bicep). I will focus only on the definitions of the container groups here:

*   **ACI Container Group for `web`:**

```json
{
    "apiVersion": "2023-05-01",
    "location": "[resourceGroup().location]",
    "name": "web-containergroup",
    "type": "Microsoft.ContainerInstance/containerGroups",
    "properties": {
        "osType": "Linux",
        "restartPolicy": "Never",
        "containers": [
            {
                "name": "web",
                "properties": {
                    "image": "my-web-app:latest",
                    "resources": {
                        "requests": {
                            "cpu": 1,
                            "memoryInGB": 1
                        }
                    },
                     "ports": [
                       {
                           "port": 80
                         }
                     ]
                }
            }
        ],
      "ipAddress": {
        "type": "Public",
          "ports": [
           {
              "port": 80,
               "protocol": "TCP"
           }
         ]
       }
    }
}
```

*   **ACI Container Group for `api`:**

```json
{
    "apiVersion": "2023-05-01",
    "location": "[resourceGroup().location]",
    "name": "api-containergroup",
    "type": "Microsoft.ContainerInstance/containerGroups",
        "properties": {
            "osType": "Linux",
            "restartPolicy": "Never",
              "containers": [
                {
                   "name": "api",
                    "properties": {
                      "image": "my-api:latest",
                       "resources": {
                           "requests": {
                               "cpu": 1,
                               "memoryInGB": 1
                            }
                        },
                         "ports": [
                          {
                             "port": 5000
                            }
                         ],
                         "environmentVariables": [
                             {
                                "name": "DATABASE_URL",
                                  "value": "db-url"
                                }
                             ]
                    }
              }
            ],
     "ipAddress": {
        "type": "Public",
          "ports": [
           {
              "port": 5000,
               "protocol": "TCP"
           }
         ]
       }
        }
}

```

Notice that, in the example above, the `depends_on` part is handled by the application itself - the `web` application will need to be configured to know where the api server is located in order to function.

**Step 3: Handling Storage and Networking**

Let's illustrate how storage is configured in ACI. Let’s imagine the API requires a persistent volume to store logs. In the ARM definition of the `api` container group, we would include a volume mount. This would require us to also configure Azure storage.

First, you need to create an Azure File Share or Azure Blob Storage. You will then need to mount that storage in your ACI container group definition.

```json
{
  "apiVersion": "2023-05-01",
  "location": "[resourceGroup().location]",
  "name": "api-containergroup",
  "type": "Microsoft.ContainerInstance/containerGroups",
    "properties": {
        "osType": "Linux",
        "restartPolicy": "Never",
        "containers": [
          {
           "name": "api",
            "properties": {
              "image": "my-api:latest",
              "resources": {
                  "requests": {
                      "cpu": 1,
                      "memoryInGB": 1
                   }
                  },
                  "ports": [
                    {
                       "port": 5000
                      }
                    ],
              "environmentVariables": [
                {
                    "name": "DATABASE_URL",
                    "value": "db-url"
                 }
                ],
                "volumeMounts": [
                  {
                    "name": "logvolume",
                    "mountPath": "/var/log/api"
                  }
                ]
            }
          }
        ],
     "ipAddress": {
          "type": "Public",
            "ports": [
             {
                "port": 5000,
                 "protocol": "TCP"
             }
           ]
         },
         "volumes": [
             {
                "name": "logvolume",
                "azureFile": {
                  "shareName": "api-logs",
                    "storageAccountName": "mystorageaccount",
                  "storageAccountKey": "storageaccountkey"
                }
            }
          ]
    }
}
```

In this example, we're using an Azure file share called `api-logs`, and mounting it to `/var/log/api` inside the container. This allows the application to persist the logs to a remote persistent location. Remember to replace `"mystorageaccount"` and `"storageaccountkey"` with your actual storage account details.

For networking between container groups, you would need to place these groups within the same Virtual Network (VNet) in Azure and configure application code to use each other’s FQDNs. If each ACI group has its own public IP, then you also configure the application to use these instead of any DNS. For more complex networking, look at options such as private link, Azure DNS, or a service mesh such as Linkerd.

**Important Considerations**

*   **Orchestration:** ACI alone doesn't offer robust orchestration capabilities, including automatic scaling and rollouts. If you need these, consider Azure Container Apps or Kubernetes.
*   **Monitoring:** Use Azure Monitor to gain visibility into container performance, CPU usage, and logs.
*   **Security:** Make sure to implement proper security and authorization configurations for each ACI container group.
*   **Configuration:** It is often advised to externalise all the configurable settings to ACI environment variables instead of hardcoding them into the container images.

**Recommended Resources**

For a deeper understanding of the concepts mentioned above, consider looking at these resources:

*   **"Cloud Native Patterns: Designing Change-Tolerant Software" by Cornelia Davis:** This book offers valuable insights into architectural patterns for building resilient cloud-native systems.
*   **"Kubernetes in Action" by Marko Luksa:** While not directly about ACI, this book is invaluable for comprehending container orchestration and how to design complex deployments, which will translate well to ACI concepts.
*   **Official Azure Container Instance Documentation:** Always consult Microsoft's official documentation for the most up-to-date information and best practices for working with ACI.

This approach should guide you in translating your Docker Compose setup to ACI. The key takeaway here is that ACI is not a direct drop-in replacement for Docker Compose environments. You'll need to adapt your application's architecture and use Azure services for networking and persistence, but the result will often provide a highly cost effective way of deploying containerized workloads. The flexibility in deployment patterns will increase significantly with knowledge and experience.
