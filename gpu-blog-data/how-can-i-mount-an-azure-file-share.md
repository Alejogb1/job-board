---
title: "How can I mount an Azure File Share to an Azure Container using an ARM template and access keys?"
date: "2025-01-30"
id: "how-can-i-mount-an-azure-file-share"
---
Direct file share mounting within an Azure Container Instance using access keys and ARM templates requires a careful orchestration of several key resource properties. The process essentially involves provisioning the container instance, defining a volume referencing the Azure File Share, and securely passing the required storage account key for authentication. This is not achieved through direct file system mounting commands within the container definition, but rather by the Azure infrastructure interpreting resource definitions in the ARM template to create the connections.

I've personally deployed numerous applications using this configuration, often finding it a more streamlined method than relying on network file system (NFS) setups when working with smaller microservices or temporary processing environments where a full-fledged virtual machine isn't necessary. The core principle revolves around declaring a volume with `azureFile` as the driver, and providing details like the storage account name, share name, and most critically, the storage account key within the container instance's properties. These properties are defined within the `properties.containers.properties.volumeMounts` and `properties.volumes` sections of the ARM template.

The following provides a detailed breakdown of the necessary steps and configuration elements, accompanied by code examples and considerations.

**Explanation of the Process**

The ARM template, in this context, primarily describes the Azure Container Instance (ACI). Within its definition, two areas require specific attention: `volumes` and `volumeMounts`. The `volumes` array defines the storage volume, in our case, an Azure File Share. The `volumeMounts` array, inside each container definition, dictates where that defined volume should be mounted inside the container's file system. Critically, the volume definition includes the `azureFile` type, requiring a `shareName`, `storageAccountName` and the `storageAccountKey` parameter, which is ideally supplied using parameterization for security reasons. This eliminates hardcoding and allows management outside of the main template. The `volumeMount` definition then specifies the `mountPath` inside the container where the file share is accessible. It's important to note that `readOnly` permissions can be declared if read-only access to the share is required.

Once the ARM template is deployed, Azure Infrastructure interprets this configuration. The ACI runtime fetches the specified storage account key from the ARM parameters during deployment (or key vault if referencing secret) and automatically mounts the provided share to the provided mountPath as part of the container's initial configuration. This is all done transparently; no specific code modifications are required within your container application itself to access the files. The storage account key is not exposed within the container environment variables or accessible within the container logs, providing security by not exposing sensitive information in plain text.

**Code Examples with Commentary**

Below are three examples illustrating this mechanism, progressively building in complexity and highlighting key aspects.

**Example 1: Basic Azure File Share Mount**

This example showcases the minimal template required for a single container mounting a file share.

```json
{
  "$schema": "https://schema.management.azure.com/schemas/2019-04-01/deploymentTemplate.json#",
  "contentVersion": "1.0.0.0",
  "parameters": {
    "containerGroupName": {
      "type": "string",
      "defaultValue": "aci-example-1"
    },
     "location": {
      "type": "string",
      "defaultValue": "[resourceGroup().location]"
    },
    "storageAccountName": {
      "type": "string"
    },
    "storageAccountKey": {
        "type": "securestring"
      },
      "shareName": {
          "type": "string"
      }

  },
  "resources": [
    {
      "type": "Microsoft.ContainerInstance/containerGroups",
      "apiVersion": "2023-05-01",
      "name": "[parameters('containerGroupName')]",
      "location": "[parameters('location')]",
      "properties": {
        "osType": "Linux",
        "containers": [
          {
            "name": "mycontainer",
            "properties": {
              "image": "mcr.microsoft.com/azuredocs/aci-helloworld",
              "resources": {
                "requests": {
                  "cpu": 1,
                  "memoryInGB": 1
                }
              },
               "volumeMounts": [
              {
                 "name": "myfileshare",
                 "mountPath": "/mnt/myfileshare"
                }
              ]
            }
          }
        ],
        "volumes": [
          {
            "name": "myfileshare",
            "azureFile": {
              "shareName": "[parameters('shareName')]",
              "storageAccountName": "[parameters('storageAccountName')]",
                "storageAccountKey": "[parameters('storageAccountKey')]"
            }
          }
        ],
       "restartPolicy": "Never"
      }
    }
  ]
}
```

*   **Commentary**: This template defines a single container named `mycontainer` which uses a basic hello world image and mounts the azure file share to `/mnt/myfileshare`.  It utilizes parameterized variables for the storage account details and share name, enhancing reusability. The `storageAccountKey` is a `securestring` type parameter, which should always be used to pass the key, never a standard string parameter for increased security.

**Example 2: Read-Only Mount and Multiple Containers**

This example demonstrates a file share being mounted as read-only, and in a scenario where two containers share the same volume (common in sidecar pattern use cases).

```json
{
  "$schema": "https://schema.management.azure.com/schemas/2019-04-01/deploymentTemplate.json#",
  "contentVersion": "1.0.0.0",
   "parameters": {
    "containerGroupName": {
      "type": "string",
      "defaultValue": "aci-example-2"
    },
     "location": {
      "type": "string",
      "defaultValue": "[resourceGroup().location]"
    },
    "storageAccountName": {
      "type": "string"
    },
    "storageAccountKey": {
        "type": "securestring"
      },
      "shareName": {
          "type": "string"
      }
  },
  "resources": [
    {
      "type": "Microsoft.ContainerInstance/containerGroups",
      "apiVersion": "2023-05-01",
      "name": "[parameters('containerGroupName')]",
      "location": "[parameters('location')]",
      "properties": {
        "osType": "Linux",
        "containers": [
          {
            "name": "appcontainer",
            "properties": {
              "image": "myregistry.azurecr.io/myapplication:latest",
               "resources": {
                "requests": {
                  "cpu": 1,
                  "memoryInGB": 1
                }
              },
               "volumeMounts": [
                {
                 "name": "sharedconfig",
                 "mountPath": "/app/config",
                   "readOnly": true
                }
              ]
            }
          },
          {
            "name": "logprocessor",
            "properties": {
              "image": "myregistry.azurecr.io/logprocessor:latest",
              "resources": {
                "requests": {
                  "cpu": 0.5,
                  "memoryInGB": 0.5
                }
              },
                "volumeMounts": [
                 {
                 "name": "sharedconfig",
                 "mountPath": "/log/config",
                    "readOnly": true
                }
              ]
            }
          }
        ],
        "volumes": [
          {
            "name": "sharedconfig",
             "azureFile": {
               "shareName": "[parameters('shareName')]",
              "storageAccountName": "[parameters('storageAccountName')]",
              "storageAccountKey": "[parameters('storageAccountKey')]"
             }
          }
        ],
         "restartPolicy": "Never"
      }
    }
  ]
}
```

*   **Commentary**: This template defines two containers: `appcontainer` and `logprocessor`, that both mount the same Azure File Share at different paths (`/app/config` and `/log/config` respectively). They both mount it as read-only due to the  `"readOnly": true` parameter. This is ideal for situations where multiple containers may need to access configuration or other shared read-only data from a file share.

**Example 3:  Utilizing Environment Variables and Secure Parameters**

This example highlights best practices such as using environment variables inside of containers and referencing storage account keys from key vault using ARM Parameterization.

```json
{
  "$schema": "https://schema.management.azure.com/schemas/2019-04-01/deploymentTemplate.json#",
  "contentVersion": "1.0.0.0",
   "parameters": {
      "containerGroupName": {
        "type": "string",
        "defaultValue": "aci-example-3"
      },
      "location": {
          "type": "string",
          "defaultValue": "[resourceGroup().location]"
      },
       "storageAccountName": {
          "type": "string"
        },
      "keyVaultName": {
          "type": "string"
      },
      "secretName": {
          "type": "string"
      },
       "shareName": {
        "type": "string"
        }
   },
   "variables": {
     "storageAccountKey": "[reference(resourceId('Microsoft.KeyVault/vaults/secrets', parameters('keyVaultName'), parameters('secretName')), '2023-07-01').properties.value]"
  },
  "resources": [
    {
      "type": "Microsoft.ContainerInstance/containerGroups",
      "apiVersion": "2023-05-01",
      "name": "[parameters('containerGroupName')]",
       "location": "[parameters('location')]",
      "properties": {
        "osType": "Linux",
        "containers": [
          {
            "name": "appcontainer",
            "properties": {
              "image": "myregistry.azurecr.io/myapplication:latest",
                "resources": {
                  "requests": {
                    "cpu": 1,
                    "memoryInGB": 1
                  }
                },
              "environmentVariables":[
                 {
                      "name": "FILESHARE_MOUNT_PATH",
                      "value": "/app/shareddata"
                  }
              ],
              "volumeMounts": [
                {
                 "name": "shareddata",
                 "mountPath": "[environmentVariable('FILESHARE_MOUNT_PATH')]"
                }
              ]
            }
          }
        ],
        "volumes": [
          {
            "name": "shareddata",
            "azureFile": {
              "shareName": "[parameters('shareName')]",
              "storageAccountName": "[parameters('storageAccountName')]",
               "storageAccountKey": "[variables('storageAccountKey')]"
            }
          }
        ],
       "restartPolicy": "Never"
      }
    }
  ]
}
```

*  **Commentary**: This template retrieves the `storageAccountKey` from an Azure Key Vault by using an ARM template `reference` function, and stores it in an template variable. It then utilizes an environment variable called `FILESHARE_MOUNT_PATH` to define the mount path inside the container, demonstrating a more flexible approach. Using this approach, we can further enhance security by restricting access to the key via Key Vault Access Policies, and we can change the mount path outside the definition itself by updating the environment variable.

**Resource Recommendations**

*   **Azure Resource Manager Templates Documentation:** This is the primary source for detailed information on all ARM template properties, including those for Azure Container Instances and their related configuration settings. Familiarizing yourself with the schema will allow you to manipulate the templates more effectively.
*   **Azure Container Instances Documentation:** Specific information about ACI properties, limits, and best practices for container deployments can be found here, complementing the ARM template specific information.
*   **Azure Storage Documentation:**  This is critical to have a fundamental understanding of Azure Storage concepts like file shares and access keys. The nuances of authentication, security, and performance are vital for proper configuration.
*   **Azure Key Vault Documentation:** Explore the key vault documentation to understand how to secure secrets such as storage account keys and how to integrate them into your ARM deployments. This approach drastically improves security, avoiding plain text credentials in code.

Using these resources, one can effectively construct and troubleshoot the necessary ARM templates to achieve file share mounting within ACI, and further refine their approach to incorporate best practices for security, modularity, and maintainability.
