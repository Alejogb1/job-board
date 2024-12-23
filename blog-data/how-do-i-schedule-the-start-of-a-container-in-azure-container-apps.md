---
title: "How do I schedule the start of a container in Azure Container Apps?"
date: "2024-12-23"
id: "how-do-i-schedule-the-start-of-a-container-in-azure-container-apps"
---

Alright, let's unpack how to schedule the start of a container in Azure Container Apps. It’s a common need, and having encountered it many times in my years of working with cloud infrastructure, I've found there's not a single silver bullet, but rather a combination of strategies depending on the specific scenario. Let me share some practical approaches, along with code examples, based on situations I’ve personally faced.

The core challenge with Azure Container Apps isn’t necessarily the *scheduling* itself; the platform is designed to manage that. The difficulty lies in triggering the container to start at a *specific* time, or under specific conditions not natively provided by the standard scaling parameters. So, what we’re really discussing is how to leverage or work *around* the container apps platform’s default behavior to achieve our targeted timing.

The simplest case is typically needing a container to only become active after deployment. This is often handled via the built-in revision system. By default, a new container revision becomes active once deployment finishes. However, what if you need to initiate a workload that *shouldn't* start immediately? Say, for example, you're processing batch data overnight and don’t want your processing container to spin up and start pulling from a queue until midnight.

One of the most robust solutions for this is using an external orchestrator or scheduler. While Azure Container Apps doesn't directly provide cron-like scheduling, Azure Logic Apps and Azure Functions paired with queues or event grids can work wonders. I vividly remember a project where we needed to run a complex data transformation process that only needed to run daily at 3am. Trying to shoehorn this into a constantly-running container was just wasteful, both in terms of resources and cost.

Here's how we tackled it. We used Azure Logic Apps:

**Code Snippet 1: Azure Logic App Definition (Simplified)**

```json
{
  "definition": {
    "$schema": "https://schema.management.azure.com/providers/Microsoft.Logic/schemas/2016-06-01/workflowdefinition.json#",
    "actions": {
      "HTTP_Action_to_Container_App": {
        "type": "Http",
        "inputs": {
          "method": "POST",
          "uri": "[concat('https://management.azure.com/subscriptions/', parameters('subscriptionId'), '/resourceGroups/', parameters('resourceGroupName'), '/providers/Microsoft.App/containerApps/', parameters('containerAppName'), '/revisions/default/activate?api-version=2023-05-01')]",
          "headers": {
            "Content-type": "application/json"
             },
          "authentication": {
            "type": "ManagedServiceIdentity",
            "identity": "[list('2023-05-01',concat('/subscriptions/', parameters('subscriptionId'),'/resourcegroups/', parameters('resourceGroupName'), '/providers/Microsoft.ManagedIdentity/userAssignedIdentities/',parameters('managedIdentityName'))).principalId]"
           }
          }
         }
       },
      "triggers": {
         "Recurrence": {
            "type": "Recurrence",
            "recurrence": {
              "frequency": "Day",
              "interval": 1,
               "startTime": "2024-07-28T03:00:00Z",
              "timeZone": "UTC",
              "schedule": {
               "hours": [
                "03"
                ],
                "minutes": [
                 "00"
               ]
              }
            }
          }
        }
  },
  "parameters": {
    "subscriptionId": {
        "type": "string"
    },
    "resourceGroupName": {
        "type": "string"
     },
    "containerAppName": {
        "type":"string"
      },
   "managedIdentityName": {
    "type":"string"
       }
  }
}
```

This Logic App uses a *recurrence trigger* set for 3:00 am UTC. Importantly, it makes a post request to the Azure Management API endpoint to activate a revision. You need to set up a user assigned managed identity with permissions to activate container app revisions and add that to the logic app. This approach gives you precise time-based control. The recurrence settings are fairly flexible, allowing for a range of schedules.

Another scenario I frequently encounter involves event-driven workloads. Rather than time, perhaps you need a container to start when a new file arrives in Azure Blob Storage, or when a message is added to a queue. In this case, Azure Functions are more suitable than Logic Apps due to their event-driven nature.

**Code Snippet 2: Azure Function (Python) triggered by a Storage Queue**

```python
import logging
import json
import requests
import os
import azure.functions as func

def main(queueItem: func.QueueMessage):
    logging.info('Python queue trigger function processed a queue item.')

    try:
        # Extract data from the queue
        queue_data = json.loads(queueItem.get_body().decode('utf-8'))

        # Prepare the activation request
        subscription_id = os.environ.get("subscriptionId")
        resource_group = os.environ.get("resourceGroupName")
        container_app_name = os.environ.get("containerAppName")
        identity_client_id = os.environ.get("identityClientId")

        api_endpoint = f"https://management.azure.com/subscriptions/{subscription_id}/resourceGroups/{resource_group}/providers/Microsoft.App/containerApps/{container_app_name}/revisions/default/activate?api-version=2023-05-01"
        
        headers = {
            "Content-type": "application/json"
        }

        # Use Managed Identity for authentication
        token_url = f"https://login.microsoftonline.com/{os.environ.get('tenantId')}/oauth2/token"

        token_response = requests.post(token_url, data={
                "grant_type": "client_credentials",
                "client_id": identity_client_id,
                "resource": "https://management.azure.com/"
            }, headers={'Content-type': 'application/x-www-form-urlencoded'})

        token_response.raise_for_status()

        token_data = token_response.json()

        auth_header = {
           'Authorization': f'Bearer {token_data["access_token"]}'
        }
      
        # Make the POST request to activate the container app revision.
        response = requests.post(api_endpoint, headers={**headers, **auth_header})
        response.raise_for_status()

        logging.info("Container App revision activated successfully.")

    except Exception as e:
        logging.error(f"Error activating container app: {str(e)}")
```

This function triggers when a message arrives in a designated Azure Storage Queue. It extracts the relevant data, obtains an authentication token using a managed identity, and makes an API call to activate the latest revision of the Container App. The function demonstrates that a managed identity can be used, but you can use service principals or user identities with suitable permissions depending on your needs. You would need to populate the app settings for tenantId, subscriptionId, resourceGroupName, containerAppName and identityClientId in your function app configuration.

Finally, let’s consider a scenario where the container itself needs to control its own start timing. In a highly distributed system, you might have a container that needs to defer the start of its workload until a prerequisite service is available or some internal condition is met.

**Code Snippet 3: Containerized Application Code (Python)**

```python
import time
import os
import logging
from azure.identity import DefaultAzureCredential
from azure.mgmt.appcontainers import AppContainersManagementClient

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def check_prerequisite():
    # Replace this with your actual prerequisite check logic
    logging.info("Checking prerequisite condition...")
    time.sleep(5)  # Simulate some work or service check
    return True # Simulate the successful prerequisite

def activate_container_app_revision(subscription_id, resource_group, container_app_name, identity_client_id):
    try:

        credentials = DefaultAzureCredential(client_id=identity_client_id)
        app_containers_client = AppContainersManagementClient(credentials, subscription_id)
        app_containers_client.container_apps.activate_revision(resource_group, container_app_name,"default")

        logging.info("Container App revision activated successfully.")
    except Exception as e:
       logging.error(f"Error activating container app revision: {str(e)}")


if __name__ == "__main__":
    subscription_id = os.environ.get("subscriptionId")
    resource_group = os.environ.get("resourceGroupName")
    container_app_name = os.environ.get("containerAppName")
    identity_client_id = os.environ.get("identityClientId")

    if check_prerequisite():
        activate_container_app_revision(subscription_id, resource_group, container_app_name,identity_client_id)
    else:
        logging.info("Prerequisite condition not met. Exiting.")

```
Here, the container application includes a function `check_prerequisite()`. Before running the main application logic, it checks that the prerequisite is met. If successful, it proceeds to use the `azure-identity` and `azure-mgmt-appcontainers` python packages to activate it's own revision using a managed identity. The container would need to be deployed with a specific start up command that activates the function. It uses the DefaultAzureCredential class to handle authentication with azure.

For further study, I'd highly recommend looking into *Enterprise Integration Patterns* by Gregor Hohpe and Bobby Woolf, which provides a great foundational understanding of integration patterns often used in these scenarios. The official Microsoft documentation on Azure Logic Apps, Azure Functions, and Azure Container Apps are also essential resources. Specifically, pay close attention to the documentation around managed identities and the Azure Management REST API. The *Designing Distributed Systems* book by Brendan Burns provides a broader perspective on the challenges and patterns around distributed applications, which is also helpful when planning the deployment and scheduling of such workloads.

In summary, scheduling the start of a container within Azure Container Apps frequently requires using external services for controlling when or how a container becomes active, especially if immediate activation is not the desired behavior. Each of these approaches provides different levels of control, and the "best" approach depends entirely on your specific requirements and architectural patterns. I hope this helps you navigate this.
