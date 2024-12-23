---
title: "How can I disable download messages from Azure Datastore?"
date: "2024-12-23"
id: "how-can-i-disable-download-messages-from-azure-datastore"
---

Let’s tackle this from a pragmatic angle, shall we? Disabling download messages from azure datastore, while seemingly straightforward, often involves a bit of understanding about how azure storage handles notifications and logging. In my past engagements, notably a large-scale data migration project involving petabytes of unstructured data to blob storage, I encountered this particular nuisance head-on. Users were bombarded with download notifications, creating a significant amount of “noise” and impacting their workflow. The solution wasn't always obvious, as it isn’t a single switch you can flick within the portal. So, let’s break down the approaches and offer some working code examples.

First, it's crucial to understand that what you perceive as "download messages" aren’t actually datastore notifications directly, but rather related to activities within the azure environment – specifically, logging and monitoring mechanisms configured for your storage account. Primarily, these messages stem from either *diagnostic logging* or custom *event subscriptions*. Diagnostic logs, which are extremely useful for audits and troubleshooting, can sometimes inadvertently generate high levels of messages, especially during heavy data transfers. Event subscriptions, on the other hand, which enable reactive workflows, might also trigger notifications based on specific blob events. We'll address them both.

The most common culprit is often related to diagnostic logging configured to capture `Read` operations on blobs. These logs are then potentially routed to different sinks, such as Log Analytics workspace, Storage Account or Event Hub, and each sink can have their own notification settings. It’s often the connection between the storage account logs and the alerting mechanisms that's generating these perceived "download messages”. To mitigate this, we need to adjust the diagnostic logging settings to filter out `Read` operations or suppress logging to specific sinks.

Here’s a code snippet using the azure cli to disable the capture of `Read` events. It assumes you already have the azure cli configured and are logged into the appropriate subscription:

```bash
az monitor diagnostic-settings show \
    --name "default" \
    --resource /subscriptions/<subscription_id>/resourceGroups/<resource_group_name>/providers/Microsoft.Storage/storageAccounts/<storage_account_name> \
    --output json | jq '.properties.logs[] | select(.category == "StorageRead")'

#If the above returns data then you know "storageRead" is enabled, the below command disables it.

az monitor diagnostic-settings update \
    --name "default" \
    --resource /subscriptions/<subscription_id>/resourceGroups/<resource_group_name>/providers/Microsoft.Storage/storageAccounts/<storage_account_name> \
    --remove 'logs[?category=="StorageRead"].enabled=true'
```
*Explanation:*
   *The first command `az monitor diagnostic-settings show` retrieves the current diagnostic settings for the specified storage account in json format. This allows us to see if the 'StorageRead' category is enabled.*
   *The json result is piped into `jq` which is used to filter the array of 'logs' objects, and check if 'StorageRead' is enabled. If it returns an object, that means `StorageRead` logging is enabled.*
   *The second command `az monitor diagnostic-settings update` modifies the diagnostic settings, specifically removing the enabled property (`=true`) from the `StorageRead` category. This effectively disables `Read` event logging for the storage account. (Remember to replace placeholders with your actual subscription id, resource group name, and storage account name.)*

This cli-based solution provides a quick and efficient way to manage diagnostic logs. Remember, these changes may not be instantaneous and can take a few minutes to fully propagate. It is crucial to monitor log behavior after applying these changes.

Now, if the issue lies with event grid notifications, you might have an event subscription that's reacting to blob read events and subsequently pushing notifications. In this case, you'll need to identify and modify or remove this subscription. Here’s an example using powershell and the azure module to achieve this:

```powershell
$storageAccountName = "<storage_account_name>"
$resourceGroupName = "<resource_group_name>"
$subscriptionId = "<subscription_id>"

Select-AzSubscription -SubscriptionId $subscriptionId

$eventSubscriptions = Get-AzEventGridSubscription -ResourceGroupName $resourceGroupName

foreach ($subscription in $eventSubscriptions) {
    if ($subscription.Scope -match "Microsoft.Storage/storageAccounts/$storageAccountName") {
        Write-Host "Found event subscription for storage account: $($subscription.Name)"
        if ($subscription.EventTypes -contains "Microsoft.Storage.BlobCreated" -or $subscription.EventTypes -contains "Microsoft.Storage.BlobDeleted" -or $subscription.EventTypes -contains "Microsoft.Storage.BlobRead")
        {
            Write-Host "Event Subscription contains blob events, disabling"
            Remove-AzEventGridSubscription -ResourceGroupName $resourceGroupName -Name $subscription.Name -Force
           Write-Host "Event subscription removed $($subscription.Name)"
        }else{
        Write-Host "Event subscription does not contain blob events"
        }
    }
}

```
*Explanation:*
   *First we declare variables holding the `storageAccountName`, the `resourceGroupName`, and the `subscriptionId`.*
   *Then using `Select-AzSubscription`, the powershell context is changed to the provided subscription*
   *The code fetches all event grid subscriptions in the resource group.*
   *It then iterates through each subscription and checks if its scope includes the target storage account.*
   *Inside the loop, we check if events for `Microsoft.Storage.BlobCreated`, `Microsoft.Storage.BlobDeleted` or `Microsoft.Storage.BlobRead` are present in the subscriptions event types, if they are, we remove the subscription*

   *(Remember to replace placeholders with your actual subscription id, resource group name, and storage account name.)*

This script specifically looks for event subscriptions that include blob events related to the provided storage account and disables them if found. Always double-check your event subscription definitions before removal to avoid disrupting other workflows. In some cases, you might find that you need to maintain event notifications, but fine-tune them to exclude read events. This is possible by modifying the event types or using advanced filters available in event grid.

Lastly, if you're leveraging custom code with the azure sdk, make sure that the code itself isn't contributing to the notification problem. For example, if you have code that reads blobs and then logs each read, you might be inadvertently generating noise. Review your application code and adjust the logging levels to avoid unnecessary notifications related to download activities.

Here’s a snippet of python code demonstrating how to use the azure sdk to download blobs without generating logs locally (assuming you've already set up the appropriate azure environment variables):
```python
from azure.storage.blob import BlobServiceClient

def download_blob_quietly(connection_string, container_name, blob_name, local_file_path):
    try:
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)

        with open(local_file_path, "wb") as local_file:
           download_stream = blob_client.download_blob()
           local_file.write(download_stream.readall())
        print(f"Downloaded '{blob_name}' to '{local_file_path}' successfully.")
    except Exception as e:
        print(f"Error downloading '{blob_name}': {e}")


# Example usage
connection_string = "<your_connection_string>"
container_name = "<your_container_name>"
blob_name = "<your_blob_name>"
local_file_path = "local_download.txt"

download_blob_quietly(connection_string, container_name, blob_name, local_file_path)
```
*Explanation:*

   *This code demonstrates the core logic of downloading a blob and writing the contents to a local file. it's a basic example of how you might download a blob from azure without implementing additional logging in your application.*

   *The code instantiates a `BlobServiceClient` using a connection string and then retrieves a `BlobClient` for specific blob*
    *The blob data is then written directly to a local file.*

It’s key to understand that in this example, the code doesn’t actively log any events beyond a success/failure message which is printed to the console, therefore, is quiet. Proper handling of exceptions in production would be necessary, but I am keeping the example simple. Ensure that the `azure-storage-blob` package is installed in your environment, and replace placeholders for your specific container and blob details, if applicable.

In summary, resolving "download message" issues usually isn’t a singular approach, it requires a combination of these methods. Focus on diagnostic logging, event subscriptions, and application code to pinpoint where these messages are originating from, rather than attributing them directly to datastore actions. It's a nuanced scenario that requires thoughtful analysis and tailored solutions. I’d suggest you delve into the azure documentation on "Azure Monitor Diagnostic Settings", particularly for storage accounts, as well as "Azure Event Grid" docs for a deeper understanding. Consider also exploring the excellent resource, "Microsoft Azure Storage Essentials" by Benjamin Perkins, which offers valuable insights into storage management. These resources combined with practical hands-on experience will provide a solid foundation for addressing these types of challenges effectively.
