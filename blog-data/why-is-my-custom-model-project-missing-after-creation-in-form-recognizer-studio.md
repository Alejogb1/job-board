---
title: "Why is my Custom Model project missing after creation in Form Recognizer Studio?"
date: "2024-12-23"
id: "why-is-my-custom-model-project-missing-after-creation-in-form-recognizer-studio"
---

Alright, let's get into this. It's frustrating, I know, when you think a custom model is happily residing in Form Recognizer Studio, only to find it has seemingly vanished into the digital ether. I’ve personally chased down quite a few disappearing acts over the years, so let me share some insights on what might be going on and how to track it down. It's rarely a bug in the core system itself, more often it's a configuration issue, a misunderstanding of the expected workflow, or sometimes just a brief delay in the platform's internal indexing.

First, let's be clear on what constitutes “creation” in this context. We’re talking about successfully training a custom model with labeled data, not just initiating a project setup. A successful training operation should, generally, return a model id and be reflected within the studio environment shortly thereafter. However, “shortly” is relative in the cloud.

The most prevalent culprit I’ve seen, and have tripped over myself several times, is related to **resource configuration** and **access control**. When you train a custom model, the operation is tied to a specific Azure Form Recognizer resource and a storage account containing the training documents. Let's say your training script is using credentials from ‘resource_a’ but your studio browser session is looking at ‘resource_b’, the model will absolutely seem to have disappeared, because it's not attached to the resource that you are actually visualizing. This is often a consequence of working with multiple resources, test environments and the like. It’s also essential to confirm you have the correct permissions configured at both the resource level and the storage container level. If the user associated with your Azure credentials lacks ‘contributor’ access, for example, you may be able to *initiate* the training, but the model might not be correctly associated with your user profile or might not be listed at all. The first thing you need to verify is whether the storage container is set to private or public. If it is set to private, you will need to configure the user identity that is used to access the storage account.

Another common cause is the sometimes-imperfect nature of **caching and indexing**. Sometimes, the studio interface doesn’t instantly reflect changes in the backend systems. A successful model training operation might be complete, but the portal cache hasn't updated yet. Clearing your browser cache or trying a different browser might seem simple, but I've seen it work more than you might expect, as have re-logging into the studio interface. While this won't explain consistent absence of models, it helps to quickly eliminate the most obvious visual problems.

Finally, consider the impact of **training dataset size and complexity.** If you're dealing with an extremely large or unusually complex set of documents, the training process can take a non-trivial amount of time. While the training itself might have finished, the necessary metadata updates and studio indexing might lag behind, so it's important to check the training status directly through the API and not only via the studio interface. It is also vital to ensure that the documents you are using for training meet the documented minimum requirement for page numbers and the variety of document types. Furthermore, if you are only trying to train with handwritten documents, consider using the neural model rather than the default template model which may be less accurate for this task.

To illustrate some of the troubleshooting steps, let’s look at a few code snippets, focusing on using the python SDK for Azure Form Recognizer (and assuming you’ve already installed the SDK):

**Snippet 1: Confirm Resource and Storage Account Connection**

```python
from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentAnalysisClient

# Replace with your actual keys and endpoints
endpoint = "https://your-form-recognizer-resource.cognitiveservices.azure.com/"
key = "YOUR-FORM-RECOGNIZER-API-KEY"
storage_account_uri="https://yourstorageaccount.blob.core.windows.net/"
storage_account_key = "YOUR-STORAGE-ACCOUNT-ACCESS-KEY"
container_name = "your-container-name"

try:
  document_analysis_client = DocumentAnalysisClient(endpoint=endpoint, credential=AzureKeyCredential(key))

  # Attempt to list models, should throw exception if there is a connection issue
  models = document_analysis_client.list_custom_models()
  print("Successfully connected to Form Recognizer resource.")

except Exception as e:
    print(f"Error connecting to Form Recognizer resource: {e}")


from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient

try:
    blob_service_client = BlobServiceClient(account_url=storage_account_uri, credential=storage_account_key)
    container_client = blob_service_client.get_container_client(container=container_name)
    blob_list = container_client.list_blobs()

    print(f"Successfully connected to storage account at {storage_account_uri} and container {container_name}.")


except Exception as e:
    print(f"Error connecting to Storage Account: {e}")

```

This snippet confirms if your python script can connect to both the form recognizer resource and the storage account used. Verify all connection information as well as access credentials as a first step of debugging.

**Snippet 2: Checking Model Training Status via API**

```python
from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentAnalysisClient

# Replace with your actual keys and endpoints
endpoint = "https://your-form-recognizer-resource.cognitiveservices.azure.com/"
key = "YOUR-FORM-RECOGNIZER-API-KEY"

# Replace with the model ID (if you have it)
model_id = "your_model_id_here"


try:
    document_analysis_client = DocumentAnalysisClient(endpoint=endpoint, credential=AzureKeyCredential(key))
    model = document_analysis_client.get_custom_model(model_id)

    if model:
        print(f"Model Id: {model.model_id}")
        print(f"Model Status: {model.status}")
        print(f"Training Started On: {model.training_started_on}")
        print(f"Training Completed On: {model.training_completed_on}")
        print(f"Training Error: {model.training_error}") # This is the error you should check for training errors.
    else:
        print(f"Model with ID '{model_id}' not found.")

except Exception as e:
    print(f"Error retrieving model: {e}")

```
This snippet retrieves the training status of the model using the model id. If the model is "ready" but still not visible, this helps to isolate the problem. If no such model is found, the issue is more likely tied to the credentials used, resource connection, or the lack of actually generating a model.

**Snippet 3: Initiating Model Training via API (Example)**
```python
from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentAnalysisClient, DocumentModelTrainingOptions

# Replace with your actual keys and endpoints
endpoint = "https://your-form-recognizer-resource.cognitiveservices.azure.com/"
key = "YOUR-FORM-RECOGNIZER-API-KEY"
storage_container_url="https://yourstorageaccount.blob.core.windows.net/your-container-name" #Make sure it's the container, not the whole storage account

try:
    document_analysis_client = DocumentAnalysisClient(endpoint=endpoint, credential=AzureKeyCredential(key))

    poller = document_analysis_client.begin_training(
        training_files_url=storage_container_url,
        model_id="your-model-id-for-training-if-any",
        options= DocumentModelTrainingOptions(model_description="Test Model")
    )

    model = poller.result()

    print(f"Model Id: {model.model_id}")
    print(f"Model Status: {model.status}")

except Exception as e:
        print(f"Error training model: {e}")

```

This last example demonstrates how you can start model training process, bypassing the Studio interface completely. If this code snippet is successful and returns a model id and the status of ready/succeeded, but the model is not visible in the portal, this further isolates the problem to the indexing of the studio interface.

For further reading, I'd recommend diving into the official Microsoft documentation for Azure Form Recognizer API and SDK, which is meticulously maintained. Specifically, the sections on custom model training and resource management are invaluable. Also, "Programming Microsoft Azure: Core Infrastructure" by David G. Hogue is helpful for a deep dive into Azure resource management, which is relevant for understanding underlying principles of your issue. For more on best practices in machine learning model management, consider "Machine Learning Engineering" by Andriy Burkov.

In my experience, the combination of these debugging approaches usually reveals the cause of the “missing” model. Remember to methodically check access, permissions, resource configurations, training status through API, and clear those browser caches! Good luck, and don't hesitate to dig deeper if you run into further snags.
