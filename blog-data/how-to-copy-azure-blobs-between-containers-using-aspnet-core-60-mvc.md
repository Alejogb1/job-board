---
title: "How to copy Azure blobs between containers using ASP.NET Core 6.0 MVC?"
date: "2024-12-23"
id: "how-to-copy-azure-blobs-between-containers-using-aspnet-core-60-mvc"
---

Alright,  I've had my share of moving blobs around in Azure over the years, and while it might seem straightforward, there are definitely nuances to keep in mind, especially when you’re aiming for efficiency and robustness in an ASP.NET Core 6.0 MVC application. The core task, copying blobs from one container to another, hinges on the Azure Storage SDK, but the “how” can significantly impact your application’s performance and even cost.

Before diving into the code, let’s establish a solid conceptual ground. We're not simply downloading and re-uploading files. That would be inefficient, especially for larger blobs. Instead, we want to utilize Azure's server-side copy operation which minimizes network traffic from our application, since the actual data transfer occurs within Azure's infrastructure. This approach is significantly faster and more cost-effective because you don't incur egress bandwidth charges.

Now, the technical specifics. The `Azure.Storage.Blobs` NuGet package is your go-to. You will primarily be interacting with the `BlobServiceClient`, `BlobContainerClient`, and `BlobClient` classes. The process involves initiating a copy operation from the source blob to a target blob and then potentially monitoring that operation's progress. This avoids the common pitfall of trying to stream blob content through your application.

Let's break down a basic scenario where you're copying a single blob. We will handle authentication and such outside of this specific snippet, let's assume that you already have the needed configurations set for your Azure Storage account. This assumes a synchronous operation for clarity but in practice you should use the async equivalents.

```csharp
using Azure.Storage.Blobs;
using Azure.Storage.Blobs.Models;

public class BlobCopier
{
    private readonly BlobServiceClient _blobServiceClient;

    public BlobCopier(string connectionString)
    {
       _blobServiceClient = new BlobServiceClient(connectionString);
    }

    public void CopyBlob(string sourceContainerName, string sourceBlobName, string targetContainerName, string targetBlobName)
    {
        BlobContainerClient sourceContainerClient = _blobServiceClient.GetBlobContainerClient(sourceContainerName);
        BlobContainerClient targetContainerClient = _blobServiceClient.GetBlobContainerClient(targetContainerName);

        BlobClient sourceBlobClient = sourceContainerClient.GetBlobClient(sourceBlobName);
        BlobClient targetBlobClient = targetContainerClient.GetBlobClient(targetBlobName);

        //Get the URI of the source blob
        var sourceBlobUri = sourceBlobClient.Uri;

        //Begin the server-side copy operation
        targetBlobClient.StartCopyFromUri(sourceBlobUri);

    }
}
```

This is a foundational example. In a production environment, you'd absolutely want to use the asynchronous methods (e.g., `StartCopyFromUriAsync`). This ensures that your MVC application isn't blocking threads while waiting for the copy operation to complete. Also you'd likely add error handling, logging, and probably some status checking on the copy operation, as described shortly.

Let’s refine this with a more advanced example that includes asynchronous behavior, progress monitoring, and a rudimentary check for completion. This example also uses the standard `Microsoft.Extensions.Configuration` for injecting settings

```csharp
using Azure.Storage.Blobs;
using Azure.Storage.Blobs.Models;
using Microsoft.Extensions.Configuration;
using System.Threading.Tasks;

public class BlobCopierAdvanced
{
    private readonly BlobServiceClient _blobServiceClient;
    private readonly IConfiguration _configuration;

    public BlobCopierAdvanced(IConfiguration configuration)
    {
        _configuration = configuration;
        var connectionString = _configuration.GetConnectionString("StorageConnectionString");
        _blobServiceClient = new BlobServiceClient(connectionString);
    }

    public async Task<bool> CopyBlobAsync(string sourceContainerName, string sourceBlobName, string targetContainerName, string targetBlobName)
    {
        BlobContainerClient sourceContainerClient = _blobServiceClient.GetBlobContainerClient(sourceContainerName);
        BlobContainerClient targetContainerClient = _blobServiceClient.GetBlobContainerClient(targetContainerName);

        BlobClient sourceBlobClient = sourceContainerClient.GetBlobClient(sourceBlobName);
        BlobClient targetBlobClient = targetContainerClient.GetBlobClient(targetBlobName);

         //Get the URI of the source blob
        var sourceBlobUri = sourceBlobClient.Uri;


        //start asynchronous copy operation
        CopyFromUriOperation copyOperation = await targetBlobClient.StartCopyFromUriAsync(sourceBlobUri);


        while(copyOperation.HasCompleted == false){
          await Task.Delay(500); //wait 500ms before polling again
          await copyOperation.UpdateStatusAsync();
        }


       if(copyOperation.HasFailed){
           return false;
       } else {
            return true;
       }

    }
}
```

In this more advanced scenario, we're using the asynchronous methods `StartCopyFromUriAsync` and `UpdateStatusAsync` for non-blocking operations. We have included a simple polling mechanism for checking the status of the operation. Real-world implementations would benefit from exponential backoff with jitter or consider utilizing event-driven mechanisms, but this demonstrates a basic working example.

Now, consider a situation where you need to copy several blobs from one container to another. Here is a potential approach for such an operation. Note that this example does not include error handling as that is very case specific and this is meant to be a general example.

```csharp
using Azure.Storage.Blobs;
using Azure.Storage.Blobs.Models;
using Microsoft.Extensions.Configuration;
using System.Threading.Tasks;
using System.Collections.Generic;
using System.Linq;

public class BatchBlobCopier
{
        private readonly BlobServiceClient _blobServiceClient;
        private readonly IConfiguration _configuration;
    public BatchBlobCopier(IConfiguration configuration)
    {
        _configuration = configuration;
        var connectionString = _configuration.GetConnectionString("StorageConnectionString");
        _blobServiceClient = new BlobServiceClient(connectionString);
    }

   public async Task<bool> CopyBlobsAsync(string sourceContainerName, string targetContainerName, List<string> blobNames){

       BlobContainerClient sourceContainerClient = _blobServiceClient.GetBlobContainerClient(sourceContainerName);
       BlobContainerClient targetContainerClient = _blobServiceClient.GetBlobContainerClient(targetContainerName);

        var tasks = blobNames.Select(async blobName => {

            BlobClient sourceBlobClient = sourceContainerClient.GetBlobClient(blobName);
            BlobClient targetBlobClient = targetContainerClient.GetBlobClient(blobName);

             //Get the URI of the source blob
            var sourceBlobUri = sourceBlobClient.Uri;
            var copyOperation = await targetBlobClient.StartCopyFromUriAsync(sourceBlobUri);

              while(copyOperation.HasCompleted == false){
                   await Task.Delay(500); //wait 500ms before polling again
                    await copyOperation.UpdateStatusAsync();
              }

             return copyOperation.HasSucceeded;
        });


        var results = await Task.WhenAll(tasks);

        return results.All(r => r == true);

    }


}
```

Here we are parallelizing multiple copy operations to increase throughput. We are also using `Task.WhenAll` to await the completion of all of our copy operations before completing. Again, logging and robust error handling should be a priority for any production system.

In terms of further learning, I highly recommend the official documentation for the Azure Storage SDK. Specifically, delve into the `Azure.Storage.Blobs` namespace, paying close attention to the classes I've mentioned. “Cloud Computing Concepts, Technology, and Architecture” by Ricardo Puttini covers various cloud service implementation and architectural patterns that you might find useful while designing scalable solutions. And of course, Microsoft's official documentation on Azure Storage is an invaluable resource. Focus particularly on the documentation related to blob storage operations and their server-side copy functionality.

To conclude, copying blobs between containers is achievable with a few lines of code using the Azure SDK. However, it's essential to embrace best practices like server-side copy, asynchronous operations, and proper progress monitoring to avoid common pitfalls. These considerations will lead to a solution that is not only correct, but also efficient and robust.
