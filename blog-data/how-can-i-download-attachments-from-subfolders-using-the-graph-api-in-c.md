---
title: "How can I download attachments from subfolders using the Graph API in C#?"
date: "2024-12-23"
id: "how-can-i-download-attachments-from-subfolders-using-the-graph-api-in-c"
---

Let’s tackle this. I remember a project back in '18 where we had a similar requirement—migrating a legacy document management system to SharePoint Online. Dealing with nested folders and their attachments through the Microsoft Graph API was, to put it mildly, a learning experience. The key is understanding the paginated nature of the API and how to efficiently traverse the folder structure. It's less about a single magic call and more about a methodical, iterative process.

The core challenge lies in the fact that the Graph API often returns results in batches, and that the folder structure can be arbitrarily deep. This means you'll need to handle pagination and recursion or iteration effectively. Also, you’ll need to understand how to differentiate between a file and a folder within the response. Let's break this down step by step, starting with how we would typically fetch items in a folder. We'll then build upon that.

First, you'll need to instantiate the `GraphServiceClient`. I'm assuming you have that setup already, along with the proper authentication mechanisms. This generally involves using an access token acquired through the appropriate authentication flow (Azure AD, for example). I won't go into that here as that's a separate, albeit important, topic. For the purposes of this explanation, let’s assume our client object is `graphClient`.

Let’s start with a basic function to fetch items from a *single* folder, then we can expand.

```csharp
    using Microsoft.Graph;
    using System.Collections.Generic;
    using System.Threading.Tasks;
    using System.IO;

    public class GraphAttachmentHandler
    {
        private readonly GraphServiceClient graphClient;

        public GraphAttachmentHandler(GraphServiceClient graphClient)
        {
           this.graphClient = graphClient;
        }

       public async Task DownloadAttachmentsFromFolder(string driveId, string folderPath, string localSavePath)
        {
            var folderId = await GetFolderId(driveId, folderPath);
            if (folderId == null) return;

             var items = await graphClient.Drives[driveId].Items[folderId].Children
                .Request()
                .GetAsync();

             if (items?.Count == 0) return;

            await ProcessItems(driveId, items, localSavePath);


             while(items.NextPageRequest != null)
            {
                items = await items.NextPageRequest.GetAsync();
                await ProcessItems(driveId, items, localSavePath);

             }

        }
     private async Task<string?> GetFolderId(string driveId, string folderPath)
        {
            var pathComponents = folderPath.Split('/').Where(s => !string.IsNullOrWhiteSpace(s)).ToArray();
            var currentId = "root";

            foreach (var component in pathComponents)
            {
               var response = await graphClient.Drives[driveId].Items[currentId].Children
                .Request()
                .Filter($"name eq '{component}'")
                .GetAsync();

                if (response.Count == 0) return null;

                currentId = response[0].Id;
            }
            return currentId;
        }


       private async Task ProcessItems(string driveId, IGraphServiceChildrenCollectionPage items, string localSavePath)
        {

             foreach (var item in items)
            {
              if(item.File != null)
               {
                    var fileStream = await graphClient.Drives[driveId].Items[item.Id].Content.Request().GetAsync();

                    using (var fs = new FileStream(Path.Combine(localSavePath, item.Name), FileMode.Create))
                    {
                        await fileStream.CopyToAsync(fs);
                    }

                    var attachments = await graphClient.Drives[driveId].Items[item.Id].Attachments.Request().GetAsync();
                  if (attachments.Count > 0)
                    {
                     await DownloadAttachments(driveId, item.Id, attachments, localSavePath);
                    }

                }

                 if (item.Folder != null)
                {
                    await DownloadAttachmentsFromFolder(driveId, $"{item.ParentReference.Path}/{item.Name}", localSavePath);
                }
            }

        }

     private async Task DownloadAttachments(string driveId, string fileId, IGraphServiceAttachmentCollectionPage attachments, string localSavePath)
        {
           foreach (var attachment in attachments)
            {
                var attachmentStream = await graphClient.Drives[driveId].Items[fileId].Attachments[attachment.Id].Content.Request().GetAsync();
                using(var fs = new FileStream(Path.Combine(localSavePath, attachment.Name), FileMode.Create))
                 {
                    await attachmentStream.CopyToAsync(fs);
                 }
             }
         }

    }
```

This first code snippet includes the core logic for recursively processing the folder tree. I have split this out into several helper methods for readability and organization. Firstly, `DownloadAttachmentsFromFolder` is the main function you'd call with the path, `driveId`, and where to download files to. `GetFolderId` is a helper function to get the folder's ID based on the path. The method `ProcessItems` differentiates between files and folders, processing accordingly. The actual file content download is done via `graphClient.Drives[driveId].Items[item.Id].Content.Request().GetAsync()`, while attachments are retrieved and handled in `DownloadAttachments`. If an item is a folder, the `DownloadAttachmentsFromFolder` is called recursively.

A couple of important points here: We are paginating the response from the children call using a `while` loop and the `NextPageRequest`, this is the correct way to get all items from the Graph API, as the response is limited to a certain amount. We are also correctly checking the response `if(item.File != null)` to determine if the `item` is a file or folder. This method will handle files within the current folder, and if the file has attachments, it will handle those as well, saving them to the `localSavePath` under the respective file name. It also recursively calls `DownloadAttachmentsFromFolder` if the item is a folder. The recursive aspect here is crucial, as we need to move through subfolders and process each level, which the example above shows.

Now, let's assume that we need to filter by file type. Modifying the above slightly, the method `ProcessItems` becomes:

```csharp
  private async Task ProcessItems(string driveId, IGraphServiceChildrenCollectionPage items, string localSavePath, string fileTypeFilter)
        {
             foreach (var item in items)
            {
              if(item.File != null && item.Name.EndsWith(fileTypeFilter, StringComparison.OrdinalIgnoreCase))
               {
                    var fileStream = await graphClient.Drives[driveId].Items[item.Id].Content.Request().GetAsync();

                    using (var fs = new FileStream(Path.Combine(localSavePath, item.Name), FileMode.Create))
                    {
                        await fileStream.CopyToAsync(fs);
                    }

                     var attachments = await graphClient.Drives[driveId].Items[item.Id].Attachments.Request().GetAsync();
                  if (attachments.Count > 0)
                    {
                     await DownloadAttachments(driveId, item.Id, attachments, localSavePath);
                    }
                }

                 if (item.Folder != null)
                {
                    await DownloadAttachmentsFromFolder(driveId, $"{item.ParentReference.Path}/{item.Name}", localSavePath, fileTypeFilter);
                }
            }

        }
```

In this snippet, we’ve added a `fileTypeFilter` parameter. The conditional `if(item.File != null && item.Name.EndsWith(fileTypeFilter, StringComparison.OrdinalIgnoreCase))` ensures that only files ending with the specified file type are downloaded. We also need to update the call to `DownloadAttachmentsFromFolder` to pass this parameter down to subsequent calls. This addition highlights how easy it is to incorporate filtering.

Finally, let’s add error handling for both network issues and exceptions during processing:

```csharp
  private async Task ProcessItems(string driveId, IGraphServiceChildrenCollectionPage items, string localSavePath, string fileTypeFilter)
        {
             foreach (var item in items)
            {
              try {
                 if(item.File != null && item.Name.EndsWith(fileTypeFilter, StringComparison.OrdinalIgnoreCase))
                  {
                     var fileStream = await graphClient.Drives[driveId].Items[item.Id].Content.Request().GetAsync();

                       using (var fs = new FileStream(Path.Combine(localSavePath, item.Name), FileMode.Create))
                         {
                             await fileStream.CopyToAsync(fs);
                         }

                     var attachments = await graphClient.Drives[driveId].Items[item.Id].Attachments.Request().GetAsync();
                     if (attachments.Count > 0)
                     {
                       await DownloadAttachments(driveId, item.Id, attachments, localSavePath);
                     }
                  }

                 if (item.Folder != null)
                {
                     await DownloadAttachmentsFromFolder(driveId, $"{item.ParentReference.Path}/{item.Name}", localSavePath, fileTypeFilter);
                 }
            } catch (ServiceException ex)
             {
                Console.WriteLine($"Graph API Error: {ex.Message}");
                // Consider more robust logging here
             }
             catch (Exception ex)
             {
                Console.WriteLine($"Error processing item {item.Name}: {ex.Message}");
                // Log error more robustly
              }
            }
        }
```
In this revised method, we've wrapped all the item processing code in a try-catch block to catch both Graph API specific `ServiceException` errors as well as general exceptions that may occur during processing, particularly during IO operations, logging the issues to the console for easy identification and debugging. This is crucial for ensuring a resilient application that doesn’t break during bulk operations. In a production scenario, I would recommend logging the errors to a more persistent storage mechanism.

For further study, I highly recommend the Microsoft Graph documentation itself, which you can find on the Microsoft developer portal. For a broader understanding of graph theory and traversal algorithms, consider studying a classic algorithm text such as "Introduction to Algorithms" by Cormen et al., which will give you the theoretical background to these concepts. Additionally, the official Microsoft Graph SDK documentation (for C# in this case) is your ultimate source of truth, and can be found on the Nuget package page, which also has reference documentation for the various classes and methods available. I encourage you to dive deep into these resources to master this crucial topic.
