---
title: "How can I download multiple attachments from an Outlook email using Java?"
date: "2025-01-30"
id: "how-can-i-download-multiple-attachments-from-an"
---
The Microsoft Graph API’s email endpoint, specifically the `/messages/{message-id}/attachments` resource, provides the most reliable and maintainable method for programmatically downloading multiple attachments from Outlook emails in a Java application. Direct interaction with proprietary protocols like IMAP or POP3, while historically possible, often presents challenges regarding authentication, protocol versioning, and vendor-specific nuances, resulting in fragile solutions. Using the Graph API, with its robust RESTful interface, greatly simplifies the process.

Here's how to accomplish this, drawing from my experience building an email processing service several years ago. The core of this task involves three main steps: first, authenticating with Microsoft Graph; second, retrieving the required message and attachment metadata; and third, actually downloading the attachment content. This breakdown assumes that you already have registered your application with Azure Active Directory, granted the necessary API permissions (likely `Mail.Read` or `Mail.ReadWrite`, and possibly `offline_access`), and have a working authentication flow to acquire an access token.

My first challenge on this project was managing the OAuth 2.0 authentication flow. While various libraries streamline the process, understanding the underlying mechanism was critical to debug issues. Once the access token was in hand, interaction with the Graph API became straightforward. The crucial element is proper construction of the API requests.

**Code Example 1: Retrieving Message Attachments**

This code snippet illustrates fetching a list of attachment metadata associated with a given email message.

```java
import com.microsoft.graph.core.ClientException;
import com.microsoft.graph.models.Attachment;
import com.microsoft.graph.models.AttachmentCollectionResponse;
import com.microsoft.graph.requests.AttachmentCollectionRequest;
import com.microsoft.graph.requests.GraphServiceClient;

import java.util.List;

public class AttachmentFetcher {

    private final GraphServiceClient<com.microsoft.graph.requests.GraphServiceClient> graphClient;

    public AttachmentFetcher(String accessToken) {
        this.graphClient = GraphServiceClient.builder().authenticationProvider(request -> {
            request.addHeader("Authorization", "Bearer " + accessToken);
        }).buildClient();
    }

    public List<Attachment> fetchAttachments(String messageId) throws ClientException {
        AttachmentCollectionRequest request = graphClient.me()
                                                          .messages(messageId)
                                                          .attachments()
                                                          .buildRequest();
        AttachmentCollectionResponse response = request.get();
        if (response != null && response.value != null) {
            return response.value;
        }
        return null;
    }
}
```

**Commentary:**

*   We utilize the Microsoft Graph SDK for Java, which abstracts away much of the complexity of constructing HTTP requests.
*   The `GraphServiceClient` is initialized with an authentication provider that injects the access token into each request’s authorization header. This is crucial for the API to authenticate the request.
*   The `fetchAttachments` method takes a message ID as input.
*   The code navigates the Graph API resource path, using the `me()` method to specify the current user, then drills down to the target message, and finally to its associated attachments.
*   The `.get()` method executes the request and returns an `AttachmentCollectionResponse`. The `response.value` contains a list of `Attachment` objects, each representing an attachment in the email. The SDK automatically handles response parsing, making the returned list readily usable.
*   Error handling is simplified with a `ClientException` being thrown in case of API errors.

**Code Example 2: Downloading a Single Attachment**

Once the metadata is retrieved, we can download the actual attachment content. This example focuses on downloading a single attachment, identified by its ID.

```java
import com.microsoft.graph.core.ClientException;
import com.microsoft.graph.requests.AttachmentRequest;
import com.microsoft.graph.requests.GraphServiceClient;

import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

public class AttachmentDownloader {

    private final GraphServiceClient<com.microsoft.graph.requests.GraphServiceClient> graphClient;

    public AttachmentDownloader(String accessToken) {
        this.graphClient = GraphServiceClient.builder().authenticationProvider(request -> {
            request.addHeader("Authorization", "Bearer " + accessToken);
        }).buildClient();
    }


    public void downloadAttachment(String messageId, String attachmentId, String destinationPath) throws ClientException, IOException {
        AttachmentRequest request = graphClient.me()
                                             .messages(messageId)
                                             .attachments(attachmentId)
                                             .buildRequest();

        InputStream is = request.content().get(); // Download the attachment

        try(OutputStream outputStream = new FileOutputStream(destinationPath)){
            byte[] buffer = new byte[8 * 1024]; // 8KB buffer
            int bytesRead;
            while ((bytesRead = is.read(buffer)) != -1) {
                outputStream.write(buffer, 0, bytesRead);
            }
        }
        is.close();
    }
}
```

**Commentary:**

*   This class is similarly instantiated with a `GraphServiceClient`.
*   The `downloadAttachment` method takes the message ID, attachment ID, and destination file path as inputs.
*   The Graph API path is constructed to specifically target the attachment with the given ID, using `.attachments(attachmentId)`.
*   The `request.content().get()` retrieves the attachment content as an `InputStream`.
*   We use a standard input/output stream pattern to read the data and write it to the specified file path. A try-with-resources block ensures proper closure of the output stream.
*   A simple buffer is used to efficiently copy the file content from the input stream to the output stream. This byte-level stream handling gives more control over the file writing operation.

**Code Example 3: Downloading All Attachments of a Message**

Building on the previous examples, here is the approach to download all attachments of a given email message. This combines fetching the attachment metadata, and subsequently downloading the content for each attachment.

```java
import com.microsoft.graph.models.Attachment;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;

public class AttachmentManager {

    private final AttachmentFetcher fetcher;
    private final AttachmentDownloader downloader;

    public AttachmentManager(String accessToken) {
        this.fetcher = new AttachmentFetcher(accessToken);
        this.downloader = new AttachmentDownloader(accessToken);
    }

    public void downloadAllAttachments(String messageId, String downloadDirectory) throws Exception {
        List<Attachment> attachments = fetcher.fetchAttachments(messageId);

        if (attachments != null) {
            for (Attachment attachment : attachments) {
                String attachmentId = attachment.id;
                String fileName = attachment.name;
                if(fileName == null){
                  fileName = "unnamed_attachment"; //handles rare cases with no name
                }
                Path destinationPath = Paths.get(downloadDirectory, fileName);
                downloader.downloadAttachment(messageId, attachmentId, destinationPath.toString());
                System.out.println("Downloaded: " + fileName);
            }
        }
    }
}

```

**Commentary:**

*   This class encapsulates the `AttachmentFetcher` and `AttachmentDownloader` instances, making it easier to manage the overall process.
*   The `downloadAllAttachments` method first retrieves the attachment metadata using the `fetchAttachments` method.
*   It then iterates through the returned list of attachments.
*   For each attachment, the `attachment.id` and `attachment.name` properties are extracted. If the attachment has no name, the code adds an 'unnamed_attachment' identifier to avoid file writing errors.
*   A destination path is created using the specified download directory and the attachment's file name, using the NIO API to handle the path construction.
*   The `downloader.downloadAttachment` is called to download the attachment's content.
*   Simple print statements are included for feedback on each downloaded attachment. Error handling is handled via exceptions.

Regarding resources, the Microsoft Graph API documentation is the primary reference. It includes detailed information on each endpoint, authentication mechanisms, and usage examples. The official Java SDK GitHub repository also provides comprehensive examples and guidance on SDK-specific use, including dependency management with Maven or Gradle. Furthermore, understanding general OAuth 2.0 concepts is vital for effective authentication. Look for resources covering the authorization grant flow, refresh tokens, and security implications. Lastly, explore Java documentation related to file I/O, specifically classes like `InputStream`, `OutputStream`, `FileOutputStream`, and `BufferedInputStream`. This will ensure proper and efficient handling of the downloaded files.
