---
title: "How can I retrieve the MIME content of nested messages via the Graph REST API?"
date: "2024-12-23"
id: "how-can-i-retrieve-the-mime-content-of-nested-messages-via-the-graph-rest-api"
---

Alright,  I remember a particularly frustrating project back in my days at *Acme Corp* where we had to pull apart email threads with intricate nesting. We were migrating a colossal email archive, and the Graph API was our primary access point. Dealing with those nested MIME structures proved, let’s say, challenging initially. The key isn't just hitting the right endpoints; it’s understanding how the Graph API represents complex MIME data and the specific properties you need to query. Let's break it down.

The Graph API, when dealing with messages, doesn't automatically expose every single part of a MIME structure in an immediately usable format. For a simple message, you can readily fetch the `body` property. However, when it comes to nested messages, which often appear as attachments of type `message/rfc822`, you won’t find a straightforward "nestedMIME" property. Instead, these nested messages are represented as attachments, and their content needs to be retrieved separately. The trick is iterative fetching and a good understanding of what properties are available and what they represent.

First things first, you need to initially retrieve the message containing the nested data. A typical Graph API request to do this might look like:

```http
GET https://graph.microsoft.com/v1.0/me/messages/{message-id}?$select=id,hasAttachments,attachments
```

This will return the message details, crucially including an array of attachments, if any exist. The important part here is `hasAttachments` and the actual `attachments` array. You will need to inspect each attachment to determine if it’s a `message/rfc822` type. This type indicates a nested message.

Now, let’s assume one of your attachments looks something like this in the response:

```json
{
    "@odata.type": "#microsoft.graph.fileAttachment",
    "id": "AAMkAGUyY2M0ODg2LTdkMzYtNDRjNS1hN2I3LWIwNTVkMzIxNWI2OAAuAAAAAABbT1qYx4q5Q479KxTqE4zSBwB70x0t517kQyR_I79q_v0YAAAAAAEGAAAACv7BwZt05SEh4cR5X-920AACXoE0wAAA=",
    "lastModifiedDateTime": "2024-11-04T12:12:12Z",
    "name": "ForwardedMessage.eml",
    "contentType": "message/rfc822",
    "size": 5678,
    "isInline": false
}
```

Notice the `"contentType": "message/rfc822"`. This signifies a nested email. To get the full MIME content, you need to make another request, this time targeting the attachment itself. We will use the attachment ID. The request looks like this:

```http
GET https://graph.microsoft.com/v1.0/me/messages/{message-id}/attachments/{attachment-id}/$value
```

The key part here is the `$value` at the end of the URL. This instructs the API to return the raw content of the attachment, which in our case, is the MIME content of the nested message. This content will be base64-encoded and it is your responsibility to decode it to access the raw content.

Now, let's look at some code examples. I’ll provide three snippets, each using a different approach/language, to demonstrate how this can be done in practice. Keep in mind you'll need to have authenticated against the graph API prior to running these.

First, here's a Python snippet using the `requests` library:

```python
import requests
import base64
import json

def get_nested_mime(message_id, access_token):
    headers = {'Authorization': f'Bearer {access_token}'}
    url = f"https://graph.microsoft.com/v1.0/me/messages/{message_id}?$select=id,hasAttachments,attachments"
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    message_data = response.json()
    if not message_data.get('hasAttachments'):
        return None # No attachments to process

    nested_mime_content = []
    for attachment in message_data['attachments']:
        if attachment.get('contentType') == 'message/rfc822':
            attachment_id = attachment['id']
            attachment_url = f"https://graph.microsoft.com/v1.0/me/messages/{message_id}/attachments/{attachment_id}/$value"
            attachment_response = requests.get(attachment_url, headers=headers)
            attachment_response.raise_for_status()
            mime_content_bytes = base64.b64decode(attachment_response.content)
            mime_content_str = mime_content_bytes.decode('utf-8', 'ignore')
            nested_mime_content.append(mime_content_str)
    return nested_mime_content
# Example usage:
# message_id = "YOUR_MESSAGE_ID"
# access_token = "YOUR_ACCESS_TOKEN"
# nested_content = get_nested_mime(message_id, access_token)
# if nested_content:
#     print("Nested MIME Content:", nested_content)
```

This script first gets the message details, and if attachments are present, it checks each one for the `message/rfc822` type. If found, it retrieves the content using a second request. The content is decoded from base64 and returned as a string.

Second, a JavaScript example (Node.js using `node-fetch`):

```javascript
const fetch = require('node-fetch');

async function getNestedMime(messageId, accessToken) {
    const headers = { 'Authorization': `Bearer ${accessToken}` };
    let url = `https://graph.microsoft.com/v1.0/me/messages/${messageId}?$select=id,hasAttachments,attachments`;

    const response = await fetch(url, { headers });
    response.ok ? null : console.error(`Failed to fetch message: ${response.status} ${response.statusText}`);

    if(!response.ok) return;
    const messageData = await response.json();

    if (!messageData.hasAttachments) {
      return null; // No attachments to process
    }
    
    const nestedMimeContent = [];

    for(let attachment of messageData.attachments){
        if(attachment.contentType === 'message/rfc822') {
            let attachmentUrl = `https://graph.microsoft.com/v1.0/me/messages/${messageId}/attachments/${attachment.id}/$value`;
            let attachmentResponse = await fetch(attachmentUrl, { headers });

            if (!attachmentResponse.ok) {
                console.error(`Failed to fetch attachment: ${attachmentResponse.status} ${attachmentResponse.statusText}`);
                continue;
            }
            const mimeContent = await attachmentResponse.text();
            const decodedMime = Buffer.from(mimeContent, 'base64').toString('utf-8');
            nestedMimeContent.push(decodedMime);
        }
    }
    return nestedMimeContent;
}
// Example usage:
// getNestedMime("YOUR_MESSAGE_ID", "YOUR_ACCESS_TOKEN")
// .then(nested_content => {
//    if(nested_content) console.log("Nested MIME Content:", nested_content);
// });
```
This JavaScript code follows a similar pattern but uses async/await and `node-fetch`. It's also important to note the use of Buffer here when decoding from base64.

Finally, a C# snippet using the Microsoft.Graph SDK:

```csharp
using Microsoft.Graph;
using Microsoft.Identity.Client;
using System.Text;
using System.Text.Json;

public class GraphHelper
{
    private GraphServiceClient _graphClient;

    public GraphHelper(string clientId, string tenantId, string clientSecret)
    {
        IConfidentialClientApplication app = ConfidentialClientApplicationBuilder
            .Create(clientId)
            .WithClientSecret(clientSecret)
            .WithAuthority(AzureCloudInstance.AzurePublic, tenantId)
            .Build();

        var scopes = new[] { "https://graph.microsoft.com/.default" };

        var authProvider = new ClientCredentialProvider(app, scopes);
        _graphClient = new GraphServiceClient(authProvider);
    }
    public async Task<List<string>> GetNestedMime(string messageId)
    {
        var message = await _graphClient.Me.Messages[messageId].Request()
            .Select(m => new { m.Id, m.HasAttachments, m.Attachments })
            .GetAsync();

         if (!message.HasAttachments.GetValueOrDefault()) {
             return null;
         }

         var nestedMimeContent = new List<string>();

        foreach (var attachment in message.Attachments)
        {
            if (attachment.ContentType == "message/rfc822")
            {
                var stream = await _graphClient.Me.Messages[messageId].Attachments[attachment.Id].Content.Request().GetAsync();
                using (var reader = new System.IO.StreamReader(stream, Encoding.UTF8))
                {
                     string base64Content = await reader.ReadToEndAsync();
                    var mimeContentBytes = Convert.FromBase64String(base64Content);
                    string mimeContent = Encoding.UTF8.GetString(mimeContentBytes);
                    nestedMimeContent.Add(mimeContent);
                }
             }
        }
      return nestedMimeContent;

    }
   //Example Usage:
   //var graphHelper = new GraphHelper("YOUR_CLIENT_ID", "YOUR_TENANT_ID", "YOUR_CLIENT_SECRET");
   //var mimeContent = await graphHelper.GetNestedMime("YOUR_MESSAGE_ID");
   //if (mimeContent != null)
   //{
   //  foreach(var content in mimeContent) {
   //   Console.WriteLine($"Nested MIME content: {content}");
   //  }
   //}
}
```

Here, I use the C# SDK directly, which streamlines the process and handles a lot of the underlying complexities. You'll need to handle the auth setup, which I've outlined for completeness. I have included how to create a client using the client secret flow.

For further exploration, I recommend looking into authoritative sources. First, the official Microsoft Graph documentation is indispensable. You can start with the “Get message” endpoint documentation and then branch off into the “Get attachment” and the concepts around handling MIME data. The book "Programming Microsoft Graph" by Marc LaFleur is also an excellent resource, especially its chapters that delve into the handling of complex data structures and attachments. These resources will solidify your understanding of the API and its intricacies.

Remember, dealing with nested MIME content is an exercise in iterative data retrieval. Start with the message, inspect for attachments, and then retrieve the content of those identified as nested messages. Decoding base64 content will be necessary, as demonstrated. These code snippets should provide a starting point for your project, but feel free to reach out if any additional guidance is required. Good luck!
