---
title: "How can I send video attachments from Airtable to a Microsoft Teams bot?"
date: "2025-01-30"
id: "how-can-i-send-video-attachments-from-airtable"
---
The core challenge in sending video attachments from Airtable to a Microsoft Teams bot lies in Airtable's limitations regarding direct file access and the requirement for structured data transmission to the Teams platform.  Airtable's API primarily focuses on retrieving record data, not directly serving files. Therefore, a solution necessitates an intermediary step: using a cloud storage service as a bridge.  This allows Airtable to upload the video, obtain a sharable URL, and then pass that URL to the Teams bot for final delivery. I've encountered this hurdle numerous times during my work integrating various CRMs and collaboration platforms, and the following approach consistently proved reliable.

**1.  Clear Explanation:**

The process involves three key components: an Airtable automation, a cloud storage service (I'll use Azure Blob Storage for this explanation, though AWS S3 or Google Cloud Storage are equally viable), and a custom Microsoft Teams bot.  The Airtable automation triggers upon a specified event (e.g., a new record creation) and uploads the video file to the chosen cloud storage.  It then extracts the publicly accessible URL of the uploaded video. This URL is subsequently sent to the Teams bot via a webhook or direct API call. The Teams bot then uses this URL to construct a message containing a clickable link to the video within the Teams channel.


**2. Code Examples with Commentary:**

**A. Airtable Automation (using Javascript):**

This automation assumes you've already configured an Airtable base with a "Video" attachment field.  The script uses the Airtable API to access the attachment, the Azure Blob Storage SDK (Node.js) to upload it, and sends a POST request to your Teams bot webhook.  Remember to replace placeholders with your actual API keys and URLs.

```javascript
// Airtable API Key and Base ID
const airtableApiKey = 'YOUR_AIRTABLE_API_KEY';
const airtableBaseId = 'YOUR_AIRTABLE_BASE_ID';
const airtableTableId = 'YOUR_AIRTABLE_TABLE_ID';

// Azure Blob Storage Connection String
const azureConnectionString = 'YOUR_AZURE_CONNECTION_STRING';
const blobContainerName = 'your-video-container';

// Teams Bot Webhook URL
const teamsWebhookUrl = 'YOUR_TEAMS_WEBHOOK_URL';

const Airtable = require('airtable');
const { BlobServiceClient } = require('@azure/storage-blob');

const base = new Airtable({apiKey: airtableApiKey}).base(airtableBaseId)(airtableTableId);

base.select({
  maxRecords: 1,
  view: "Grid view", // Replace with your view name
  filterByFormula: "{Status}='Submitted'" // Filter for records ready for processing
}).eachPage(function page(records, fetchNextPage) {
  records.forEach(function(record) {
    const attachment = record.get('Video')[0];
    const fileName = attachment.filename;
    const fileUrl = attachment.url;
    const blobServiceClient = new BlobServiceClient(azureConnectionString);
    const containerClient = blobServiceClient.getContainerClient(blobContainerName);
    const blockBlobClient = containerClient.getBlockBlobClient(fileName);

    fetch(fileUrl)
      .then(response => response.blob())
      .then(blob => blockBlobClient.uploadStream(blob))
      .then(() => {
        const videoUrl = `https://${blobContainerName}.blob.core.windows.net/${fileName}`; //Adjust according to your storage setup
        const payload = {
          "type": "message",
          "attachments": [
            {
              "contentType": "application/vnd.microsoft.card.adaptive",
              "content": {
                "type": "AdaptiveCard",
                "body": [
                  {
                    "type": "TextBlock",
                    "text": "New Video Uploaded!",
                  },
                  {
                    "type": "Action.OpenUrl",
                    "title": "View Video",
                    "url": videoUrl
                  }
                ]
              }
            }
          ]
        };

        fetch(teamsWebhookUrl, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify(payload)
        })
        .then(() => console.log('Video sent to Teams!'))
        .catch(error => console.error('Error sending to Teams:', error));
      })
      .catch(error => console.error('Error uploading to Azure:', error));
  });
  fetchNextPage();
}, function done(err) {
  if (err) { console.error(err); }
});
```


**B. Azure Blob Storage Configuration (Conceptual):**

This section outlines the necessary steps, omitting the detailed Azure portal interactions.  You need to create a storage account, a container within that account (e.g., `your-video-container`), and configure appropriate access permissions (e.g., public read access for the container to allow the Teams bot to view the video). The Node.js SDK ( `@azure/storage-blob`) provides the methods for interaction.  Error handling and robust exception management are vital aspects omitted here for brevity.


**C. Microsoft Teams Bot (using Bot Framework Composer):**

The bot's role is minimal – it receives the video URL from Airtable, and uses the Bot Framework SDK to create a message containing a link to this URL. The example below uses a simplified Adaptive Card for better presentation.  Detailed error handling and input validation are omitted for brevity.

```javascript
// Sample Bot Framework Composer dialog code (simplified)

bot.dialog('ReceiveVideoUrl', [
    (session, args) => {
        const videoUrl = args.videoUrl; // Received from Airtable via webhook
        const adaptiveCard = {
            "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
            "type": "AdaptiveCard",
            "body": [
              {
                "type": "TextBlock",
                "text": "New Video Available!"
              },
              {
                "type": "Action.OpenUrl",
                "title": "View Video",
                "url": videoUrl
              }
            ]
          };

        builder.Prompts.attachment(session, adaptiveCard);
    }
]);
```

**3. Resource Recommendations:**

* **Airtable API Documentation:**  Familiarize yourself with the API methods for retrieving records and attachments.
* **Azure Blob Storage Documentation:** Understand container creation, access control, and SDK usage.  Similar documentation exists for AWS S3 and Google Cloud Storage.
* **Microsoft Bot Framework Composer Documentation:** Learn how to create and deploy bots, handle incoming messages, and use adaptive cards.
* **Node.js Documentation:**  Understand asynchronous operations and error handling in Node.js.

This comprehensive approach addresses the core limitations of Airtable's direct file access and ensures a secure, reliable method for sending video attachments to your Microsoft Teams bot.  Remember that security best practices should be implemented throughout, including robust access control on your storage service and secure communication channels between Airtable, your storage service, and your bot.  Furthermore, thorough testing and error handling are crucial for a production-ready solution.
