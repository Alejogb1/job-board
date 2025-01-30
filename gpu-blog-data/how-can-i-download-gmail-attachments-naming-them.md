---
title: "How can I download Gmail attachments, naming them after the email subject?"
date: "2025-01-30"
id: "how-can-i-download-gmail-attachments-naming-them"
---
Gmail’s API presents a structured approach to interacting with user email data, specifically circumventing the need for unreliable screen scraping methods when automating attachment downloads. I’ve built similar systems for document processing pipelines, and the key lies in leveraging the ‘users.messages.get’ endpoint combined with Base64 decoding for the message body and attachments. The challenge is often not in accessing the data, but in correctly structuring the API calls and handling potentially multipart messages.

First, accessing the message requires its unique ID. The Gmail API uses the ‘users.messages.list’ endpoint to search for emails matching specific criteria, such as a sender or subject keyword. This returns a list of message IDs. Once the desired message ID is obtained, the ‘users.messages.get’ endpoint, with the ‘format=raw’ parameter, fetches the full email content, including its encoded parts. Crucially, this raw format avoids potential parsing issues caused by Gmail’s default processing.

Decoding these parts requires understanding the structure of MIME messages. An email can have multiple parts, including the plain text body, HTML content, and attachments. These are typically separated by boundary strings within the raw message. Once the boundary markers are identified, attachments, which are themselves encoded in Base64, can be extracted. This process can be achieved programmatically using most modern languages, but handling different encoding schemes and attachment types is important. Furthermore, the API has rate limits. Implementations should be designed to accommodate these limitations with retry mechanisms and exponential backoff algorithms for stability.

Now, for specific examples:

**Example 1: Python with Google API Client Library**

```python
from googleapiclient.discovery import build
from google.oauth2 import service_account
import base64
import email
import os

# Replace with path to your service account JSON file
SERVICE_ACCOUNT_FILE = 'path/to/your/service_account.json'
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

def get_gmail_service():
    creds = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    return build('gmail', 'v1', credentials=creds)

def download_attachments(subject_keyword):
    service = get_gmail_service()
    results = service.users().messages().list(userId='me', q=f'subject:"{subject_keyword}"').execute()
    messages = results.get('messages', [])

    for message in messages:
        msg = service.users().messages().get(userId='me', id=message['id'], format='raw').execute()
        msg_str = base64.urlsafe_b64decode(msg['raw']).decode('utf-8')
        mime_msg = email.message_from_string(msg_str)

        for part in mime_msg.walk():
            if part.get_content_maintype() == 'multipart':
                continue
            if part.get('Content-Disposition') is None:
                continue
            filename = part.get_filename()
            if filename:
                subject_filename = mime_msg.get('Subject').replace("/", "-").replace(":", "-") # Sanitize subject
                att_filename = f"{subject_filename}_{filename}"
                file_data = part.get_payload(decode=True)
                if file_data:
                    try:
                        with open(att_filename, 'wb') as f:
                            f.write(file_data)
                        print(f"Downloaded: {att_filename}")
                    except IOError as e:
                        print(f"Error writing file: {e}")


if __name__ == '__main__':
    subject = "Your Subject Here"  # Modify the subject here.
    download_attachments(subject)
```
This Python example utilizes the Google API Client Library. It authenticates using a service account JSON file, searches for emails based on a subject keyword, fetches the raw email content, and then iterates through the MIME parts to locate attachments. The attachment's content is then written to a file. Importantly, the filename is constructed using the email subject. Error handling is included to manage potential file writing issues.

**Example 2: Node.js with Google APIs Library**

```javascript
const { google } = require('googleapis');
const { JWT } = require('google-auth-library');
const fs = require('fs').promises;
const base64 = require('js-base64').Base64;
const path = require('path');


async function downloadAttachments(subjectKeyword) {
    const keys = require('./path/to/your/service_account.json'); // Path to your service account json
    const client = new JWT({
        email: keys.client_email,
        key: keys.private_key,
        scopes: ['https://www.googleapis.com/auth/gmail.readonly'],
    });

    const gmail = google.gmail({ version: 'v1', auth: client });


    try{
        const response = await gmail.users.messages.list({
            userId: 'me',
            q: `subject:"${subjectKeyword}"`,
        });
        const messages = response.data.messages;

        if (messages && messages.length > 0) {
          for (const message of messages){
            const msgRes = await gmail.users.messages.get({
              userId: 'me',
              id: message.id,
              format: 'raw'
            })

            const raw_msg = base64.decode(msgRes.data.raw.replace(/-/g, '+').replace(/_/g, '/'));

            const parsed_message = await parseMIME(raw_msg)
            if(parsed_message.attachments && parsed_message.attachments.length > 0){
                const subject = parsed_message.subject.replace(/[/:]/g, '-'); // Sanitize subject
                for (const attachment of parsed_message.attachments){
                    const filename = `${subject}_${attachment.filename}`
                    try {
                      await fs.writeFile(filename, attachment.data, {encoding: 'binary'})
                      console.log(`Downloaded: ${filename}`);
                    } catch (error) {
                      console.error(`Error writing file: ${error}`)
                    }
                }
            }
           }
        }else {
          console.log('No messages found with that subject.')
        }
    } catch (error) {
        console.error(`An error occurred: ${error}`);
    }

}


async function parseMIME(raw_msg){
    const email = require('mailparser');
    const parsed = await email.simpleParser(raw_msg);
    return parsed;
}


downloadAttachments("Your Subject Here"); // Modify the subject here
```

This Node.js example employs the ‘googleapis’ library, using a service account for authentication. Similar to the Python example, it fetches messages based on a subject, obtains the raw message, and parses it using the mailparser library. This node example additionally sanitizes the subject and uses binary encoding for writing attachments. It demonstrates using asynchronous functions for managing file I/O operations and integrates a function to parse the MIME email structure.

**Example 3: Command-Line with Curl and jq**

```bash
#!/bin/bash

# Replace with your actual access token. Obtained through OAuth2.
ACCESS_TOKEN="your_access_token"
SUBJECT="Your Subject Here" # Modify subject here.

MESSAGE_IDS=$(curl -s -H "Authorization: Bearer ${ACCESS_TOKEN}" \
    "https://gmail.googleapis.com/gmail/v1/users/me/messages?q=subject:\"${SUBJECT}\"" \
     | jq -r '.messages[].id')

while read -r MESSAGE_ID; do
    RAW_MESSAGE=$(curl -s -H "Authorization: Bearer ${ACCESS_TOKEN}" \
        "https://gmail.googleapis.com/gmail/v1/users/me/messages/${MESSAGE_ID}?format=raw" \
        | jq -r '.raw' | sed 's/-/+/g' | sed 's/_/\//g' | base64 -d)

    SUBJECT_SAN=$(echo "$SUBJECT" | sed 's/[/:/]/_/g')

    echo "$RAW_MESSAGE" | grep 'Content-Disposition: attachment' -A 10000 | while read -r line
    do
    if [[ "$line" == *"filename="* ]]
    then
        FILENAME=$(echo "$line" | sed 's/.*filename="\(.*\)"/\1/' )
        FILENAME_SAN="${SUBJECT_SAN}_${FILENAME}"
        CONTENT_START=$(echo "$RAW_MESSAGE" | grep -n  "$line" | cut -d ":" -f1)
        CONTENT_START=$((CONTENT_START + 1))
        CONTENT_END=$(echo "$RAW_MESSAGE" | grep -n  '--boundary' -A 10000 | grep -n  "$line" -B 10000| grep -n -- '--boundary' | tail -n1 | cut -d ':' -f1)
        CONTENT_END=$((CONTENT_END -1))
        if [[ $CONTENT_END -gt 0 ]]
        then
         CONTENT_ENCODED=$(echo "$RAW_MESSAGE"| head -n  $CONTENT_END | tail -n $((CONTENT_END - CONTENT_START + 1)))
         echo "$CONTENT_ENCODED" | base64 -d > "$FILENAME_SAN"
         echo "Downloaded: $FILENAME_SAN"
        else
            echo "No attachment found"
        fi
    fi
  done
done <<< "$MESSAGE_IDS"
```

This Bash script demonstrates a command-line approach using curl and jq for JSON parsing.  Authentication uses an access token which needs to be manually obtained. It retrieves the raw message and then uses grep to parse the attachment sections by looking for the 'Content-Disposition: attachment' header. It encodes the attachment data, and uses base64 to decode into the intended file. This method showcases a different approach to handling attachments, focusing on a shell-based workflow suitable for simple automation. Note this is more sensitive to changes in the formatting of email messages than the other options. It also includes more fragile boundary detection based on the assumption of a single boundary.

In summary, each approach requires a varying level of complexity and dependencies. The Python example offers readability with a more robust API interaction. The Node.js example introduces a asynchronous execution model. The shell script approach provides a flexible command-line alternative suitable for a Linux environment. When selecting an approach, it’s critical to consider project requirements, familiarity with each language or technology, and desired level of complexity.

Further investigation into email parsing libraries beyond the specific examples shown here is essential. These resources often streamline the process of handling multipart messages, especially when dealing with diverse email structures. Security best practices for API key management and handling potentially sensitive email data should also be strictly adhered to.
