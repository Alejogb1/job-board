---
title: "How can I save emails as drafts using Python and Gmail's IMAP?"
date: "2025-01-30"
id: "how-can-i-save-emails-as-drafts-using"
---
The IMAP protocol's `APPEND` command, specifically when used with the `\Draft` flag, provides the mechanism for saving emails as drafts within a Gmail mailbox. This functionality circumvents the need for sending the email immediately and allows for iterative composition and review.

My experience managing email processing pipelines for a small organization showed that draft creation through IMAP was essential for situations where an email required multiple stages of approval or content refinement before dispatch. Direct use of the Gmail API for drafts, while possible, often introduced greater complexity when operating within the constraints of existing IMAP-based infrastructure. Therefore, leveraging IMAP’s native capabilities proved more efficient.

The foundational requirement for saving a draft is establishing an IMAP connection and then correctly formatting the message. An email, in its raw form, includes headers defining characteristics like sender, recipient, subject, and content type, followed by the actual message body. When saving a draft, these headers need to be constructed and appended appropriately. Critically, you must include the `\Draft` flag in the `APPEND` command, indicating that this message is to be stored in the drafts folder. Without this flag, the email would be appended as a standard message to the currently selected mailbox, which is generally undesirable. Gmail does not inherently support a 'create draft' command separate from appending an email with the draft flag; it interprets it within this construct of appending.

Here is my first example demonstrating this process:

```python
import imaplib
import email
from email.mime.text import MIMEText
from email.utils import formatdate

def save_draft_imap(server, username, password, to_addr, subject, body):
    try:
        mail = imaplib.IMAP4_SSL(server)
        mail.login(username, password)
        mail.select("INBOX") # or '[Gmail]/Drafts'
        # Create a MIMEText object for the email body
        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = username
        msg['To'] = to_addr
        msg['Date'] = formatdate(localtime=True)

        # Convert message to a string, including headers
        raw_email = msg.as_string()
        # Note the \Draft flag here
        mail.append('[Gmail]/Drafts', '\\Draft', imaplib.Time2Internaldate(email.utils.localtime()), raw_email.encode('utf-8'))
        mail.close()
        mail.logout()
        print("Draft saved successfully.")

    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage:
if __name__ == '__main__':
    server = "imap.gmail.com"
    username = "your_email@gmail.com"
    password = "your_app_password" # Using app password, not main password
    to_addr = "recipient@example.com"
    subject = "Test Draft via IMAP"
    body = "This is the body of the draft email."
    save_draft_imap(server, username, password, to_addr, subject, body)
```

This code begins by connecting to the Gmail IMAP server using SSL. The `login` method authenticates the user, and we then select the `INBOX`. While drafts can be directly added to the `[Gmail]/Drafts` folder, appending to inbox first then moving/copying offers a more predictable behavior. I have opted to use the draft folder directly here for brevity, which works reliably given the `\Draft` flag's presence. Then, a `MIMEText` object is created and populated with message details and crucial headers like `Subject`, `From`, `To`, and `Date`, which are essential for properly formatted emails. The `as_string()` method transforms this structured object into a complete email string. Critically, this string must be byte-encoded before the `APPEND` command, so `raw_email.encode('utf-8')` is used. The `\Draft` flag is specified in the `APPEND` call, guaranteeing the message is saved as a draft. Finally, the connection is closed.

A common situation encountered involved adding attachments to drafts. The `email` module requires a multipart MIME structure for this. Example two illustrates the process:

```python
import imaplib
import email
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from email.utils import formatdate

import os

def save_draft_with_attachment_imap(server, username, password, to_addr, subject, body, attachment_path):
    try:
        mail = imaplib.IMAP4_SSL(server)
        mail.login(username, password)
        mail.select("INBOX") # or '[Gmail]/Drafts'

        msg = MIMEMultipart()
        msg['Subject'] = subject
        msg['From'] = username
        msg['To'] = to_addr
        msg['Date'] = formatdate(localtime=True)

        msg.attach(MIMEText(body))

        if attachment_path and os.path.exists(attachment_path):
             with open(attachment_path, "rb") as attachment_file:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(attachment_file.read())
                encoders.encode_base64(part)
                part.add_header('Content-Disposition', f'attachment; filename="{os.path.basename(attachment_path)}"')
                msg.attach(part)
        
        raw_email = msg.as_string()
        mail.append('[Gmail]/Drafts', '\\Draft', imaplib.Time2Internaldate(email.utils.localtime()), raw_email.encode('utf-8'))
        mail.close()
        mail.logout()
        print("Draft with attachment saved successfully.")

    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage:
if __name__ == '__main__':
    server = "imap.gmail.com"
    username = "your_email@gmail.com"
    password = "your_app_password"
    to_addr = "recipient@example.com"
    subject = "Test Draft with Attachment via IMAP"
    body = "This is the body of the draft email with attachment."
    attachment_path = "example.txt" # Replace with an actual file
    # Create an example file if it does not exist
    if not os.path.exists(attachment_path):
        with open(attachment_path, 'w') as f:
            f.write("This is an example attachment")

    save_draft_with_attachment_imap(server, username, password, to_addr, subject, body, attachment_path)
```

In this code, the core structure is similar to the first example, but it utilizes `MIMEMultipart` to create a message container. The text body is attached using `MIMEText`. The attachment handling uses `MIMEBase` to encode the file content using base64 encoding, necessary for transferring binary data through email. We must set the `Content-Disposition` header to indicate this is an attachment and also specify a filename. This multipart message ensures both text and file data are correctly interpreted. Again, we encode this as a string before appending it to the drafts folder with the draft flag.  The file is read in binary mode because attachments are often non-textual.

Beyond text and attachments, emails may include HTML formatting. Example three shows this implementation:

```python
import imaplib
import email
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import formatdate

def save_draft_html_imap(server, username, password, to_addr, subject, html_body):
    try:
        mail = imaplib.IMAP4_SSL(server)
        mail.login(username, password)
        mail.select("INBOX")

        msg = MIMEMultipart('alternative') # Important for HTML and plain-text alternatives
        msg['Subject'] = subject
        msg['From'] = username
        msg['To'] = to_addr
        msg['Date'] = formatdate(localtime=True)

        # Create a plain text version for email clients that don't support HTML
        text_part = MIMEText("Please view this email with HTML enabled.", 'plain')

        # Add HTML content
        html_part = MIMEText(html_body, 'html')

        msg.attach(text_part)
        msg.attach(html_part)

        raw_email = msg.as_string()
        mail.append('[Gmail]/Drafts', '\\Draft', imaplib.Time2Internaldate(email.utils.localtime()), raw_email.encode('utf-8'))
        mail.close()
        mail.logout()
        print("HTML draft saved successfully.")

    except Exception as e:
         print(f"An error occurred: {e}")

# Example usage:
if __name__ == '__main__':
    server = "imap.gmail.com"
    username = "your_email@gmail.com"
    password = "your_app_password"
    to_addr = "recipient@example.com"
    subject = "Test HTML Draft via IMAP"
    html_body = """
        <html>
          <head></head>
          <body>
             <h1>HTML Draft Example</h1>
            <p>This email is formatted using <b>HTML</b>.</p>
          </body>
        </html>
        """
    save_draft_html_imap(server, username, password, to_addr, subject, html_body)
```

In this instance, a `MIMEMultipart` message is again utilized but initialized as an `alternative` structure. This allows both plain text and HTML variants of the message to be included. The plain text part is presented to those clients that do not support HTML rendering, providing a fallback and improving accessibility.  The HTML content is added as a separate part with the content type set to ‘html’. This inclusion of both versions increases the robustness of the email across various clients. Again, this is encoded as a string and appended to the drafts folder with the draft flag.

In practice, it is important to consider security best practices such as storing credentials securely and utilizing app passwords instead of the primary account password for IMAP access, as demonstrated in these examples. When troubleshooting issues, logging IMAP server responses can provide valuable diagnostics, and this can be enabled using the IMAP library. Additionally, rate limiting your requests can help prevent temporary blocks from Gmail’s servers.

For further study, I suggest focusing on the official documentation for the Python `email` module, which provides extensive detail on MIME structures and how to assemble emails programmatically. Also, RFC 3501, which defines the IMAP protocol, is highly recommended for understanding the intricacies of IMAP commands, such as the `APPEND` command, and its various flags and arguments. Studying MIME standards will also significantly enhance the ability to construct complex emails correctly, including managing different encodings and attachment types. Finally, familiarizing oneself with the specific implementation details in Gmail is critical as Gmail does not strictly follow all IMAP protocol standards.
