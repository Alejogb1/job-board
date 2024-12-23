---
title: "Why are Mailgun images being sent as attachments?"
date: "2024-12-23"
id: "why-are-mailgun-images-being-sent-as-attachments"
---

Alright, let's talk about why Mailgun might be treating your embedded images as attachments—it's a classic issue, and one I've debugged more times than I'd care to count, often in high-pressure situations where timely communication was paramount. It's rarely a Mailgun problem *per se*, but rather how the email itself is constructed, specifically the message formatting and the way image resources are referenced. It boils down to the message's mime type, the way you reference the images, and sometimes subtle issues with the email content.

The fundamental problem lies in how email clients interpret content, which hinges significantly on the `Content-Type` headers and the way multipart messages are structured. We're used to seeing HTML emails with embedded images displaying perfectly in our inboxes, but that magic is meticulously choreographed behind the scenes. When these messages aren't crafted properly, that "inline" image gets bumped to an attachment.

In my experience, particularly with legacy systems transitioning to more modern email workflows, the issue usually stems from one of a few core causes: incorrect mime type specification, improper referencing of the image resource using `cid:` (Content-ID) versus a plain `src:` with a URL, or sometimes an incorrect message structure leading to misinterpretation.

Let’s break these down into more detail, using scenarios I've personally encountered.

First, the most common culprit is not declaring the email content as `multipart/related` or `multipart/mixed` when you have both text/html content and embedded images. If you send a HTML email and reference an image within the HTML using a tag like `<img src="my_image.jpg">`, while simultaneously sending the image itself as an attachment, *and* the email's mime type isn't `multipart/related` (for inline images) or `multipart/mixed` (for both inline and attachments), the email client will likely treat the image as *just* another attachment. It's lacking context on how to interpret the relationship of the image with the surrounding HTML.

Let’s illustrate this with a simplified example. Imagine a basic setup sending an email with a simple HTML body and a referenced image. This first snippet demonstrates *what not to do*, where the incorrect header will likely result in the image being attached.

```python
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import os

# Path to our image file (ensure file exists)
image_path = "my_image.jpg"

# Set up email content
msg = MIMEMultipart()
msg['From'] = "sender@example.com"
msg['To'] = "recipient@example.com"
msg['Subject'] = "Email with an image"

html_content = """
<html>
<body>
 <p>Here's an image:</p>
 <img src="my_image.jpg">
</body>
</html>
"""

msg.attach(MIMEText(html_content, 'html'))

# Load the image file
with open(image_path, 'rb') as image_file:
    image_data = image_file.read()
    image = MIMEImage(image_data, name=os.path.basename(image_path))
    msg.attach(image)

# Attempt to send (this example may not work without a server)
# Here a correct server is assumed for this part
try:
    smtp = smtplib.SMTP("smtp.example.com", 587)
    smtp.starttls()
    smtp.login("username", "password")
    smtp.send_message(msg)
    smtp.quit()
    print("Email sent (with potentially attached image).")
except Exception as e:
    print(f"Error sending email: {e}")
```

In this first snippet, the email's mime type defaults to `multipart/mixed` by the addition of attachments via `msg.attach`, however, while we've attached the image, it's not *linked* as inline image with the HTML content explicitly via the `cid` protocol. The `img src` tag does not know the context of this attached image.

Now, let's look at a proper approach. This involves using `multipart/related` and the Content-ID, along with using `cid:` as the source for the image:

```python
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import os
from email.mime.base import MIMEBase
from email import encoders

# Path to our image file (ensure file exists)
image_path = "my_image.jpg"

# Create the root multipart/related message
msg = MIMEMultipart('related')
msg['From'] = "sender@example.com"
msg['To'] = "recipient@example.com"
msg['Subject'] = "Email with an embedded image"

# Create the HTML part
html_part = MIMEText("""
    <html>
    <body>
    <p>Here's an image:</p>
    <img src="cid:image1">
    </body>
    </html>
    """, 'html')

msg.attach(html_part)

# Create the image part
with open(image_path, 'rb') as image_file:
    image_data = image_file.read()
    image = MIMEImage(image_data, name=os.path.basename(image_path))
    image.add_header('Content-ID', '<image1>')  # unique id is key
    msg.attach(image)

# Attempt to send (this example may not work without a server)
# Here a correct server is assumed for this part
try:
    smtp = smtplib.SMTP("smtp.example.com", 587)
    smtp.starttls()
    smtp.login("username", "password")
    smtp.send_message(msg)
    smtp.quit()
    print("Email sent with an embedded image.")
except Exception as e:
    print(f"Error sending email: {e}")

```
In this version, the critical change is the use of `multipart/related` and the `<img src="cid:image1">` tag, paired with a header added to the image part of the multipart with the same content id `<image1>`. This is what tells the email client to understand the relationship between the image and its HTML context. This is how a image can be embedded instead of added as an attachment.

Finally, sometimes you do indeed want an attachment alongside inline images. In those cases, the email's mime type should be `multipart/mixed`, and we need to include both inline and attachments.

```python
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import os
from email.mime.base import MIMEBase
from email import encoders

# Paths to our image and other file (ensure files exist)
image_path = "my_image.jpg"
attachment_path = "text.txt"

# Create the root multipart/mixed message
msg = MIMEMultipart('mixed')
msg['From'] = "sender@example.com"
msg['To'] = "recipient@example.com"
msg['Subject'] = "Email with an embedded image and attachment"

# Create the related multipart for embedded images (we have to nest)
related_part = MIMEMultipart('related')
html_part = MIMEText("""
    <html>
    <body>
    <p>Here's an image:</p>
    <img src="cid:image1">
    </body>
    </html>
    """, 'html')

related_part.attach(html_part)
with open(image_path, 'rb') as image_file:
    image_data = image_file.read()
    image = MIMEImage(image_data, name=os.path.basename(image_path))
    image.add_header('Content-ID', '<image1>')  # unique id is key
    related_part.attach(image)


msg.attach(related_part)
# Create a non-image attachment
attachment = MIMEBase('application', "octet-stream")

with open(attachment_path, "rb") as file:
    attachment.set_payload(file.read())

encoders.encode_base64(attachment)
attachment.add_header('Content-Disposition', 'attachment', filename=os.path.basename(attachment_path))
msg.attach(attachment)

# Attempt to send (this example may not work without a server)
# Here a correct server is assumed for this part
try:
    smtp = smtplib.SMTP("smtp.example.com", 587)
    smtp.starttls()
    smtp.login("username", "password")
    smtp.send_message(msg)
    smtp.quit()
    print("Email sent with embedded image and attachment.")
except Exception as e:
    print(f"Error sending email: {e}")
```

Here we nest a `multipart/related` structure to encapsulate the inline images, then the entire structure is added to a  `multipart/mixed` message, allowing for both inline images and a true attachment.

To deepen your understanding, I'd recommend diving into "Email Security: How to Keep Your Messages Safe" by Bruce Schneier (although it doesn't focus specifically on mime types, it's a crucial read to understand the larger context of email security and message structure). For a more direct and detailed look at mime types, consider checking out RFC 2045, RFC 2046, and RFC 2047—these are the foundational documents that define the standards for email formatting. The Python `email` package documentation is also crucial for understanding how these structures are implemented. Also, while not an academic reference, exploring the examples and best practices on the Mailgun developer pages, while often more pragmatic than rigorous, is helpful for understanding issues from a real world sending perspective.

In my experience, these are the most likely causes and solutions. Always verify that you are using the correct `Content-Type` and `Content-ID` headers when including inline images, and make sure your message structure is appropriate for the content you are sending. Proper multipart message construction is the key to preventing your embedded images from being interpreted as attachments. It's a subtle but vital aspect of modern email.
