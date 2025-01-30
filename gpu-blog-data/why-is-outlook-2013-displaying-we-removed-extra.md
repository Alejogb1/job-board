---
title: "Why is Outlook 2013 displaying 'We removed extra line breaks' when receiving messages sent with a '.' from Unix?"
date: "2025-01-30"
id: "why-is-outlook-2013-displaying-we-removed-extra"
---
The observed behavior in Outlook 2013, where the message "We removed extra line breaks" appears upon receiving emails containing a single "." character sent from a Unix-based system, stems from a fundamental incompatibility between the line-ending conventions used by these two operating systems.  This isn't a bug within Outlook 2013 per se, but rather a consequence of how different systems handle newline characters.

My experience working with email server configurations and troubleshooting client-side display issues for over a decade has provided ample exposure to this specific problem.  The core issue lies in the differing newline character representations. Unix-like systems (including Linux and macOS) typically use a single line feed character (`\n`) to denote a new line.  Conversely, Windows systems, including those running Outlook 2013, traditionally employ a carriage return and line feed sequence (`\r\n`).  When a Unix system sends an email containing only a ".", followed by a `\n`, Outlook interprets this as an incomplete line ending.  It then proceeds to "correct" the perceived formatting error by removing the perceived "extra" line break (`\n`), thereby triggering the warning message.

This behavior is not limited to a single "." character; any single character followed by only a `\n` will likely provoke the same response.  The warning isn't an error in the strictest sense but rather an indication that Outlook has performed an automatic line-ending conversion.  While this conversion generally aims for improved readability, it highlights a potential issue in the email's encoding and transport.  Proper handling necessitates ensuring consistent line endings throughout the email composition and transmission process.

Let's examine this through several code examples.  These examples demonstrate how the line endings are handled differently in various contexts and how they might lead to the observed outcome.  I’ll utilize Python for its concise syntax and wide availability.


**Example 1: Sending an email from a Unix-like system without proper line ending conversion.**

```python
import smtplib
from email.mime.text import MIMEText

msg = MIMEText(".")
msg['Subject'] = 'Test Email'
msg['From'] = 'sender@example.com'
msg['To'] = 'recipient@example.com'

with smtplib.SMTP('localhost') as server:  # Replace with your SMTP server details
    server.send_message(msg)
```

In this example, the `MIMEText` object, although creating a message body containing only ".", doesn't inherently specify the line ending.  The default behavior of the `smtplib` library on a Unix-like system will likely use `\n`, leading to the line-ending conflict within Outlook.


**Example 2:  Sending the email with explicit line ending conversion using Python's `smtplib` library (not recommended for robustness).**

```python
import smtplib
from email.mime.text import MIMEText

msg = MIMEText(".\r\n") # Explicitly adding \r\n
msg['Subject'] = 'Test Email'
msg['From'] = 'sender@example.com'
msg['To'] = 'recipient@example.com'

with smtplib.SMTP('localhost') as server:  # Replace with your SMTP server details
    server.send_message(msg)
```

This example forces the line ending to `\r\n`, which is expected by Outlook.  However, relying solely on this method is fragile; it assumes the receiving mail server and client correctly handle `\r\n` and doesn't address potential issues with other mail clients.  A more robust approach is discussed below.


**Example 3:  Illustrating the line-ending difference and utilizing a more robust approach with a dedicated email library.**

This example, focusing on a more sophisticated email library, allows for greater control over message encoding and line endings, improving cross-platform compatibility. While the details of the `yagmail` library are beyond this immediate scope, the principle remains the same: explicit control over encoding ensures the integrity of line endings during transmission.

```python
#This example requires the 'yagmail' library
import yagmail

yag = yagmail.SMTP('sender@example.com', 'password') #Replace with your SMTP credentials

contents = ['.']
yag.send(to='recipient@example.com', subject='Test Email', contents=contents, encoding='utf-8')
```

Here, we use a library explicitly designed for email sending, handling encoding and potentially MIME type issues in a more robust way than the basic `smtplib`.  Proper encoding prevents character encoding misinterpretations, improving reliability.  While the encoding doesn't directly solve the newline issue, it's an important aspect of preventing related communication problems.


Addressing the root cause effectively requires a multi-pronged strategy:

1. **Server-side Configuration:** Ensure your mail server correctly handles line endings.  Many mail servers allow for configuration options to standardize line endings for outgoing messages, converting `\n` to `\r\n` regardless of the originating system.  Consult your mail server's documentation for appropriate settings.

2. **Client-side Handling (If feasible):** If server-side modification is not an option, consider client-side pre-processing of messages before sending.  This involves explicitly adding `\r\n` before sending.  However, this is generally less desirable due to potential for compatibility problems.

3. **Email Library Usage:**  Utilize well-maintained email libraries (such as the `yagmail` example demonstrated above) which offer higher-level abstractions and handle encoding and line-endings more reliably.

4. **Header Inspection:** Examine the email headers.  Inconsistencies or missing headers pertaining to character encoding might indirectly influence Outlook's interpretation of line breaks.

In conclusion, the Outlook 2013 "We removed extra line breaks" warning arises from a mismatch in newline character conventions.  While a quick fix like adding `\r\n` might appear sufficient,  a thorough approach involving server-side configuration or sophisticated email libraries is recommended for long-term stability and cross-platform compatibility, thereby avoiding potential future email delivery and display issues.



**Resource Recommendations:**

* The official documentation for your specific mail server.
* Python's `smtplib` module documentation.
* Comprehensive documentation of any email library you choose to use.
* RFC 5322 (Internet Message Format).
* A good introductory text on networking and internet protocols.
