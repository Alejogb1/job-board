---
title: "Why does Python SMTP authentication fail when sending mail from Gmail?"
date: "2025-01-30"
id: "why-does-python-smtp-authentication-fail-when-sending"
---
Gmail, while providing a convenient SMTP service, introduces specific security protocols that frequently cause authentication failures for Python scripts attempting to send email. The core issue is Gmail’s requirement for “Less Secure App Access” or its modern equivalent, “App Passwords,” both of which are frequently misunderstood and poorly configured by users. Many tutorials often present basic SMTP configuration, overlooking these crucial Gmail-specific steps, leading to repeated authentication failures despite seemingly correct credentials.

The failure stems from Gmail’s enhanced security measures aimed at preventing unauthorized access to user accounts. Basic SMTP authentication, involving simply providing a username (your email address) and password, is considered insufficient by Gmail, particularly when the origin of the connection is an automated script. Gmail’s default security settings disable such simple connections, interpreting them as potential security threats. Therefore, unless explicitly enabled by the user, a Python script utilizing standard SMTP configurations will likely encounter authentication errors.

I've encountered this issue multiple times while building internal tools for automated email reporting, primarily when transitioning from testing locally using a simple mail server to production environments that required using Gmail. The standard Python `smtplib` module is perfectly capable of communicating with Gmail’s SMTP server, but its success depends on the user taking the necessary steps to enable the connection on the Gmail account settings side. Three primary conditions must be satisfied: the correct server address and port must be provided, the user must enable the appropriate access settings within their Google Account, and the user must utilize the correct password type based on the enabled access settings.

Let’s explore some common scenarios and the code adjustments required to resolve them. The first, and most common, issue arises when a user attempts to use their standard Gmail password. This will predictably lead to an authentication failure, even if the username (email address) is accurate. This is because standard authentication is disabled by default by Google. The following code snippet illustrates a failed attempt to connect using standard credentials.

```python
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

sender_email = "your_email@gmail.com"
receiver_email = "recipient_email@example.com"
password = "your_gmail_password"  # This is where the problem lies
smtp_server = "smtp.gmail.com"
smtp_port = 587


message = MIMEMultipart()
message["From"] = sender_email
message["To"] = receiver_email
message["Subject"] = "Test Email"

body = "This is a test email using the regular Gmail password."
message.attach(MIMEText(body, "plain"))

try:
    server = smtplib.SMTP(smtp_server, smtp_port)
    server.starttls() # Initiate TLS encryption
    server.login(sender_email, password)
    text = message.as_string()
    server.sendmail(sender_email, receiver_email, text)
    server.quit()
    print("Email sent successfully.")
except Exception as e:
    print(f"Error: {e}")
```

The critical element here is the `password` variable. Using your regular Gmail account password will result in an error message from the SMTP server, typically indicating an authentication failure, even if the email address and server settings are correctly configured. This code does not account for Google’s security measures and would fail under default security settings.

The next code example, therefore, focuses on the deprecated but still occasionally employed method of enabling “Less Secure App Access.” Within a Google Account's settings, you can locate a section labeled Security, and then "Less secure app access." By enabling this, you allow external applications, like a Python script using `smtplib`, to bypass the modern security protocols and connect to your Gmail account using your regular password. **It's important to note this is a less secure practice and not recommended for production systems.** However, for testing and learning purposes, this configuration is often used.

```python
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

sender_email = "your_email@gmail.com"
receiver_email = "recipient_email@example.com"
password = "your_gmail_password" # Regular password will now work with less secure access enabled
smtp_server = "smtp.gmail.com"
smtp_port = 587


message = MIMEMultipart()
message["From"] = sender_email
message["To"] = receiver_email
message["Subject"] = "Test Email"

body = "This email was sent using Less Secure App Access."
message.attach(MIMEText(body, "plain"))

try:
    server = smtplib.SMTP(smtp_server, smtp_port)
    server.starttls() # Initiate TLS encryption
    server.login(sender_email, password)
    text = message.as_string()
    server.sendmail(sender_email, receiver_email, text)
    server.quit()
    print("Email sent successfully.")
except Exception as e:
    print(f"Error: {e}")
```

This example is structurally identical to the previous one; however, with “Less Secure App Access” enabled in your Google account settings, this code will now successfully authenticate and send the test email using your Gmail password. While seemingly resolving the problem, the underlying security risk of enabling less secure access remains a significant concern. This approach should be treated as a temporary solution and not used for production deployments due to its inherent vulnerability.

The preferred and more secure method to overcome authentication failures involves the use of "App Passwords." These passwords are uniquely generated for specific applications, granting access to your account without exposing your main password. Instead of using your regular Gmail password, you generate an app password within your Google account security settings. This approach is highly recommended, as it isolates access to your email account to the specific application that needs to connect.

Here’s the code sample utilizing an App Password:

```python
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

sender_email = "your_email@gmail.com"
receiver_email = "recipient_email@example.com"
password = "your_generated_app_password"  # The App Password
smtp_server = "smtp.gmail.com"
smtp_port = 587

message = MIMEMultipart()
message["From"] = sender_email
message["To"] = receiver_email
message["Subject"] = "Test Email"

body = "This email was sent using an App Password."
message.attach(MIMEText(body, "plain"))

try:
    server = smtplib.SMTP(smtp_server, smtp_port)
    server.starttls()
    server.login(sender_email, password)
    text = message.as_string()
    server.sendmail(sender_email, receiver_email, text)
    server.quit()
    print("Email sent successfully.")
except Exception as e:
    print(f"Error: {e}")
```

The key difference here is the content of the `password` variable.  Instead of your regular password, a generated App Password is used. This significantly enhances security since the app password can be revoked, limiting any potential damage if compromised. Utilizing App Passwords is the most secure way to programmatically access Gmail’s SMTP server and should be favored when developing email automation tools. The procedure to generate app passwords requires navigating to the Security settings section of your Google account and following the prompts to create one.  The generated app password is then substituted for the standard password in the script.

To further understand and debug SMTP connection issues with Gmail, I would recommend consulting the official Python documentation for the `smtplib` module; this is a definitive guide for understanding its functionality. Secondly, a thorough review of Google’s security help documentation concerning "Less Secure App Access" and "App Passwords" will clarify the necessary configuration steps for your Google Account. Lastly, various online tutorial websites and blog posts offer guidance, but be sure to cross-reference these with the official documentation and Google’s official help resources to ensure accuracy. Prioritize understanding the security implications when choosing a method for authentication. Using App Passwords is the recommended approach and avoids the inherent security risks of enabling “Less Secure App Access.” These resources collectively will provide a deeper understanding of email protocols, specifically SMTP, and enhance your troubleshooting capabilities, enabling you to build reliable, secure email automation tools.
