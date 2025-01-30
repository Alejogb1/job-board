---
title: "Why does Django Gmail authentication fail with valid credentials?"
date: "2025-01-30"
id: "why-does-django-gmail-authentication-fail-with-valid"
---
Django Gmail authentication failures, despite using valid credentials, often stem from Google's stringent security policies regarding "less secure app access" and OAuth 2.0 requirements, particularly when attempting SMTP interactions. I've personally encountered this frequently in development and testing phases, debugging projects attempting to email confirmations and notifications via Gmail. The issue isn’t usually flawed credentials, but rather the authorization mechanisms and configurations implemented on Google's side and how Django integrates with them.

The core of the problem lies in Google's gradual deprecation of basic authentication methods, which Django’s `EMAIL_HOST_USER` and `EMAIL_HOST_PASSWORD` settings traditionally rely on for SMTP connections. For security, Google increasingly encourages OAuth 2.0 based authentication, necessitating a more complex process than simply supplying a username and password. When a Django project tries to connect to a Gmail SMTP server using basic authentication, and “less secure app access” is disabled (or no longer an option), Google will reject the login attempt despite technically “valid” credentials.

To break down the solution process, one must understand that Django's built-in email backend for SMTP relies on establishing a connection with a server using a provided hostname, port, username and password. The issue arises when Google's authentication mechanisms block this direct connection via password. So, we can approach this from two broad angles. The first, less secure approach (now deprecated by Google for new accounts), involves enabling "less secure app access" in Google's account settings, effectively bypassing the stricter security requirements for password based authentication. The second approach, and the recommended method, involves utilizing OAuth 2.0. This approach requires creating credentials in Google's Developer Console, which can then be used by Django to authenticate with Gmail's API via a specific client library.

Let me illustrate the problem with a failed example first:

```python
# settings.py (Incorrect configuration with basic authentication that will likely fail)

EMAIL_BACKEND = 'django.core.mail.backends.smtp.EmailBackend'
EMAIL_HOST = 'smtp.gmail.com'
EMAIL_PORT = 587
EMAIL_USE_TLS = True
EMAIL_HOST_USER = 'your_email@gmail.com'  # Incorrect
EMAIL_HOST_PASSWORD = 'your_password'      # Incorrect
```

This code snippet attempts to connect to Gmail using a username and password within `settings.py`. If “less secure app access” is disabled, it will generate a `smtplib.SMTPAuthenticationError`, stating it cannot log into Gmail with the provided credentials, even if they are correct. These are valid credentials in the traditional sense, but Google's security stance has rendered them ineffective with SMTP when not explicitly permitted.

The “less secure app access” workaround isn't a sustainable practice as Google continues to phase it out, and this poses a vulnerability and is generally discouraged for production deployments. To truly integrate Django with Gmail in a secure and future-proof manner, we must move to OAuth 2.0. This involves a setup on Google’s Developer Console to generate the necessary client credentials. These credentials, once set up, are used by Django to gain an access token from Gmail's API. This token is then used for authentication with SMTP, bypassing the traditional password method.

Here is an example of a successful Django setup using OAuth 2.0, requiring the installation of a suitable client library for handling the OAuth 2.0 flow, such as the `google-api-python-client` package, and its related `google-auth-httplib2` and `google-auth-oauthlib` dependencies.

```python
# settings.py (Correct configuration with OAuth 2.0)

EMAIL_BACKEND = 'django.core.mail.backends.smtp.EmailBackend'
EMAIL_HOST = 'smtp.gmail.com'
EMAIL_PORT = 587
EMAIL_USE_TLS = True
EMAIL_HOST_USER = 'your_email@gmail.com'  # Still required but not used for authentication
EMAIL_HOST_PASSWORD = ''                 # Not required or used
EMAIL_USE_SSL = False
# Custom email backend using OAuth 2.0
EMAIL_BACKEND = 'path.to.your.oauth_backend.OAuth2Backend' # See below for implementation
```

This configuration differs fundamentally; we’re setting the `EMAIL_HOST_PASSWORD` to an empty string, acknowledging that our traditional password is no longer used for authentication. We also include a new value for `EMAIL_BACKEND` which will point towards custom code that takes care of the OAuth exchange. We will need to create our custom email backend.

```python
# oauth_backend.py (Implementation of custom email backend)

import smtplib
import email.utils
from email.mime.text import MIMEText
from google.oauth2 import service_account
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google.oauth2 import service_account
from django.core.mail.backends.base import BaseEmailBackend
import os
import pickle
from googleapiclient.discovery import build

class OAuth2Backend(BaseEmailBackend):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.credentials = self.get_credentials()

    def get_credentials(self):

         creds = None
         # The file token.pickle stores the user's access and refresh tokens, and is
         # created automatically when the authorization flow completes for the first
         # time.
         if os.path.exists('token.pickle'):
             with open('token.pickle', 'rb') as token:
                 creds = pickle.load(token)
         # If there are no (valid) credentials available, let the user log in.
         if not creds or not creds.valid:
             if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
             else:
                # Load credentials
                creds = Credentials.from_service_account_file('path_to_your_credentials.json', scopes=['https://mail.google.com/'])

             # Save the credentials for the next run
             with open('token.pickle', 'wb') as token:
                 pickle.dump(creds, token)

         return creds

    def open(self):
        self.connection = smtplib.SMTP(self.host, self.port)
        self.connection.ehlo()  # Identify with the server
        self.connection.starttls()  # Enable TLS for encryption
        self.connection.ehlo()  # Re-identify for TLS

        if self.credentials:
            # Convert credentials to smtplib object
            auth_string = self.credentials.token

            self.connection.login(self.username, auth_string)

        else:
             raise Exception('Missing OAuth 2.0 credentials')


        return self.connection

    def close(self):
        if self.connection:
            self.connection.close()
            self.connection = None

    def send_messages(self, email_messages):
        if not email_messages:
            return False
        new_conn_created = not self.connection
        if not self.connection:
            self.open()

        sent_messages = 0
        for message in email_messages:
             sent_messages += self._send(message)

        if new_conn_created:
            self.close()

        return sent_messages

    def _send(self, email_message):
       """A helper method that sends the email using the open connection."""
       try:
           msg_text = email_message.message().as_bytes()
           self.connection.sendmail(
              email_message.from_email,
              email_message.recipients(),
              msg_text
            )

       except smtplib.SMTPException:
            if not self.fail_silently:
               raise
            return False
       return True
```
This custom backend retrieves OAuth 2.0 credentials, and then on connection, takes the access token and authenticates with Gmail using a direct login. This requires storing and loading access tokens, usually via pickling and refreshing expired ones. The credentials are obtained via a service account and a credentials JSON file that is downloaded from Google cloud after setting up the relevant application and service account in Google's Developer Console.

In terms of resources, delving into the official Google documentation on their OAuth 2.0 process is paramount. Specific documentation regarding the Gmail API and service account setup will provide clarity, along with information on generating service account credentials, the scope configuration (permissions required), and token handling. Exploring existing open-source projects that implement similar OAuth flows can offer insights into best practices. The python package `google-api-python-client` provides thorough guides and demos for API authentication and usage. It is important to note that these methods and specific configurations are subject to Google's updates, and therefore keeping track of Google's security announcements and changes is vital for long-term reliable implementations.
