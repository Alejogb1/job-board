---
title: "How can I download Office 365 messages using IMAP and OAuth 2?"
date: "2024-12-23"
id: "how-can-i-download-office-365-messages-using-imap-and-oauth-2"
---

Let’s dive straight into the thick of it. Authenticating with Microsoft's Graph API, specifically for accessing email data using IMAP, and combining that with OAuth 2.0, isn't always straightforward. I remember one particularly challenging project back in '21 – a migration from an old on-premise exchange server to O365 where we needed to pull a lot of mailbox data programmatically. We definitely hit some snags, mostly around handling the access tokens gracefully.

The core problem lies in the fact that while IMAP itself is an older protocol, Microsoft requires you to use OAuth 2.0 for authentication when accessing O365 resources. That means you can't just use a simple username and password. Instead, you have to jump through the hoops of the OAuth authorization flow. It involves acquiring access and refresh tokens using an application you've registered in your Azure Active Directory tenant. The refresh token allows you to renew your access token without re-prompting the user every single time.

The process is generally this: register an application in Azure AD, obtain client credentials, and use the appropriate libraries to perform the OAuth dance, then configure your imap connection using the access token. It's a bit involved, so I’ll walk through the essential parts with some concrete code examples. I am going to use Python, as it’s fairly common in these contexts. However the principles translate over to other environments such as C# with the MSAL libraries.

Before we get to the code, some conceptual background is crucial. You need to understand that Azure AD applications have two main types: delegated and application permissions. Delegated permissions mean you're acting on behalf of a user, requiring user consent, whereas application permissions mean your application is acting as itself and requires admin consent. In the context of email, it is quite common to require user delegation. We would need the `IMAP.AccessAsUser.All` delegated permission to access the user mailbox.

, let's get to the first snippet, which shows the OAuth flow using the `msal` library:

```python
import msal
import asyncio

async def acquire_token(client_id, client_secret, authority, scopes):
    config = {
        "client_id": client_id,
        "client_secret": client_secret,
        "authority": authority
    }
    app = msal.ConfidentialClientApplication(**config)

    # Try to get token from cache first, if available
    result = app.acquire_token_silent(scopes, account=None)

    if not result:
        flow = app.initiate_device_flow(scopes=scopes)
        print(flow['message'])
        result = app.acquire_token_by_device_flow(flow)
    
    if 'access_token' in result:
       return result['access_token']

    else:
       print(result.get('error'))
       print(result.get('error_description'))
       print(result.get('correlation_id'))
       return None


async def main():
    client_id = "YOUR_CLIENT_ID" # Replace with your application client id
    client_secret = "YOUR_CLIENT_SECRET"  # Replace with your application client secret
    authority = "https://login.microsoftonline.com/YOUR_TENANT_ID" # Replace with your tenant id
    scopes = ["IMAP.AccessAsUser.All"] # Delegated permission to access the user mailbox using IMAP

    access_token = await acquire_token(client_id, client_secret, authority, scopes)

    if access_token:
        print("Access token acquired successfully!")
        # Now you'd proceed with connecting to the IMAP server, see next example
    else:
        print("Error acquiring token.")

if __name__ == '__main__':
   asyncio.run(main())
```

This code snippet leverages the `msal` library, which is the Microsoft Authentication Library, to handle the authorization flow. You'd need to install it first using `pip install msal`. Replace the placeholder strings with your Azure AD application details and tenant id. This specific method uses the device flow, which is suitable for non-interactive applications. The `acquire_token_silent` checks the token cache before going to the authentication flow.

The next crucial step is using that access token to connect to the IMAP server. For this, we’d use the built-in `imaplib` module in Python. Here's how to establish an authenticated IMAP connection:

```python
import imaplib
import asyncio
import ssl
import base64


async def connect_imap_oauth(access_token):
    imap_server = "outlook.office365.com"
    imap_port = 993

    try:
      ssl_context = ssl.create_default_context()
      mail_server = imaplib.IMAP4_SSL(host=imap_server, port=imap_port, ssl_context=ssl_context)
      auth_string = f'user={mail_server.user}\x01auth=Bearer {access_token}\x01\x01'
      auth_string_b = auth_string.encode('utf-8')
      auth_string_b64 = base64.b64encode(auth_string_b).decode('utf-8')

      mail_server.authenticate('XOAUTH2', lambda x: auth_string_b64)
      print("Connected to IMAP server successfully using OAuth2!")
      mail_server.select("INBOX")
      # From here you can fetch mail and process accordingly
      return mail_server
    except Exception as e:
        print(f"Error connecting to IMAP server: {e}")
        return None


async def main():
  # assume access_token was acquired from the previous example
  client_id = "YOUR_CLIENT_ID" # Replace with your application client id
  client_secret = "YOUR_CLIENT_SECRET"  # Replace with your application client secret
  authority = "https://login.microsoftonline.com/YOUR_TENANT_ID" # Replace with your tenant id
  scopes = ["IMAP.AccessAsUser.All"] # Delegated permission to access the user mailbox using IMAP

  access_token = await acquire_token(client_id, client_secret, authority, scopes)

  if access_token:
    mail_server = await connect_imap_oauth(access_token)
    if mail_server:
      # Fetch the unread mails in the INBOX
      status, data = mail_server.search(None, 'UNSEEN')
      if status == 'OK':
        for num in data[0].split():
          status, data = mail_server.fetch(num, '(RFC822)')
          if status == 'OK':
            print(f"Fetched message: {num}")
            # Process the raw message data here
      mail_server.close()
      mail_server.logout()
  else:
    print("Error acquiring access token.")


if __name__ == '__main__':
    asyncio.run(main())
```

Here, we create a secure SSL connection to the Microsoft IMAP server. The critical part is the authentication step. Instead of a username and password, we use the `XOAUTH2` method, passing a specially formatted string containing the access token, encoded in base64, as the authentication token. We then demonstrate the simplest way of fetching all unread emails from the INBOX folder. You could easily extend this to fetch only mails since a specific date, or all mails in a different folder. Remember, when finished you have to close and logout from the connection.

Finally, while these examples do a good job showcasing the fundamentals, error handling should be addressed, and it's essential to consider pagination and rate limits from Microsoft. You may want to investigate the `imaplib` documentation to get more familiar with all the options it provides. I'd also suggest studying the OAuth 2.0 specification directly to have a deeper understanding of how it works and how to get the most out of the Microsoft Graph API.

For further reading, I'd highly recommend "Programming Microsoft Dynamics CRM 2016" by Jim Steger for a very general understanding of Microsoft integration and authentication concepts, even if the focus is on CRM, the foundational auth concepts are identical. Additionally, "OAuth 2.0: The Authorization Protocol" by Eran Hammer is a good choice, which goes deeper into the low-level details of the standard. Finally, keep up-to-date with Microsoft's official documentation on the Graph API and MSAL libraries for any changes or new features. Don’t try to build this from scratch, leverage the libraries provided. It can be complex and tricky to get it all right, but focusing on a systematic approach to understand how each part works will definitely get you there.
