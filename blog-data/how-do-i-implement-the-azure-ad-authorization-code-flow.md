---
title: "How do I implement the Azure AD authorization code flow?"
date: "2024-12-23"
id: "how-do-i-implement-the-azure-ad-authorization-code-flow"
---

Alright, let's tackle this. I’ve spent a fair bit of time wrestling with authentication flows, especially the authorization code flow with Azure AD, which, trust me, has its nuances. I remember back when we were transitioning from our legacy authentication system to a more robust, cloud-centric one—it wasn't smooth sailing at first, but we got there. Let's break down how to implement that flow, covering the key steps and considerations.

The authorization code flow is designed for secure delegation of user identity to your application without directly exposing their credentials. It's a crucial aspect of modern application security, particularly when dealing with sensitive resources. The core idea revolves around a two-step process involving an intermediary authorization server—in this case, Azure AD.

First, the user is redirected to Azure AD to authenticate. Upon successful authentication, Azure AD provides a temporary authorization code. This code is then exchanged by your application's backend for access and refresh tokens. These tokens allow your application to access the protected resources the user has permission to view. This two-step approach, using the authorization code as an intermediary, enhances security.

Let’s get into the practical implementation, looking at the necessary steps and code.

**Step 1: Initiating the Authorization Request**

This is where the process starts. Your application needs to construct a URL that redirects the user to Azure AD's authorization endpoint. This URL must contain several parameters, including your application's client ID, the redirect URI, the requested scopes, and the response type (which should be "code" for the authorization code flow).

Here's a Python snippet using the `requests` library to generate the authorization URL:

```python
import requests
import urllib.parse
import uuid

def generate_authorization_url(client_id, redirect_uri, scopes, tenant_id):
    state = str(uuid.uuid4())  # Generate a unique state
    params = {
        'client_id': client_id,
        'redirect_uri': redirect_uri,
        'response_type': 'code',
        'scope': ' '.join(scopes),
        'state': state,
        'response_mode': 'query', # Ensures response parameters are returned as a query string

    }
    encoded_params = urllib.parse.urlencode(params)
    authorization_endpoint = f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/authorize"
    authorization_url = f"{authorization_endpoint}?{encoded_params}"
    return authorization_url, state

# Example usage:
client_id = "YOUR_CLIENT_ID" # replace with your application's client id.
redirect_uri = "http://localhost:8080/callback" # URL where Azure AD will send the authorization code.
scopes = ["User.Read", "openid", "offline_access"]
tenant_id = "YOUR_TENANT_ID" # replace with your AAD tenant id.

url, state = generate_authorization_url(client_id, redirect_uri, scopes, tenant_id)
print(f"Generated URL: {url}")
print(f"Generated State: {state}")
# Redirect the user to the URL
```

In the above, we crafted a function that encapsulates the URL generation process. The `state` parameter is crucial to protect against CSRF attacks, associating each redirect with a particular user session. When Azure AD redirects back, the provided `state` will be compared with our stored state to ensure no malicious redirections have been attempted. The `response_mode` is explicitly set to `query` to make sure the parameters are returned in the query string.

**Step 2: Handling the Authorization Code Callback**

After the user authenticates with Azure AD, it redirects them back to the configured redirect URI. This redirect includes the authorization code as a query parameter. At this stage, you need to extract the code and the returned `state`.

Here's a simplified example using the Flask framework to handle the callback in Python:

```python
from flask import Flask, request, redirect
import requests

app = Flask(__name__)

# Replace with your previously generated and stored state
expected_state = "YOUR_STATE_HERE" # Use the state that was generated before redirecting.

@app.route('/callback')
def callback():
    code = request.args.get('code')
    state = request.args.get('state')

    if not code or not state:
        return "Error: Missing code or state in callback.", 400
    if state != expected_state:
      return "Error: Invalid state in callback.", 400

    print(f"Received Authorization Code: {code}")

    # Exchange the code for tokens (next step)
    return f"Code received, now exchanging for tokens. Code: {code}", 200

if __name__ == '__main__':
    app.run(debug=True, port=8080)
```

In this snippet, we extract the `code` and `state` parameters from the callback URL. As mentioned, we verify if the state provided by Azure AD matches the state generated earlier to prevent CSRF.

**Step 3: Exchanging the Authorization Code for Tokens**

Once we have the authorization code, we need to exchange it for access and refresh tokens. This exchange occurs directly between your application's backend and Azure AD. A POST request to Azure AD’s token endpoint is required, including the client ID, client secret (if your application requires a secret), redirect URI, and the authorization code.

Here's an example of that process:

```python
import requests
import json

def exchange_code_for_tokens(client_id, client_secret, redirect_uri, code, tenant_id):

    token_endpoint = f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token"

    data = {
        'client_id': client_id,
        'client_secret': client_secret,
        'code': code,
        'redirect_uri': redirect_uri,
        'grant_type': 'authorization_code'
    }
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded'
    }

    try:
      response = requests.post(token_endpoint, data=data, headers=headers)
      response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
      tokens = response.json()
      return tokens
    except requests.exceptions.RequestException as e:
      print(f"Error during token exchange: {e}")
      return None


# Example usage:
client_id = "YOUR_CLIENT_ID"  # replace with your application's client id
client_secret = "YOUR_CLIENT_SECRET"  # This is optional and should be kept secure.
redirect_uri = "http://localhost:8080/callback" # Redirect Uri used in previous steps
auth_code = "PASTE_AUTH_CODE_HERE" # Authorization code recieved from redirect.
tenant_id = "YOUR_TENANT_ID" # replace with your AAD tenant id.

tokens = exchange_code_for_tokens(client_id, client_secret, redirect_uri, auth_code, tenant_id)
if tokens:
    print(f"Access Token: {tokens.get('access_token')}")
    print(f"Refresh Token: {tokens.get('refresh_token')}")

```
In the above, `exchange_code_for_tokens` makes a POST request to the token endpoint. If the exchange is successful, the response will contain an access token and a refresh token. I've included a basic error check using `response.raise_for_status()` to surface any potential errors during the request.

**Important Considerations**

*   **Token Storage:** Securely store your tokens, especially the refresh token. Avoid storing them in the browser's local storage or session storage. Consider more secure methods like HTTP-only cookies or server-side storage.
*   **Refresh Tokens:** Use the refresh token to obtain new access tokens when they expire, so you don't need to redirect the user every hour.
*   **Scopes:** Understand the different scopes available and request only the necessary permissions for your application. Over-requesting permissions is bad practice.
*   **Security:** Be very careful with your application's client secret. Avoid embedding it in the source code, especially if the application is a client-side one. Use secure secret management tools when working with server applications.
*   **Error Handling:** Implement robust error handling at each step of this process. Logging errors is crucial to identify issues early in development.

**Resources:**

For a deeper dive into this subject, I highly recommend exploring the official Microsoft documentation for Azure Active Directory and OAuth 2.0. Specifically, “Understanding the OAuth 2.0 authorization code flow” on the Microsoft Learn website is an excellent starting point. Additionally, the book "OAuth 2 in Action" by Justin Richer and Antonio Sanso is a comprehensive resource for anyone looking to master OAuth 2.0 and is something I personally consult from time to time. Finally, reviewing the RFC 6749 (The OAuth 2.0 Authorization Framework) will provide a deeper theoretical foundation of OAuth principles.

Implementing the authorization code flow may seem complex initially, but it’s crucial for building robust and secure applications. By understanding these fundamental steps, and paying close attention to the associated considerations, you’ll be well on your way to integrating with Azure AD effectively.
