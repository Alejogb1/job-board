---
title: "How do I implement the Azure AD authorization code flow?"
date: "2025-01-30"
id: "how-do-i-implement-the-azure-ad-authorization"
---
The Azure Active Directory (Azure AD) authorization code flow is fundamental to securing modern web applications, enabling them to access protected resources on behalf of users. It’s not a simple one-step process; understanding the nuances is crucial to implementation. I've seen countless applications stumble due to misinterpretations of the specifications, often resulting in security vulnerabilities or non-functional authentication. My experience in migrating a large legacy platform to a microservices architecture reliant on Azure AD deeply ingrained this process.

The core principle is redirecting the user to Azure AD for authentication and authorization, then receiving a code that can be exchanged for access and refresh tokens. This multi-step process ensures that the client application never handles user credentials directly, a key security best practice. The flow is initiated by the application crafting an authorization request and redirecting the user’s browser to the Microsoft Identity Platform’s authorization endpoint. This request includes several crucial parameters, such as the client ID, redirect URI, and requested scopes.

The user authenticates at the Microsoft login page, granting consent to the requested permissions. Upon successful authentication, Azure AD redirects the user back to the specified redirect URI, appending an authorization code to the URL. The application then extracts this code, which acts as a temporary credential. It's vital this code be exchanged immediately for security tokens, as it has a short lifespan. This exchange happens through a backchannel request to the Microsoft Identity Platform token endpoint, never exposing the code in the browser. The server-side application utilizes the code and its client secret to obtain access and refresh tokens from Azure AD. These tokens then authorize access to protected resources. The access token is used to make requests to those resources, while the refresh token can be employed to obtain a new access token when the original one expires, preventing frequent re-authentication.

Let’s examine the implementation using different coding languages. I will showcase a simplified approach omitting error handling and focusing on the crucial steps. Remember to install required libraries, such as `msal` for Python or `azure-identity` for .NET.

**Code Example 1: Python using MSAL**

```python
import msal
import requests

# Configuration (replace with your actual values)
CLIENT_ID = "your_application_client_id"
CLIENT_SECRET = "your_application_client_secret"
AUTHORITY = "https://login.microsoftonline.com/your_tenant_id"
REDIRECT_URI = "http://localhost:8080/callback"
SCOPES = ["User.Read"]

# 1. Build the Authorization Request URL
app = msal.ConfidentialClientApplication(
    CLIENT_ID,
    authority=AUTHORITY,
    client_credential=CLIENT_SECRET,
)
auth_url = app.get_authorization_request_url(
    scopes=SCOPES,
    redirect_uri=REDIRECT_URI,
    state="arbitrary_state",
)
print("Navigate to: " + auth_url)

# 2. Receive the Authorization Code (hypothetical redirect handling)
code = "returned_code_from_redirect_uri" # Simulate the returned code from the redirect

# 3. Exchange the Code for Tokens
result = app.acquire_token_by_authorization_code(
    code,
    scopes=SCOPES,
    redirect_uri=REDIRECT_URI,
)

if "access_token" in result:
    access_token = result["access_token"]
    print("Access Token:", access_token)

    # Example of making an authenticated API request
    headers = {'Authorization': f'Bearer {access_token}'}
    graph_api_endpoint = 'https://graph.microsoft.com/v1.0/me'
    response = requests.get(graph_api_endpoint, headers=headers)
    if response.status_code == 200:
      print("User Info: " , response.json())
    else:
      print("Error accessing Graph API")
else:
  print(result.get("error"))
  print(result.get("error_description"))

```

This Python example utilizes the `msal` library. The first part generates the authorization URL, which would be used to redirect a user’s browser.  The `app.get_authorization_request_url` function creates this URL based on provided client ID, redirect URI and scopes. After the user logs in, the simulated `code` variable represents the returned authorization code in the redirect URI. Subsequently, the `acquire_token_by_authorization_code` function is used to exchange this code with the server for tokens. Finally, a sample API call against Microsoft Graph is made.

**Code Example 2: C# using .NET Azure Identity Library**

```csharp
using Azure.Identity;
using Microsoft.Identity.Client;
using System;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Threading.Tasks;


public class AuthCodeFlow
{
  private static string ClientId = "your_application_client_id";
  private static string TenantId = "your_tenant_id";
  private static string RedirectUri = "http://localhost:8080/callback";
  private static string[] Scopes = new string[] { "User.Read" };

  public static async Task RunAsync()
  {

        //1. Build the Authorization Request URL
        var pcaBuilder = PublicClientApplicationBuilder.Create(ClientId)
           .WithAuthority(AzureCloudInstance.AzurePublic, TenantId)
           .WithRedirectUri(RedirectUri);
        var pca = pcaBuilder.Build();
        var authResult = await pca.AcquireTokenInteractive(Scopes).ExecuteAsync();
      
        if (authResult != null) {
           string accessToken = authResult.AccessToken;
           Console.WriteLine("Access Token: " + accessToken);


            // Example of making an authenticated API request
            using (HttpClient client = new HttpClient())
            {
                client.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("Bearer", accessToken);

                HttpResponseMessage response = await client.GetAsync("https://graph.microsoft.com/v1.0/me");
                if (response.IsSuccessStatusCode)
                {
                    string content = await response.Content.ReadAsStringAsync();
                    Console.WriteLine("User Info: " + content);
                }
                else
                {
                  Console.WriteLine("Error accessing Graph API");
                }
            }
        }
        else{
          Console.WriteLine("Error acquiring access token");
        }
    }
    public static void Main(string[] args)
    {
       RunAsync().Wait();
    }

}
```

Here, the C# code employs the `Azure.Identity` and `Microsoft.Identity.Client` libraries. Unlike the Python example, it directly uses the interactive flow via `AcquireTokenInteractive`.  This is because, in a desktop application or similar environment, the user's browser can be opened, and the authorization process managed directly.  The `pcaBuilder` sets up the client application with the required parameters. Once authentication succeeds and an access token is acquired, a Microsoft Graph API call is made. This example, different from the Python example, handles the redirect within the interactive login prompt.

**Code Example 3: Javascript using MSAL Browser**

```javascript
import * as msal from "@azure/msal-browser";


const msalConfig = {
    auth: {
      clientId: 'your_application_client_id',
      authority: 'https://login.microsoftonline.com/your_tenant_id',
      redirectUri: 'http://localhost:8080/callback'
    }
};
const myMSALObj = new msal.PublicClientApplication(msalConfig);

const loginRequest = {
    scopes: ["User.Read"]
};


async function handleLogin() {
    myMSALObj.loginPopup(loginRequest)
        .then(response => {
             if (response && response.accessToken) {
                console.log("Access Token: " + response.accessToken);
                callGraphApi(response.accessToken)

           }
           else
           {
             console.log("No Access token")
           }
        })
        .catch(error => {
           console.log("Error: " + error)
        });
}

async function callGraphApi(accessToken){
  const headers = new Headers();
  const bearer = `Bearer ${accessToken}`;
  headers.append("Authorization", bearer);

  const options = {
      method: "GET",
      headers: headers
  };
  const graphEndpoint = "https://graph.microsoft.com/v1.0/me";
    fetch(graphEndpoint,options)
          .then(response => response.json())
          .then(data => console.log("User Info: " + JSON.stringify(data)))
          .catch(error => console.log("Error calling Graph API: " + error));
}

// Call handleLogin() when needed, e.g. triggered by button click.
```

This JavaScript example, using the `@azure/msal-browser` library, demonstrates how authentication is managed within a web application. It uses a popup window via `loginPopup` to trigger authentication.  Upon a successful login, it extracts the access token and calls `callGraphApi` to demonstrate an authenticated request.  The JavaScript example is designed to be client-side only, meaning the secret is not present.

**Resource Recommendations**

When implementing Azure AD authorization, I recommend referring to the following resources: the official Microsoft Identity Platform documentation (specifically the section on the authorization code flow), along with dedicated guides for each programming language used; also, documentation for the specific libraries you choose for your language, such as `msal` and `azure-identity`, is invaluable. Finally, scrutinize the security recommendations relating to token management; the Microsoft Security Compliance Toolkit may be a useful starting point. Ensure you thoroughly comprehend the implications of each configuration option, such as client IDs, redirect URIs, and scopes.  I've found that spending sufficient time with the documentation significantly reduces the risk of introducing security issues or encountering implementation bottlenecks. Properly understanding the nuances of the flow, the token lifecycle, and potential edge cases will ensure a secure and robust authentication experience for users.
