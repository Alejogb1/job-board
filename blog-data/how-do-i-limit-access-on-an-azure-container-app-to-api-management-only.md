---
title: "How do I limit access on an Azure Container App to API Management only?"
date: "2024-12-16"
id: "how-do-i-limit-access-on-an-azure-container-app-to-api-management-only"
---

Okay, let's tackle this. I've seen this scenario play out a few times, and it's a common requirement when you're trying to create a secure and well-defined access layer for your microservices, particularly when using Azure Container Apps. The aim, as you’ve stated, is to restrict direct access to your container app and instead funnel all traffic through your Azure API Management (APIM) instance. This pattern helps with rate limiting, authentication, authorization, and many other crucial aspects of a robust api landscape.

The crux of the issue lies in understanding how to leverage Azure’s networking and identity features to effectively block direct access, while simultaneously allowing your APIM service to reach the container app. We achieve this primarily through a combination of network restrictions and authentication controls. It's not a single switch; rather, it’s a carefully orchestrated setup involving a few moving parts.

My experience has shown that a layered approach works best. The first line of defense is *network-based*. We're going to use a combination of a private environment, if possible, and, if not, utilize service tags to achieve this. Secondly, we'll employ some form of *authentication* to verify that any request attempting to reach the container app truly originates from your APIM service. Let’s examine each layer.

Firstly, when deploying the container app, it is advantageous, if possible, to place the container app in a private virtual network (VNet). Using a VNet integrated container app allows for private connectivity and inherently blocks public internet access. Within this VNet, your APIM service must also be deployed into the same virtual network. This is not always feasible depending on your organization’s infrastructure limitations. Let's discuss the scenario without a VNet, as it's more prevalent and presents the challenges we will face here.

If you don’t have the luxury of a private VNet, which is common in early-stage projects or when infrastructure is already established, we rely on service tags. Service tags, in Azure, represent a group of IP address prefixes from a given Azure service. We're going to use the ‘ApiManagement’ service tag in the network restrictions of the Container App.

Here is an example of how you would define this using an Azure Resource Manager (ARM) template, which provides the infrastructure as code approach I often prefer, though the same outcome is achievable via the Azure portal or the Azure CLI.

```json
{
    "type": "Microsoft.App/containerApps",
    "apiVersion": "2022-03-01",
    "name": "[parameters('containerAppName')]",
    "location": "[parameters('location')]",
    "properties": {
       "configuration": {
           "ingress": {
               "external": true,
               "targetPort": 8080,
               "allowInsecure": false,
               "traffic": [
                   {
                       "weight": 100,
                       "latestRevision": true
                   }
               ],
             "ipSecurityRestrictions": [
                  {
                      "name": "AllowApiManagementOnly",
                      "description": "Allow traffic only from Azure Api Management service.",
                      "action": "Allow",
                      "ipAddressRange": null,
                      "serviceTag": "ApiManagement"
                    }
                  ]

           },
        "secrets": [],
        "registries": []
       },
       "template": {
         "containers": [
            {
              "name": "my-container",
              "image": "[parameters('containerImage')]"
            }
         ]
       }
    }
}
```

Notice the `ipSecurityRestrictions` section within the `configuration.ingress`. Here, we're specifying that only traffic originating from the Azure APIM service tag is permitted. This means that any attempt to access the container app directly from the outside will be blocked by Azure’s networking layer. This significantly restricts access and is the first step to ensuring our stated goal. However, this relies entirely on the source IP, which, while secure, isn't granular enough to differentiate between your APIM service and another APIM instance potentially within that same subnet, which might be another organization, if using public Azure infrastructure.

This brings us to the second, and critical, layer: *authentication*. We need to guarantee that the request, even though coming from a ‘valid’ IP, actually comes from your specific APIM instance. For this, we rely on mutual TLS (mTLS), or client certificates, as well as request authentication with API keys for APIM to provide a verifiable identity. This is often implemented using API keys and is a straightforward approach to authentication between services when a service-to-service auth is not required. The container app will validate the key and, if present and correct, allow access to the app. If the key is missing or incorrect, the app should reject the request with a 401/403 response. This is also applicable if you chose to use mTLS. Here is a sample configuration for your Container App which validates an API Key.

```python
from flask import Flask, request, abort
import os

app = Flask(__name__)

API_KEY = os.environ.get('API_KEY')

@app.before_request
def validate_api_key():
    api_key = request.headers.get('X-API-Key')
    if not api_key or api_key != API_KEY:
        abort(401, description="Invalid API key")

@app.route('/')
def hello():
    return 'Hello from the secured container app!'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)

```

In this simplified Flask application (it can be another tech stack just as easily), before handling each request, it checks for the existence of the `X-API-Key` header. If it's either missing or doesn't match the expected API_KEY, the request is rejected with a 401 Unauthorized status code. The `API_KEY` value is stored as an environment variable in the container app, ensuring it’s configurable and can be changed safely without rebuilding your application.

Now, on the APIM side, you'll need to configure your policy to include this header on each request it forwards to the container app. You will also need to ensure that you retrieve the value of the key from a secure location; for example Azure Key Vault. Here is an example of how you might set that in the APIM policy.

```xml
<policies>
    <inbound>
        <base />
        <set-header name="X-API-Key" exists-action="override">
            <value>@{
                    return context.Variables["api-key"]
                }</value>
        </set-header>
    </inbound>
    <backend>
        <base />
    </backend>
    <outbound>
        <base />
    </outbound>
    <on-error>
        <base />
    </on-error>
</policies>
```

In this APIM policy, we're retrieving the api key from a variable `api-key`, which would have been set via Key Vault. We then add this as a header on all outbound calls to the container app. The container app can now validate the incoming request.

In practical deployment, I always favor a layered approach, combining network restrictions with authentication. Using these methods reduces the attack surface dramatically, by effectively making the container app unreachable from the outside except via APIM.

For further reading, I recommend exploring 'Cloud Native Patterns' by Cornelia Davis for deeper insight into the architectural considerations of cloud-native microservices. For Azure specific details, the official Azure documentation on Container Apps and API Management provides the most up-to-date and granular information. A deep understanding of OAuth2.0 and OIDC is also crucial for more advanced scenarios. 'OAuth 2 in Action' by Justin Richer and Antonio Sanso is an excellent resource for this. These resources, combined with a practical approach, should enable you to effectively restrict access to your container apps as per the stated requirements.
