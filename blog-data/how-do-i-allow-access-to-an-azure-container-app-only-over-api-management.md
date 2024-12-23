---
title: "How do I allow access to an Azure Container App only over Api Management?"
date: "2024-12-23"
id: "how-do-i-allow-access-to-an-azure-container-app-only-over-api-management"
---

Alright, let's tackle this. I remember a project a few years back where we needed to secure a microservices backend running in Azure Container Apps. The client had a strict policy: no direct access to the containers, everything had to go through API Management (APIM). It seemed straightforward enough, but the devil, as always, was in the details. This isn't just about slapping a firewall rule; it requires a nuanced approach to networking and identity.

The core principle here is to restrict access to the container app's ingress point and then configure API Management to proxy requests to it. First, we’ll isolate the container app within a virtual network (VNet), limiting public access. Then, we’ll configure APIM to communicate with the app internally, bypassing public internet exposure. Finally, we will rely on APIM's robust authentication and authorization policies to protect the backend. Think of it as a guardhouse (APIM) securing the inner sanctum (Container App).

Here’s the breakdown, step-by-step, along with the technical reasoning behind each move:

**1. Virtual Network Integration:**

The first thing we need to do is ensure the container app is not exposed publicly. This means deploying it within a VNet. When creating a container app, you should opt for a managed environment that integrates with a VNet. Specifically, you want to create an internal ingress to allow communication only within the vnet, this will remove the requirement to have a public facing ip address and makes the container app only accesible inside the vnet. The process is relatively straightforward, but you must remember this crucial bit: you need to have your vnet setup already, including the correct subnet delegation for the container app. This is commonly done in the resource definition or through the azure portal itself. The subnet needs to delegate its access to 'Microsoft.Web/managedEnvironments' resource provider.

Here's a simplified example of how this configuration might look in an ARM template. (Please note, actual production templates are much more comprehensive, this is for illustrative purpose.):

```json
{
  "$schema": "https://schema.management.azure.com/schemas/2019-04-01/deploymentTemplate.json#",
  "contentVersion": "1.0.0.0",
  "parameters": {
        "managedEnvironmentName": {
            "type": "string"
        },
    "vnetResourceId": {
            "type": "string"
        },
        "subnetResourceId": {
            "type": "string"
        },
    "location": {
        "type": "string"
    }
  },
  "resources": [
      {
            "apiVersion": "2023-05-01",
            "type": "Microsoft.App/managedEnvironments",
            "name": "[parameters('managedEnvironmentName')]",
            "location": "[parameters('location')]",
             "properties": {
                "vnetConfiguration": {
                    "internal": true,
                  "subnetId": "[parameters('subnetResourceId')]"
                }
              }

        },
    {
      "type": "Microsoft.App/containerApps",
      "apiVersion": "2023-05-01",
      "name": "mycontainerapp",
      "location": "[parameters('location')]",
      "dependsOn": [
       "[resourceId('Microsoft.App/managedEnvironments', parameters('managedEnvironmentName'))]"
      ],
      "properties": {
        "managedEnvironmentId": "[resourceId('Microsoft.App/managedEnvironments', parameters('managedEnvironmentName'))]",
         "configuration": {
          "ingress": {
            "external": false,
            "targetPort": 8080
           }
        },
        "template": {
          "containers": [
            {
              "name": "mycontainer",
              "image": "myimage:latest"
            }
          ]
        }
      }
    }

  ]
}
```

Key points here: `vnetConfiguration.internal` is set to `true`, and `ingress.external` is set to `false`. Notice also the `subnetId` reference pointing to a previously created subnet. This ensures that the container app has only internal network visibility.

**2. API Management Configuration:**

Now that your container app is tucked away safely in a VNet, it's time to configure APIM to act as the gatekeeper. This is where you set up the communication path between APIM and your container app. The critical part here is configuring the backend to resolve to your container app's fully qualified domain name (FQDN) in its private DNS zone. When the container app is in internal mode, Azure sets up a DNS record in a private DNS zone, to resolve the container app's hostname to its internal ip address. APIM needs to access this. If you are using an Azure provided DNS, APIM will generally resolve this automatically. If not, ensure the proper dns resolution is in place for the APIM service.

Here's a simplified example of a APIM backend configuration within an ARM template:

```json
{
  "$schema": "https://schema.management.azure.com/schemas/2019-04-01/deploymentTemplate.json#",
    "contentVersion": "1.0.0.0",
        "parameters": {
        "apiManagementServiceName": {
            "type": "string"
        },
    "containerAppHostName": {
      "type": "string"
    },
    "location": {
      "type": "string"
    }
  },
  "resources": [
    {
      "type": "Microsoft.ApiManagement/service/backends",
      "apiVersion": "2022-08-01",
      "name": "[concat(parameters('apiManagementServiceName'), '/containerappbackend')]",
     "location": "[parameters('location')]",
      "properties": {
        "url": "[concat('https://', parameters('containerAppHostName'))]",
        "protocol": "http",
        "credentials": null
      }
    }

  ]
}
```

In this snippet, the `url` property points to the FQDN of the container app. Critically the `protocol` is set to "http", since container apps do not support https on internal ingresses. APIM needs to be set to use http for its request. APIM itself, however, will still be using https with the client and this process is managed via APIM and its settings.

**3. API Policy Setup:**

Finally, and equally crucial, you need to define API policies in APIM to enforce access control. You'll use APIM policies to map incoming API requests to the backend and enforce authorization. This will typically involve a combination of subscription keys, OAuth 2.0, or other forms of authentication.

Here's an example of how a policy might look for a simple case, where the backend requires no extra authentication:

```xml
<policies>
  <inbound>
    <base />
    <set-backend-service base-url="https://backend.contoso.com/" />
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

Note this is a simple, example policy and is included to help visualize how api policies work. You would need to modify this for the necessary headers and other requirements needed to connect with your backend API. I will also note it is generally not a good idea to just pass on all requests to the backend.

**Key Considerations:**

*   **DNS Resolution:** Ensure APIM can resolve your container app's internal DNS. If the default Azure DNS isn’t sufficient, you may need to configure a custom DNS resolver in the VNet.
*   **Authentication:** Carefully consider your authentication method. API management provides many tools, such as client certificates, OAuth 2.0, and subscription keys, to authenticate clients.
*   **Rate Limiting and Throttling:** Implement policies within APIM to protect your backend from abuse.
*   **Network Security Groups:** Review the NSGs for both your APIM service and your container app to ensure you are not blocking needed communication.
*   **Logging and Monitoring:** Setup logging and monitoring for both your APIM and your container apps to see where requests are going and how your service is working, and quickly identify any potential problems.

**Recommended Resources:**

For a deeper dive, I'd recommend these authoritative sources:

*   **Azure documentation:** The official Microsoft documentation is an excellent starting point. Specific sections on container app networking and APIM configurations will be particularly useful.
*   **"Cloud Native Patterns" by Cornelia Davis:** This book provides architectural patterns for modern cloud application development, with details that are pertinent to this scenario.
*  **Microsoft’s Azure Architecture Center:** This resource provides reference architectures and best practices for building highly scalable and secure systems on Azure.

In practice, it took us several iterations to get this right, but the benefits of increased security and centralized API management were well worth it. The key takeaway is to understand the interplay between the different Azure services and to implement a layered security approach. By carefully configuring the vnet, setting up the correct dns resolution, and the backend connection in API management, it's possible to get your Azure container app to be accessed only through api management.

Remember to start small and iterate. Don’t be afraid to experiment in a non-production environment. This approach, while seemingly complex, is essential for securing modern, cloud-native applications.
