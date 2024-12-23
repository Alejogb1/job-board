---
title: "Why can I not limit access to an Azure Container App over Api Management?"
date: "2024-12-23"
id: "why-can-i-not-limit-access-to-an-azure-container-app-over-api-management"
---

Alright, let's unpack this. I've definitely bumped into this particular headache a few times in the past – the frustration of trying to tightly control access to an azure container app using api management and finding it, well, less than straightforward. it's not a simple yes-or-no scenario, and the subtleties often get in the way. so, let’s dive into why this can be more challenging than you might expect and what steps you can take to effectively secure your container app.

the core issue here stems from how api management (apim) and azure container apps (aca) are designed to interact, particularly when it comes to network boundaries and authentication. apim acts as a reverse proxy, sitting at the edge of your network and directing traffic based on its defined policies. aca, on the other hand, lives within its own environment, often behind a virtual network (vnet). this separation is crucial for isolation and security.

the fundamental hurdle you're likely encountering is that apim's access restrictions, especially when implemented at the api policy level, don’t natively translate to direct network-level enforcement at the aca endpoint. apim can authenticate and authorize requests, but that doesn't necessarily mean your container app automatically refuses connections from sources not explicitly allowed by apim. if your aca app's ingress setting allows public access, it will listen to requests from *any* source, irrespective of apim’s policies.

think of it like this: apim acts as a gatekeeper checking credentials at the front gate, but the back door (your aca endpoint) remains unlocked unless you explicitly close it. the challenge, therefore, is not about stopping requests from reaching aca—apim will forward those once authenticated. the key is preventing connections directly to aca outside of the authorized flow through apim. this boils down to correctly configuring both the apim policies *and* the aca ingress settings, particularly around network restrictions.

here’s a practical example from a past project, where we had a microservice running inside an aca environment. we were initially puzzled as to why, despite all our authorization policies defined in apim, we could still bypass apim and access the aca directly through its public endpoint. we quickly realized that the public ingress of the aca was the key vulnerability. the solution wasn't in more aggressive apim rules, but in tightening aca's own network settings.

let’s illustrate this with a code example – or, more accurately, configuration snippets as this is largely about settings rather than application code:

**snippet 1: apim policy example (xml):**

```xml
<policies>
  <inbound>
    <base />
    <validate-jwt header-name="authorization" failed-validation-httpcode="401" failed-validation-message="Unauthorized. Invalid token.">
      <openid-config url="https://login.microsoftonline.com/{tenant-id}/v2.0/.well-known/openid-configuration" />
      <required-claims>
        <claim name="scp" separator=" " match="all">
          <value>api.read</value>
        </claim>
      </required-claims>
    </validate-jwt>
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

this policy demonstrates a fairly common setup: validate a json web token (jwt) in the authorization header and confirm that the token contains the `api.read` scope. if the token is invalid or the scope is missing, apim will reject the request with a 401. however, this is apim specific and doesn't impact if a direct request to the container app is made bypassing api management.

**snippet 2: aca ingress configuration (arm template snippet):**

```json
{
  "properties": {
     "ingress": {
        "external": true,
        "targetPort": 80,
        "transport": "auto",
        "traffic": [
            {
              "weight": 100,
              "latestRevision": true
            }
        ],
        "allowInsecure": false
      },
    "configuration": {
        "registries": [],
        "secrets": []
    },
     "template": {
        "containers":[
         {
          "name": "mycontainer",
          "image": "myregistry.azurecr.io/myimage:latest",
          "resources": {
            "cpu": 0.5,
            "memory": "1Gi"
          }
         }
        ]
     },
    "environmentId": "[parameters('environmentId')]"
   },
   "apiVersion": "2022-10-01",
   "type": "Microsoft.App/containerApps"
}

```

this template configures external ingress to be 'true', meaning that the aca endpoint is publicly accessible over the internet, although access can be controlled with azure authorization policies. but this does not restrict access from bypassing APIM which is the main problem. in this setup, even with the apim policy in place, an attacker could bypass apim by directly accessing the aca endpoint, assuming the aca’s domain is known and no further network restrictions are set up within the network containing the aca.

**snippet 3: aca ingress configuration with internal vnet setting (arm template snippet):**

```json
{
  "properties": {
    "ingress": {
        "external": false,
         "targetPort": 80,
        "transport": "auto",
        "traffic": [
            {
              "weight": 100,
              "latestRevision": true
            }
        ],
        "allowInsecure": false,
        "ipSecurityRestrictions": [
           {
             "description": "allow access from api management subnet",
             "ipAddressRange": "[parameters('apimSubnetPrefix')]",
             "action": "Allow"
           }
         ]
      },
    "configuration": {
      "registries": [],
      "secrets": []
    },
    "template": {
       "containers": [
          {
            "name": "mycontainer",
            "image": "myregistry.azurecr.io/myimage:latest",
            "resources": {
             "cpu": 0.5,
             "memory": "1Gi"
            }
           }
       ]
    },
    "environmentId": "[parameters('environmentId')]"
  },
  "apiVersion": "2022-10-01",
  "type": "Microsoft.App/containerApps"
}
```

here's where it becomes effective: we've changed the `external` setting to `false`, which means only resources inside the virtual network where aca resides can reach the app. we've also added the `ipSecurityRestrictions` property to explicitly only allow traffic from the subnet where apim is deployed. this configuration forces all requests to go through apim; direct access to the container app is now blocked. by making the app internal, you're limiting all traffic to the virtual network, with access being controlled through apim deployed in the same network.

to effectively secure your setup, therefore, here's what i recommend:

1.  **disable public ingress on aca:** set the `external` ingress property in aca to false. this will make the container app inaccessible from the public internet, forcing traffic to be routed through your virtual network.
2.  **use virtual network integration:** ensure both apim and aca reside in the same (or peered) virtual network. this allows apim to access the aca via its internal address.
3.  **implement ip access restrictions in aca:** configure the `ipSecurityRestrictions` setting in aca to only allow traffic from the apim subnet (as shown in snippet 3). this acts as a network-level firewall, reinforcing the access control from apim.
4.  **leverage mutual tls authentication (mtls):** for added security, consider enabling mtls between apim and aca. this adds an additional layer of authentication and is especially beneficial when dealing with highly sensitive data.

it's also essential to stay current with azure updates and best practices. for authoritative guidance, I'd suggest focusing on the official azure documentation, specifically articles pertaining to networking in azure container apps and azure api management. for a more in-depth exploration of secure microservices architecture patterns, "building microservices" by sam newman is a must-read. also, the book "cloud native patterns" by kukreja et al. provides excellent guidelines on patterns for designing secure and scalable cloud applications. always refer to the microsoft learn modules on azure networking too, as this is the basis for most of these interactions.

the trick isn't solely in api management's rules, but also in implementing network-level restrictions in your azure container apps and using apim as the exclusive access point. by combining those techniques, you can effectively control access and keep things secure. it’s a multi-layered approach, and each layer adds to the overall security of your system.
