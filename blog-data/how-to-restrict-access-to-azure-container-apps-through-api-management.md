---
title: "How to restrict access to Azure Container Apps through API Management?"
date: "2024-12-23"
id: "how-to-restrict-access-to-azure-container-apps-through-api-management"
---

Let’s jump straight into this; we’re talking about a common, yet crucial, security consideration when deploying containerized applications on azure: restricting access to your azure container apps through api management. I've encountered this scenario a few times, most notably when we were rolling out a microservices architecture across multiple teams and needed a centralized gateway to enforce security policies and manage routing. It's not as straightforward as flipping a switch, but with a proper understanding of the components, it becomes manageable.

The underlying principle here revolves around not exposing your container apps directly to the internet. Instead, we want users, or other services, to communicate with them solely through Azure API Management (APIM). Think of APIM as your bouncer – it checks credentials, handles rate limiting, applies policies, and then, only if everything checks out, forwards the request to the correct container app. The first step is configuring your container app to only accept traffic from apim, essentially disabling public access.

Let's begin by looking at how you would achieve this, technically. We’ll primarily be concerned with configuring access restrictions on the Container App and correctly routing traffic through the API Management gateway.

The first key piece is the *Ingress settings* of your container app. By default, container apps often have public ingress enabled, meaning anyone with the app's url can reach it. This, naturally, won't do for our purposes. We must restrict access to internal traffic only, and this means that we are relying on Azure's internal virtual network or private link to be able to reach the container app. While the exact steps may vary slightly depending on the method you're using to configure your resources (e.g., the azure portal, bicep, cli), here’s a conceptual example of how you might configure this using something like a bicep template, since I find bicep’s syntax to be quite clear and easily read:

```bicep
resource containerApp 'Microsoft.App/containerApps@2023-05-01' = {
  name: 'my-container-app'
  location: location
  properties: {
    environmentId: environment.id
    configuration: {
        ingress:{
            external: false
            targetPort: 80
        }
    }
    template: {
       // Container image and other deployment configurations.
    }
  }
}
```

In the above snippet, the `external: false` configuration is the crucial part. It disables external ingress, meaning that your container app won’t respond to requests from the internet directly. The `targetPort` specifies which port the container app is listening on internally. This setup ensures that no external client can bypass apim to get to the application.

Now, we move to setting up APIM to route requests correctly to our container app. The key here is to understand that apim is going to be acting as a proxy. We first need to set up the *backend* in api management to represent the container app. Apim needs to know how to reach your container app internally. If you have exposed your container app via a private link, this would be set up as a private endpoint in your backend. Here’s how that might look, again, conceptually:

```bicep
resource apimBackend 'Microsoft.ApiManagement/service/backends@2022-08-01' = {
  parent: apimService
  name: 'container-app-backend'
  properties: {
    title: 'Container App Backend'
    description: 'Backend pointing to the internal container app'
    url: 'https://<your-container-app-fqdn>' //Internal FQDN here
    protocol: 'http' // or 'https' if using certificates
    credentials:{
      // If required use this for custom headers/credentials when communicating with the backend
    }
  }
}
```

In this code, `url` is *critical*. This should not be the public URL of your container app, but rather the internal fully qualified domain name of your app as resolved through a private link or an internal DNS entry. The `protocol` should reflect your container app's internal service configuration. I would generally advise using `https` and properly configuring certificates but for the purpose of demonstration, I have used `http`. The `credentials` section would be used if you have particular authentication needs, for example, if your container app needs to have headers set or have a secret sent.

With the backend configured, the next step is creating an api within the api management service that uses this backend. This is when we start defining the *operations* or api calls that can be made through APIM. Here's an example:

```bicep
resource apiOperation 'Microsoft.ApiManagement/service/apis/operations@2022-08-01' = {
    parent: api
    name: 'get-data'
    properties: {
        displayName: 'Get Data Operation'
        method: 'GET'
        urlTemplate: '/data' // how this endpoint is exposed on APIM
        templateParameters: []
        responses: [{
            statusCode: 200,
            description: 'successful request'
        }]
        backendId: apimBackend.id
    }
}
```

This code snippet defines a `get` operation. The key setting here is `backendId`, which connects this operation to our container app's backend. The `urlTemplate` shows how this route is presented on the APIM interface, and `method` is the http method. Essentially, a request to `<your-apim-url>/data` will, when processed by apim, be forwarded to the internal route that is the `url` of your container app endpoint.

Now that we have the core components configured, a few essential considerations remain. Authentication is paramount – generally you'll want to use an authentication mechanism that APIM is able to validate before the request ever reaches your container app. This might be api keys, oauth 2.0, or other authentication flows. Rate limiting policies, caching, transformations, or other functionalities can also be layered onto the api using api management’s policy configurations.

I have found that proper monitoring and logging are also vital in diagnosing issues. Enable logging on apim, on your container app, and on any other related azure services to assist in understanding traffic flows and troubleshooting.

For more in-depth knowledge, I would recommend the official Azure documentation for both API Management and Container Apps, particularly the sections regarding networking. *Microsoft Azure Architect Technologies* by Michael Collier, James Bannan, and Eric Boyd provides an excellent and broad perspective on azure technologies, and specific chapters detail networking and security considerations within azure that you'll find useful. Also, the official Microsoft Learn modules, specifically on Api Management and Container Apps, offer more detailed guides and practical lab exercises. These are continuously updated, so they are generally reliable.

This configuration ensures that traffic to your container apps is controlled through a single, secure gateway (apim), giving you a greater level of security and more granular control over how your containerized applications are accessed. The patterns outlined here are the foundation of securing containerized microservices within azure. Although the provided code is conceptual, it is illustrative of the required architectural patterns and configuration steps. Remember that the exact steps may vary depending on your setup, but the underlying principles remain the same.
