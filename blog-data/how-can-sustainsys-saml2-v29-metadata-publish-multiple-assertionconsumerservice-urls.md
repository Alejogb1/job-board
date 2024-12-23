---
title: "How can SustainSys Saml2 v2.9 metadata publish multiple AssertionConsumerService URLs?"
date: "2024-12-23"
id: "how-can-sustainsys-saml2-v29-metadata-publish-multiple-assertionconsumerservice-urls"
---

Alright, let’s tackle this. The intricacies of configuring multiple `AssertionConsumerService` URLs within SustainSys.Saml2, particularly version 2.9, aren't exactly front-and-center in the documentation, which can sometimes be a source of frustration. I've personally seen teams grapple with this, especially when rolling out a multi-tenant application where different tenant configurations need to point to distinct endpoints for SAML responses. We’ve always found a solution by stepping methodically through the configuration options and really understanding how the library interprets that. Let me share what I’ve learned.

The short version: you can't directly configure multiple *alternating* `AssertionConsumerService` URLs for a single service provider entity within the SustainSys.Saml2 configuration that would be used for selection by the IdP. Instead, the SAML 2.0 specification, which the library follows, permits only one *default* `AssertionConsumerService` URL at a time within the metadata. The library can, however, be configured to handle requests sent to different ACS endpoints, but those need to be selected by mechanisms other than simple metadata configurations. Let's break down how to achieve this, focusing on the practicalities that arise in real-world implementations.

The core issue stems from how SAML metadata is structured. The `<SPSSODescriptor>` element within a service provider’s metadata only allows for a single default `AssertionConsumerService` binding and location (url) to be declared at a time for each binding type (e.g. HTTP-POST, HTTP-REDIRECT). So, if you’re expecting the IdP to pick between different ACS locations based on some preference within the metadata itself, you’ll find that isn't supported. The metadata primarily dictates where the assertion should be *normally* delivered, not a set of options.

Now, while the SAML specification restricts a single default ACS URL in the metadata, the SustainSys.Saml2 library is flexible enough to handle requests sent to *other* configured ACS URLs, and that is the key. The library has a concept of registered `AssertionConsumerService` endpoints, which are internally mapped and recognized beyond the one published in the metadata. This functionality allows for advanced routing based on several factors, such as the incoming request’s `RelayState` parameter (in the form of a hidden form field or query parameter) or by directly extracting the ACS URL from the request itself.

Here's how it generally works in practice: You configure the *default* ACS URL within your service provider's metadata settings, as usual. Then, you register additional, non-default ACS URLs within your application's SAML configuration, usually done programmatically. When a SAML response is received, the library checks if the URL to which the response was posted matches *any* of the registered ACS URLs, not just the default one. If a match is found, the response is processed; otherwise, the library may throw an error.

Here's a code illustration in c# that includes three examples. Assume we are using a startup class for a asp.net core application where we wire up our services:

```csharp
// Example 1: Registering Additional AssertionConsumerService URLs in startup
public void ConfigureServices(IServiceCollection services)
{
    // Existing service configuration...

    services.AddSaml(options =>
    {
        options.SPOptions.EntityId = new EntityId("https://your-sp.com/saml");

        //Default ACS URL as per usual
        options.SPOptions.AssertionConsumerServiceLocations.Add(new AssertionConsumerService
            {
              Binding = Saml2BindingType.HttpRedirect,
              Location = new Uri("https://your-sp.com/acs")
           });
       //This is the method used in the sample. It is important to note that if you have the same binding to multiple urls, you must use unique index on each one.
       options.SPOptions.RegisterAssertionConsumerService(Saml2BindingType.HttpRedirect,
              new Uri("https://your-sp.com/acs"), index:0);

          //register an alternate acs url
       options.SPOptions.RegisterAssertionConsumerService(Saml2BindingType.HttpRedirect,
              new Uri("https://your-sp.com/acs/tenant1"), index:1);

       options.SPOptions.RegisterAssertionConsumerService(Saml2BindingType.HttpRedirect,
              new Uri("https://your-sp.com/acs/tenant2"), index:2);


        // Metadata settings (this will generate the default url into the metadata)
        options.SPOptions.MetadataPath = "/metadata"; //or configure to your specific path
        // ... Other SAML Options
    });

    // ... Other service registrations
}


```

In Example 1, we are explicitly registering multiple additional ACS URLs using the `RegisterAssertionConsumerService` method, we register three: the same `https://your-sp.com/acs`, and two additional `https://your-sp.com/acs/tenant1`, and `https://your-sp.com/acs/tenant2`. The index is important here as each url, *for the same binding* needs to have a unique index.

The default acs url is still explicitly configured. When generating the metadata, only the *default* one will appear within the published metadata, but the library internally knows about the additional URLs. This will allow you to direct responses to either of these registered URLs and the library will still process them correctly.

Now, here’s how you would, in your application code, extract the ACS url information for purposes of constructing the request itself:

```csharp
 //Example 2: Constructing an AuthnRequest that directs to a specific ACS URL
  public IActionResult InitiateSamlAuth(string tenantIdentifier){
        var samlService = HttpContext.RequestServices.GetService<ISaml2Service>();
        var authRequest = samlService.CreateAuthnRequest(
        new Uri("https://your-idp.com/saml"),
             relayState:"somestate",
              acsUri: tenantIdentifier switch{
                 "tenant1" => new Uri("https://your-sp.com/acs/tenant1"),
                 "tenant2" => new Uri("https://your-sp.com/acs/tenant2"),
                 _ => new Uri("https://your-sp.com/acs")
                });
          return Redirect(authRequest.RequestUrl.AbsoluteUri);
       }
```
In example 2, we can dynamically select the ACS url to be used for a given authentication request. We can base it on the tenant, or any other arbitrary logic we would like. The method above shows a simplistic version that matches against tenant id. A more sophisticated application might extract the appropriate value from the user or session and store it to determine the specific ACS endpoint before constructing the auth request.

Here's example three, of how you might go about matching this on the backend based on a url (in an MVC endpoint controller, for example)

```csharp
 // Example 3: Matching based on ACS URL
[HttpPost("acs/tenant1")]
[HttpPost("acs")]
[HttpPost("acs/tenant2")]
public async Task<IActionResult> AssertionConsumerService()
{
    var samlService = HttpContext.RequestServices.GetService<ISaml2Service>();
    var result = await samlService.ReceiveSamlResponseAsync(Saml2BindingType.HttpRedirect,HttpContext,false);
    if (result.IsSuccessful){
       //process the result here for your application
      return View("Success");
    }
    else
    {
      return View("Failure");
    }
}
```

In example 3, the key here is that you provide multiple endpoints in your routing logic in order to facilitate handling requests from different ACS urls. The library will look for a match on all registered urls when you receive the request, *not just the metadata configured acs url*.

For further reading on this topic, I strongly suggest starting with the official SAML 2.0 specification documents. They are quite detailed, but a careful reading will give you a solid foundation. Specifically, pay close attention to the sections on metadata and the `SPSSODescriptor` element. Also, the "OpenSAML" java implementation is an excellent resource, as it goes deeply into metadata management, though it's not .net specific, much of the concepts apply directly to SustainSys. For a .NET focused approach, the official sustain sys github documentation is also extremely helpful. Finally, the book "Programming Web Services with SOAP" by Doug Tidwell, among other things, describes in detail message structures for web services, and is a good resource for understanding core concepts of xml schemas.

In closing, remember that while you can't declare multiple alternative ACS URLs directly in the metadata, the SustainSys.Saml2 library allows flexible registration and handling of multiple ACS endpoints. It enables more nuanced and sophisticated service configurations, but requires a firm understanding of the underlying concepts to effectively utilize.
