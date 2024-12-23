---
title: "How do I configure Sustainsys.Saml2 to accept unsigned logout responses?"
date: "2024-12-23"
id: "how-do-i-configure-sustainsyssaml2-to-accept-unsigned-logout-responses"
---

Alright, let's tackle this. Accepting unsigned logout responses with Sustainsys.Saml2… it's a situation I've encountered firsthand, and it's definitely one that requires a nuanced approach, especially in production environments. I remember a project back in '18 where we were integrating with an identity provider that, shall we say, wasn't the most rigorous about signing logout responses. Dealing with that brought some specific challenges. The short answer, of course, is that you *can* configure Sustainsys.Saml2 to do this, but you have to carefully consider the security implications.

The core issue is that the saml specification, particularly for production deployments, strongly recommends signing all assertions and messages, including logout responses, to ensure integrity and non-repudiation. An unsigned response is inherently vulnerable to tampering. Someone could potentially spoof the identity provider and initiate a forced logout or worse. However, in specific, controlled scenarios, like during development, integration testing, or when dealing with a very limited test environment, accepting unsigned responses can be a temporary necessity. Never ever consider this acceptable in a real production setting.

The mechanism within Sustainsys.Saml2 that governs this behaviour is the `SignatureValidation` property, or rather the lack of it, during `AuthnResponse` processing in the various implementations of `Saml2Message`. To allow unsigned responses, we need to essentially instruct the library to skip signature verification during logout response processing. Note, I'm focusing on logout responses; generally, you wouldn’t want to turn this off for login responses or assertions.

Here are three ways to achieve this, going from the least to most flexible:

**Example 1: Configuration through `saml2.config`**

This method relies on adjusting the configuration file directly. It’s the most straightforward way to enable the behaviour, although it’s somewhat limited in that it’s a blanket setting for your entire application. The disadvantage is this can not be done on specific logouts from specific identity providers - its an 'all or nothing' approach.

In your `saml2.config` file, within the `<serviceProviders>` section (or `<identityProviders>` if you're dealing with a logout *request* that you need to accept unsigned), you would add a `allowUnsignedLogoutResponses="true"` attribute to the relevant provider. It looks something like this:

```xml
<serviceProviders>
    <add entityId="https://your-app.com/saml" 
        allowUnsignedLogoutResponses="true"
       ...
    </add>
</serviceProviders>
```
Or if you are the idp accepting the request, not a service provider:

```xml
<identityProviders>
    <add entityId="https://example-idp.com/metadata"
        allowUnsignedLogoutResponses="true"
        ...
    </add>
</identityProviders>
```

After making this change, Sustainsys.Saml2 will bypass signature validation on logout responses from that specific service provider or identity provider. This is quick and easy, but it applies globally for all logout responses from that single defined entity, making it less flexible for fine-grained control. If you are trying to test your own custom idp and it does not yet have reliable signing, this could be acceptable for local testing.

**Example 2: Programmatic Control within a Custom `LogoutResponse` Handler**

A more adaptable approach is to create a custom handler that inherits from the `Saml2Message` implementations and overrides the signature validation logic. This is definitely preferred over simply setting a flag in the config. This approach lets you implement more complex logic; perhaps only bypassing validation under specific circumstances or for specific tenants. In order to implement this, we also need a factory class for injection.

First we define our class that inherits from the necessary type, it will be `LogoutResponse`.

```csharp
using Sustainsys.Saml2;
using Sustainsys.Saml2.Messages;

public class MyCustomLogoutResponse : LogoutResponse
{
    public MyCustomLogoutResponse(
        LogoutResponse logoutResponse) : base(logoutResponse)
    {
    }


    protected override void ValidateSignature(EntityId idpEntityId, string expectedSigner)
    {
        // Simply skip validation, this is where you can add custom logic.
        // For this example, we're just skipping entirely.
        return;
    }
}
```
Next, we create a factory that can create the `MyCustomLogoutResponse` object:

```csharp
using Sustainsys.Saml2;
using Sustainsys.Saml2.Messages;

public class CustomMessageFactory : MessageFactory
{
    public override Saml2Message Create(Saml2Message samlMessage)
    {

            if (samlMessage is LogoutResponse logoutResponse)
            {
               return new MyCustomLogoutResponse(logoutResponse);
            }

        return base.Create(samlMessage);
    }
}

```
Then we need to instruct `Sustainsys.Saml2` to use our new factory:

```csharp
services.AddAuthentication().AddSaml2(options =>
{

//Other configuration items

options.MessageFactory = new CustomMessageFactory();

});
```

Here, we’ve effectively overridden the core validation step, allowing unsigned messages to pass through. This offers far more control than just the configuration option, allowing you to inject logic based on the specific logout response or the request itself, in the override of `ValidateSignature`.

**Example 3: Using the Events System to modify the response**

This approach leverages the `Events` system in `Sustainsys.Saml2` which provides callbacks during the processing of various saml messages. It lets us examine the incoming logout response before validation is attempted, and then bypass validation if certain criteria is met.

First you would define an event callback with signature `Func<LogoutResponseReceivedContext, Task>`.

```csharp
using Sustainsys.Saml2;
using Sustainsys.Saml2.Web;

public class CustomSamlEvents
{
  public async Task OnLogoutResponseReceived(LogoutResponseReceivedContext context)
  {
      // Your logic here to determine if validation should be skipped.

       if (context.Response.IsSigned) {
        // do nothing
       } else {
            context.Response.IsSigned = true;
       }
  }
}
```

Then hook it up to the `AddSaml2` builder:

```csharp
services.AddAuthentication().AddSaml2(options =>
{
//Other configuration items

options.Events = new Saml2Events
 {
      OnLogoutResponseReceived = new CustomSamlEvents().OnLogoutResponseReceived
 };

});
```
Here, we’ve modified the `IsSigned` property on the `LogoutResponse` before processing to indicate we have validated the signature, regardless of whether it was signed in the first place. This method offers great flexibility, allowing you to implement complex custom logic, even checking for an accepted list of providers that do not sign responses. It can also be easily unit tested without having to mock out the message parsing itself.

**Considerations and Resources**

Remember, disabling signature validation on logout responses isn't something you should take lightly. You’re opening up a potential vulnerability. It’s imperative you have absolute control over the environments where this is done.

For a deeper understanding of the saml protocol, I would recommend *“Understanding SAML”* by Peter J. Davis. It provides an excellent overview and a detailed look at the mechanisms involved. Also, carefully review the specification document, “*Assertions and Protocols for the OASIS Security Assertion Markup Language (SAML) V2.0*, because you need to be familiar with the security implications you introduce when deviating from it. Additionally, the Sustainsys.Saml2 GitHub repository has excellent example configuration and documentation that can help fine tune your specific implementation. Finally, familiarize yourself with common saml implementation vulnerabilities, to avoid them.

In closing, while accepting unsigned logout responses is technically achievable with Sustainsys.Saml2, it's a decision that should be made with a full understanding of the associated risks and a strong mitigation plan. Always lean towards the more secure configuration and only make exceptions when you absolutely have to and understand the risk involved.
