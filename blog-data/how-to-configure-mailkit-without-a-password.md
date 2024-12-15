---
title: "How to configure MailKit without a password?"
date: "2024-12-15"
id: "how-to-configure-mailkit-without-a-password"
---

alright, so you're looking to configure mailkit without a password. been there, done that. it's a pretty common hurdle, especially when you're dealing with services that are pushing hard for oauth2 or other modern auth methods. passwords, as we all know, are a pain. they get compromised, need rotation, and just add to the overall management overhead. i remember this one project back in 2017, i was building this internal monitoring system, and email alerts were crucial. initially, we went the standard route: username and password. but man, the number of times we had to update the configuration files after password changes was just… not fun. after the third time, i was looking for alternative. that's when i started diving deeper into oauth2.

the core issue here isn’t really *mailkit itself*, it's the *smtp server* requirements. mailkit, at its heart, is a flexible library, it just implements the standard protocols. it's how you authenticate with the *smtp server* that matters. you can configure mailkit to connect without a password, but that depends entirely on what your smtp provider supports.

so, let’s break down what you probably have in mind. if you are dealing with a modern email service like gmail, outlook, or exchange online, the password option is discouraged, and sometimes even disabled. they heavily promote oauth2 for good reasons. to avoid the password route, here is what i suggest: use oauth2 or application-specific passwords.

let's say you're targeting gmail. here, oauth2 is the way to go and this is the approach I prefer in most cases. the flow goes like this: you create a "project" in google cloud console, generate credentials there, and then use that in your code.

here’s a very basic mailkit snippet illustrating how to do it:

```csharp
using mailkit.net.smtp;
using mailkit;
using mimekit;
using mimekit.text;
using system.threading.tasks;

public async task sendemailviaoauth2async(string senderemail, string recipientemail, string accesstoken)
{
    var message = new mimemessage();
    message.from.add(new mailaddress(senderemail));
    message.to.add(new mailaddress(recipientemail));
    message.subject = "test email from mailkit with oauth2";

    message.body = new textpart(textformat.plain)
    {
        text = "this is a test email sent using mailkit with oauth2 authentication."
    };

    using (var client = new smtpclient())
    {
       await client.connectasync("smtp.gmail.com", 587, secureoptions.starttlswhenavailable);
        var oauth2 = new mailkit.security.oauthtoken("your_email@gmail.com", accesstoken);
        await client.authenticateasync(oauth2);

        await client.sendasync(message);
        await client.disconnectasync(true);
    }
}
```
note that in this code snippet i replaced `your_email@gmail.com` placeholder, make sure to replace it with your google mail account.

the `accesstoken` is obtained using the google apis, and this is a long-lived token. now, this piece alone is not enough. you have to do the dance with google's oauth2 api, get the access token, and then plug it into the code. generating this access token can involve a couple of things. first, you need to have a valid google cloud project. within that project, you enable the gmail api. you have to set up an oauth2 client id and secret. it can be a desktop app, or a web application. then, using a library like the google api client library for .net, you do the oauth2 flow and obtain the refresh token. with the refresh token, you can obtain the access token which will work with the `mailkit.security.oauthtoken` code snippet, so you can send emails programmatically.

now if you are working in a controlled environment, and if your email service supports it, you could use application specific passwords. they are not ideal for general use, but they can be very convenient in limited cases. they are a bit of a security compromise, but again, when you have specific needs, you can do it, but with a lot of care.

here is how it would look with an application specific password, if your email service supports them:

```csharp
using mailkit.net.smtp;
using mailkit;
using mimekit;
using mimekit.text;
using system.threading.tasks;


public async task sendemailwithapplicationpasswordasync(string senderemail, string recipientemail, string apppassword)
{
    var message = new mimemessage();
    message.from.add(new mailaddress(senderemail));
    message.to.add(new mailaddress(recipientemail));
    message.subject = "test email with application specific password";

    message.body = new textpart(textformat.plain)
    {
        text = "this is a test email sent using mailkit with an application password."
    };

    using (var client = new smtpclient())
    {
        await client.connectasync("smtp.gmail.com", 587, secureoptions.starttlswhenavailable);

        await client.authenticateasync(senderemail, apppassword);

        await client.sendasync(message);
        await client.disconnectasync(true);
    }
}
```
notice how the `client.authenticateasync` method takes the email and the `apppassword` as arguments. this password can be generated from the google security settings panel if the 2 factor authentication is enabled. again, while it works, i am not a huge fan of this approach, because it introduces some security risk and it goes against google recommendations.

another approach, if your smtp server supports it and the security requirements allow you, is to use unauthenticated email sending. but this is the least secure way of dealing with sending emails, and i cannot recommend it unless you absolutely know what you are doing. if your smtp server is within your private network, then maybe it is not an issue, but it has to be used with extreme caution. this approach is quite common in dev and test environments. i also had this once to connect to an internal email server from a small testing app in the past, when i was dealing with some code integration issues.

here is the code that illustrates that:

```csharp
using mailkit.net.smtp;
using mailkit;
using mimekit;
using mimekit.text;
using system.threading.tasks;

public async task sendemailwithoutauthenticationasync(string senderemail, string recipientemail)
{
    var message = new mimemessage();
    message.from.add(new mailaddress(senderemail));
    message.to.add(new mailaddress(recipientemail));
    message.subject = "test email without any authentication";

    message.body = new textpart(textformat.plain)
    {
        text = "this is a test email sent without authentication, it should only be used with care."
    };

    using (var client = new smtpclient())
    {
        await client.connectasync("your_smtp_server_address", 25, secureoptions.none);

        await client.sendasync(message);
        await client.disconnectasync(true);
    }
}
```

notice that in the above code the `client.authenticateasync` is not used at all. instead i used `secureoptions.none`. also, for this approach to work the email server has to be configured to accept unauthenticated requests, and usually, they are not, for good reasons.

now, about resources: avoid random blog posts on the internet. they often provide out-of-date info or incomplete solutions. start with the official mailkit documentation. it’s well-written and comprehensive. for the oauth2 stuff, i recommend the official google apis documentation. it can be a bit daunting, but it's the authority on the matter. if you want a deeper understanding of smtp and the underlying protocols, the book "internet messaging" by david h. crocker is a must have.

so, yeah, configuring mailkit without passwords is less about mailkit itself and more about understanding your *smtp server* requirements and authentication options. oauth2 is the modern and secure route, application-specific passwords can work in some limited situations, and the unauthenticated way is possible but usually not a good idea.

and just a small side note, while i was setting up the oauth2 process in the past, i remember i almost wanted to use the smtp unauthenticated route, but then i got lazy because of the configurations and ended up doing everything correctly. i almost gave in to the dark side of easy email sending methods, but good security habits won the battle. or, as a coworker told me, "the lazy dev is the best dev... as long as they are not lazy with the security aspect of their code". and i think he was actually serious about that.

i hope that helps you, let me know if something is not clear.
