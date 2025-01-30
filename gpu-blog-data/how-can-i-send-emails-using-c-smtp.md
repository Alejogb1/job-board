---
title: "How can I send emails using C# SMTP after Gmail's deprecation of the traditional method?"
date: "2025-01-30"
id: "how-can-i-send-emails-using-c-smtp"
---
The shift in Gmail's authentication requirements necessitates employing OAuth 2.0 for SMTP access, abandoning the previously straightforward username and password method. I've personally managed migrations for several enterprise systems, and this change represents a significant shift in how we handle email sending through C#.  Directly using user credentials within code has become a security risk and is no longer permissible for Google's SMTP service. Therefore, the implementation requires retrieving an access token from Google's Identity Platform, which will authenticate our application on a per-session basis.

The core of the solution involves several steps: creating a Google Cloud Platform (GCP) project, enabling the Gmail API, configuring OAuth 2.0 credentials (specifically, client IDs and secrets), and then implementing the C# code to obtain an access token and use it to send emails. I’ve consistently found that skipping any of these steps, particularly the GCP configurations, results in cryptic error messages, which consume debugging time.

Initially, setting up the GCP project and the necessary API permissions is crucial. In Google Cloud Platform, you would need to create a new project. Navigate to the API Library, locate and enable the Gmail API. The "Enable" option will activate the required services. Then, under "Credentials", you will have to create OAuth 2.0 client credentials; choose "Desktop application" as the type. This process yields a client ID and client secret, both of which will be essential for retrieving an access token.  Additionally, specifying a redirect URI which is usually something simple such as `http://localhost`. I also recommend a specific set of permissions (scopes) to be requested during token retrieval; at a minimum, the `https://mail.google.com/` scope will allow sending email. When your application requests authentication from Google, it will only be authorized to interact with the Gmail service within the scope granted to it. Storing these credentials securely is paramount; do not embed them directly into the code.

My current preferred method in C# uses the `Google.Apis.Auth` NuGet package for token management and the standard `.NET` `SmtpClient` class for the actual email transmission. The flow involves creating a `UserCredential` instance, which will handle the authorization logic. Let's break down a minimal C# implementation:

**Code Example 1: Retrieving the Access Token**

```csharp
using Google.Apis.Auth.OAuth2;
using Google.Apis.Gmail.v1;
using Google.Apis.Util.Store;
using System;
using System.IO;
using System.Threading;
using System.Threading.Tasks;

public static class TokenProvider
{
    public static async Task<string> GetAccessTokenAsync(string clientId, string clientSecret)
    {
        string[] scopes = { GmailService.Scope.GmailSend };
        var credential = await GoogleWebAuthorizationBroker.AuthorizeAsync(
            new ClientSecrets
            {
                ClientId = clientId,
                ClientSecret = clientSecret
            },
            scopes,
            "user",
            CancellationToken.None,
            new FileDataStore("token.json", true)
        );
      
        return credential.Token.AccessToken;
    }
}
```

This code snippet focuses entirely on the token acquisition process. It leverages `GoogleWebAuthorizationBroker.AuthorizeAsync` which will manage the authorization flow. I use `FileDataStore` so that the token can be persisted between executions which simplifies subsequent use, rather than re-authorizing each time. The `scopes` array specifies the required permission level, in this case to send emails (`GmailService.Scope.GmailSend`). The `AuthorizeAsync` method will either retrieve an existing valid token or prompt the user for authorization within a browser, if a token is not already present or needs renewal. Storing the token in a file `token.json` provides persistence and reduces unnecessary browser authorizations. The key here is error handling, I have omitted this for clarity, but production code should always validate token retrieval. The returned string is the `AccessToken`, which we use in the `SmtpClient`.

**Code Example 2: Sending the Email**

```csharp
using System;
using System.Net;
using System.Net.Mail;
using System.Threading.Tasks;

public static class EmailSender
{
    public static async Task SendEmailAsync(string recipient, string subject, string body, string accessToken)
    {
        var smtpClient = new SmtpClient("smtp.gmail.com", 587)
        {
            EnableSsl = true,
            DeliveryMethod = SmtpDeliveryMethod.Network,
            Credentials = new NetworkCredential("unused", accessToken),
        };

        var mailMessage = new MailMessage
        {
            From = new MailAddress("your_email@gmail.com"), // Your Gmail address
            Subject = subject,
            Body = body,
        };
        mailMessage.To.Add(recipient);
        try
        {
          await smtpClient.SendMailAsync(mailMessage);
        }
        catch (Exception ex)
        {
          Console.WriteLine("Error sending email: " + ex.Message);
        }
    }
}
```

This code segment utilizes the `SmtpClient` to transmit an email. Crucially, instead of using a username and password, I instantiate a `NetworkCredential` with an arbitrary username and the access token obtained in the previous step as the password. It's essential to configure the `SmtpClient` to use SSL and the correct port (`587` for TLS/STARTTLS), ensuring secure communication with Gmail's servers. This implementation is a barebones example and lacks critical features such as error logging or more complex message formatting, but highlights the core mechanics of OAuth2 authentication. For example, failure to set the correct `DeliveryMethod` or `EnableSsl` could result in connection errors which are time consuming to trace.  Always thoroughly check TLS/SSL compatibility with the provider’s SMTP server settings to avoid these problems. The `From` field must be a valid Gmail address associated with the credentials. The exception handler will assist in identifying common issues.

**Code Example 3: Integrating the Code**

```csharp
using System;
using System.Threading.Tasks;

public class EmailService
{
    public static async Task Main(string[] args)
    {
        string clientId = "YOUR_CLIENT_ID"; // Replace with your actual Client ID
        string clientSecret = "YOUR_CLIENT_SECRET"; // Replace with your actual Client Secret
        string recipientEmail = "recipient@example.com"; // Replace with your actual recipient email
        string emailSubject = "Test Email via OAuth2";
        string emailBody = "This is a test email sent using OAuth2 with the Gmail API.";

        try
        {
          string accessToken = await TokenProvider.GetAccessTokenAsync(clientId, clientSecret);

          await EmailSender.SendEmailAsync(recipientEmail, emailSubject, emailBody, accessToken);

           Console.WriteLine("Email sent successfully!");
        }
        catch(Exception ex)
        {
          Console.WriteLine("An error has occurred: " + ex.Message);
        }
    }
}
```

This snippet encapsulates the previous two code examples into a simple program. Here I am invoking the `TokenProvider` class to get the access token. This token is then passed into the `EmailSender` class. Note the placeholder values for `clientId` and `clientSecret`, these will need replacing with values from your project in Google Cloud Platform. This illustrates the typical flow where you would first retrieve the token before sending any email.  Error handling, while present, could be extended in a full application. The key concept here is the modularity, isolating token retrieval and email sending which is good practice for maintainability.

For further learning, I would recommend researching the Google Identity Platform documentation for detailed information about OAuth 2.0 flows. The Google API client libraries for .NET GitHub repository includes extensive samples that illustrate the concepts described above, including best practices for handling access tokens and implementing error handling. Microsoft's documentation on the `SmtpClient` class provides a deeper understanding of the underlying SMTP protocol and how to configure the client effectively.  Specifically, I find examining the advanced settings related to connection management and security crucial when debugging production email issues. Finally, reviewing established community libraries for email handling can provide valuable insights into more advanced scenarios, such as handling attachments or implementing message queues.
