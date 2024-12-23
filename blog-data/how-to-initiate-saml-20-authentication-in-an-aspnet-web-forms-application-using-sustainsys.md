---
title: "How to initiate SAML 2.0 authentication in an ASP.NET Web Forms application using Sustainsys?"
date: "2024-12-23"
id: "how-to-initiate-saml-20-authentication-in-an-aspnet-web-forms-application-using-sustainsys"
---

Alright, let's tackle this one. I've spent a fair amount of time dealing with federated authentication, specifically SAML 2.0, and the intricacies of hooking it up in various .net environments. It's not always a walk in the park, particularly in older Web Forms apps where things can get a little… bespoke. We'll focus on doing this using the sustainsys library, as it's my go-to for saml in .net. It's robust and handles a lot of the heavy lifting. Let’s break down the process, covering the key steps, and then I'll walk through a few code snippets.

Essentially, initiating a saml 2.0 authentication flow with sustainsys involves a couple of fundamental actions: redirecting the user to the identity provider (idp), and then subsequently handling the assertion response when they are sent back. The whole process starts with the user clicking 'login' or accessing a protected resource, at which point your application, acting as the service provider (sp), needs to send an authentication request to the idp. Sustainsys simplifies this by providing a clear api for doing just that.

Before diving into code, understand the foundational elements: your application requires metadata about the idp (endpoint urls, signing certificates, etc), and your idp needs metadata about your sp (redirect url, signing certificates, entity id, etc). Getting these right, matching, is critical. If these pieces aren't aligned properly, you’ll face cryptic errors and a frustrating debugging session. This is usually the largest source of pain in SAML integration. Double, triple-check your metadata on both sides.

Now, let’s look at how you'd implement this in a typical ASP.NET web forms application. First, I’m assuming you have already installed the sustainsys package via nuget (typically the 'sustainsys.saml2' package), and configured some rudimentary settings in the `web.config`. I'm using the `<sustainsys.saml2>` section within the `configuration` in the `web.config` to store and configure things. The examples below, while specific to what we’re doing here, still need to be configured with actual metadata values for your specific IdP. Don’t try to copy/paste this without changing relevant bits.

**Example 1: Redirecting to the Identity Provider (Login Initiation)**

Let's start with the code responsible for initiating the SAML login process when the user tries to access a protected resource, or explicitly clicks the 'login' button. This usually happens in an aspx page or code-behind file.

```csharp
using System;
using System.Web;
using Sustainsys.Saml2;
using Sustainsys.Saml2.Configuration;

public partial class Login : System.Web.UI.Page
{
    protected void Page_Load(object sender, EventArgs e)
    {
        if (!IsPostBack)
        {
            var options = new Options
            {
                ReturnUrl = Request.Url.ToString(),
                ForceAuthn = true // Example: force the user to reauthenticate on every login
            };
            
            try
            {
               var response = Saml2Service.Authenticate(options);

                if (response.Status == Saml2AuthenticationStatus.Redirect)
                {
                   Response.Redirect(response.Location); // Redirect to IDP
                } else {
                   throw new Exception("Saml2Service did not return a redirect");
                }
               
            }
            catch (Exception ex)
            {
                //Log exception, handle properly and display a message for the user
                //this can be an issue with metadata misconfiguration, for instance
               Response.Write("An error occurred during login initialization: " + ex.Message);
            }
        }
    }
}
```

Here, `Saml2Service.Authenticate()` does all the heavy lifting. It takes an `Options` parameter, where you can define things like the `ReturnUrl` (where to send the user after successful authentication) and `ForceAuthn`. Notice the `Try-Catch` block as its important to log and react to issues at the authentication inititiation stage. You redirect the user using the location that sustainsys gives you. It's essentially building the required authentication request.

**Example 2: Handling the SAML Assertion Response**

After the idp authenticates the user, it sends the browser back to your application (specifically, to the configured acs url). You need to capture that response and extract the user's identity. This handling of the assertion is typically done within a class derived from the `Saml2Controller`, usually the `AssertionConsumerServiceController`. Sustainsys handles a large portion of this automatically by using the registered middleware to bind it to the configured acs url.

Here's a snippet of how this might look in your custom acs controller:

```csharp
using System;
using System.Web;
using Sustainsys.Saml2;
using Sustainsys.Saml2.Mvc;
using Sustainsys.Saml2.Web;
using System.Security.Claims;


public class AssertionConsumerServiceController : Saml2Controller
{
        public override ActionResult Index()
        {
           var result = this.GetResult();
           if (result.Status == CommandResultStatus.Success)
           {
              var samlIdentity = result.Principal as ClaimsPrincipal;
              if (samlIdentity != null)
               {
                  
                   var nameIdentifier = samlIdentity.FindFirst(ClaimTypes.NameIdentifier)?.Value; // Example retrieval of a Name Identifier
                   var name = samlIdentity.FindFirst(ClaimTypes.Name)?.Value; // Example retrieval of a name

                   // Store in session, or ticket, as appropriate for your specific app. 
                   HttpContext.Current.Session["saml_name_identifier"] = nameIdentifier;
                   HttpContext.Current.Session["saml_name"] = name;
                   
                   var returnUrl = Saml2Service.GetReturnUrl(this.Request);
                   if(!string.IsNullOrWhiteSpace(returnUrl)){
                        return Redirect(returnUrl);
                   }
                   else{
                        return Redirect("/default.aspx"); //or whatever the default is when no returnUrl was sent
                   }


               }
           }
           else {
               // Handle error scenarios gracefully
               return new HttpStatusCodeResult((int)System.Net.HttpStatusCode.InternalServerError, "Saml authentication failed");

           }


           return new HttpStatusCodeResult((int)System.Net.HttpStatusCode.InternalServerError, "Invalid controller handling");


        }

}
```

The `Index()` method in your custom `AssertionConsumerServiceController` uses `GetResult` to parse the SAML response. If it's successful, we cast the returned principal into a `ClaimsPrincipal`, which contains the assertions made by the idp. You can access the user's information using claim types (like `ClaimTypes.NameIdentifier`, `ClaimTypes.Name`, etc). It’s a good practice to set the session values to hold the authentication context of the user. And of course, as always, catch and react to errors gracefully.

**Example 3: Retrieving SAML session information to check if a user is authenticated**

Let's now consider how you would access the session in your application. This is often necessary when checking whether a user is authenticated or to retrieve authentication related information. This information can be in session variables, or more complex such as in a cookie.

```csharp
using System;
using System.Web;
using System.Security.Principal;

public partial class Default : System.Web.UI.Page
{
    protected void Page_Load(object sender, EventArgs e)
    {
        if (HttpContext.Current.Session["saml_name_identifier"] != null)
        {
            // User is authenticated
             var name = (string)HttpContext.Current.Session["saml_name"];

           WelcomeLabel.Text = "Welcome back, " + name + "!";
           LoginButton.Visible = false;
           LogoutButton.Visible = true;
           ProtectedButton.Visible = true;
        }
        else
        {
           WelcomeLabel.Text = "Not logged in yet";
           LoginButton.Visible = true;
           LogoutButton.Visible = false;
           ProtectedButton.Visible = false;
        }
    }

    protected void LoginButton_Click(object sender, EventArgs e)
    {
       Response.Redirect("/login.aspx");
    }

    protected void LogoutButton_Click(object sender, EventArgs e)
    {
        //Clear the session values
        Session.Remove("saml_name_identifier");
        Session.Remove("saml_name");

        Response.Redirect("/default.aspx");
    }

     protected void ProtectedButton_Click(object sender, EventArgs e)
     {
           Response.Redirect("/protected.aspx");
     }

}
```

In this example, when the `Default.aspx` page is loaded, we check for the session values set during the authentication. Based on those values, we display a specific message and set visibility to the relevant elements to either trigger the login or logout flow. This type of code needs to be implemented for every page that requires authentication. There is a way to do it with better code using handlers (using middleware), which is something worth considering.

For more in-depth knowledge, I'd recommend checking out the official documentation for Sustainsys.Saml2, which is comprehensive. Also, "Understanding Identity and Access Management" by Ivor Macfarlane provides a solid theoretical foundation for anyone getting into the field. Furthermore, studying the SAML 2.0 specification directly (available on OASIS) can offer a deeper grasp of the protocol itself.

Remember, SAML is intricate, so be prepared to debug a bit. I’ve had to troubleshoot my fair share of metadata errors and certificate mismatches. Start with simple use cases and progressively increase complexity. The most important advice is to meticulously test and double-check configurations. These small details will save you countless hours of headaches down the line. The Sustainsys library makes the developer experience much smoother, so embrace it. Good luck.
