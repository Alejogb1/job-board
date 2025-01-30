---
title: "How do I update user claims in IdentityServer4 (.NET Core 3.1)?"
date: "2025-01-30"
id: "how-do-i-update-user-claims-in-identityserver4"
---
IdentityServer4, built upon the principles of OAuth 2.0 and OpenID Connect, stores user-related data as claims within access and identity tokens. These claims represent facts about the user, such as their name, email address, roles, and permissions. Modifying these claims after the initial user authentication necessitates a clear understanding of IdentityServer4's pipeline and the available extension points. I've encountered scenarios where stale claims led to authorization failures, highlighting the critical need for a robust claim update mechanism.

The challenge arises because IdentityServer4, by design, aims to be stateless with respect to user data beyond authentication. The user data itself typically resides in a separate user store (e.g., a database or Active Directory). Therefore, we cannot directly modify the claims within IdentityServer4's internal mechanisms. Instead, we must intercept the token issuance process, retrieve fresh user data from the store, and then inject the updated claims into the outgoing tokens.

The primary method for updating user claims is through the implementation of a custom `IProfileService`. This service acts as an intermediary, invoked by IdentityServer4 at key points in the token issuance flow, enabling modification of claims before they are embedded within tokens. The interface defines two crucial methods: `GetProfileDataAsync` and `IsActiveAsync`.

`GetProfileDataAsync` is called when an access or identity token is being generated. This is where we retrieve the most up-to-date user information from our data store and transform it into claims.  The method receives a `ProfileDataRequestContext` which contains information about the subject and what type of claims are requested (e.g., requested via scopes). We use the subject ID to look up the corresponding user entity from the data store, and then proceed to construct the updated claims.

`IsActiveAsync` is invoked during token validation, checking if the user is still considered active. This method provides an opportunity to enforce user state, such as account lockouts. Again, the method receives a `IsActiveContext` containing the user’s subject and we use this to determine if the user is valid for token refresh.

Implementing a custom `IProfileService` requires injecting any necessary data access services via constructor injection.  This promotes modularity and facilitates unit testing.  It’s important to note that both methods are asynchronous and non-blocking operations should be performed.  Additionally, caching mechanisms should be considered to optimize performance, ensuring frequent data lookups are avoided.

Below are three code examples demonstrating the implementation of a custom `IProfileService`, highlighting various aspects and common scenarios.

**Example 1: Basic Claim Update**

This example showcases a simple update of claims.  It assumes a User entity with Name and Email properties, stored via a UserDataService, accessible via constructor injection.  Note the use of the `AddClaims` helper function to ease claim creation.

```csharp
using IdentityServer4.Models;
using IdentityServer4.Services;
using System.Collections.Generic;
using System.Security.Claims;
using System.Threading.Tasks;

public class CustomProfileService : IProfileService
{
    private readonly IUserDataService _userDataService;

    public CustomProfileService(IUserDataService userDataService)
    {
        _userDataService = userDataService;
    }

    public async Task GetProfileDataAsync(ProfileDataRequestContext context)
    {
        var subjectId = context.Subject.FindFirst(ClaimTypes.NameIdentifier)?.Value;

        if (subjectId == null)
        {
            return;
        }
        
        var user = await _userDataService.GetUserByIdAsync(subjectId);

        if (user != null)
        {
             var claims = new List<Claim>
            {
                new Claim(ClaimTypes.Name, user.Name),
                new Claim(ClaimTypes.Email, user.Email)

            };

           context.IssuedClaims.AddRange(claims);
        }
    }

    public async Task IsActiveAsync(IsActiveContext context)
    {
      var subjectId = context.Subject.FindFirst(ClaimTypes.NameIdentifier)?.Value;

        if (subjectId == null)
        {
             context.IsActive = false;
              return;
        }
      
      var user = await _userDataService.GetUserByIdAsync(subjectId);

         context.IsActive = user != null && user.IsActive;
    }
}

// Example User Data Interface
public interface IUserDataService
{
    Task<User> GetUserByIdAsync(string id);
}

// Example User Class
public class User
{
    public string Id { get; set; }
    public string Name { get; set; }
    public string Email { get; set; }
    public bool IsActive {get; set;}
}


```
*Commentary:* The `GetProfileDataAsync` method retrieves a user record from `_userDataService` based on the provided subject ID. It then creates claims for Name and Email and adds these claims to the context’s `IssuedClaims` collection. The `IsActiveAsync` checks that the user is both valid and active, setting `context.IsActive` to control refresh token activity.

**Example 2: Role-Based Claims**

Building upon the first example, this showcases how to include role claims based on user permissions. Assuming a User entity now has a Roles property, representing their assigned roles in the application.

```csharp
using IdentityServer4.Models;
using IdentityServer4.Services;
using System.Collections.Generic;
using System.Security.Claims;
using System.Threading.Tasks;

public class CustomProfileService : IProfileService
{
    private readonly IUserDataService _userDataService;

    public CustomProfileService(IUserDataService userDataService)
    {
        _userDataService = userDataService;
    }

    public async Task GetProfileDataAsync(ProfileDataRequestContext context)
    {
          var subjectId = context.Subject.FindFirst(ClaimTypes.NameIdentifier)?.Value;

        if (subjectId == null)
        {
            return;
        }
        
        var user = await _userDataService.GetUserByIdAsync(subjectId);

        if (user != null)
        {
             var claims = new List<Claim>
            {
                new Claim(ClaimTypes.Name, user.Name),
                new Claim(ClaimTypes.Email, user.Email),
             };
                
            // add role claims
            if(user.Roles != null && user.Roles.Count > 0 )
            {
                foreach (var role in user.Roles)
                {
                    claims.Add(new Claim(ClaimTypes.Role, role));
                 }
            }
             context.IssuedClaims.AddRange(claims);
        }

    }

    public async Task IsActiveAsync(IsActiveContext context)
    {
        var subjectId = context.Subject.FindFirst(ClaimTypes.NameIdentifier)?.Value;

        if (subjectId == null)
        {
             context.IsActive = false;
              return;
        }
      
      var user = await _userDataService.GetUserByIdAsync(subjectId);

         context.IsActive = user != null && user.IsActive;
    }
}

// Example User Data Interface
public interface IUserDataService
{
    Task<User> GetUserByIdAsync(string id);
}

// Example User Class
public class User
{
    public string Id { get; set; }
    public string Name { get; set; }
    public string Email { get; set; }
    public List<string> Roles { get; set; }
        public bool IsActive {get; set;}
}
```
*Commentary:* In addition to basic claims, this implementation iterates over a `Roles` property within the `User` entity and generates role claims using `ClaimTypes.Role`.  This ensures that applications that leverage role-based authorization can correctly utilize the token.

**Example 3: Handling External Authentication Providers**

This example demonstrates handling different types of subject IDs, which can come from external identity providers (e.g., Google, Facebook) where the subject claim is not always  `ClaimTypes.NameIdentifier`.

```csharp
using IdentityServer4.Models;
using IdentityServer4.Services;
using System.Collections.Generic;
using System.Security.Claims;
using System.Threading.Tasks;

public class CustomProfileService : IProfileService
{
    private readonly IUserDataService _userDataService;

    public CustomProfileService(IUserDataService userDataService)
    {
        _userDataService = userDataService;
    }

    public async Task GetProfileDataAsync(ProfileDataRequestContext context)
    {
      
        var subjectId = context.Subject.FindFirst(ClaimTypes.NameIdentifier)?.Value;
        
       if(subjectId == null)
        {
          // attempt to find other valid subject claim
          subjectId = context.Subject.FindFirst("sub")?.Value; // common external claim
        }
        
        if(subjectId == null)
        {
          return;
        }
          
        var user = await _userDataService.GetUserBySubjectIdAsync(subjectId);

        if (user != null)
        {
            var claims = new List<Claim>
            {
                new Claim(ClaimTypes.Name, user.Name),
                new Claim(ClaimTypes.Email, user.Email),
            };

             if(user.Roles != null && user.Roles.Count > 0 )
             {
               foreach (var role in user.Roles)
                {
                  claims.Add(new Claim(ClaimTypes.Role, role));
                }
            }
            context.IssuedClaims.AddRange(claims);
        }
     }


    public async Task IsActiveAsync(IsActiveContext context)
    {
        var subjectId = context.Subject.FindFirst(ClaimTypes.NameIdentifier)?.Value;
            if (subjectId == null)
            {
             // attempt to find other valid subject claim
            subjectId = context.Subject.FindFirst("sub")?.Value; // common external claim
            }
            if(subjectId == null)
        {
             context.IsActive = false;
              return;
        }
      
      var user = await _userDataService.GetUserBySubjectIdAsync(subjectId);

         context.IsActive = user != null && user.IsActive;
    }
}

// Example User Data Interface
public interface IUserDataService
{
    Task<User> GetUserBySubjectIdAsync(string subjectId);
}

// Example User Class
public class User
{
    public string SubjectId { get; set; }
    public string Name { get; set; }
    public string Email { get; set; }
    public List<string> Roles { get; set; }
    public bool IsActive {get; set;}
}
```
*Commentary:* In this example, the code checks for the standard `ClaimTypes.NameIdentifier`. If that claim isn't found, it checks for a `sub` claim which is a common identifier used by external providers such as Google,  and uses this as the user key to find the user. The IUserDataService interface method has also been updated to allow for the lookup. This highlights the need to handle various subject claim types.

To activate a custom profile service, it must be registered in the IdentityServer4’s dependency injection container within your `Startup.cs` file:

```csharp
public void ConfigureServices(IServiceCollection services)
{
    // existing service configuration...

    services.AddIdentityServer()
        .AddDeveloperSigningCredential()
         .AddInMemoryApiResources(Config.Apis)
        .AddInMemoryClients(Config.Clients)
        .AddTestUsers(Config.Users)
        .AddProfileService<CustomProfileService>();

    services.AddTransient<IUserDataService, ExampleUserDataService>();

    // other configurations...
}
```

The `AddProfileService` extension method registers the custom implementation, replacing the default. The `IUserDataService` service will also need to be added as a transient service.

For further study, the official IdentityServer4 documentation provides comprehensive information about `IProfileService` and token customization. Also beneficial are texts covering OAuth 2.0 and OpenID Connect standards, fostering a deeper understanding of the underlying protocols and best practices.  Examining the source code of the IdentityServer4 project can also provide invaluable insights into the implementation details. Finally, articles and blogs focused on security best practices for token handling and user management are highly recommended.
