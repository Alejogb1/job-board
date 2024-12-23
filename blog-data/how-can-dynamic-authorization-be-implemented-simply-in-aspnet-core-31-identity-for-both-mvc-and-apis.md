---
title: "How can dynamic authorization be implemented simply in ASP.NET Core 3.1 Identity for both MVC and APIs?"
date: "2024-12-23"
id: "how-can-dynamic-authorization-be-implemented-simply-in-aspnet-core-31-identity-for-both-mvc-and-apis"
---

Okay, let’s unpack dynamic authorization in ASP.NET Core 3.1 Identity, specifically for both MVC applications and APIs. It’s a challenge I faced firsthand back in my time on the ‘Project Chimera’ team, when we moved from a static role-based system to something more granular and contextual. We definitely needed a solution that didn't drown us in configuration files or constant deployments.

Fundamentally, dynamic authorization pivots from a predefined, role-based model to one that uses contextual information at runtime. Rather than just asking “is this user an admin?”, we might ask “can this user modify *this specific document* at *this particular time*, based on their permissions and the state of the application?”. That's the shift in thinking.

The core concept relies on the `IAuthorizationPolicyProvider` and `IAuthorizationHandler` interfaces within ASP.NET Core’s authorization framework. The standard policy provider primarily looks for policy attributes on controllers or actions. We'll bypass that default mechanism and create something that interrogates context at runtime.

First, let’s handle the policy provider. Instead of looking for attribute markers, we'll create a custom policy provider that parses a convention from our application, or pulls the policy criteria from some centralized location like a database or even a file. I've leaned towards database storage in most implementations, allows for more dynamic changes. Here’s an example showing a simplified, database-driven policy provider. This example expects the policies to be saved in a database table named `AuthorizationPolicies`:

```csharp
using Microsoft.AspNetCore.Authorization;
using Microsoft.Extensions.Options;
using System.Threading.Tasks;
using System.Collections.Generic;
using System.Linq;
using Microsoft.EntityFrameworkCore;
using System;

public class DatabasePolicyProvider : IAuthorizationPolicyProvider
{
    private readonly IOptions<AuthorizationOptions> _options;
    private readonly AuthorizationDbContext _dbContext;

    public DatabasePolicyProvider(IOptions<AuthorizationOptions> options, AuthorizationDbContext dbContext)
    {
        _options = options;
        _dbContext = dbContext;
    }

    public Task<AuthorizationPolicy> GetDefaultPolicyAsync()
    {
        return Task.FromResult(_options.Value.DefaultPolicy);
    }


    public Task<AuthorizationPolicy> GetFallbackPolicyAsync()
    {
        return Task.FromResult(_options.Value.FallbackPolicy);
    }

    public async Task<AuthorizationPolicy> GetPolicyAsync(string policyName)
    {
        if (string.IsNullOrEmpty(policyName))
        {
            return null;
        }

        var policyDefinition = await _dbContext.AuthorizationPolicies
                            .FirstOrDefaultAsync(p => p.PolicyName == policyName);

        if (policyDefinition == null)
        {
             return null;
        }

        var policyBuilder = new AuthorizationPolicyBuilder();

        // Assuming your policyDefinition has a property 'Requirements' that
        // contains a serialized list of requirement strings.
        // The mapping should be done in a separate service if it is complex.
        foreach(var requirementString in policyDefinition.Requirements.Split(','))
        {
            switch(requirementString)
            {
                case "DocumentOwner":
                   policyBuilder.AddRequirements(new DocumentOwnerRequirement());
                   break;
                case "GroupMember":
                    policyBuilder.AddRequirements(new GroupMemberRequirement());
                    break;
                // Add other handlers based on your domain logic
            }
        }

        return policyBuilder.Build();
    }
}

public class AuthorizationPolicy
{
    public int Id { get; set; }
    public string PolicyName { get; set; }
    public string Requirements { get; set; }
}

public class AuthorizationDbContext : DbContext
{
    public AuthorizationDbContext(DbContextOptions<AuthorizationDbContext> options) : base(options) {}

    public DbSet<AuthorizationPolicy> AuthorizationPolicies { get; set; }
}
```

This `DatabasePolicyProvider` fetches authorization requirements based on the `policyName` from the database. The requirements are then turned into instances of `IAuthorizationRequirement`. This allows for a single `[Authorize]` attribute with custom policy names that will be resolved dynamically at runtime.

Next comes the handler itself, which dictates if the user meets the requirements set by the policy. We need an implementation of `IAuthorizationHandler` for each type of requirement you intend to implement. Here's an example implementation for a `DocumentOwnerRequirement` as was listed in the previous policy provider example:

```csharp
using Microsoft.AspNetCore.Authorization;
using System.Threading.Tasks;
using System;
using Microsoft.AspNetCore.Http;

public class DocumentOwnerRequirement : IAuthorizationRequirement
{
    // Empty, used as a type marker
}


public class DocumentOwnerHandler : AuthorizationHandler<DocumentOwnerRequirement>
{
    private readonly IHttpContextAccessor _httpContextAccessor;
    public DocumentOwnerHandler(IHttpContextAccessor httpContextAccessor)
    {
        _httpContextAccessor = httpContextAccessor;
    }

    protected override Task HandleRequirementAsync(AuthorizationHandlerContext context, DocumentOwnerRequirement requirement)
    {
        var documentId = _httpContextAccessor.HttpContext.Request.RouteValues["documentId"];
         if (documentId == null) { return Task.CompletedTask; }
        // Example implementation assumes that the authenticated user's id can be pulled from the current identity.
        var userId = context.User.FindFirst(System.Security.Claims.ClaimTypes.NameIdentifier)?.Value;

       // In a real application you'd fetch the document from the databse
       // and check if this user is indeed the owner, the logic is ommitted for brevity
       var isOwner = CheckIfUserOwnsDocument(userId, documentId.ToString());

        if (isOwner)
        {
            context.Succeed(requirement);
        }
        return Task.CompletedTask;
    }
    private bool CheckIfUserOwnsDocument(string userId, string documentId)
    {
        //Actual implementation ommitted, for example, using ORM or an API call
         return userId == "1" && documentId =="1";
    }
}
```

This handler now has access to the current `HttpContext` and can use route data, query parameters or anything else that is relevant to it, to determine if the request should be authorized. This approach is incredibly powerful when dealing with API endpoints, where route parameters or other parts of the HTTP request are highly contextual.

Finally, to tie it all together, you need to register these services in your `Startup.cs` or `Program.cs`:

```csharp
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Configuration;


public class Startup
{

    public Startup(IConfiguration configuration)
    {
        Configuration = configuration;
    }

    public IConfiguration Configuration { get; }

    public void ConfigureServices(IServiceCollection services)
    {
        services.AddDbContext<AuthorizationDbContext>(options =>
            options.UseInMemoryDatabase("AuthorizationDB")); //use your preferred database provider

        services.AddHttpContextAccessor();
        services.AddScoped<IAuthorizationHandler, DocumentOwnerHandler>();
        //Add the group handler and others
         services.AddScoped<IAuthorizationHandler, GroupMemberHandler>();
        services.AddSingleton<IAuthorizationPolicyProvider, DatabasePolicyProvider>();
        services.AddControllers();
    }

    public void Configure(IApplicationBuilder app, IWebHostEnvironment env)
    {
          app.UseRouting();

            app.UseAuthorization();

            app.UseEndpoints(endpoints =>
            {
                endpoints.MapControllers();
            });
    }
}
```

With these registrations, any controller or API endpoint can be decorated with the `[Authorize(Policy = "YourPolicyName")]` attribute, and the `DatabasePolicyProvider` will determine what requirements are necessary. The handlers will then determine if the user should be authorized based on the current context of the request.

The beauty of this setup lies in its flexibility and control. Dynamic policy updates don’t require application redeployments, as changes to policy criteria in the data source are reflected instantly. This model, while slightly more complex to set up, provides much more flexibility than a purely static role-based system. I recommend reading “Programming Microsoft ASP.NET Core” by Dino Esposito for a good grounding on ASP.NET Core's authorization mechanisms. Furthermore, "Patterns of Enterprise Application Architecture" by Martin Fowler can offer broader guidance on this kind of architectural design. These sources were fundamental to how we tackled similar challenges on Project Chimera, and I found the knowledge they provided to be crucial for a successful implementation. This is a solid approach for implementing dynamic authorization that works well across both MVC and API use cases, making the authentication process a lot more intelligent and manageable in the long term.
