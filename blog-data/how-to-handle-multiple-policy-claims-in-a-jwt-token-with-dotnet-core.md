---
title: "How to handle multiple policy claims in a JWT token with Dotnet core?"
date: "2024-12-15"
id: "how-to-handle-multiple-policy-claims-in-a-jwt-token-with-dotnet-core"
---

alright,  multiple policy claims in jwt, right? i’ve been down this road a few times, and it can get a bit messy if you're not careful. dotnet core’s authentication system is pretty flexible but needs a bit of a nudge when dealing with complex jwt claims. 

first off, the core issue is that the standard claims-based authorization in dotnet core often assumes a single value per claim type. when a jwt comes along and has, say, multiple `role` claims or a custom `permission` claim with an array of values, things can start to fail silently. that's the fun part, debugging those silent failures!

i remember back in my early days working on this e-commerce platform, i had this really cool idea of a super user role that could, you know, basically do everything. we encoded it by adding a "superuser" value to the `role` claim in jwt. then, as the platform got more complex, other roles came into existence: "admin", "editor", "customer". so, we ended up needing multiple roles per user, not just one single role, and the existing dotnet core setup just choked. the middleware was only looking for *one* value, and it wasn’t picking up on the fact that a user could have multiple roles. we had to scramble quite a bit, and that’s when i got into understanding how claims work under the hood.

so, how do we get around this? the secret sauce is in how you configure your authentication handler and how you define your policies. here’s the lowdown.

the common approach is to leverage the `claimsidentity` and its helper methods. we need to basically iterate through those claims and see if they contain any of the claims we need.

here’s a simple example of how to define a policy that checks for multiple allowed role values:

```csharp
services.AddAuthorization(options =>
{
    options.AddPolicy("RequireRoles", policy =>
    {
       policy.RequireClaim("role");
       policy.RequireAssertion(context =>
       {
          var roles = context.User.Claims.Where(c => c.Type == "role").Select(c => c.Value);
          return roles.Any(role => role == "admin" || role == "superuser");
        });
    });
});
```
in this snippet, we're adding a policy named `"requireroles"`. it first checks if the `role` claim exists, then within the `requireassertion` part, we are manually reading the `role` claim, mapping it to value and checking if any of them exist either as `"admin"` or `"superuser"`.

we're using `requireclaim` first as a simple pre-check to make sure that a basic claim exists, if not we skip the assertion, it's a performance trick i picked up years ago.

now, if you want more complex claims logic, like permission-based authorization, or want to use more dynamic approaches you can use the claims principle object directly. here's how you can validate a `permissions` claim that contains an array of permission strings:

```csharp
services.AddAuthorization(options =>
{
    options.AddPolicy("RequirePermissions", policy =>
    {
         policy.RequireAssertion(context =>
            {
               var permissionsClaim = context.User.Claims.FirstOrDefault(c => c.Type == "permissions");
                if (permissionsClaim == null)
                {
                    return false;
                }

               try
               {
                   var permissions = System.Text.Json.JsonSerializer.Deserialize<string[]>(permissionsClaim.Value);
                   if(permissions == null) return false;
                   return permissions.Contains("read") || permissions.Contains("edit");
                }
                catch
                {
                   return false;
                }
            });
    });
});
```

in this case, the `permissions` claim holds a json array. we're deserializing it, doing a null check for security reasons and then checking if the array contains either `"read"` or `"edit"` permission. obviously, you could add more permissions there, or handle it with a more robust approach. also the serialization can be customized, this example does not use the `System.text.json` options, but it is possible for better control.

now, to take it up a notch, you can abstract the permission handling logic into a custom authorization handler. here's a glimpse of that. first you need to define a requirement class for the policy:

```csharp
public class PermissionRequirement : IAuthorizationRequirement
{
    public string[] AllowedPermissions { get; }
    public PermissionRequirement(string[] allowedPermissions)
    {
        AllowedPermissions = allowedPermissions;
    }
}
```

then the handler itself:

```csharp
public class PermissionHandler : AuthorizationHandler<PermissionRequirement>
{
    protected override Task HandleRequirementAsync(AuthorizationHandlerContext context, PermissionRequirement requirement)
    {
        var permissionsClaim = context.User.Claims.FirstOrDefault(c => c.Type == "permissions");
        if (permissionsClaim == null)
        {
            return Task.CompletedTask;
        }
           try
           {
               var permissions = System.Text.Json.JsonSerializer.Deserialize<string[]>(permissionsClaim.Value);
                if(permissions == null) return Task.CompletedTask;
               if (permissions.Any(permission => requirement.AllowedPermissions.Contains(permission)))
                {
                    context.Succeed(requirement);
                }
            }
            catch
            {
            }
       return Task.CompletedTask;
    }
}
```

and this is how you hook it up in your `startup.cs`:

```csharp
services.AddAuthorization(options =>
{
    options.AddPolicy("RequireSpecificPermissions", policy =>
    {
        policy.Requirements.Add(new PermissionRequirement(new []{"read", "delete"}));
    });
});
services.AddSingleton<IAuthorizationHandler, PermissionHandler>();
```

this is a more modular approach, where you can create authorization requirements as you need and apply them in your code.

so yeah, this isn’t a walk in the park the first time you see it. i’ve spent many nights scratching my head at similar issues. it’s all about understanding the underlying mechanism of claims in dotnet core and how to leverage it to your needs. it’s like, why did the web dev cross the road? to get to the other site! (my best programmer joke, i know, i should work on it).

for learning more about this i highly recommend checking out the official microsoft documentation on authorization and policies in dotnet core. also, “programming microsoft asp.net mvc” by dino esposito, while slightly dated, it has deep knowledge that is still valid today, it helped me a lot at the start of my career. the book goes into the detail of authorization and it's not just about the syntax, but the logic behind. and i also found ‘asp.net core in action’ by andrew lock to be a good resource on the implementation aspect. those books were my go to during my learning time. they delve into the underlying principles better than any online tutorial can.

hope this makes sense, remember, don’t be shy to try things, test different approaches and experiment. that’s how we learn the best, through trial and error. also, test your code. always test your code.
