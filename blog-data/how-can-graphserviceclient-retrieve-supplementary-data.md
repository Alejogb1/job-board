---
title: "How can GraphServiceClient retrieve supplementary data?"
date: "2024-12-23"
id: "how-can-graphserviceclient-retrieve-supplementary-data"
---

,  It's not uncommon to find ourselves needing more than just the basic data returned by a `GraphServiceClient`. I've been down this path multiple times, particularly when dealing with complex user profiles and resource relationships in large enterprise environments. The standard requests often get you the core properties, but the supplementary data – those additional fields or expanded relationships – require a bit more finesse. The `GraphServiceClient`, when used within the microsoft graph api ecosystem, is exceptionally versatile in this regard, thankfully.

The key to retrieving this supplementary data lies in understanding the available mechanisms and implementing them correctly within your request. There are primarily two ways we achieve this: using `$select` to specify additional properties, and using `$expand` to retrieve related resources. We can also combine these for very specific data sets.

First, let's address `$select`. This query parameter allows you to explicitly define which properties you want returned in the response. By default, the microsoft graph api will return a default set of properties for most objects. If you need others, you *must* specify them using `$select`. It is not implicit. I recall a particularly frustrating debugging session where I spent hours trying to understand why a specific field wasn’t appearing in the response, only to find it was a case of neglecting `$select`. This isn't just a case of adding extra fields; it's also an important method to *reduce* the data volume, improving efficiency, especially when dealing with many objects.

Here’s a C# snippet using the .net Graph SDK to illustrate this:

```csharp
using Microsoft.Graph;
using Microsoft.Identity.Client;

// Assuming you have initialized the GraphServiceClient named graphClient

public async Task<User> GetUserWithAdditionalProperties(string userId)
{
    var user = await graphClient.Users[userId]
        .Request()
        .Select(user => new
        {
            user.Id,
            user.DisplayName,
            user.Mail,
            user.UserPrincipalName,
            user.OfficeLocation,
            user.CreatedDateTime,
            user.JobTitle,
            user.Department
        })
        .GetAsync();

    return user;
}
```

In this code, we’re not just getting the default properties for a user. We're specifically asking for `OfficeLocation`, `CreatedDateTime`, `JobTitle`, and `Department`, along with the common identifiers, by using `$select` effectively. If you omitted the select statement, you would only receive a subset of this information, which would lack the additional details you were trying to retrieve.

Now, let's talk about `$expand`. This parameter deals with retrieving relationships between resources. For example, if you're fetching a user, you might want to also get their manager or the groups they belong to. Simply requesting a user will not bring related resources without explicitly stating this using `$expand`. I once worked on an application that relied heavily on user-manager relationships. The first iteration failed miserably because I didn’t use `$expand`, and I was left with a lot of queries to try and pull the managers manually which just added to the total round trip times.

Here's another example, this time showing how to fetch a user along with their direct reports:

```csharp
using Microsoft.Graph;
using Microsoft.Identity.Client;

// Assuming you have initialized the GraphServiceClient named graphClient

public async Task<User> GetUserWithDirectReports(string userId)
{
    var user = await graphClient.Users[userId]
        .Request()
        .Expand("directReports")
        .Select(user => new
        {
           user.Id,
           user.DisplayName,
           user.DirectReports // Note: We include this to make sure we don't get a default list of a 'user'
        })
        .GetAsync();

    return user;
}
```

Here, `$expand("directReports")` is crucial. Without it, the `directReports` navigation property would be absent from the response. Note the `$select` to include the `directReports` - this is an important point to remember; without it, you *may* get empty list for `directReports`, or just a list of Id's if you're lucky. It forces the server to return the actual objects associated with the relationship rather than just links or identifiers. This means the returned user object will now have a `directReports` property that contains a collection of `user` objects representing their direct subordinates, *not just* their ids.

We can further combine these mechanisms for truly comprehensive requests. If, for example, you want a user, their manager’s name, department, and then a count of how many people their manager supervises, it’s achievable with some nested `$expand` and `$select` usage. This can get complex, but understanding how to construct these queries is important for efficient data retrieval.

```csharp
using Microsoft.Graph;
using Microsoft.Identity.Client;
using System.Linq;

// Assuming you have initialized the GraphServiceClient named graphClient
public async Task<User> GetUserWithManagerDetailsAndManagerReportCount(string userId)
{
    var user = await graphClient.Users[userId]
       .Request()
       .Select(u=> new {
            u.Id,
            u.DisplayName,
            u.Mail,
            u.UserPrincipalName
       })
        .Expand("manager($select=displayName, department;$expand=directReports($select=id))")
        .GetAsync();


    if(user.Manager!=null)
    {
      var numberOfDirectReports = user.Manager.DirectReports?.Count ?? 0;
      Console.WriteLine($"Manager {user.Manager.DisplayName} has {numberOfDirectReports} direct reports in department {user.Manager.Department}");
     }

    return user;
}

```
In this last example, we fetch a user's basic properties, *and* expand the manager relationship. Within the expanded manager resource, we request the manager’s display name, department and in turn, their `directReports`, all with the appropriate `$select` calls. This way you can gather hierarchical and interconnected data within a single, albeit slightly more complex, request. You can see how we can use the `.Select()` method in a few places to limit the amount of information returned by the graph api. This method in particular improves the speed and efficiency of the application.

For deeper understanding, I’d strongly suggest exploring the official Microsoft Graph documentation. Look into the OData query language specifications for fine-tuning your requests. "OData - The Query Language for the Web" is a useful starting point. Also, consider reviewing "Microsoft Graph API reference," it's where you'll find information on all the different resources, their relationships, and properties. And for the specific implementation within .NET, digging into the source code of `Microsoft.Graph` library on GitHub can also help in understanding the behind the scenes of `GraphServiceClient` interactions with the API. Additionally, reading the "Microsoft Graph SDK documentation" for your specific language can be useful.

In conclusion, retrieving supplementary data isn't inherently difficult, but it requires a clear understanding of `$select` and `$expand`, coupled with a careful consideration of the structure of the data you require. These examples should give you a solid foundation to build from.
