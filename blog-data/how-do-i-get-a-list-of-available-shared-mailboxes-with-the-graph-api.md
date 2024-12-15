---
title: "How do I get a list of available shared mailboxes with the Graph API?"
date: "2024-12-15"
id: "how-do-i-get-a-list-of-available-shared-mailboxes-with-the-graph-api"
---

alright, let's tackle this. getting a list of shared mailboxes using the graph api is something i've definitely had to deal with a few times. it’s not always as straightforward as you might hope, but it's certainly doable. i remember spending a solid afternoon on this a few years back when we were migrating user permissions across a large tenant and needed to audit shared mailbox access – it was a mess at first, but hey, we got it figured out.

first off, you're not going to find a single endpoint that just dumps all the shared mailboxes for you. the graph api works a bit more granularly than that. you have to approach it by querying user objects and then filtering for specific mailbox types. the key here is understanding that shared mailboxes are technically user objects in azure active directory (aad), just with specific properties set to identify them as shared mailboxes rather than regular user accounts.

the main differentiator we need to look for is the `mailboxType` property. this property for a regular user mailbox will typically be "user", while a shared mailbox will have it set to "shared". so, our approach will involve listing all user objects and then filtering based on this property. also, be aware that you will need the correct permission, `user.read.all` or `user.readbasic.all` should be sufficient if you want to query all users. if you want a subset of users, you will need to narrow down the filter scope and permissions accordingly.

let me show you a basic example using the graph api via an http request. this is the kind of code i typically use to perform a manual test against the endpoint before actually embedding into an application. you’d send this as a get request to the microsoft graph api:

```http
get https://graph.microsoft.com/v1.0/users?$filter=mailboxType eq 'shared'&$select=id,displayName,userPrincipalName
authorization: bearer {your_access_token}
```

this will return a json payload with the `id`, `displayname`, and `userprincipalname` of all user objects with the `mailboxType` property set to 'shared'.

now, that’s the basic http request. but if you’re in a scripting context, for example, using powershell you might want a script like this:

```powershell
#requires -modules Microsoft.Graph
#requires -version 7.0
connect-mggraph -scopes "user.read.all"

$sharedmailboxes = get-mggraphuser -filter "mailboxType eq 'shared'" -property id,displayname,userprincipalname

if ($sharedmailboxes)
{
    foreach ($mailbox in $sharedmailboxes) {
        write-host "mailbox id: $($mailbox.id)"
        write-host "mailbox name: $($mailbox.displayname)"
        write-host "mailbox upn: $($mailbox.userprincipalname)"
        write-host "------------------------"
    }
} else {
    write-host "no shared mailbox found."
}

disconnect-mggraph
```

this powershell snippet first makes sure we have the necessary module installed and version. then establishes a connection, and get the list of shared mailboxes, and finally, disconnects, pretty straight forward. this is my usual go-to when doing quick and dirty things against the graph api.

however, let's say you're working in a more structured environment and want to do this programmatically in c#. it looks a bit more involved but is still fairly easy to get working with the microsoft graph sdk.

```csharp
using microsoft.graph;
using microsoft.graph.auth;
using microsoft.identity.client;
using system;
using system.collections.generic;
using system.threading.tasks;

public class graphapihelper
{
    private graphserviceclient _graphclient;

    public graphapihelper(string tenantid, string clientid, string clientsecret)
    {
        var clientcredential = clientcredentialprovider.create(tenantid, clientid, clientsecret);
        _graphclient = new graphserviceclient(clientcredential);
    }

    public async task<list<user>> getsharedmailboxes()
    {
        var sharedmailboxes = new list<user>();
        var request = _graphclient.users.request().filter("mailboxType eq 'shared'").select("id,displayname,userprincipalname").top(999);
        var results = await request.getasync();
        if (results?.count > 0)
        {
            sharedmailboxes.addrange(results.currentpage);
            while (results.nextpage != null)
            {
                results = await results.nextpage.getasync();
                sharedmailboxes.addrange(results.currentpage);
            }
        }

        return sharedmailboxes;
    }

    public static async task main(string[] args)
    {
        //replace the following values with your own
        string tenantid = "your_tenant_id";
        string clientid = "your_app_client_id";
        string clientsecret = "your_app_client_secret";

        var graphhelper = new graphapihelper(tenantid, clientid, clientsecret);
        var mailboxes = await graphhelper.getsharedmailboxes();
        if (mailboxes?.count > 0)
        {
            foreach (var mailbox in mailboxes)
            {
                console.writeline($"mailbox id: {mailbox.id}");
                console.writeline($"mailbox name: {mailbox.displayname}");
                console.writeline($"mailbox upn: {mailbox.userprincipalname}");
                console.writeline("------------------------");
            }
        }
        else
        {
            console.writeline("no shared mailbox found.");
        }
    }
}
```

this c# example is using the microsoft graph sdk with client credential flow. it's doing the same filtering by `mailboxType` and retrieves the user's properties. the `top(999)` is to get the max number of results on the first page, then if results.nextpage is not null it will request the next page until no more results are found, because the graph api response has paging. you'll need to install the microsoft.graph and microsoft.identity.client nuget packages for this to work. this method is much more robust for production-ready code since it deals with the graph's paging.

one thing to keep in mind is that if you have a really large number of users, you will hit graph api throttling limits and using pagination properly is crucial to avoid that. when i was migrating user permissions, we hit this pretty fast, so make sure to implement it well in your project to deal with a large volume of users.

also, you have to be mindful of the permissions you are using. `user.read.all` or `user.readbasic.all` are the most obvious ones, but depending on your specific needs you might be able to use more restrictive permissions. less permissions is always the best practice.

in terms of further reading, i recommend exploring the official microsoft graph documentation (it's constantly updated). microsoft also has very helpful code samples. additionally, the book “programming microsoft graph” by eric schneider is a very good resource if you want to go deeper into the subject. the book "microsoft 365 development" by vesa jutila and waldek mast is also a good pick. they tend to go into the intricacies of how these systems work.

that's pretty much it. it's not really that hard once you understand the fundamentals of how shared mailboxes are structured within the aad model and understand what specific properties are needed. also, don’t forget to have your access token. what is the most used api in the graph? the “get” api.
