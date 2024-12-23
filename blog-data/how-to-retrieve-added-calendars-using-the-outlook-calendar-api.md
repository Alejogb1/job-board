---
title: "How to retrieve added calendars using the Outlook Calendar API?"
date: "2024-12-23"
id: "how-to-retrieve-added-calendars-using-the-outlook-calendar-api"
---

,  I’ve spent a fair amount of time working with the Outlook Calendar API, particularly around the complexities of managing not just primary calendars, but those additional ones users often create or subscribe to. The initial assumption is usually, “get calendars, done,” but it's never quite that simple, is it? Let's break down how to reliably retrieve these added calendars, drawing from experiences where a straightforward approach simply didn't cut it.

Fundamentally, when you’re querying the Outlook Calendar API, it’s essential to understand the underlying data model. There's a distinct difference between a user's primary calendar and the collection of secondary, or added, calendars. These added calendars aren't just copies; they have their own unique properties and identifiers. The standard graph api endpoint `/me/calendars` often gets you the user's default calendar, but won't necessarily give the others without some additional care.

So how do we go about this properly? The secret sauce often lies in leveraging a more precise query, sometimes using filters or expanding properties to get all of the calendars you need. The key is the `calendarGroup` relationship. Calendars aren't typically listed at the top level if they aren't primary, they're often associated with `calendarGroup` objects.

Let's move on to specifics. First, a basic example, using the microsoft graph api, that will show the primary and group calendars using C# which utilizes the Microsoft Graph SDK.

```csharp
using Microsoft.Graph;
using Microsoft.Identity.Client;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

public class CalendarService
{
    private GraphServiceClient _graphClient;

    public CalendarService(string clientId, string tenantId, string clientSecret)
    {
      var scopes = new[] {"Calendars.Read"};
        IConfidentialClientApplication confidentialClientApplication = ConfidentialClientApplicationBuilder
            .Create(clientId)
            .WithClientSecret(clientSecret)
            .WithAuthority(new Uri($"https://login.microsoftonline.com/{tenantId}"))
            .Build();
          
        var authProvider = new ClientCredentialProvider(confidentialClientApplication, scopes);
        _graphClient = new GraphServiceClient(authProvider);
    }

   public async Task<List<Calendar>> GetCalendars()
   {
        List<Calendar> allCalendars = new List<Calendar>();
        var userCalendars = await _graphClient.Me.Calendars
            .Request()
            .GetAsync();

        if (userCalendars?.Count > 0){
            allCalendars.AddRange(userCalendars);
        }

        var calendarGroups = await _graphClient.Me.CalendarGroups.Request().GetAsync();
        if (calendarGroups?.Count > 0)
        {
          foreach (var group in calendarGroups) {
            var groupCalendars = await _graphClient.Me.CalendarGroups[group.Id].Calendars.Request().GetAsync();
            if(groupCalendars?.Count > 0) {
              allCalendars.AddRange(groupCalendars);
            }
          }
        }
      return allCalendars;
    }
}
```

This snippet shows how to fetch all calendars, both at the root user level and those nested within calendar groups, which is often missed. This is important to realize as some calendars could be in groups. This approach fetches all user calendars which are not in a group, and then gets all user calendar groups, and iterates through each one to return the calendars in the groups.

Now, let's explore an equivalent example but this time in JavaScript with Node.js, which uses the Microsoft Graph SDK as well:

```javascript
const { Client } = require('@microsoft/microsoft-graph-client');
const { ConfidentialClientApplication } = require('@azure/msal-node');

async function getCalendars(clientId, tenantId, clientSecret) {
    const scopes = ["Calendars.Read"];

    const cca = new ConfidentialClientApplication({
        auth: {
            clientId: clientId,
            authority: `https://login.microsoftonline.com/${tenantId}`,
            clientSecret: clientSecret,
        }
    });
    const tokenRequest = {
        scopes: scopes,
      };
    const authProvider = {
          getAccessToken: async () => {
                const result = await cca.acquireTokenByClientCredential(tokenRequest);
                 return result.accessToken;
          }
    };
    const graphClient = Client.initWithMiddleware({
      authProvider: authProvider
    });

    let allCalendars = [];

    try {
         const userCalendars = await graphClient.api('/me/calendars').get();
        if (userCalendars?.value) {
            allCalendars = allCalendars.concat(userCalendars.value);
        }

         const calendarGroups = await graphClient.api('/me/calendarGroups').get();

        if (calendarGroups?.value) {
            for (const group of calendarGroups.value) {
               const groupCalendars = await graphClient.api(`/me/calendarGroups/${group.id}/calendars`).get();
                 if (groupCalendars?.value) {
                  allCalendars = allCalendars.concat(groupCalendars.value);
                }
            }
        }
      return allCalendars;
    } catch (error) {
        console.error("Error fetching calendars:", error);
        throw error;
    }
}
```

Similar to the C# example, this JavaScript snippet fetches all calendars at the root and within groups. Understanding how to perform this across different languages and SDKs reinforces the pattern, which is essential when dealing with various environments. The core logic is retrieving the root calendars, then fetching calendar groups, and then the calendars within each group.

Finally, lets show how this can be done directly in a web application making API calls directly using javascript and the browser's fetch API. This example is meant for web use cases, where you may not have access to the MSAL packages available for more controlled back-end scenarios.

```javascript
async function getCalendars(accessToken) {
  const graphAPIEndpoint = 'https://graph.microsoft.com/v1.0';
  let allCalendars = [];

  try {
        const rootCalendarsResponse = await fetch(`${graphAPIEndpoint}/me/calendars`, {
        method: 'GET',
        headers: {
          'Authorization': `Bearer ${accessToken}`,
          'Content-Type': 'application/json'
        }
    });

        const rootCalendarsData = await rootCalendarsResponse.json();
        if(rootCalendarsData?.value) {
          allCalendars = allCalendars.concat(rootCalendarsData.value);
        }
        
        const groupsResponse = await fetch(`${graphAPIEndpoint}/me/calendarGroups`, {
          method: 'GET',
          headers: {
              'Authorization': `Bearer ${accessToken}`,
              'Content-Type': 'application/json'
          }
        });
        const groupsData = await groupsResponse.json();
        if(groupsData?.value) {
            for (const group of groupsData.value) {
               const groupCalendarsResponse = await fetch(`${graphAPIEndpoint}/me/calendarGroups/${group.id}/calendars`, {
                  method: 'GET',
                  headers: {
                      'Authorization': `Bearer ${accessToken}`,
                      'Content-Type': 'application/json'
                    }
                });
                const groupCalendarsData = await groupCalendarsResponse.json();
                  if (groupCalendarsData?.value) {
                      allCalendars = allCalendars.concat(groupCalendarsData.value);
                  }
            }
         }

       return allCalendars;

    } catch (error) {
      console.error("Error getting Calendars", error);
      throw error;
    }
}
```
This JavaScript snippet demonstrates making direct calls to the API, fetching root calendars, then calendar groups, and finally the calendars in the groups. It uses `fetch` and is suitable for browser environments, which is a very different context than the previous two examples, thus it requires manual authorization and error handling that is a bit different than using a SDK.

To take this further, it’s crucial to delve deeper into the official documentation for the Microsoft Graph API. Specifically, I recommend the *Microsoft Graph documentation* itself (start with the section for calendars), along with any *training materials specific to the Graph SDKs*. Another worthwhile resource is the book *Programming Microsoft Graph* by Brian T. Jack, which offers more detailed examples. Don't ignore the example snippets provided within Microsoft's own documentation as well, as they often demonstrate the most up to date methods and approaches. These are crucial for understanding the nuances of API pagination, throttling, and change tracking – especially as your data needs grow more complex.

The practical point I've stressed here is, and based on experience, is to ensure to inspect each level of object hierarchy provided by the API. Don’t assume that a simple `/me/calendars` is all you’ll need. By understanding the role of the `calendarGroups` relationship, you’ll be well on your way to reliably retrieve all added calendars, avoiding those frustrations we all sometimes stumble into with APIs. Also be wary of caching and keep an eye out for API change logs; the Graph API continues to evolve. Hope this helps.
