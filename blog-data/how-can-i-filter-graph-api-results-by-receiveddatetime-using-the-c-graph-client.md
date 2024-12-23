---
title: "How can I filter Graph API results by ReceivedDateTime using the C# Graph client?"
date: "2024-12-23"
id: "how-can-i-filter-graph-api-results-by-receiveddatetime-using-the-c-graph-client"
---

Alright, let's dive into filtering Graph API results by `ReceivedDateTime` using the C# Graph client. It’s something I've dealt with extensively, especially when building custom email processing tools for a previous employer – that’s where the need for precise filtering really hits home. The Graph API, while powerful, has nuances when dealing with date and time, and getting your filters spot on is critical.

First off, the key here is understanding how the Graph API interprets date and time values, and specifically, the syntax you need to use in your OData filter query. It's not as simple as just plugging in a DateTime object directly. The Graph API requires date and time to be represented as ISO 8601 formatted strings. That means you're looking at something like `2023-10-27T10:00:00Z`, where `Z` signifies UTC timezone.

Now, let's get into the C# client specifics. You won’t directly manipulate a `DateTime` object within the `filter` method of the Graph client. You'll build your filter string with the correctly formatted date and time and then feed that string into the query. This avoids type mismatch and ensures the Graph API correctly interprets your date constraints. I've seen many developers make that mistake of trying to pass C# DateTime objects into the filter without properly converting them to the required string format, so I've developed the following method which is my usual go-to when handling such operations

**Example 1: Filtering for emails received after a specific date and time:**

```csharp
using Microsoft.Graph;
using System;
using System.Threading.Tasks;
using System.Collections.Generic;

public class GraphEmailExample
{
    public static async Task Main(string[] args)
    {
      //initialize a graph client, you will need to register an app on Azure and handle authentication.
      //for brevity's sake, let's assume you already have a `GraphServiceClient` instance called `graphClient`

      GraphServiceClient graphClient = /* initialization of graph client */;

      DateTimeOffset filterTime = new DateTimeOffset(2023, 10, 26, 12, 0, 0, TimeSpan.Zero); // UTC, October 26, 2023, 12:00:00
      string filterString = $"receivedDateTime ge {filterTime.UtcDateTime.ToString("o")}";

      try {

        var messages = await graphClient.Me.MailFolders.Inbox.Messages
            .Request()
            .Filter(filterString)
            .GetAsync();

          if (messages?.Value == null || messages.Value.Count == 0)
          {
            Console.WriteLine("No emails found matching the criteria.");
          }
          else
          {
              Console.WriteLine($"Found {messages.Value.Count} emails matching the specified criteria");

            foreach (var message in messages.Value)
            {
              Console.WriteLine($"Subject: {message.Subject}, Received: {message.ReceivedDateTime}");
            }
          }

       } catch (ServiceException ex) {

        Console.WriteLine($"Error: {ex.Message} Status code: {ex.StatusCode}");
       }

    }
}
```

Here, we're building a filter string that looks like, for instance, `receivedDateTime ge 2023-10-26T12:00:00Z`. The `ge` operator stands for "greater than or equal to".  The crucial part is `filterTime.UtcDateTime.ToString("o")` which ensures the proper ISO 8601 format. The `o` format specifier guarantees the UTC time is represented with the `Z` suffix and includes the necessary precision which the graph api expects.

**Example 2: Filtering for emails received within a specific date range:**

Let's move on to something slightly more advanced, such as filtering within a specific date range. This involves using multiple filter conditions combined using logical operators. In a project where I was migrating an on-prem email server to O365, filtering based on received date ranges was crucial for migrating emails in batches.

```csharp
using Microsoft.Graph;
using System;
using System.Threading.Tasks;
using System.Collections.Generic;


public class GraphEmailRangeExample
{
    public static async Task Main(string[] args)
    {

      //assuming the same `graphClient` initialization as above

      GraphServiceClient graphClient = /* initialization of graph client */;

      DateTimeOffset startTime = new DateTimeOffset(2023, 10, 20, 0, 0, 0, TimeSpan.Zero); // UTC, October 20, 2023, 00:00:00
      DateTimeOffset endTime = new DateTimeOffset(2023, 10, 27, 12, 0, 0, TimeSpan.Zero);   // UTC, October 27, 2023, 12:00:00

      string filterString = $"receivedDateTime ge {startTime.UtcDateTime.ToString("o")} and receivedDateTime le {endTime.UtcDateTime.ToString("o")}";

      try {

      var messages = await graphClient.Me.MailFolders.Inbox.Messages
           .Request()
           .Filter(filterString)
           .GetAsync();

          if (messages?.Value == null || messages.Value.Count == 0)
          {
            Console.WriteLine("No emails found matching the criteria.");
          }
          else
          {
              Console.WriteLine($"Found {messages.Value.Count} emails matching the specified criteria");
            foreach (var message in messages.Value)
            {
              Console.WriteLine($"Subject: {message.Subject}, Received: {message.ReceivedDateTime}");
            }
          }
      } catch (ServiceException ex) {

       Console.WriteLine($"Error: {ex.Message} Status code: {ex.StatusCode}");
      }
    }
}
```

Here, we use the `ge` (greater than or equal to) and `le` (less than or equal to) operators connected by an `and`. This effectively specifies a window of time where you want to retrieve your emails. Again, note how the `ToString("o")` ensures the correct formatting. This is something I learned the hard way after several failed attempts and debugging sessions.

**Example 3: Filtering for emails received before a specific date and time:**

Finally, just to show the flexibility, you might need to find emails that were received *before* a specific point. Here's how you'd do that:

```csharp
using Microsoft.Graph;
using System;
using System.Threading.Tasks;
using System.Collections.Generic;

public class GraphEmailBeforeExample
{
    public static async Task Main(string[] args)
    {
      //assuming the same `graphClient` initialization as above

      GraphServiceClient graphClient = /* initialization of graph client */;

      DateTimeOffset beforeTime = new DateTimeOffset(2023, 10, 28, 0, 0, 0, TimeSpan.Zero); // UTC, October 28, 2023, 00:00:00

        string filterString = $"receivedDateTime lt {beforeTime.UtcDateTime.ToString("o")}";

    try {
    var messages = await graphClient.Me.MailFolders.Inbox.Messages
             .Request()
             .Filter(filterString)
             .GetAsync();

        if (messages?.Value == null || messages.Value.Count == 0)
        {
          Console.WriteLine("No emails found matching the criteria.");
        }
        else
        {
            Console.WriteLine($"Found {messages.Value.Count} emails matching the specified criteria");
           foreach (var message in messages.Value)
           {
             Console.WriteLine($"Subject: {message.Subject}, Received: {message.ReceivedDateTime}");
           }
        }
    } catch (ServiceException ex) {

       Console.WriteLine($"Error: {ex.Message} Status code: {ex.StatusCode}");
    }
    }
}
```

This example uses the `lt` operator for "less than". It's important to note that the OData filter expressions are case-sensitive, so make sure to use the correct casing for properties like `receivedDateTime`.

In terms of further resources, I'd recommend diving into the *OData specification* itself. Understanding how OData filters work is fundamental when dealing with the Graph API. Look for the official documentation on OData filters, specifically focusing on the `datetime` type and the filter operators. Specifically, the ISO 8601 standard is essential for all of this. The *Microsoft Graph API documentation* is also your primary source for specifics on the filterable properties and syntax. The Graph API documentation has lots of examples and specific properties you can filter on, and it's good to consult the latest version of that resource for the most up-to-date practices. You will also benefit from reading the *Microsoft .Net Documentation* specifically on `DateTimeOffset` and string formatting. Understanding `DateTimeOffset` objects is crucial when dealing with timezones in a server side application. Lastly, the *Microsoft Authentication Library (MSAL) documentation* will also be useful to you for authentication, which you will need for your code to work correctly. This documentation explains the authentication flows required by the graph API and is essential for proper implementation.

In summary, when using the C# Graph client to filter by `receivedDateTime`, always ensure that you format your date and time values as ISO 8601 strings, specifically using the `"o"` format specifier when calling `ToString()` on your `DateTimeOffset` objects. Build your filter strings meticulously using the appropriate comparison operators (`ge`, `le`, `lt`, etc.) and logical operators (`and`, `or`) as needed. Finally, always check the official Microsoft Graph and OData documentation for the most up-to-date information. These techniques should enable you to build effective filters on date and time with the graph API.
