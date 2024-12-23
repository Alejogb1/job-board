---
title: "how to measure exactly what data out in azure web app?"
date: "2024-12-13"
id: "how-to-measure-exactly-what-data-out-in-azure-web-app"
---

 so you're asking about figuring out exactly what data is flowing in and out of an Azure Web App that's a classic problem I've wrestled with a fair bit myself over the years Let's break this down because there isn't one single magic bullet solution it's more of a toolbox approach and what you reach for depends a lot on what exactly you're trying to measure

First up you gotta differentiate between network traffic and app-level data flow Network traffic is the raw bytes going across the wire http requests responses etc while app-level is like the data your app actually processes the actual database queries the json payloads and so on Network stuff you're often interested in the bandwidth usage how many connections you're seeing maybe even some latency data App data well that’s about the structure of the data the size of requests how often certain API endpoints are hit etc

Let’s start with the network side of things I’ve found a couple of approaches that are generally useful

**Network Monitoring Azure Networking Tools**

Azure itself provides some built-in monitoring tools that can give you a decent overview of network traffic for your Web App especially if your web app is deployed with other Azure services like virtual networks This is where the Azure Monitor tool comes into play

You are likely to use Network Watcher and its associated services like NSG flow logs If you use Network Watcher for your VM on the same virtual network as your Web App you can capture and query the flow logs. Network Watcher lets you view network flows going in and out of your network interface. It can be a bit coarse but it's a good starting point

Here's how you might enable NSG flow logs using the Azure CLI it's usually my first go to method

```bash
az network watcher flow-log create \
  --resource-group MyResourceGroup \
  --location WestUS2 \
  --nsg MyNetworkSecurityGroup \
  --storage-account MyStorageAccount \
  --enabled true
```

Once enabled the logs go to the storage account and you can analyze them later you can also use log analytics to query these logs that is a way to see the network traffic volume and source of request

Now the drawback with these is they are not very specific they don't tell you what data is being passed only that there is traffic flowing in and out its just raw network traffic so you can see things like the source IP destination IP and protocol but not say what that data was about or if it was malformed or an application layer error that did not propagate to the lower layers

**Application Insights**

Now when we talk about the *application level* data flow this is where Application Insights really shines It's not just about performance metrics it can also tell you what your app is sending and receiving this is where you can really dive into how your API is behaving This requires some instrumentation in your app but the payoff is worth it for this specific type of analysis

I've spent many a late night debugging complex APIs and Application Insights helped me narrow down all of the weird corner cases. There was this time I was using signalR and my front end was working perfectly but the back end was acting weird and then I had to use app insights to look at the messages and it turns out it was a client side version incompatibility. App Insights for sure helped me narrow it down.

To start using Application Insights you have to add the NuGet package to your Web App project. Usually the ASPNET core project

```csharp
// In your Startup.cs (ASP.NET Core)
public void ConfigureServices(IServiceCollection services)
{
    // ... other service registrations
    services.AddApplicationInsightsTelemetry();
}
```

This gives you basic telemetry out of the box request duration exceptions you know all the basic stuff. You can also log custom events that you want to see like the content of a specific data request or the response. Let's say you're making an external API call you can log that and how long it takes and the result.

```csharp
// Sample Application Insights logging
using Microsoft.ApplicationInsights;

public class MyService
{
  private readonly TelemetryClient _telemetryClient;
  public MyService(TelemetryClient telemetryClient)
  {
     _telemetryClient = telemetryClient;
  }

  public async Task ProcessRequestAsync(string id)
  {
    _telemetryClient.TrackEvent("Request Started", new Dictionary<string, string> {{"id",id}});
   try {
      // Do something
       _telemetryClient.TrackEvent("Request Completed", new Dictionary<string, string> {{"id",id}});
   }
    catch(Exception e) {
      _telemetryClient.TrackException(e, new Dictionary<string, string> {{"id",id}});
       throw;
    }
  }
}
```

With the code like this you can track exactly when a function is called and if the error happens there so you can use this and track what parameters are passed and the result you get back. This gives you a good overall view and you can even go to the telemetry explorer to query what requests have been happening based on your own properties. For instance you can select only a specific id that you want to track

Now if you need to get even more into the nitty gritty you may need to go deeper. In my early days I ended up doing this by manually logging but now I know a better way.

**Advanced Data Analysis tools**

 Sometimes App Insights isn't enough to get super specific details about the data itself If you need to see the exact payloads or if you want more flexibility you'll need to go another level deeper You could do some form of custom logging in your app itself

One method is to add a middleware to track all requests and responses in a format you can control. This can be very powerful but it also means adding your own stuff and if done wrong it might be slow or expose private data you should be careful

Here's an example of middleware that logs requests and responses:

```csharp
// Example middleware for logging
using System;
using System.IO;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Logging;

public class LoggingMiddleware
{
    private readonly RequestDelegate _next;
    private readonly ILogger _logger;

    public LoggingMiddleware(RequestDelegate next, ILoggerFactory loggerFactory)
    {
        _next = next;
        _logger = loggerFactory.CreateLogger<LoggingMiddleware>();
    }

    public async Task InvokeAsync(HttpContext context)
    {
        string requestBody = await ReadRequestBody(context.Request);
        _logger.LogInformation($"Request: {context.Request.Method} {context.Request.Path} Body: {requestBody}");
        var originalBodyStream = context.Response.Body;

       using var responseBody = new MemoryStream();
            context.Response.Body = responseBody;
        await _next(context);
         responseBody.Seek(0, SeekOrigin.Begin);
        string responseBodyString = await new StreamReader(responseBody).ReadToEndAsync();
        responseBody.Seek(0, SeekOrigin.Begin);

        _logger.LogInformation($"Response: {context.Response.StatusCode} Body: {responseBodyString}");
        await responseBody.CopyToAsync(originalBodyStream);

    }

     private async Task<string> ReadRequestBody(HttpRequest request)
    {
        request.EnableBuffering();
        using var stream = new StreamReader(request.Body);
        var body = await stream.ReadToEndAsync();
        request.Body.Seek(0, SeekOrigin.Begin);
        return body;
    }
}

// In your Startup.cs Configure method add the line app.UseMiddleware<LoggingMiddleware>();
```

This gives you full control over logging request and responses with body This can get expensive but if you are testing or debugging it can help. You can configure the logger to be stored where you want.

**Resource Recommendations**

Instead of just giving you links I find its better to point you to some core resources:

*   "Microsoft Azure Architectures" by John Savill. This book gives an overview of the Azure ecosystem and it touches all of these aspects in different contexts it will give you an understanding of the platform and how the different tools interoperate
*   "Cloud Native Patterns" by Cornelia Davis has also helped me think about how to apply these data monitoring techniques within microservices. This book has helped me decouple the problem with application performance and data analysis and gave me some good patterns to follow when measuring all of the things I need to measure
*   The official Azure documentation is always a solid go-to if you are more of a reference person
*   For specifics on Application Insights "Microsoft Application Insights" is always the best place to be

**Final thoughts**

Measuring data flow isn’t a simple thing, it’s a layered problem You use Network Watcher for low level network details and then use App Insights for the application level stuff. You also might need to add your own custom logging if you need something specific It is important that you use the right tool for the right job and you always have to remember to think of the cost this can be expensive so make sure you are not doing more logging than you need If you have questions or want to delve into specific parts I can dive deeper. It's not rocket science...well unless you're dealing with the data coming from a rocket then maybe it is
