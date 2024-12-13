---
title: "the servicepointmanager does not support proxies https error?"
date: "2024-12-13"
id: "the-servicepointmanager-does-not-support-proxies-https-error"
---

Okay so servicepointmanager proxies not playing nice huh Been there done that got the t-shirt literally I think I have a couple of those floating around from old projects involving flaky network setups and even worse API integrations This issue is like a classic case of "I thought I set everything up right but the universe hates me" situation that I encountered way too many times and I can understand the frustration especially when you are trying to get a system working with some third party APIs This is not an uncommon issue as you have probably guessed since you are asking here so let's break down what's likely going on and how I usually try to get out of it

First off ServicePointManager is a beast and it's supposed to handle these things for you You would expect that setting the proxy via the configuration files system level or the app config is enough and it should just work It doesn't always work like that and you are not alone There are more layers to this onion that you might initially think. So what usually happens when you set the proxy and you get HTTP errors It's basically that the ServicePointManager isn't picking up your proxy settings like you expect it to and probably ignoring it The ServicePointManager handles requests in batches using connection pooling which means that it will try to reuse connections when it can. And sometimes the proxy settings get messy when reusing connections if there were some connection failures and your settings get ignored leading to the error you are seeing

So before we get into specific code and configuration try checking a few obvious things first. Are you 100% sure your proxy is up and running You should be able to access resources through your browser using your proxy settings It might sound too basic but sometimes its not the code but a simple configuration failure or service being down at the network level Also check that the proxy address port number and username/password if you have it is correct and the proxy is available from the machine that your code is running If you use a network firewall or something related to a security program make sure that the ports related to the proxy are allowed

Once you have confirmed the proxy is actually working now it gets more interesting

Here is what I usually do the first step I've learned that sometimes overriding the ServicePointManager directly is more reliable than hoping it picks up config settings.

```csharp
using System;
using System.Net;

public class ProxyHelper
{
    public static void ConfigureProxy(string proxyAddress, int proxyPort, string proxyUser = null, string proxyPassword = null)
    {
         var proxy = new WebProxy(proxyAddress, proxyPort);

        if (!string.IsNullOrEmpty(proxyUser) && !string.IsNullOrEmpty(proxyPassword))
        {
            proxy.Credentials = new NetworkCredential(proxyUser, proxyPassword);
        }

         ServicePointManager.DefaultConnectionLimit = 100;
         ServicePointManager.SecurityProtocol = SecurityProtocolType.Tls12 | SecurityProtocolType.Tls11 | SecurityProtocolType.Tls;
         WebRequest.DefaultWebProxy = proxy;

         Console.WriteLine("Proxy configured successfully.");
     }
}

```

This snippet explicitly sets the proxy on `WebRequest.DefaultWebProxy` this should be sufficient in most cases This is a general-purpose version that covers authentication if required I also like to configure the `SecurityProtocol` to support the last versions of TLS and to increase the connection pool because sometimes the default 2 connections can be a bottleneck and the requests can timeout The `ServicePointManager.DefaultConnectionLimit` is there just to try to avoid any kind of connection limitations by the pooling mechanism

Now if that doesn't work there's a little thing you need to check about the httpclient implementation that you are using to make the call. The `HttpClient` uses a different approach for managing connections and proxies than the old `HttpWebRequest` The previous code might not apply to the newer `HttpClient` class because it uses internally a handler that can be configured differently. I usually try to apply the settings to the httpclient handler directly like this:

```csharp
using System;
using System.Net;
using System.Net.Http;

public class ProxyHttpClientHelper
{
  public static HttpClient CreateHttpClientWithProxy(string proxyAddress, int proxyPort, string proxyUser = null, string proxyPassword = null)
    {
        var proxy = new WebProxy(proxyAddress, proxyPort);

         if (!string.IsNullOrEmpty(proxyUser) && !string.IsNullOrEmpty(proxyPassword))
        {
            proxy.Credentials = new NetworkCredential(proxyUser, proxyPassword);
        }

        var handler = new HttpClientHandler
        {
            Proxy = proxy,
            UseProxy = true,
           ServerCertificateCustomValidationCallback = (sender, cert, chain, sslPolicyErrors) => true // for testing only remove in production

        };
       return new HttpClient(handler);
    }
}
```

Here, I create a `HttpClientHandler` set the proxy there and ensure the `UseProxy` property is set to `true` This is usually more reliable when dealing with the http client. Notice the `ServerCertificateCustomValidationCallback` I'm adding this in this example it is useful for debugging purposes only or for very specific cases. This will skip certificate validation and it is really bad practice for production but I added it here because I saw this being the cause of the issues many times while I was debugging. It allows you to quickly check that it is not a certificate issue by turning it on. Always turn it off once you solved the underlying cause and understand why the certificate was the issue

Also be aware that when you are working with very old frameworks or libraries they are sometimes not aware of the settings configured in `app.config` or are not compatible with them. I saw a colleague debugging an issue for almost a full day and he was using an old library that was actually ignoring the `app.config` settings all along and no configuration will have worked and no code will have solved this If the previous code snippets are not working check the libraries and the framework you are using to make the request. Sometimes the issue is in the framework itself and not your code or the proxy

Sometimes these configurations can be tricky to debug especially with the old HttpWebRequest class and the ServicePointManager if you don't have any logs enabled at your proxy level You might even encounter some other issues such as certificate problems while debugging your code through the proxy I highly recommend you use fiddler or wireshark to debug the traffic while you test your application with the proxy enabled. Wireshark is my usual choice but both are good options here. It's like being a detective except instead of a magnifying glass you have packet capture.

Finally if you are working with async calls be sure to configure the proxy before any async calls are made This is a race condition and may create problems that are difficult to debug. If you set it up after making a call it won't be applied in the current request You could create a `ProxyConfig` class and add it to the dependency injection and it would be applied only once when the application starts. It is a good practice to configure all external resources settings before starting any kind of network calls.

```csharp
public class ProxyConfig
{
   public string ProxyAddress {get;set;}
   public int ProxyPort {get;set;}
   public string ProxyUser {get;set;}
   public string ProxyPassword {get;set;}

   public void Configure()
   {
      if (!string.IsNullOrEmpty(ProxyAddress) && ProxyPort > 0)
         ProxyHelper.ConfigureProxy(ProxyAddress,ProxyPort,ProxyUser,ProxyPassword);
   }
}
```
And when you start your application before the first http request call configure proxy.

```csharp
  var config = new ProxyConfig(){
    ProxyAddress = "192.168.1.1",
    ProxyPort = 8080,
    ProxyUser = "proxy_user",
    ProxyPassword = "proxy_password"
    };
  config.Configure();

  //Start making network calls from here..
```

A resource that I recommend you study if you are deep diving into this area is "TCP/IP Illustrated, Volume 1: The Protocols" by W. Richard Stevens. It will provide you with a deep understanding of how network protocols operate under the hood. This is a must for any network engineer and anyone that wants to understand the network layer. If you want to go deeper on C# http and web related stuff check "Pro ASP.NET Core 3" by Adam Freeman. It is a good resource to have and will give you some insights about how http request works in .net with the new core version of the framework.

Hope this helps you solve your issues and if you have more questions come back here. I know proxy configurations can be painful and debugging can feel like walking through a maze but it is doable with a little bit of persistence.
