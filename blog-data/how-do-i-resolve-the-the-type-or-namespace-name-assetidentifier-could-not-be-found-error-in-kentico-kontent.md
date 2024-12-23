---
title: "How do I resolve the 'The type or namespace name 'AssetIdentifier' could not be found' error in Kentico Kontent?"
date: "2024-12-23"
id: "how-do-i-resolve-the-the-type-or-namespace-name-assetidentifier-could-not-be-found-error-in-kentico-kontent"
---

 Encountering "the type or namespace name 'AssetIdentifier' could not be found" within a Kentico Kontent project, particularly after seemingly straightforward setup, is something I’ve definitely navigated more than a few times. It's often less about a core Kentico issue and more about how our development environment interacts with the SDK and our code referencing it. It signals that your project isn’t able to resolve the `AssetIdentifier` type, usually because the necessary Kentico Kontent delivery SDK assembly isn’t properly referenced or the version is incompatible. I've found that the root cause almost always boils down to these scenarios, each requiring a slightly different fix.

In my early days with Kontent, I recall spending a solid afternoon troubleshooting this on a client project. It was frustrating, seemingly out of nowhere, but tracing back through the build logs and meticulously checking the nuget package references, I was able to identify the missing link. So let's break down how to get this resolved with some concrete examples.

First, the most common culprit, especially after a project setup or update, is an incorrect or missing NuGet package reference. The `AssetIdentifier` type, and indeed all the delivery client types, reside within the official Kontent.Delivery package. You need to explicitly include this as a dependency in your project. If you're not sure, inspect your project file (usually a `.csproj` for .NET projects). Open this in your text editor, and look for something like this within the `<ItemGroup>` section:

```xml
<PackageReference Include="Kentico.Kontent.Delivery" Version="[YOUR_VERSION]" />
```

The `[YOUR_VERSION]` should be a legitimate version number of the delivery SDK. If this is missing completely or has the wrong version (for instance, a mismatch with other libraries or dependencies), this will lead to this error. If it’s absent, you'll need to add it using the nuget package manager. I frequently use the command-line interface (CLI) for this:

```bash
dotnet add package Kentico.Kontent.Delivery
```
or, if you need a specific version:

```bash
dotnet add package Kentico.Kontent.Delivery -v <version_number>
```

Make sure you use the correct version number here. Checking the Kentico official documentation (I recommend the "Kontent.ai Delivery SDK .NET" documentation) always gives the most up-to-date recommendations for version compatibility between the SDK and other tools, especially if you're running an older version of .NET.

Second, let's consider a slightly less obvious but equally common issue: namespace collisions or import issues. Even when you have the correct NuGet package, your code needs to explicitly import the relevant namespace. Specifically, the `AssetIdentifier` resides within the `Kentico.Kontent.Delivery` namespace. You need to have a `using` statement (or its equivalent in other languages) to tell your code how to find this type. Here’s an illustrative example in C#:

```csharp
using System;
using System.Threading.Tasks;
using Kentico.Kontent.Delivery; // Crucial import!

public class ContentFetcher
{
    private readonly IDeliveryClient _deliveryClient;

    public ContentFetcher(IDeliveryClient deliveryClient)
    {
        _deliveryClient = deliveryClient;
    }

    public async Task<ContentItem> GetContentItem(string codename)
    {
      //Example of using AssetIdentifier
      var response = await _deliveryClient.GetItemAsync<ContentItem>(codename,
            new List<IQueryParameter>{
                new DepthParameter(1)
            });

      if (response.Item != null)
          return response.Item;

      return null;
    }
    // An example of how to reference AssetIdentifier
    public async Task<Asset> GetAsset(AssetIdentifier assetIdentifier)
    {
       var response = await _deliveryClient.GetAssetAsync(assetIdentifier);
        return response.Asset;
    }

    public async Task Main()
    {
      //Dummy implementation of IAssetIdentifier
      var assetIdentifier = new TestAssetIdentifier("test");
      var asset = await GetAsset(assetIdentifier);
    }
    
    public class TestAssetIdentifier : AssetIdentifier
    {
        public string id {get;set;}
        public TestAssetIdentifier(string id) :base()
        {
           this.id = id;
        }
    }
}
//Example of using Kontent object
public class ContentItem {
    public string title {get;set;}
}
```

Here, the `using Kentico.Kontent.Delivery;` statement at the top is mandatory. Without it, the compiler simply won’t know where to look for the `AssetIdentifier` type, leading to the dreaded error. If you are sure you have this `using` but are still experiencing this issue, ensure you have a clean build (delete the `bin` and `obj` folders). Sometimes older build artifacts or cached references can cause these strange behaviors. In my experience, particularly on complex projects, this is surprisingly helpful.

Finally, while less frequent in standard setups, version mismatches between the NuGet package and other project dependencies can also manifest as this error. For instance, an older version of the delivery package might not support a type present in a newer version of, say, the .net framework. This can be tricky to diagnose since the error message is so localized to that specific type. If the previous steps haven't resolved your error, a good approach is to carefully review all your project's NuGet packages. Ensure they're compatible with each other and with your target runtime environment. A technique that has worked for me in the past has been to add the `Microsoft.Extensions.Logging.Console` logging provider. Then I check the output during application startup for binding redirect errors that point to version mismatches. If you are on .net core 3.1 or later, and you have any transitive dependencies referencing an older version of the same dependency, the console output will typically contain those binding redirect issues which you can resolve using `<AutoGenerateBindingRedirects>true</AutoGenerateBindingRedirects>` in the project file.

Let's put that together with a second example, showcasing the importance of the namespace and SDK usage in a more specific use case, fetching an asset:

```csharp
using System;
using System.Threading.Tasks;
using Kentico.Kontent.Delivery; // Ensure the correct namespace is included

public class AssetFetcher
{
    private readonly IDeliveryClient _deliveryClient;

    public AssetFetcher(IDeliveryClient deliveryClient)
    {
        _deliveryClient = deliveryClient;
    }

    public async Task<Asset> GetAssetByExternalId(string externalId)
    {
        //Example using the assetIdentifier
        var assetIdentifier = new AssetIdentifier(externalId);

       var response = await _deliveryClient.GetAssetAsync(assetIdentifier);

        if (response.Asset != null)
            return response.Asset;

        return null;
    }
    public async static Task Main()
    {
        //Dummy implementation of client
        var deliveryOptions = new DeliveryOptions() {
           ProjectId = "test",
        };
        var deliveryClient = DeliveryClientBuilder.Create(deliveryOptions).Build();
        AssetFetcher fetcher = new AssetFetcher(deliveryClient);
        var asset = await fetcher.GetAssetByExternalId("someExternalId");
        if(asset != null)
        {
            Console.WriteLine(asset.Name);
        }

    }

}
```

This example illustrates that you don't create an instance of `AssetIdentifier` on your own. You'll need to use a constructor of the `AssetIdentifier` to pass the external id of the asset. The key takeaway here is the explicit use of `Kentico.Kontent.Delivery` namespace with the `using` statement. Without it, your code will simply fail to recognize and resolve `AssetIdentifier`.

And finally, lets address how to instantiate the delivery client.

```csharp
using System;
using Kentico.Kontent.Delivery;
using Microsoft.Extensions.DependencyInjection;

public class ClientFactory
{
    public static IDeliveryClient CreateDeliveryClient(string projectId, string previewApiKey = null, string environmentId = null)
    {

      var deliveryOptions = new DeliveryOptions() {
           ProjectId = projectId,
           UsePreviewApi = !string.IsNullOrWhiteSpace(previewApiKey),
           PreviewApiKey = previewApiKey,
           EnvironmentId = environmentId
        };
      return DeliveryClientBuilder.Create(deliveryOptions).Build();
    }
    public static void Main()
    {
        var client = CreateDeliveryClient("your_project_id", "previewKey", "environmentId");
        Console.WriteLine("Client has been created and is ready to go.");
    }

}
```

This simple example focuses on the proper initialization of the Delivery client, which is used by the previous code snippets. The `DeliveryClientBuilder` from the delivery sdk is responsible for creation of the Delivery Client. The constructor also takes into account the environment settings using the `DeliveryOptions` object.

In summary, resolving the "The type or namespace name 'AssetIdentifier' could not be found" error predominantly involves verifying your NuGet package references, ensuring correct namespace imports, checking for version incompatibilities, and ensuring the `DeliveryClient` is initialized correctly. If all these bases are covered and you're still facing this issue, it might be helpful to take a step back, try a clean build, and then re-evaluate each potential cause. This methodical approach has served me well over the years, and it should certainly help you to get back on track. Always consult the official Kentico documentation and the delivery SDK documentation for the most accurate and up-to-date information and guidance. A deep dive into "Dependency Injection in .NET" can also help to understand issues related to the client instantiation.
