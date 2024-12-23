---
title: "How can I pull an image from Azure Container Registry using the .NET SDK?"
date: "2024-12-23"
id: "how-can-i-pull-an-image-from-azure-container-registry-using-the-net-sdk"
---

Alright, let's dive into pulling images from Azure Container Registry (acr) using the .net sdk. I've spent a fair bit of time working with cloud infrastructure, particularly around containerization, so this is something I've wrestled with firsthand quite a few times. Specifically, I recall a project involving microservices deployed on kubernetes where we needed to automate image retrieval as part of our ci/cd pipeline; that experience certainly shaped my understanding of this process. It’s not always straightforward, but with the right approach, it’s entirely manageable.

The core concept hinges on utilizing the `azure.identity` and `azure.containerregistry` nuget packages. We'll leverage managed identities to ensure secure authentication without hardcoding credentials. This approach is highly recommended as best practice, especially for production environments.

First off, let's break down the overall process. Authenticating with acr is the initial hurdle; you'll need to establish a secure, authenticated session. Then, you'll use the `containerregistryclient` to specify your acr endpoint and the target image to pull. The .net sdk simplifies much of the low-level http interactions, allowing us to focus on the core logic.

Here’s how I would approach this, generally:

1.  **Authentication:** Set up a managed identity on the resource (e.g., virtual machine, app service, azure function) doing the pulling or use a service principal. This is handled by the `azure.identity` package, abstracting away the intricacies of token management.
2.  **Client Initialization:** Using the authenticated identity, instantiate a `containerregistryclient` pointing to your specific acr endpoint.
3.  **Image Manifest Retrieval:** Retrieve the image manifest. This is metadata describing the image.
4.  **Image Layers Retrieval:** Pull the individual image layers referenced in the manifest. This typically involves a series of `getblob` calls.
5.  **Reconstructing the Image:** In some scenarios, you may need to store these layers locally for local use (e.g., with docker cli). However, if you’re pulling to deploy to kubernetes, your deployment tool handles all that so this is often not needed in that context.

Now, let’s look at concrete examples using code snippets. These snippets will assume you have the necessary nuget packages installed (`azure.identity`, `azure.containerregistry`).

**Example 1: Basic Image Manifest Retrieval**

This example focuses on pulling the image manifest, which is often the first step when programmatically interacting with a container image. This shows how to get the manifest for a specific image and tag.

```csharp
using Azure.Identity;
using Azure.Containers.ContainerRegistry;
using System;
using System.Threading.Tasks;

public class AcrManifestExample
{
    public static async Task Main(string[] args)
    {
        string acrName = "your-acr-name"; //replace with your acr name
        string imageName = "your-image-name"; // replace with your image name
        string imageTag = "latest"; // replace with the image tag

        string acrEndpoint = $"https://{acrName}.azurecr.io";

        var credential = new DefaultAzureCredential();
        var client = new ContainerRegistryClient(new Uri(acrEndpoint), credential);

        try
        {
            var manifest = await client.GetManifestAsync(imageName, imageTag);
            Console.WriteLine($"Successfully retrieved manifest for {imageName}:{imageTag}. Digest: {manifest.Value.Digest}");

            //you can further inspect manifest content here such as the layers
            foreach (var layer in manifest.Value.Layers)
            {
                 Console.WriteLine($"Layer Digest: {layer.Digest}, Size: {layer.Size}");
            }


        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error retrieving manifest: {ex.Message}");
        }
    }
}
```

In this snippet, we are first setting up the `acrEndpoint`, image name, and image tag. We then initialize the `containerregistryclient` using `DefaultAzureCredential`, which attempts to authenticate based on available azure authentication mechanisms – including managed identities. Finally, we call `getmanifestasync` to retrieve the manifest object. The manifest contains the layers and digest used to construct the container image.

**Example 2: Pulling an Image Layer**

Building upon the previous example, this one focuses on pulling a single image layer. Often you will need to pull each layer in sequence to compose the final image.

```csharp
using Azure.Identity;
using Azure.Containers.ContainerRegistry;
using System;
using System.IO;
using System.Threading.Tasks;

public class AcrPullLayerExample
{
    public static async Task Main(string[] args)
    {
        string acrName = "your-acr-name";  //replace with your acr name
        string imageName = "your-image-name"; // replace with your image name
        string imageTag = "latest";  // replace with the image tag

        string acrEndpoint = $"https://{acrName}.azurecr.io";
        var credential = new DefaultAzureCredential();
        var client = new ContainerRegistryClient(new Uri(acrEndpoint), credential);

        try
        {
            var manifest = await client.GetManifestAsync(imageName, imageTag);
            if (manifest.Value.Layers.Count == 0) {
                 Console.WriteLine("No layers found in this image manifest.");
                 return;
            }

            var firstLayer = manifest.Value.Layers[0];
            Console.WriteLine($"Pulling layer with digest: {firstLayer.Digest}");

            var blobResponse = await client.GetBlobAsync(firstLayer.Digest);

            if (blobResponse.Value == null) {
                Console.WriteLine("Could not retrieve image blob stream.");
                return;
            }
            string localFilePath = $"layer_{firstLayer.Digest}.tar.gz"; // or a path to where you want to save the layer

             using (var fileStream = File.Create(localFilePath))
            {
                await blobResponse.Value.Content.CopyToAsync(fileStream);
             }


            Console.WriteLine($"Successfully downloaded layer to {localFilePath}");

        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error pulling layer: {ex.Message}");
        }
    }
}
```

Here, after obtaining the manifest, we extract the digest of the first layer. We use this digest to retrieve the blob (the layer content) via `getblobasync`. Then we save the layer to a local file. Note that you will likely want to loop through all the layers of the manifest and pull them individually.

**Example 3: Handling Anonymous Access**

While managed identities are generally recommended, sometimes your setup may require using anonymous access for public registries. This is not recommended for anything that isn't a publicly hosted image because of security concerns. This example shows how to retrieve a manifest anonymously.

```csharp
using Azure.Containers.ContainerRegistry;
using System;
using System.Threading.Tasks;

public class AnonymousAcrExample
{
    public static async Task Main(string[] args)
    {
        string acrName = "your-public-acr"; // replace with your public acr name
        string imageName = "your-image-name"; // replace with your image name
        string imageTag = "latest"; //replace with the image tag


        string acrEndpoint = $"https://{acrName}.azurecr.io";

        var client = new ContainerRegistryClient(new Uri(acrEndpoint)); //notice we did not provide credentials here

        try
        {
            var manifest = await client.GetManifestAsync(imageName, imageTag);
             Console.WriteLine($"Successfully retrieved manifest anonymously for {imageName}:{imageTag}. Digest: {manifest.Value.Digest}");


        }
        catch (Exception ex)
        {
            Console.WriteLine($"Error retrieving manifest anonymously: {ex.Message}");
        }
    }
}
```

In this example, notice how we initialize the `containerregistryclient` without passing credentials; this falls back to anonymous access. This works fine for publicly accessible images, but not for private registries.

For a deeper understanding of container registry internals, I'd suggest referencing *Docker Deep Dive* by Nigel Poulton. It provides a comprehensive look at the low-level components of a container registry and how image layers are constructed. Also, the official azure documentation on `azure.containers.containerregistry` nuget package is indispensable – it provides detailed api references and usage guidelines. Furthermore, familiarize yourself with the open container initiative (oci) specification for container image formats; this will give a deeper knowledge of the data structure you’re handling. Finally, the official azure documentation on authentication with container registries is a great place to learn best practices for securing your registries.

These examples should give you a solid foundation for pulling images from acr using the .net sdk. Remember to tailor the code to your specific needs and carefully consider security implications, particularly with regard to authentication and credential management.
