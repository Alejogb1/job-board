---
title: "How do I access image inspection properties in an Azure Container Registry?"
date: "2024-12-23"
id: "how-do-i-access-image-inspection-properties-in-an-azure-container-registry"
---

Alright, let's tackle this. It's a question that, believe it or not, I've spent a fair bit of time on in previous projects. Back when we were shifting our microservices to a more containerized setup, the need to programmatically inspect image properties within our Azure Container Registry (acr) became pretty crucial for things like security scanning and automated compliance checks. So, let me share what I've learned over time.

Accessing image inspection properties in an acr isn't a single, straightforward operation. It involves interacting with the acr apis, primarily through the Azure CLI, but also through other available tools and libraries depending on your needs. At its core, what you're retrieving is a set of metadata associated with each image, things like its creation time, operating system, size, and crucially, any discovered vulnerabilities or security findings (if you've configured vulnerability scanning, that is).

Now, the Azure portal provides a decent visual representation, but it's rarely enough for automation or in-depth analysis. For those scenarios, programmatically accessing these details is paramount. I've typically approached this in one of three primary ways, based on specific requirements:

1. **Using the Azure CLI:** This is often the most accessible entry point, especially if you're already comfortable with command-line interfaces. The `az acr` command group provides several subcommands for image management. To inspect properties, you'd use the `az acr repository show-tags` or `az acr repository show-manifests` command. The `show-tags` subcommand retrieves information about the tags applied to images within a repository, and `show-manifests` fetches details about image manifests, which contain the underlying image layer metadata.

2. **Leveraging the Azure SDKs (specifically Python or .NET):** When integrating acr data access into an application or an automated pipeline, the Azure SDKs offer a much more structured and programmatic approach. You interact directly with the apis through strongly typed objects and methods, making it easier to parse the responses and manage complex workflows.

3. **Directly Interacting with the ACR REST API:** While less common, it's the most granular approach, offering maximum control. This involves sending HTTP requests to the acr's api endpoints. You'd need to handle authentication, request formation, and response parsing yourself, but it's beneficial if you need a very lightweight solution or are working in an environment where the Azure CLI or SDKs aren't feasible.

Let's break down each of these methods with examples.

**Example 1: Using the Azure CLI**

Let's say you want to get a list of image tags and their respective digests for a repository named `my-image` within an acr named `myregistry`. Here's the command you'd use:

```bash
az acr repository show-tags --registry myregistry --repository my-image --output json
```

The `--output json` parameter is particularly useful because it gives you the result in json format, making it easy to pipe into other tools or processes. The output will look something like this (truncated for brevity):

```json
[
  {
    "digest": "sha256:abcdef1234567890...",
    "name": "latest",
    "timestamp": "2024-07-26T10:00:00Z"
  },
  {
     "digest": "sha256:0987654321fedcba...",
     "name": "v1.0",
     "timestamp": "2024-07-25T14:30:00Z"
  }
]
```

You can use `jq` (a lightweight command-line JSON processor) to further filter or manipulate this data. For instance, to extract only the tag names, you could do something like:

```bash
az acr repository show-tags --registry myregistry --repository my-image --output json | jq -r '.[].name'
```

This would output each tag on a separate line:

```
latest
v1.0
```

**Example 2: Using the Azure SDK for Python**

For more complex scenarios, like parsing vulnerability scans, using the python sdk is often a more effective route. You'd first install the needed library:

```bash
pip install azure-identity azure-containerregistry
```

And then, using the following code (remember to set your environment variables appropriately):

```python
from azure.identity import DefaultAzureCredential
from azure.containerregistry import ContainerRegistryClient

acr_name = "myregistry"
repository_name = "my-image"

credential = DefaultAzureCredential()
client = ContainerRegistryClient(f"https://{acr_name}.azurecr.io", credential)

try:
    tags = client.list_tag_properties(repository_name)
    for tag in tags:
        print(f"Tag: {tag.name}")
        manifest = client.get_manifest(repository_name, tag.digest)
        print(f"  Digest: {manifest.digest}")
        if manifest.annotations and "org.cncf.oras.image.scan-status" in manifest.annotations:
            scan_status = manifest.annotations["org.cncf.oras.image.scan-status"]
            print(f"  Scan Status: {scan_status}")

except Exception as e:
   print(f"Error occurred: {e}")
```

This code snippet demonstrates how to list tags, retrieve the associated manifest, and then, crucially, access the image scan status (if it exists). The `org.cncf.oras.image.scan-status` annotation is commonly used to store vulnerability scan results. This approach provides a more programmatic way to process and analyze data than directly using the command line.

**Example 3: Direct REST API Interaction (Illustrative)**

Directly using the acr rest api is more involved, but it's valuable to understand how it all works under the hood. Here’s a conceptual example using `curl` (it requires proper authentication setup, such as using a service principal or managed identity). You would first need to acquire an authentication token (using the `az account get-access-token` command), and then use the token in the header of the request:

```bash
TOKEN=$(az account get-access-token --resource https://management.azure.com | jq -r '.accessToken')
acr_name="myregistry"
repo_name="my-image"
tag_name="latest"

curl -s -H "Authorization: Bearer $TOKEN" "https://$acr_name.azurecr.io/v2/$repo_name/manifests/$tag_name" -H "Accept: application/vnd.docker.distribution.manifest.v2+json"
```

The output is a detailed manifest document in json format, which includes information about layers, configuration details, and, crucially, annotations. Again, you would need to parse this to extract relevant properties.

In summary, accessing image properties in acr offers multiple avenues. The Azure CLI is a good starting point for ad-hoc analysis. The Azure SDKs are suitable for integration into applications and pipelines, and direct rest api access offers the most control at the expense of added complexity. Which approach is optimal really depends on your specific scenario and requirements.

For further exploration into these methods and the underlying technologies, I highly recommend delving into:

*   **Docker Image Specification:** Understand the structure of a docker image manifest, available in the official Docker documentation, which will be critical for parsing results.
*   **Azure Container Registry Documentation:** The official Microsoft documentation is invaluable. Specifically, focus on the rest api references and the Azure CLI documentation for acr.
*   **Azure SDK for Python/Dotnet Documentation:** Depending on your language preference, the SDK's api documentation provides detailed information on each function and object.
*   **The Open Container Initiative (oci) specifications:** For deep dives into the underlying container image specifications and standards, particularly if you find yourself working with low-level tooling.

Hope this helps clarify things. Let me know if you have more questions.
