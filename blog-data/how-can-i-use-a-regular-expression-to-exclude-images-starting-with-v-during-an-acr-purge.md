---
title: "How can I use a regular expression to exclude images starting with 'v' during an ACR purge?"
date: "2024-12-23"
id: "how-can-i-use-a-regular-expression-to-exclude-images-starting-with-v-during-an-acr-purge"
---

Alright, let's tackle this ACR purge challenge. It's a scenario I've bumped into a few times back when I was managing a sprawling containerized application on Azure. The situation calls for targeted pruning, and regular expressions are indeed the weapon of choice for that precision. The core concept, of course, is to *selectively* exclude images matching a specific pattern during the cleanup. We're not just blindly nuking everything; we're carefully preserving what’s necessary.

The key here, and where a lot of people initially stumble, is that regular expressions in these contexts often interact with tooling-specific syntax and nuances. It’s not just the pure regex pattern itself that matters, but how it’s interpreted within the tool you’re using, be it the `az acr` cli or any other similar interface. In our case, it’s about excluding images. This means we need to craft a regex that accurately *identifies* the unwanted images, and the tooling will then work to exclude them, effectively retaining anything that *doesn't* match that pattern.

Specifically, you want to exclude images that *start* with 'v'. To do that, the regex is going to center on the anchor character `^`, which indicates the beginning of the string, followed by a simple `v`. So our base pattern will be `^v`. Now, the way you use this depends on how you're actually triggering this purge, but the fundamental regex remains the same. Let me break down a few scenarios that you might encounter with different tooling interfaces and provide relevant code snippets.

First, let's look at using the Azure CLI (`az acr`). This is probably the most common way people interact with ACR cleanup, and it allows you to define exclusion patterns using `--filter` argument, and this argument accepts a regular expression.

```bash
az acr repository delete --name myacr --image "^v" --dry-run --yes --force
```

In this snippet, `myacr` is your ACR name and `^v` is our regex pattern. The `--dry-run` option is there for safety. I’ve learned the hard way to use it to make absolutely certain that I’m about to delete what I intended. I recommend keeping the `--dry-run` in, removing it only when you’re completely confident. The `--yes` and `--force` flags bypass confirmation prompts, which are useful in automated scripts but please proceed with caution. This will *exclude* images starting with 'v'. It's the default behavior of `--image` with regular expression when using the `az acr` command. So, we're not *selecting* images that start with v. We’re saying, *exclude* anything starting with v. It can seem counterintuitive at first, and I spent a lot of time checking and double-checking these commands in the past.

Now, let's say you're not dealing directly with the CLI, but a more programmatic interface, possibly through a programming language. Let's assume you are using python and need to use SDK. Here’s how that would look, focusing on the filtering aspect:

```python
from azure.identity import DefaultAzureCredential
from azure.mgmt.containerregistry import ContainerRegistryManagementClient
from azure.mgmt.containerregistry.models import  DeleteTagProperties


credential = DefaultAzureCredential()
subscription_id = "your_subscription_id"
resource_group = "your_resource_group"
acr_name = "myacr"
repository = "myrepo"
tag_to_exclude = "^v.*"


acr_client = ContainerRegistryManagementClient(credential, subscription_id)

purge_params = DeleteTagProperties(
    tag_properties={"name": tag_to_exclude},
)

result = acr_client.repositories.begin_delete_tag(
    resource_group, acr_name, repository, purge_params,
).result()


print(f"Deletion initiated for tags matching '{tag_to_exclude}'.")
```

In this Python example, which leverages the azure SDK, we're using `begin_delete_tag` but the crucial part is how we define the `tag_to_exclude`. Here, I've used `^v.*` to match tags that begin with "v" and can be followed by any other characters.  Again, we’re setting up a filter; the SDK handles the logic for ensuring anything matching this pattern is *excluded* from the deletion. It's less about explicitly saying "delete this if it doesn’t start with 'v'", and more about specifying the images we *don't* want to keep. The tooling, in turn, takes that exclusion pattern into account.

Finally, let’s consider a third scenario, where you’re using a more advanced configuration tool or custom script that might leverage the docker API directly to identify tags, and then perform an ACR purge. The filtering process would be a bit more manual but still rely on the regex. You would first retrieve a list of tags using API. And filter them. The following example assumes that you've a list of tags, say `all_tags` and then filters it. The following example uses a plain python regex without any SDK.

```python
import re

all_tags = ["v1.0", "image1", "v2.0-dev", "image2", "vabc"]
exclude_pattern = "^v"
filtered_tags = [tag for tag in all_tags if not re.match(exclude_pattern, tag)]

print(filtered_tags)
```

In this case we are getting all the tags using API call before and the filter is done using regex match in the python code. Note that here the filter is performed *before* any delete action takes place. In this case `filtered_tags` only contains the images that don't start with `v`. You will then iterate through the `filtered_tags` to perform the delete action.

Now, a few key things that tend to be overlooked during these operations. First, *always* test your regex carefully. Start with a simple regex, build incrementally, and use the `--dry-run` or a similar test mode where possible. It’s a good practice to use tools like regex101.com to validate your patterns. Second, ACR's tag deletion process can be intricate, especially for multi-architecture images. It's important to understand the relationship between tags and manifests in your registry. Third, when you're doing more complex filtering, you should consult the official documentation. For Azure, look at the Azure documentation specific to the `az acr` command or the Python SDK for the Azure Container Registry. The official documentation typically provides detailed guidance on regex support and any limitations that might apply, along with examples.

For a more in-depth dive into regular expressions, I’d highly recommend "Mastering Regular Expressions" by Jeffrey Friedl. It’s a classic and provides a thorough understanding of the mechanics and subtleties of regexes. You may also find "Regular Expression Pocket Reference" by Tony Stubblebine as a compact and helpful resource for quick reference. For the Azure-specific aspects, the official Microsoft Azure documentation is your best bet – I always start there. The key is to practice, test, and consult reliable sources, and slowly, it becomes second nature.
