---
title: "How can I rename tags when copying an Azure container registry?"
date: "2024-12-23"
id: "how-can-i-rename-tags-when-copying-an-azure-container-registry"
---

Alright, let's tackle this. I've seen this scenario play out more times than I care to count, especially during large infrastructure migrations or when teams are adopting more stringent naming conventions. It's not a trivial copy-paste; there’s a bit more nuance when it comes to preserving your sanity while shifting container images between Azure container registries and keeping your tags consistent.

The fundamental issue here is that Azure Container Registry (acr) copy operations, in their most basic form, aren't equipped with a simple ‘rename-on-copy’ flag. This means a straight copy operation faithfully replicates your image along with its existing tags to the destination registry. That’s helpful, but sometimes, and probably in your case, it's simply not the desired behaviour. You might have legacy naming conventions, or perhaps a new organizational structure requires adjustments. So, how do we wrangle it?

The core solution hinges on a combination of strategies. We can’t directly rename during the push (a simple `docker push` won’t magically retag on the destination), so we have to do a little pre and post-processing. I'm going to explain this in three practical approaches I've used in past projects, focusing on different tools and the situations where they’re most appropriate.

**Approach 1: Leveraging `docker pull` and `docker tag`**

This is probably the most straightforward approach for smaller, manual operations or where scripting is acceptable. We effectively pull the image using the original tag, re-tag it with the new nomenclature, and then push it to the destination registry. This method relies heavily on the command-line docker toolchain.

Here's how that generally looks in practice:

```bash
#!/bin/bash

# Variables
source_registry="sourceacr.azurecr.io"
source_image="my-image"
source_tag="v1.0.0-old"
dest_registry="destacr.azurecr.io"
dest_image="my-image"
dest_tag="v1.0.0-new"


# 1. Login to both source and destination registries
az acr login --name $source_registry
az acr login --name $dest_registry

# 2. Pull the image from the source registry
docker pull $source_registry/$source_image:$source_tag

# 3. Retag the image with the new tag
docker tag $source_registry/$source_image:$source_tag $dest_registry/$dest_image:$dest_tag

# 4. Push the retagged image to the destination registry
docker push $dest_registry/$dest_image:$dest_tag


echo "Image retagged and pushed successfully."
```

In this example, we use `docker pull` to grab the image from the source, `docker tag` to create a new tag that uses the destination registry details and desired tag, and finally, `docker push` to deploy that to the target. This is a relatively simple process, and it’s easily scripted for handling multiple images or tags. The drawback, of course, is the need to have Docker installed and be capable of interacting with container registries. Furthermore, the `az acr login` steps are essential; otherwise, your docker commands will fail with authentication issues. This approach is excellent for smaller projects or manual image handling where a fast, direct approach is preferred.

**Approach 2: Using the Azure CLI with `az acr import` and `--tag` Parameter**

For more programmatic approaches, and especially when dealing within the Azure ecosystem, `az acr import` is your friend. It's more efficient than the pull-re-tag-push cycle if you can avoid downloading the image. Essentially, it copies the image metadata directly within Azure's infrastructure. This is my usual preference when working on automated pipelines, as it reduces network transfer bottlenecks.

Here’s a sample command:

```bash
#!/bin/bash

# Variables
source_registry="sourceacr.azurecr.io"
source_image="my-image"
source_tag="v1.0.0-old"
dest_registry="destacr.azurecr.io"
dest_image="my-image"
dest_tag="v1.0.0-new"

# Import the image with a new tag.
az acr import \
    --name $dest_registry \
    --source $source_registry/$source_image:$source_tag \
    --image $dest_image:$dest_tag

echo "Image imported and retagged successfully."
```

Notice the `--source` and `--image` parameters here. The `--source` specifies the source registry, repository, and tag. The `--image` parameter, on the other hand, specifies the destination registry and the *new* tag we want to assign the image. This functionality lets us modify tags during the copy process within Azure’s network and is typically much faster. This approach is brilliant for deployments where performance and automated workflows are key. It sidesteps the need for a local Docker environment, which is a bonus when operating within serverless or cloud-based CI/CD pipelines.

**Approach 3: Utilizing a script with `az acr repository list-tags` & `az acr import` (for more dynamic operations)**

This third method combines aspects of the previous approaches but provides more dynamic capabilities. It allows for pulling tag lists and then iterating, so it's fantastic for situations where you're dealing with many images or where the specific tag transformations are based on patterns or runtime logic. Here’s an example:

```bash
#!/bin/bash

# Variables
source_registry="sourceacr.azurecr.io"
dest_registry="destacr.azurecr.io"
source_image="my-image"


# Get list of tags
tags=$(az acr repository show-tags --name $source_registry --image $source_image --query "[].name" -o tsv)

# Iterate through the tags and perform import
while IFS= read -r tag; do
  # Simple replacement. Adjust as needed.
  new_tag="${tag/old/new}"

  az acr import \
    --name $dest_registry \
    --source $source_registry/$source_image:$tag \
    --image $source_image:$new_tag
  echo "Image $source_image:$tag copied with new tag: $source_image:$new_tag"
done <<< "$tags"

echo "All images retagged and copied."
```

This script does the following: First, it obtains a list of tags for a given repository via `az acr repository show-tags`, then iterates through each tag. Within the loop, it generates a `new_tag` (in this simple example replacing the string `old` with `new`, which you’d adjust per your situation). Finally, the `az acr import` command is used to copy each image with its retagged nomenclature. This technique really shines when you have a complex set of images or tag transformation rules. You can add sophisticated logic in this script, including date-based tags or version number manipulation.

**Key considerations:**

*   **Authentication:** Always ensure your service principal or user has the correct access permissions to both the source and destination container registries.
*   **Testing:** Always test any renaming script or process on a non-production registry first before applying the same process to your live environment.
*   **Tagging schemes:** Consider implementing consistent tagging conventions to simplify future migrations. Avoid overly generic naming; descriptive and semantic naming helps a lot in debugging and operations.
*   **Tooling choices:** The right tool depends heavily on your use case. Command-line operations are good for quick tasks, while Azure CLI commands are essential for scaling and automated workflows.

**Recommended resources:**

*   **"Docker Deep Dive" by Nigel Poulton**: This book will give you a foundational understanding of how docker images are structured and how tagging works behind the scenes.
*   **Azure documentation on `az acr` CLI commands**: Specifically, go through the documentation for `az acr login`, `az acr import`, and `az acr repository`.
*   **"The Azure Cloud Native Handbook"**: Although broader, this provides excellent context for managing containers within the larger azure ecosystem. This covers best practices regarding container registry management and security.

In closing, renaming tags during an acr copy isn’t natively supported via a single command. However, by leveraging tools like docker, the Azure CLI, and a touch of scripting, you can achieve the desired result in a manageable and scalable way. Start simple, test thoroughly, and you'll navigate these challenges with a lot less stress.
