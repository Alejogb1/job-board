---
title: "How can I efficiently clean all repositories in Azure Container Registry?"
date: "2024-12-23"
id: "how-can-i-efficiently-clean-all-repositories-in-azure-container-registry"
---

,  It's a problem I’ve certainly faced a few times during my tenure managing containerized deployments. Cleaning up Azure Container Registry (acr) repositories efficiently is not just about deleting images, it’s about managing storage costs, maintaining security, and ensuring a clean build and deployment pipeline. In my early days, I had a few near misses with bloated acrs costing a small fortune in storage, so a robust cleanup strategy became paramount.

The challenge isn’t merely about blindly deleting things. We need to intelligently target outdated, untagged, or unused images while also being cognizant of potentially disrupting active deployments. So, let's break down my approach, which I’ve refined over years through trials, errors, and a whole lot of scripting.

Essentially, the core of efficient acr cleanup revolves around three main aspects: understanding the image lifecycle, leveraging the acr cli tools, and implementing automation. I've learned the hard way that relying on manual deletion through the portal becomes unsustainable very quickly.

**Understanding the Image Lifecycle and Tagging**

Before we get into the code, we need to understand why repositories get cluttered. Ideally, container image tags should be semantic and representative of their purpose and version. I've seen far too many repositories that devolve into a chaotic mix of `latest`, `v1`, `v2`, `build-123`, and so on. This often results in a difficult-to-manage mess. A more effective approach involves a consistent tagging strategy that incorporates both version information and potentially environment or build IDs. For example, something like `myapp:v1.2.3-dev-build456` is significantly clearer than just `myapp:latest`.

Also, images are not static. They are constantly updated, and older versions become redundant. This churn generates a lot of “untagged” manifests, the underlying descriptions of container images. These manifests can take up significant space and should be regularly removed. Therefore, cleaning up acr means both cleaning the tagged images and these untagged manifests. We also want to clean up manifests which are not referenced by any tags.

**Using the Azure CLI (acr cli) for cleanup**

The Azure cli, specifically the `acr` extension, is your best friend here. I've built my scripts around it. The `acr` cli has all the functionality you need to identify, filter, and remove unwanted images. Crucially, it allows us to operate at scale. The graphical portal can be useful for occasional checks, but it just doesn’t scale for routine cleanup activities.

Let's walk through a few practical code snippets that I've frequently used.

**Code Snippet 1: Deleting Untagged Manifests Older Than 7 Days**

This snippet targets manifests that are not referenced by any tags and were pushed more than a week ago. The age check provides a grace period to prevent accidental deletion of actively being developed images.

```bash
#!/bin/bash

REGISTRY_NAME="your_acr_name" # Replace with your ACR name
RESOURCE_GROUP="your_resource_group" # Replace with your resource group name
DAYS_OLD=7

# Get untagged manifests older than the specified days
MANIFESTS=$(az acr repository show-manifests \
        --name $REGISTRY_NAME \
        --resource-group $RESOURCE_GROUP \
        --query "[?tags[0]==null && (datetime() - properties.createdOn).total_days > $DAYS_OLD].digest" \
        -o tsv)

# Loop through and delete the untagged manifests
if [ -n "$MANIFESTS" ]; then
    for MANIFEST in $MANIFESTS; do
      echo "Deleting manifest $MANIFEST"
      az acr repository delete --name $REGISTRY_NAME \
          --image "$MANIFEST" \
          --resource-group $RESOURCE_GROUP -y
    done
else
    echo "No untagged manifests older than $DAYS_OLD days found."
fi
```

Here, we are using `--query` parameter in `az acr repository show-manifests` to filter manifests. We look for manifests where the `tags` array is empty and check if its `createdOn` property is older than 7 days. The `datetime()` function combined with the manifest's creation date and the `total_days` property of the result gives us the age in days, letting us perform the age filtering. The result is then passed on to the delete command, which removes the untagged manifests.

**Code Snippet 2: Deleting all Images in a Repository older than a specified date except tagged with 'latest'**

This one is for deleting images older than a specified date, but it has an exclusion for images tagged 'latest'. This is a common pattern where `latest` is used to signify the current production image and shouldn’t be deleted by the cleanup process.

```bash
#!/bin/bash

REGISTRY_NAME="your_acr_name"  # Replace with your ACR name
RESOURCE_GROUP="your_resource_group"  # Replace with your resource group name
REPOSITORY="your_repo_name" # Replace with your repo name
CUTOFF_DATE="2024-01-01T00:00:00Z" # Replace with your cutoff date, in ISO format

# Get manifests older than the cutoff date that are not tagged latest
MANIFESTS=$(az acr repository show-manifests \
    --name $REGISTRY_NAME \
    --resource-group $RESOURCE_GROUP \
    --repository $REPOSITORY \
    --query "[?!(tags[0]=='latest') && datetime(properties.createdOn) < datetime('$CUTOFF_DATE')].digest" \
    -o tsv)

# Loop through and delete the matching manifests
if [ -n "$MANIFESTS" ]; then
    for MANIFEST in $MANIFESTS; do
        echo "Deleting manifest $MANIFEST from repository $REPOSITORY"
        az acr repository delete --name $REGISTRY_NAME \
            --image "$REPOSITORY@$MANIFEST" \
            --resource-group $RESOURCE_GROUP -y
    done
else
    echo "No manifests older than $CUTOFF_DATE found for repository $REPOSITORY (excluding 'latest' tag)"
fi

```

Here, we use `!tags[0]=='latest'` as a condition to exclude any images with latest tag. Also, we utilize the `datetime()` function again to make our comparison. Notice that in this case we retrieve the repository manifest, and explicitly pass it to delete command in format "repository@manifest", rather than just manifest.

**Code Snippet 3: Deleting all but the last N images in a repository**

This script is useful for maintaining a rolling history, while keeping the size of the repository in check. You could retain, say, the last 10 builds of a given container image.

```bash
#!/bin/bash

REGISTRY_NAME="your_acr_name"  # Replace with your ACR name
RESOURCE_GROUP="your_resource_group"  # Replace with your resource group name
REPOSITORY="your_repo_name" # Replace with your repo name
KEEP_COUNT=5  # The number of last images to keep

# Get all manifests for the repository, sorted by creation date (newest first)
MANIFESTS_JSON=$(az acr repository show-manifests \
    --name $REGISTRY_NAME \
    --resource-group $RESOURCE_GROUP \
    --repository $REPOSITORY \
    --order "time_desc" \
    --output json )


MANIFEST_COUNT=$(echo $MANIFESTS_JSON | jq '. | length')


if [[ "$MANIFEST_COUNT" -gt "$KEEP_COUNT" ]]; then

    DELETE_COUNT=$((MANIFEST_COUNT - KEEP_COUNT))
    MANIFESTS_TO_DELETE=$(echo $MANIFESTS_JSON | jq --arg num $DELETE_COUNT  '.[0:$num] | .[].digest' | tr -d '"')

    if [ -n "$MANIFESTS_TO_DELETE" ]; then
       echo "Found $($DELETE_COUNT) manifests to delete"
       for MANIFEST in $MANIFESTS_TO_DELETE; do
            echo "Deleting manifest $MANIFEST"
            az acr repository delete --name $REGISTRY_NAME \
              --image "$REPOSITORY@$MANIFEST" \
              --resource-group $RESOURCE_GROUP -y
        done
    else
        echo "No manifests to delete in $REPOSITORY"
    fi

else
     echo "No manifests to delete in $REPOSITORY, already less or equal than $KEEP_COUNT"
fi
```

Here, we introduce the `--order "time_desc"` option on `show-manifests` to sort the images by their creation date, with the newest coming first. We also leverage `jq`, which is a command line tool for processing JSON, to perform some manipulation and get the desired manifests to delete. Then we loop through them and delete as before.

**Automation**

These scripts are only as good as their execution cadence. To truly clean an acr efficiently, you’ll want to automate this process. I've found that using a combination of Azure DevOps pipelines and Azure functions works well for this. A schedule trigger can run scripts periodically, and event triggers can be used to react to certain conditions, like every time a new image is pushed to acr.

**Recommendations**

For a deeper dive, I’d recommend these resources:

1.  **"Docker in Action" by Jeff Nickoloff:** While focused on Docker, it provides an essential understanding of image layers and manifests, crucial for managing any container registry.
2.  **"Programming Azure" by Jesse Liberty:** This covers the core concepts of the Azure CLI, including the `acr` extension, which is vital for all of the scripting we've discussed.
3.  **The official Azure Container Registry documentation:** This should be a primary source for specifics on available commands, options, and parameters of the `acr` CLI. Microsoft also has detailed articles on best practices for container image management.

In conclusion, efficient acr cleanup is not a single action but an ongoing process involving clear image lifecycle management, a solid understanding of cli tools, and robust automation. This approach not only reduces storage costs but also enhances the security and reliability of your containerized deployments. It’s a commitment that pays dividends in the long run.
