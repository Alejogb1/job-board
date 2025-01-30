---
title: "How do I retrieve the latest image from an Azure Container Registry's show-tags list?"
date: "2025-01-30"
id: "how-do-i-retrieve-the-latest-image-from"
---
A crucial aspect of managing containerized applications in Azure is consistently deploying the most recent image from an Azure Container Registry (ACR). Manually selecting the latest tag from the `acr show-tags` output becomes cumbersome and error-prone, especially in automated deployment pipelines. The challenge lies in programmatically extracting the tag that corresponds to the most recently updated image. I've faced this repeatedly when configuring CI/CD pipelines that required reliable, automated access to the newest image version without relying on static tag names like "latest."

The fundamental problem is that `acr show-tags` returns a JSON array of tags, which are not inherently ordered by creation or update time. While some tag naming conventions might incorporate timestamps or semantic versioning, these are not universally guaranteed. Therefore, we need to parse the JSON output and utilize the `lastUpdateTime` field, present in the metadata for each tag, to ascertain the most recent entry. This requires extracting, converting, and comparing timestamps.

I’ll illustrate this process using a combination of `az cli` commands, coupled with scripting languages to parse the resulting output. My experience shows that the command-line tools alone are not sufficient; some form of scripting is usually needed.

**Example 1: Using Bash with `jq`**

Bash, coupled with `jq` (a lightweight command-line JSON processor), is very efficient for this task in a Linux environment. I frequently use this combination in my Gitlab CI/CD pipelines.

```bash
#!/bin/bash

REGISTRY_NAME="yourregistryname"
IMAGE_NAME="yourimagename"

#Retrieve the tags and their metadata in JSON format
TAG_DATA=$(az acr repository show-tags --registry $REGISTRY_NAME --image $IMAGE_NAME --output json)

#Extract the tag name with the maximum lastUpdateTime using jq
LATEST_TAG=$(echo "$TAG_DATA" | jq -r 'sort_by(.lastUpdateTime) | last | .name')

# Print the latest tag
echo "Latest tag: $LATEST_TAG"

```

This script works as follows:

1.  `REGISTRY_NAME` and `IMAGE_NAME` variables are defined to make the script more generic.
2.  `az acr repository show-tags ... --output json` retrieves the JSON output for the specified repository image. The `--output json` is critical to ensure the output is parsable by `jq`.
3.  `jq -r 'sort_by(.lastUpdateTime) | last | .name'` is the core command:
    *   `sort_by(.lastUpdateTime)` sorts the JSON array of tags in ascending order of their `lastUpdateTime` field. This uses the ISO 8601 timestamp string for comparison.
    *   `last` selects the last element of the sorted array, which will be the tag with the most recent update.
    *   `.name` extracts the `name` property (the actual tag string) from the selected object.
    *   The `-r` flag ensures `jq` outputs the raw string rather than a quoted string.
4. The result is assigned to `LATEST_TAG` variable, which is printed to the console.

This approach provides a robust solution directly in a shell environment, which is highly useful in server environments where installing specialized tools may be cumbersome.

**Example 2: Python**

Python offers more flexibility and powerful data manipulation options, especially when dealing with complex filtering requirements. I commonly employ Python for more involved pipeline scripts, requiring interaction with other APIs.

```python
import subprocess
import json
from datetime import datetime

def get_latest_acr_tag(registry_name, image_name):
    """Retrieves the most recently updated tag from an Azure Container Registry."""
    try:
        result = subprocess.run(
            [
                "az",
                "acr",
                "repository",
                "show-tags",
                "--registry",
                registry_name,
                "--image",
                image_name,
                "--output",
                "json",
            ],
            capture_output=True,
            text=True,
            check=True
        )
        tag_data = json.loads(result.stdout)
        
        if not tag_data:
             return None
        
        latest_tag = None
        latest_time = datetime.min
        
        for tag in tag_data:
            tag_time = datetime.fromisoformat(tag['lastUpdateTime'].replace("Z", "+00:00"))
            if tag_time > latest_time:
                latest_time = tag_time
                latest_tag = tag['name']
        return latest_tag
    except subprocess.CalledProcessError as e:
            print(f"Error running az cli command: {e}")
            return None
    except json.JSONDecodeError as e:
            print(f"Error decoding JSON from az cli command: {e}")
            return None

if __name__ == "__main__":
    REGISTRY_NAME = "yourregistryname"
    IMAGE_NAME = "yourimagename"
    latest_tag = get_latest_acr_tag(REGISTRY_NAME, IMAGE_NAME)

    if latest_tag:
        print(f"Latest tag: {latest_tag}")
    else:
        print("Could not retrieve the latest tag.")

```

The script breaks down as follows:

1.  The `get_latest_acr_tag` function encapsulates the logic. It uses `subprocess.run` to execute the Azure CLI command and captures its standard output. The `check=True` option ensures the script throws an exception on a non-zero exit code from the subprocess, thus aiding debugging and error detection.
2.  The `json.loads` function parses the JSON output from the command.
3.  It initializes `latest_time` to the minimum datetime value and `latest_tag` to None.
4.  It iterates through each tag in the `tag_data` list. Crucially, `datetime.fromisoformat` parses the `lastUpdateTime`, which is an ISO 8601 formatted string (adding "+00:00" to the timestamp for timezone handling, as fromisoformat requires timezone information if the input includes timezone information). The comparison is made, updating `latest_time` and `latest_tag` when a more recent entry is found.
5. The main part of the script calls the `get_latest_acr_tag` and prints the result, or informs the user if something fails.

Python's datetime library ensures that the comparison of the update timestamps is accurate. The error handling also makes the code more robust than the shell script.

**Example 3: PowerShell**

For environments that favor Windows or PowerShell scripts, this approach provides a suitable solution. I sometimes find this useful when managing Azure resources directly from Windows development machines.

```powershell
$RegistryName = "yourregistryname"
$ImageName = "yourimagename"

# Get the JSON output of the acr show-tags command.
$tagData = az acr repository show-tags --registry $RegistryName --image $ImageName --output json | ConvertFrom-Json

# Sort the tags by lastUpdateTime and pick the last one
if ($tagData) {
    $latestTag = $tagData | Sort-Object {$_.lastUpdateTime} | Select-Object -Last 1
    if ($latestTag) {
      Write-Host "Latest tag: $($latestTag.name)"
    } else {
        Write-Host "No tags found."
    }

} else {
  Write-Host "Error: Could not retrieve the tag data."
}
```

This script executes as follows:

1.  The `$RegistryName` and `$ImageName` variables are defined.
2.  The command `az acr repository show-tags ... --output json` is executed to retrieve the JSON response. The output is piped into the `ConvertFrom-Json` cmdlet which converts the JSON text to PowerShell objects.
3.  The `$tagData` variable now contains an array of tag objects that can be easily manipulated.
4.  The tags are sorted using `Sort-Object {$_.lastUpdateTime}`. The `$_` refers to the current object within the pipeline.
5.  `Select-Object -Last 1` picks out the last element from the sorted collection, which corresponds to the latest tag.
6.  The script checks if the resulting $latestTag is valid, and outputs the correct message.

PowerShell has built-in cmdlets for processing JSON and objects, making it a very concise and convenient solution within that environment.

**Resource Recommendations:**

For further knowledge and more nuanced approaches, I strongly advise reviewing the official Azure documentation for the `az acr` CLI commands. Understanding how to use the `--query` parameter in `az cli` can be powerful for other filtering and retrieval operations not covered by the `sort` option. Experimenting with other JSON processing tools like `yq` can be beneficial if you desire tools with different parsing characteristics. General scripting best practices for each language (Bash, Python, PowerShell) are also important to consider for maintainability and robustness of your solutions. Finally, consulting resources related to date-time manipulation in the chosen language will improve the accuracy and effectiveness of your code.
