---
title: "How can I retrieve creation dates of CRI-O images using a list images call?"
date: "2025-01-30"
id: "how-can-i-retrieve-creation-dates-of-cri-o"
---
The CRI-O container runtime, unlike Docker, does not directly expose image creation dates within the output of its `crio images` command. This absence stems from CRI-O's focus on efficient container execution rather than comprehensive image management metadata. Extracting this information requires querying the underlying container image storage, typically managed by a container image registry and leveraging tools that interact with the registry's API or the local image storage itself. The information is not stored in the image list command's output.

My experience in automating large-scale container deployments highlighted the recurring need for image creation time information. During our development pipeline, accurately tracking when images were built became crucial for debugging deployment issues and understanding the provenance of running container versions. Specifically, when analyzing resource consumption discrepancies across environments, the build timestamp often proved a valuable data point. Given the limitations of `crio images`, achieving this required a shift in approach, focusing on direct interaction with the registry or utilizing tools that examine the image's manifest.

**Understanding the Problem**

The `crio images` command provides a concise summary of locally stored images but lacks the granularity to display detailed metadata like image creation timestamps. The typical output presents columns such as IMAGE ID, IMAGE NAME, and SIZE. This brevity is intentional, aligning with CRI-O's role as a lean container runtime. In contrast, Docker’s `docker images` command often reveals the “CREATED” timestamp directly, an aspect not replicated by CRI-O's command. Therefore, a different methodology is needed to extract this information within a CRI-O environment. The data we seek is indeed available, it simply isn't part of the standard `crio images` output.

To retrieve the desired creation timestamp, we must consider that container images are constructed from layered file systems and a configuration manifest, stored as a JSON document. This manifest typically contains metadata detailing when the image was built. This manifest, however, is not directly accessed via the CRI-O command line. The manifest is stored either in the container registry or, in a limited form, locally on disk. The specific approach to retrieve the creation timestamp depends on where the image originates and whether we have access to the image on the local machine.

**Methodology & Tools**

The strategy I've found most effective consists of accessing the manifest from the source registry. For publicly available images, one can use tools designed to interact with image registries using the registry's API. For images residing in private registries, authentication to that registry is also required. If the image is only available locally, we can examine the manifest through specialized container utilities.

Several tools facilitate the analysis of container images and their manifests. For interaction with registries, I commonly employ `skopeo`, a command-line utility for copying and inspecting container images. `skopeo` can directly pull manifest data from a registry, enabling us to extract the desired creation date. In situations where direct registry access isn't possible, tools like `podman` or `buildah` can be used. These tools often provide interfaces to inspect the layers and configuration of a container image, including the time the manifest was created.

**Code Examples and Explanation**

Here are three examples demonstrating different scenarios to access the creation date metadata.

**Example 1: Using `skopeo` to inspect a public image**

```bash
#!/bin/bash
image_name="quay.io/podman/hello:latest"

manifest_json=$(skopeo inspect docker://${image_name} --format 'json')

created_date=$(echo "$manifest_json" | jq -r '.created')

echo "Image: ${image_name}"
echo "Created Date: ${created_date}"

```

*   **Explanation:** This script utilizes `skopeo` to fetch the manifest of a publicly available image ("quay.io/podman/hello:latest"). The `skopeo inspect` command, with the `docker://` prefix, retrieves the image manifest. The output is formatted as JSON and then parsed via `jq` (a JSON processing tool) to extract the `.created` field, which contains the timestamp. This value is then printed to the console. This example highlights fetching the manifest from a registry.

**Example 2: Using `podman` to inspect a locally available image**

```bash
#!/bin/bash
image_name="localhost/my-local-image:latest"

# Ensure image exists locally
if ! podman image exists $image_name > /dev/null 2>&1; then
    echo "Image not found locally: $image_name"
    exit 1
fi

manifest_json=$(podman inspect $image_name | jq -r '.[0].Config.Config.Created')

echo "Image: ${image_name}"
echo "Created Date: ${manifest_json}"
```

*   **Explanation:** This example assumes a container image is stored locally. The script begins by checking if the image, `localhost/my-local-image:latest`, is available using `podman image exists`. It then leverages `podman inspect` to retrieve a detailed description of the image. `jq` is used again to navigate through the JSON structure to reach the `Config.Config.Created` field, containing the creation time. Notice the structure is different than the `skopeo` JSON. This demonstrates the variation in accessing metadata based on the tool and image source.

**Example 3: Combining `skopeo` with image ID**

```bash
#!/bin/bash
# Assume we've obtained a local image ID from crio images
image_id="sha256:f54a2d1887839a6d987b31574d8920e28803d557d79dd723a4d7389ddf278895"

# We need to find the registry name based on crio's output. This requires additional parsing
image_name=$(crio images | grep $image_id | awk '{print $2}')

# Extract the registry component if it's there
registry_name=$(echo "$image_name" | awk -F/ '{print $1}')

if [ "$registry_name" == "localhost" ] || [[ "$registry_name" != "" && "$registry_name" != "${image_name}" ]] ; then
    skopeo inspect docker://$image_name --format 'json' | jq -r .created
else
   echo "This example only works with images where the registry name is available"
   exit 1;
fi


```

*   **Explanation:** This script begins by simulating a scenario where an image ID is known from the `crio images` output (it is extracted using `grep` and `awk`). The script must then attempt to extract the registry location (if any) from the image name provided by `crio`. It checks if the first component of image name is `localhost` or if there is any registry name at all. If a registry name is identified, we can use `skopeo` to fetch the manifest based on the constructed registry image name. This process illustrates a more involved approach, incorporating `crio images` output and registry lookup. A limitation of this script is the assumption that the image name is retrievable from crio and in a format usable for querying an external registry with `skopeo`.

**Resource Recommendations**

For further exploration and understanding of container image handling, the following resources are useful:

*   **`skopeo` Documentation:** Provides comprehensive details on its capabilities for inspecting and manipulating container images.
*   **`podman` Documentation:** Details the command's utility for container management and image inspection on local machines.
*   **`jq` Documentation:** Necessary for parsing JSON data, especially when working with container manifest files.
*   **OCI Image Specification:** A deep dive into the structure and format of container images.
*   **Container registry documentation:** Each registry typically provides information on its API and how to access image manifests.

In summary, while `crio images` doesn't directly present image creation times, it's possible to retrieve this metadata by accessing the image manifest from the registry using `skopeo`, `podman` or other similar tools. Understanding this process is critical for managing and debugging containerized applications. The methods described here offer a reliable path to accessing that information using readily available command-line utilities.
