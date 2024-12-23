---
title: "How do I list image layer digests in a Docker registry?"
date: "2024-12-23"
id: "how-do-i-list-image-layer-digests-in-a-docker-registry"
---

,  I recall back in the early days of our cloud infrastructure, we had a particularly knotty issue with rogue container images bloating our storage. It became critically important to understand exactly what was inside these images, and specifically, how to inspect their individual layers. Listing image layer digests is fundamental to that process, and it’s something I’ve found myself revisiting more often than I initially expected. The approach isn’t particularly complicated, but the specifics can sometimes trip people up.

The core principle here is that a Docker image isn't a single monolithic file. It's constructed from a sequence of read-only layers, each identified by a unique content addressable hash, the *digest*. These digests aren't just random identifiers; they're cryptographic hashes calculated from the layer's content, ensuring integrity and enabling deduplication across images. This means that if two images share a layer, that layer is stored only once in the registry, saving space.

The registry, thankfully, provides the necessary endpoints to query for these digests. It’s not a matter of directly accessing the image files themselves, but rather using the registry’s api to navigate the image metadata. This interaction typically involves http requests following the docker registry http api v2. I’ll walk you through it.

We typically start by identifying an image by its name and tag in the registry. Then, we need to make a call to fetch the manifest, which contains the configuration data of the image, including the layer digests. The manifest itself is a json document that includes various metadata about the image and references to the image layers.

Here’s a simplified explanation, followed by three code examples demonstrating how we can accomplish this, using bash, python, and go:

**Conceptual Step-by-step Process**

1.  **Authenticate (if necessary):** Some registries require authentication. This usually involves obtaining a token from an authentication endpoint. For private registries, this token typically needs to be included in subsequent requests. Public registries often don’t require this.
2.  **Fetch the Manifest:** This is achieved by making a `get` request to the registry, specifically to an endpoint like `v2/<image_name>/manifests/<image_tag>`. The response includes the manifest.
3.  **Parse the Manifest:** The manifest is returned as a json document. We need to parse it and look for the layer information, typically located under a key such as `layers` or `fsLayers`, depending on the image format.
4.  **Extract Digests:** The values under these keys are the cryptographic hashes of the image layer. We then extract and can report these as needed.

Now, let’s get into the code examples. I will use `docker.io/library/alpine:latest` as the target image because it is a publicly accessible image in the docker hub. However, the principle applies to any registry following the same API conventions.

**Example 1: Using Bash and `curl` and `jq`**

This approach is helpful for quick, interactive investigations using standard command-line tools. I often reach for this when initially exploring a registry.

```bash
#!/bin/bash

image="library/alpine"
tag="latest"
registry="https://registry-1.docker.io"

token=$(curl -s "https://auth.docker.io/token?service=registry.docker.io&scope=repository:$image:pull" | jq -r '.token')

manifest=$(curl -s -H "Authorization: Bearer $token" \
  -H "Accept: application/vnd.docker.distribution.manifest.v2+json" \
  "$registry/v2/$image/manifests/$tag")

echo "Layer Digests for $image:$tag:"

jq -r '.layers[].digest' <<< "$manifest"
```

This script fetches an auth token, then uses it to pull down the manifest via the appropriate registry endpoint, and then uses `jq` to extract the digests. The output will be each digest listed on its own line.

**Example 2: Using Python**

Python provides a nice way to make programmatic requests and handle the json output. I usually prefer this for more complex tasks or when building automation.

```python
import requests
import json

image = "library/alpine"
tag = "latest"
registry = "https://registry-1.docker.io"

auth_url = "https://auth.docker.io/token"
auth_params = {"service": "registry.docker.io", "scope": f"repository:{image}:pull"}
auth_response = requests.get(auth_url, params=auth_params)
auth_response.raise_for_status()
token = auth_response.json()["token"]


manifest_url = f"{registry}/v2/{image}/manifests/{tag}"
headers = {
    "Authorization": f"Bearer {token}",
    "Accept": "application/vnd.docker.distribution.manifest.v2+json"
}
manifest_response = requests.get(manifest_url, headers=headers)
manifest_response.raise_for_status()
manifest_data = manifest_response.json()

print(f"Layer Digests for {image}:{tag}:")
for layer in manifest_data.get("layers", []):
    print(layer["digest"])
```

This python script makes use of the `requests` library to retrieve the token and then the manifest and extracts the digests from the json response. The output will also be each digest on its own line.

**Example 3: Using Go**

Go is excellent for systems programming and can handle network requests efficiently. When I need the performance of a compiled language, I typically use Go.

```go
package main

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
)

type authResponse struct {
	Token string `json:"token"`
}

type manifestResponse struct {
	Layers []struct {
		Digest string `json:"digest"`
	} `json:"layers"`
}


func main() {
	image := "library/alpine"
	tag := "latest"
	registry := "https://registry-1.docker.io"

	authURL := "https://auth.docker.io/token"
	authParams := url.Values{}
	authParams.Add("service", "registry.docker.io")
	authParams.Add("scope", fmt.Sprintf("repository:%s:pull", image))

	authResp, err := http.Get(authURL + "?" + authParams.Encode())
    if err != nil {
        fmt.Println("Error fetching auth token:", err)
		os.Exit(1)
    }
    defer authResp.Body.Close()
	
	if authResp.StatusCode != http.StatusOK {
		fmt.Println("Authentication failed with status code:", authResp.StatusCode)
        os.Exit(1)
	}


    authBody, err := io.ReadAll(authResp.Body)
    if err != nil {
        fmt.Println("Error reading auth response:", err)
		os.Exit(1)

    }

    var authData authResponse
    err = json.Unmarshal(authBody, &authData)
	if err != nil {
		fmt.Println("Error decoding auth response:", err)
        os.Exit(1)

	}


	manifestURL := fmt.Sprintf("%s/v2/%s/manifests/%s", registry, image, tag)

	req, err := http.NewRequest("GET", manifestURL, nil)
    if err != nil {
        fmt.Println("Error creating request:", err)
        os.Exit(1)
    }
    req.Header.Set("Authorization", "Bearer "+authData.Token)
    req.Header.Set("Accept", "application/vnd.docker.distribution.manifest.v2+json")


    manifestResp, err := http.DefaultClient.Do(req)
    if err != nil {
        fmt.Println("Error fetching manifest:", err)
        os.Exit(1)
    }
    defer manifestResp.Body.Close()

	if manifestResp.StatusCode != http.StatusOK {
		fmt.Println("Failed to fetch manifest with status code:", manifestResp.StatusCode)
		os.Exit(1)
	}

	manifestBody, err := io.ReadAll(manifestResp.Body)
    if err != nil {
        fmt.Println("Error reading manifest response:", err)
		os.Exit(1)

    }
    var manifestData manifestResponse
    err = json.Unmarshal(manifestBody, &manifestData)
    if err != nil {
		fmt.Println("Error decoding manifest response:", err)
        os.Exit(1)
	}

	fmt.Printf("Layer Digests for %s:%s:\n", image, tag)
    for _, layer := range manifestData.Layers {
        fmt.Println(layer.Digest)
    }
}
```

This go program also fetches the token, then uses it to fetch the manifest, and it parses the json data structure extracting the digests and printing each to its own line.

**Further Learning**

For a detailed understanding of the Docker registry http api v2, I strongly recommend reviewing the official documentation at [distribution/distribution](https://github.com/distribution/distribution/blob/main/docs/spec/api.md). Specifically, the sections related to manifest retrieval and layer retrieval are key here. Additionally, *“Docker in Action”* by Jeff Nickoloff and *“Programming Kubernetes”* by Michael Hausenblas and Stefan Schimanski can provide useful background information on container and image management. The official docker docs at [https://docs.docker.com/](https://docs.docker.com/) are invaluable to review the docker image and layer structure concepts.

In conclusion, extracting image layer digests is crucial for effective image analysis and storage management. As you can see, the core process remains consistent across different environments. Understanding how to interact with the docker registry via the API is a core competency that any professional working with docker should possess. It's about more than just running containers; it's about understanding the fundamental pieces that make up a container image and this is what allowed us to track down the bloated images back in those early days of the infrastructure.
