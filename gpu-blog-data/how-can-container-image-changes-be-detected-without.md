---
title: "How can container image changes be detected without pulling, across all registries?"
date: "2025-01-30"
id: "how-can-container-image-changes-be-detected-without"
---
Container image changes, particularly when considering multiple registries, present a significant challenge for efficient CI/CD pipelines. Directly pulling an image simply to check for alterations consumes valuable network resources and time. The solution hinges on leveraging image digests, specifically their immutability, to detect changes without needing to download the entire image content. This technique, which I have implemented in previous infrastructure projects, enables significantly faster detection of alterations.

The core principle relies on understanding that a container image, identified by a name and a tag (e.g., `my-image:latest`), can also be uniquely identified by its digest. The digest, generated through a cryptographic hash of the image's layers and manifest, is immutable. Any change to the image’s contents, regardless of how minor, will produce a different digest. Thus, by comparing the current digest against a stored one, we can effectively determine if an image has been modified. The challenge, however, is efficiently retrieving these digests across diverse registries without pulling the full image.

The process primarily involves using registry APIs, which are specific to the container registry being used (Docker Hub, Quay, Google Container Registry, Amazon ECR, etc.). Each registry provides an API endpoint for retrieving the manifest, from which the digest can be extracted. While the API calls differ slightly between registries, the underlying concept and the necessary HTTP requests remain consistent: we perform a HEAD or a GET request to retrieve the manifest. We then parse the JSON manifest to locate the digest, which is usually part of the `config` or `manifest` section in the response.

Retrieving the manifest often necessitates authentication; therefore, correctly handling authentication headers is crucial when interacting with the registry APIs. The authentication mechanism is, again, specific to the registry. For example, Docker Hub relies on token-based authentication obtained through the `docker login` command, while Google Container Registry utilizes service account credentials. Proper configuration of access credentials for the script is, therefore, paramount.

I've found in my work that using a small command-line utility wrapping these API calls often proves the most practical. This type of utility allows for automation and easy integration into various workflows. Further, implementing retry mechanisms and robust error handling is critical, as temporary network issues can cause instability when interacting directly with these APIs. I often include exponential backoff mechanisms to mitigate transient errors.

Let’s illustrate this with a few practical code examples. These assume a basic understanding of HTTP requests and JSON parsing. The following examples are written in Python, given its broad use in DevOps, and emphasize the core concepts while omitting excessive error handling for conciseness.

**Example 1: Detecting changes on Docker Hub**

This example demonstrates retrieving a digest from Docker Hub for a public image.

```python
import requests
import json

def get_dockerhub_digest(image_name, tag):
    url = f"https://registry.hub.docker.com/v2/{image_name}/manifests/{tag}"
    headers = {
        "Accept": "application/vnd.docker.distribution.manifest.v2+json"
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
    manifest = response.json()
    return manifest["config"]["digest"]

if __name__ == "__main__":
    image = "library/nginx"
    tag = "latest"
    digest = get_dockerhub_digest(image, tag)
    print(f"The digest for {image}:{tag} is: {digest}")
```

This snippet constructs the specific URL required by the Docker Hub API. It then sets the correct `Accept` header to request the v2 manifest format. After a successful GET request, it extracts the digest from the `config` section of the parsed JSON response and prints it to the console.

**Example 2: Detecting changes on Google Container Registry (GCR)**

This example shows how to interact with the GCR API using a service account. It uses Google's client library for authentication. Note this requires Google application default credentials to be set up; the specifics are beyond this scope.

```python
import google.auth.transport.requests
import google.auth
import requests
import json

def get_gcr_digest(image_name, tag, project_id):
    credentials, project = google.auth.default()
    auth_req = google.auth.transport.requests.Request()
    credentials.refresh(auth_req)
    
    url = f"https://{project_id}-docker.pkg.dev/v2/{image_name}/manifests/{tag}"
    headers = {
        "Accept": "application/vnd.docker.distribution.manifest.v2+json",
        "Authorization": f"Bearer {credentials.token}"
    }

    response = requests.get(url, headers=headers)
    response.raise_for_status()
    manifest = response.json()
    return manifest["config"]["digest"]

if __name__ == "__main__":
    image = "my-gcr-image"
    tag = "latest"
    project = "my-project-id"
    digest = get_gcr_digest(image, tag, project)
    print(f"The digest for {image}:{tag} on GCR is: {digest}")
```

This example builds upon the previous one by incorporating authentication specific to GCR. It retrieves credentials using Google's client library, then sets the Authorization header using the fetched OAuth token. The core logic for retrieving and parsing the manifest remains consistent. This example assumes the user has the `google-auth` and `requests` libraries installed.

**Example 3: Generic function adapting to different registries**

This example creates a more generic function that can adapt to Docker Hub and GCR, given proper setup. It uses a simplistic string-based approach for demonstration. More robust solutions would likely use dedicated configuration management techniques.

```python
import requests
import json
import google.auth.transport.requests
import google.auth

def get_digest(image_name, tag, registry_type=None, project_id=None):
    
    if registry_type == "gcr":
        credentials, project = google.auth.default()
        auth_req = google.auth.transport.requests.Request()
        credentials.refresh(auth_req)
        url = f"https://{project_id}-docker.pkg.dev/v2/{image_name}/manifests/{tag}"
        headers = {
            "Accept": "application/vnd.docker.distribution.manifest.v2+json",
            "Authorization": f"Bearer {credentials.token}"
        }
    elif registry_type == "dockerhub":
        url = f"https://registry.hub.docker.com/v2/{image_name}/manifests/{tag}"
        headers = {
            "Accept": "application/vnd.docker.distribution.manifest.v2+json"
        }
    else:
        raise ValueError(f"Unsupported registry type: {registry_type}")

    response = requests.get(url, headers=headers)
    response.raise_for_status()
    manifest = response.json()
    return manifest["config"]["digest"]


if __name__ == "__main__":
   
    image_dockerhub = "library/ubuntu"
    tag_dockerhub = "latest"
    digest_dockerhub = get_digest(image_dockerhub, tag_dockerhub, registry_type="dockerhub")
    print(f"Docker Hub digest: {digest_dockerhub}")
    
    image_gcr = "my-gcr-image"
    tag_gcr = "latest"
    project = "my-project-id"
    digest_gcr = get_digest(image_gcr, tag_gcr, registry_type="gcr", project_id=project)
    print(f"GCR digest: {digest_gcr}")
```

This expanded code creates a single function capable of handling both Docker Hub and GCR. It uses a simple `if/elif` structure, which can be further expanded to accommodate other registries. This function also reinforces the core pattern: building the URL, adding the appropriate headers (including authorization), fetching the manifest, and extracting the digest.

In my experience, the key to a successful implementation is not only the retrieval of digests, but also their effective storage and management. I often utilize a key-value store (such as Redis or etcd) to persist digest information, which allows for fast comparison during pipeline runs. This reduces the need to make frequent API requests to registries, further improving efficiency.

For those wishing to deepen their understanding of container registries and their APIs, I suggest referring to the official API documentation of Docker Hub, Google Container Registry, Amazon ECR and others. Exploring the underlying specifications for container image formats, like the OCI Image Specification, can also provide a deeper context. In addition, resources detailing OAuth 2.0 will provide further insight into authentication flows, necessary for interacting with secure registries. I have found that combining practical experimentation, similar to these examples, with theoretical background from the official documentation leads to the deepest understanding of the topic. This methodology, which I have employed across many projects, proves most efficient for tackling complex challenges like this.
