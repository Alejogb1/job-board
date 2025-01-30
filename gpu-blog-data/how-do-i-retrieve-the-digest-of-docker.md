---
title: "How do I retrieve the digest of Docker image layers?"
date: "2025-01-30"
id: "how-do-i-retrieve-the-digest-of-docker"
---
Docker image layers, fundamentally, are stored as content-addressable blobs, each identified by a unique SHA256 hash – its digest. Understanding how to access and utilize these digests is crucial for image management, particularly for advanced workflows such as vulnerability scanning, caching strategies, and reproducible builds. Over the course of several years developing containerized deployment pipelines, I've often needed to programmatically inspect these layers, and the process isn't always as straightforward as simply listing the image with `docker images`.

The digest of a layer is not directly exposed when using the standard `docker images` command. This command primarily displays a summary of locally stored images, referencing them by tags and a shorter image ID derived from the manifest's configuration hash. To access layer digests, one needs to interact with the Docker API, specifically by retrieving the image manifest, a JSON document containing vital information about the image, including the digests of its constituent layers. This requires understanding the structure of a Docker image manifest and how to query the Docker daemon for it.

The most direct way to retrieve layer digests is through the `docker manifest inspect` command, but this tool primarily works at the image *manifest* level, and isn't as directly informative about *layer* digests. Moreover, you don’t get layer size information directly from it, which is often necessary for optimization purposes. The manifest itself provides a list of descriptors, each referencing a specific layer blob, containing the required digest (the hash) and size. To retrieve this, we need to make calls to the Docker Daemon's API, typically through a scripting language or programming interface. Let's consider a method using the Docker CLI, coupled with `jq` (a command-line JSON processor), which provides a flexible approach for parsing API responses.

Here is a common way I achieve this using shell scripting and jq:

```bash
#!/bin/bash

# Set the image name
IMAGE_NAME="your-image:tag"

# Get the image ID
IMAGE_ID=$(docker images -q $IMAGE_NAME)

if [ -z "$IMAGE_ID" ]; then
  echo "Error: Image '$IMAGE_NAME' not found locally."
  exit 1
fi


# Get the manifest using docker inspect and extract layer digests and sizes
docker inspect "$IMAGE_ID" | jq -r '.[0].RootFS.Layers[] as $layer | {digest: $layer, size: ( . |  sub("^sha256:","") | @base64d | fromjson | .size )}'

```

**Commentary:** This script first obtains the ID of the requested image. The image ID can also be a SHA256 digest of the image manifest for non-tagged images.  It then uses `docker inspect`, piping the resulting JSON output to `jq`. `jq` filters the output to extract elements under `.[0].RootFS.Layers[]`, and for each element, it extracts both the layer's digest, and by performing the operation on each layer `sub("^sha256:","") | @base64d | fromjson | .size` we get the layer size from manifest. It outputs each layer's digest and size as JSON objects. This example provides a good starting point for automating layer analysis.

Moving beyond shell scripting, it's equally feasible to achieve this programmatically, such as within a Python environment. Python's docker SDK provides a more controlled and structured way to interact with the Docker API. The following code snippet demonstrates fetching layer digests and sizes via Python:

```python
import docker
import json
import base64


def get_image_layer_digests(image_name):
    client = docker.from_env()
    try:
        image = client.images.get(image_name)
        manifest_data = client.api.inspect_image(image.id)
    except docker.errors.ImageNotFound:
        print(f"Error: Image '{image_name}' not found.")
        return None
    
    layer_info = []
    
    for layer_digest_with_prefix in manifest_data['RootFS']['Layers']:
        layer_digest = layer_digest_with_prefix.replace("sha256:","")
        decoded_data = base64.b64decode(layer_digest).decode()
        layer_data = json.loads(decoded_data)
        layer_size = layer_data['size']
        
        layer_info.append({'digest': layer_digest, 'size': layer_size})

    return layer_info
    
if __name__ == '__main__':
    image_name = "your-image:tag"
    layer_digests = get_image_layer_digests(image_name)
    if layer_digests:
        for layer in layer_digests:
           print(f"Digest: {layer['digest']}, Size: {layer['size']}")


```

**Commentary:** This Python script utilizes the Docker SDK to retrieve the image's manifest data. The `client.images.get()` method retrieves an image object from a tag. The core of the function resides in extracting the layer information from the ‘RootFS’ section within manifest_data and iterates through each layer by accessing the layers attribute. It decodes it from base64 and then uses a JSON parser to obtain the layer's size. Finally, it appends each layer digest and its size to a list and prints the layer information.  This more structured approach is better suited for integration into larger application logic or pipelines. It handles potential `ImageNotFound` errors and gives a cleaner output than the shell based approach.

Lastly, while less common, accessing layer digests directly from a container’s filesystem during runtime can also provide insights, particularly when debugging or performing analysis within a container. While not a standard practice, this method demonstrates a deep understanding of Docker's internal file structures. The container's writable layer exists on top of the read-only image layers, accessible within the container at the mount path. This can be programmatically inspected, however, this method is not appropriate for most production environments, is resource intensive, and potentially introduces unwanted dependencies on the operating environment. This example is for demonstration purposes only.

```python
import docker
import os
import json
import subprocess
import base64

def get_container_layer_info(container_name):
    client = docker.from_env()
    try:
        container = client.containers.get(container_name)
        inspect_data = container.attrs
    except docker.errors.NotFound:
        print(f"Error: Container '{container_name}' not found.")
        return None
    
    layer_ids = inspect_data['GraphDriver']['Layers']

    layer_info = []
    for layer_id in layer_ids:
        # Remove the "sha256:" prefix
        layer_id = layer_id.replace("sha256:","")
        decoded_data = base64.b64decode(layer_id).decode()
        layer_data = json.loads(decoded_data)
        layer_size = layer_data['size']

        layer_info.append({'digest': layer_id, 'size': layer_size})

    return layer_info

if __name__ == '__main__':
    container_name = "your-container-name"
    layer_digests = get_container_layer_info(container_name)
    if layer_digests:
        for layer in layer_digests:
            print(f"Digest: {layer['digest']}, Size: {layer['size']}")

```

**Commentary:** This script takes a container name as input, using the Docker Python SDK to retrieve container information.  This script retrieves the layer IDs from the GraphDriver information on the container, decoding them from base64 in order to obtain the layer sizes. For clarity, the layer ids output by docker inspect are the layer’s digest with an additional prefix, similar to the previous examples. As with the previous script this is then cleaned to only contain the digest, and the layer's size is output.  Again, note that directly accessing layer data from inside the container filesystem should generally be avoided except for very specific debug or analysis reasons.

For further exploration, it is highly recommend looking into the official Docker documentation, which provides detailed explanations of the image manifest format. Additionally, resources focusing on container security best practices, particularly those that delve into image layer analysis, can enhance your understanding and improve your ability to leverage layer digests in various applications. Explore documentation on the Docker Engine API which outlines endpoints and structure of the underlying interface for manipulating containers and images. Furthermore, studying the `jq` manual will refine your ability to quickly extract and process JSON output from command line tools.  These resources, though not providing direct code examples, offer the knowledge needed to approach similar challenges confidently.  Through my experience, leveraging the core principles outlined has proven vital in building robust and efficient containerized applications.
