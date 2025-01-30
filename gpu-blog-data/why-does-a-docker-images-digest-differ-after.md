---
title: "Why does a Docker image's digest differ after pushing to GitHub Container Registry?"
date: "2025-01-30"
id: "why-does-a-docker-images-digest-differ-after"
---
The discrepancy in Docker image digests between a locally built image and its counterpart pushed to GitHub Container Registry stems from the inclusion of a layer containing metadata specific to the registry itself. This metadata is not present in the locally built image and is appended during the push process, leading to a differing digest.  This is a fundamental aspect of the image layering system and the immutability of Docker digests. My experience troubleshooting deployment pipelines for large-scale microservices highlighted this repeatedly.

**1. Explanation:**

A Docker image is composed of layers. Each layer represents a change in the filesystem.  The digest, a cryptographic hash (typically SHA256), is calculated based on the entire content of these layers. When you build an image locally, you create a set of layers reflecting your application's code, dependencies, and operating system components.  This forms a complete image, but it lacks the final layer added by the registry during the push operation.

GitHub Container Registry, like other container registries, adds a final layer during the push process that includes metadata pertinent to its internal management.  This layer contains information crucial for the registry's functionality, such as security attributes, image manifests, and access control lists.  The inclusion of this registry-specific layer modifies the cumulative content of the image. Because the digest is a function of all layer contents, a change in even one layer, as minimal as the addition of registry metadata, necessitates a new digest.

This is not a bug; rather, it's a designed feature reflecting the immutable nature of Docker images.  The digest acts as a unique identifier for a specific image build.  Modifying any layer, regardless of how seemingly insignificant, invalidates the previous digest and necessitates the generation of a new one to ensure the integrity and traceability of the image.

Furthermore, the registry might also perform some actions, like scanning for vulnerabilities, that impact the image content. While those actions might not add visible files to the filesystem, their effect is reflected in the final manifest, which in turn influences the final layer and the resulting digest.

Therefore, comparing the digest generated locally and the one in the registry is not about verifying the contents in a file-by-file manner, but rather about verifying the complete, immutable state of the image as understood by the registry.


**2. Code Examples and Commentary:**

The following examples illustrate the process and the resulting digest difference.  These are simplified for demonstration purposes; real-world scenarios might involve more complex Dockerfiles.

**Example 1: Simple Node.js Application**

```dockerfile
# Dockerfile
FROM node:16

WORKDIR /app

COPY package*.json ./

RUN npm install

COPY . .

CMD ["node", "server.js"]
```

This Dockerfile creates a simple Node.js application.  Building this locally and then pushing it to the GitHub Container Registry will result in different digests.  The local digest reflects the layers created during the local build process. The registry digest incorporates the added registry metadata layer.

**Example 2: Multi-stage Build**

```dockerfile
# Dockerfile
FROM node:16 AS builder

WORKDIR /app

COPY package*.json ./

RUN npm install

COPY . .

RUN npm run build

FROM node:16-alpine

WORKDIR /app

COPY --from=builder /app/dist ./dist

CMD ["node", "dist/index.js"]
```

This example uses a multi-stage build, a common practice for optimizing image sizes.  The principle remains the same: the local digest before the push differs from the registry digest after the push because of the registry's metadata layer.  Even with optimized layers, the registry still adds its own.

**Example 3:  Illustrating Manifest Changes (Conceptual)**

This example can't be fully represented in code as it directly involves the internal registry operations. However, the concept is vital.

Imagine a scenario where a vulnerability scan is performed by the registry after pushing the image.  The scan reveals vulnerabilities that are then recorded in the image manifest.  This change in the manifest, while not visibly altering files within the image layers, will still alter the digest because the digest encompasses the complete manifest. This is also crucial as security scans are an important part of the image workflow.

The key takeaway is the registry digest reflects the fully registered and potentially scanned image state, which adds information beyond the locally built image content.


**3. Resource Recommendations:**

1.  The official Docker documentation.  This provides comprehensive details on image building, layering, and digests.  It also covers registry interactions and best practices.

2.  The GitHub Container Registry documentation. Specific details on how GitHub handles image pushing, metadata, and vulnerability scanning can be found here.  This is crucial for understanding the context-specific metadata that GitHub may append to images.

3.  A good book on containerization and Docker.  These often dive deep into the underlying mechanisms of image construction, managing layers, and interacting with registries. They usually also cover security best practices related to image building and deployment.



In summary, the difference in digests is not an error but a consequence of the registry's inherent functioning, ensuring integrity and traceability of images throughout their lifecycle. The registry adds its own layer, changing the complete image makeup and thus necessitating a new digest.  Understanding this is critical for managing deployments and troubleshooting in a containerized environment.
