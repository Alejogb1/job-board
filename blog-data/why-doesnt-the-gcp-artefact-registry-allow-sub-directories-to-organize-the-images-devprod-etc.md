---
title: "Why doesn't the GCP Artefact registry allow sub-directories to organize the images dev,prod etc?"
date: "2024-12-23"
id: "why-doesnt-the-gcp-artefact-registry-allow-sub-directories-to-organize-the-images-devprod-etc"
---

Alright,  I’ve spent more than a few years wrangling container images across various environments, and the question of subdirectory organization within GCP's Artifact Registry is one that’s definitely surfaced, and for good reason. It’s not uncommon to want a clean demarcation between, say, development and production builds, and intuitively, directories would seem like the logical way to do that. But the design of Artifact Registry, like many container registries, intentionally eschews this hierarchical approach for several very good reasons, revolving around versioning, tagging, and overall management efficiency. Let me walk you through the specifics, drawing from some personal projects where I initially expected, and then had to work *with*, the registry's design.

The core of the issue is that Artifact Registry doesn't treat repository "paths" like a traditional file system. Instead of folders, it utilizes a flat namespace with a structured image naming convention based on the format: `[HOSTNAME]/[PROJECT-ID]/[REPOSITORY-NAME]/[IMAGE-NAME]:[TAG]` or `[HOSTNAME]/[PROJECT-ID]/[REPOSITORY-NAME]/[IMAGE-NAME]@[DIGEST]`. This format, while seemingly less intuitive than a directory-based structure, is fundamental to how containers are versioned and managed in practice. The crucial element here is the `[TAG]` or `@DIGEST`.

The absence of direct support for subdirectories isn't a limitation born of oversight, it's a deliberate design choice stemming from how container registries handle immutability. When you build a container, particularly for a production deployment, you want a guarantee that what was tested is exactly what is running, and the only way to guarantee that is by addressing the image by its immutable digest. Tags, on the other hand, are mutable references, essentially pointers that can be changed to point to a different image digest. If you could have nested directories, you’d introduce a lot of complexity around how tags are interpreted and resolved within that structure, potentially leading to serious discrepancies between what your systems are *meant* to be running versus what they *actually* are. For a team managing many services, this is a potential reliability nightmare.

Instead of relying on directory hierarchies, Artifact Registry encourages you to leverage repositories and image names in a consistent way with a clear tagging and versioning strategy. It’s crucial to think of the `[IMAGE-NAME]` part in the fully qualified name as an *identifier* of your service, not a path to the location in the registry file system. A good practice, for instance, is to include the service or application name here. The actual separation between, say, 'dev' and 'prod' is achieved through judicious use of tags (for mutable references) and digests (for immutable references to specific builds).

Let’s look at a few examples of how this works in practice using Docker CLI commands:

**Example 1: Tagging for Different Environments**

Let's say I have a simple application called `my-app`. Instead of trying to put my development builds into a subdirectory, I use tags to differentiate builds:

```bash
# Assume my-app is built and tagged as 'latest'
docker tag my-app:latest us-central1-docker.pkg.dev/my-project/my-repository/my-app:dev-latest
docker push us-central1-docker.pkg.dev/my-project/my-repository/my-app:dev-latest

# Similarly, after thorough testing, push to production.
docker tag my-app:latest us-central1-docker.pkg.dev/my-project/my-repository/my-app:prod-v1
docker push us-central1-docker.pkg.dev/my-project/my-repository/my-app:prod-v1
```

In this example, I’ve used tags like `dev-latest` and `prod-v1` to clearly demarcate my environments. This allows me to track which images are designated for development versus production, without introducing a directory system that would complicate resolution. I am tagging *the same image* built and only changing the tags in the registry.

**Example 2: Using Digests for Immutability**

Now, to further illustrate immutability and why relying on digests is crucial:

```bash
# After pushing a 'dev' tag to the registry, I look up its digest.
docker manifest inspect us-central1-docker.pkg.dev/my-project/my-repository/my-app:dev-latest  | grep digest

# This returns a digest like 'sha256:abcdef123456...'. Let’s use this to pull the exact image.
docker pull us-central1-docker.pkg.dev/my-project/my-repository/my-app@sha256:abcdef123456...

# If later, the 'dev-latest' tag is moved (changed to a new image digest), using the digest still fetches that specific build.
```

Here, by pulling using the digest, I am ensuring that, even if the `dev-latest` tag is updated, my deployed image remains exactly what I intended. This is critical in a production setup, where changes must be explicit and controlled. The digest gives you an immutable identifier and an audit trail for deployment.

**Example 3: Using Custom Image Naming Conventions**

Another strategy I've seen work well is embedding environment information in the image name itself. For instance:

```bash
docker tag my-app:latest us-central1-docker.pkg.dev/my-project/my-repository/my-app-dev:latest
docker push us-central1-docker.pkg.dev/my-project/my-repository/my-app-dev:latest

docker tag my-app:latest us-central1-docker.pkg.dev/my-project/my-repository/my-app-prod:v1
docker push us-central1-docker.pkg.dev/my-project/my-repository/my-app-prod:v1
```

By appending `-dev` or `-prod` to the image name, this provides another level of segregation and allows for the use of different tags in different "namespaces" if you will. This can be particularly helpful when dealing with pipelines that use dynamic image naming based on the environment they are running in.

For further in-depth study, I'd recommend exploring the following resources. You should be able to access them without too much trouble:

1.  **Docker's Documentation on Image Tagging and Versioning:** The official Docker documentation provides a thorough understanding of how image tags, digests, and manifests work. Specifically, look into their documentation on image specification and how image resolution works in the container runtime.
2.  **"Programming Google Cloud Platform" by Rui Costa and Drew Hodun:** While not solely about Artifact Registry, this book delves into many aspects of GCP, including its container-related offerings. This will give you a better grasp of how Artifact Registry fits into the larger cloud environment and the rationales behind its design. It covers general topics on GCP, with sufficient coverage on related services, providing you with the full picture.
3. **"Kubernetes in Action" by Marko Lukša:** Though not directly about registries, understanding how Kubernetes utilizes image references is crucial. Understanding the way Kubernetes pulls images with specific tags and digests will make the design choices in registries like Artifact Registry more clear. This is a really deep look into the inner workings of Kubernetes.

In conclusion, while it might initially seem counterintuitive, the absence of subdirectories in Artifact Registry isn't a design flaw; it's a deliberate choice that prioritizes versioning, immutability, and efficient management of container images. By understanding and embracing the flat namespace with appropriate tagging strategies, you can achieve clear separation between environments and maintain a reliable deployment pipeline. The focus shifts from folders to a structured image naming convention, ultimately leading to more robust and manageable container deployments. I've found that once you adjust to this way of thinking, it becomes very natural, and, indeed, much more powerful.
