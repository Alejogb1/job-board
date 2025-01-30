---
title: "Why is NerdCTL failing to use a local image?"
date: "2025-01-30"
id: "why-is-nerdctl-failing-to-use-a-local"
---
NerdCTL's inability to utilize a locally built Docker image often stems from a mismatch between the image's name as registered within the Docker daemon and how NerdCTL references it.  My experience debugging this issue across numerous Kubernetes deployments points to several common sources of error, primarily revolving around image tagging and the interaction between the Docker context and NerdCTL's container runtime configuration.  This response will detail the underlying mechanisms and offer practical solutions.

**1. Clear Explanation:**

NerdCTL, as a command-line tool for interacting with Kubernetes, relies on Docker to manage container images. When you build a Docker image locally, it's stored within the Docker daemon's registry.  This local registry is distinct from remote registries like Docker Hub or private repositories.  NerdCTL must be explicitly directed to use the local image by providing the correct image name and tag.  Failure occurs when this name doesn't precisely align with the image's name as presented by the Docker daemon's `docker images` command.  Errors can arise from improperly tagged images, incorrect naming conventions (especially regarding the inclusion of the repository prefix), or even the use of a misconfigured Docker context.  Further, ensure that NerdCTL is properly configured to communicate with your Docker daemon; a misconfiguration in the Docker socket path can lead to similar failures.  Finally, a lack of necessary permissions within the user context running NerdCTL can also prevent access to the locally built images.


**2. Code Examples with Commentary:**

**Example 1: Incorrect Tagging**

Let's assume I've built an image named `my-app` with a tag `latest`.  The correct way to refer to it in my `nerdctl run` command is:

```bash
nerdctl run --rm -d -p 8080:8080 my-app:latest
```

However, I often encounter the following mistake: omitting the tag entirely, relying on `latest` implicitly. While Docker might accept this, NerdCTL may behave unpredictably and might try to pull the image from a remote registry instead of using the local version. This leads to a failure to run the container if the image isn't present in the remote registry.

```bash
# Incorrect - omitting the tag
nerdctl run --rm -d -p 8080:8080 my-app
```


**Example 2: Repository Prefix Mismatch**

Another common pitfall involves the image repository name.  During development, we frequently omit the repository prefix.  The Docker daemon might implicitly assign a default repository, often empty or set to `<none>`.  However, NerdCTL may explicitly require the full repository path. During a recent project involving a private registry, I discovered that omitting the `<registry>/<project>` prefix caused a similar issue:

```bash
# Incorrect - Missing repository prefix; assumes the image is in a default registry which often won't be the case.
nerdctl run --rm -d -p 8080:8080 my-app:v1.0
```


The corrected command, including the explicit repository path is shown below.  Note that using `localhost` as the registry will attempt to pull the image from the local Docker daemon:

```bash
# Correct - Includes the explicit repository path
nerdctl run --rm -d -p 8080:8080 localhost/my-app:v1.0
```


**Example 3: Docker Context and NerdCTL Configuration**

In more complex environments, the Docker context becomes crucial.  I've had situations where I was working with multiple Docker contexts, each pointing to different Docker daemons or even remote machines.  If my NerdCTL instance was not pointed to the correct Docker context containing the local image, the `nerdctl run` command would fail.  To illustrate, let's consider a scenario with two contexts, 'default' and 'local-dev':

```bash
# Check current Docker context.  This should point to the context containing the local image
docker context ls

# Switch to the appropriate context if needed.  Replace 'local-dev' with your context name.
docker context use local-dev

# Now run the NerdCTL command.  NerdCTL should inherit this context.
nerdctl run --rm -d -p 8080:8080 localhost/my-app:v1.0
```

Failing to manage the Docker context properly can lead to unexpected behavior.  Ensure that the context used by NerdCTL matches the one where the image resides.  Always verify your current context before running commands that involve local images.


**3. Resource Recommendations:**

I highly recommend consulting the official NerdCTL documentation.  Understanding the command-line arguments and configuration options, especially those related to container runtimes and image pulling, is essential.  The Docker documentation on image management, tagging conventions, and registry interaction is also invaluable. Thoroughly exploring the differences between local and remote image registries and how they are addressed within the context of the command-line interfaces will prove useful.  Finally, revisiting the fundamental concepts of Docker and Kubernetes will solidify the underlying principles at play here.


In conclusion, consistently verifying image names, tags, and repository prefixes, coupled with meticulous management of your Docker context, will significantly reduce instances where NerdCTL fails to utilize your locally built images.  The core principle remains ensuring a precise alignment between how the image is stored by the Docker daemon and how it's referenced in your NerdCTL commands.  By addressing these potential points of failure, one can achieve a smooth and efficient workflow when integrating locally built Docker images into your Kubernetes deployments with NerdCTL.
