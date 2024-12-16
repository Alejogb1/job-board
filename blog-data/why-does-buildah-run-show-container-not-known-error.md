---
title: "Why does `buildah run` show container not known error?"
date: "2024-12-16"
id: "why-does-buildah-run-show-container-not-known-error"
---

Alright, let's tackle this "container not known" error with `buildah run`. It’s a classic snag, and I’ve certainly encountered it a few times myself, usually during the frantic pace of a late-night deployment push or when experimenting with some new containerization strategy. The core problem, as you might suspect, revolves around how `buildah` manages and references containers, specifically when they haven’t been properly committed or tagged after creation.

The `buildah run` command isn’t designed to operate directly on the image layers that `buildah` builds up in its working directories. Instead, it needs a container instance, which is essentially an editable snapshot of an image. When you use `buildah from <image>`, `buildah container` creates a working container instance based on this image but does not give that instance a name. This container instance is where you make changes, install packages, and perform all the necessary tweaks. When you execute `buildah run`, it expects to act upon a *named* container instance, not just the working build context. When the error "container not known" appears, it usually means `buildah run` cannot locate the named container you specified in its arguments.

Let me walk you through what typically happens and how to avoid this, drawing from some projects where I’ve had to debug similar issues. It’s not uncommon to see newcomers use `buildah from <image>`, then immediately try `buildah run <command>`. This workflow skips a crucial step, the naming and management of that created container. We must create a container instance and give it a name before we can refer to it.

There are two main scenarios where this can occur: one, you haven't actually created a container with the `buildah from` command and instead are trying to use an image, or two, you created one and didn’t persist it before the instance was cleaned up. I'll illustrate this better with the following code examples.

**Scenario 1: Incorrectly attempting `buildah run` without first creating a named container**

Imagine you have a Dockerfile that you want to build using `buildah`. Here's a simple example where we try to run a command after building the container image.

```bash
#!/bin/bash
# Assume you have a Dockerfile in the current directory

# Step 1: build the image
buildah bud -t my-test-image .

# Step 2: Try to run a command using buildah run with the image name (WRONG)
buildah run my-test-image echo "Hello from within the image"

# output: container my-test-image not known
```

Here, we’ve made the classic mistake. `buildah run` doesn't take the image name as an argument, and tries to use it as if it is the name of a created container. It is trying to locate a container named `my-test-image`, and of course, it can’t, because we haven't created one yet. You’ve built an *image*, not a *container instance*. The fix here involves creating a container from the image.

**Scenario 2: Correctly creating and then running within a container**

Here’s the rectified version.

```bash
#!/bin/bash
# Assume you have a Dockerfile in the current directory

# Step 1: build the image
buildah bud -t my-test-image .

# Step 2: Create a container instance from the image and name it "my-test-container"
container_id=$(buildah from my-test-image)
buildah commit $container_id my-test-image-for-use
buildah container my-test-container

# Step 3: Run a command inside the *named* container
buildah run my-test-container echo "Hello from within the container"

# Step 4: Cleaning up the container (optional, but good practice)
buildah rm my-test-container
buildah rmi my-test-image-for-use
```

In this modified example, after building the image, we create a container instance using `buildah from`, naming the image using variable substitution to extract the resulting container id. We then commit the image to a new intermediate image and create the container using `buildah container <container name>.` Now, when we run `buildah run my-test-container echo "Hello from within the container"`, `buildah` can find a container with the name `my-test-container` and execute the command.

**Scenario 3: An extended example with modifications to the container**

Here’s a slightly more complex scenario where we modify the container instance and then run a command. This mimics a more real-world build process.

```bash
#!/bin/bash
# Assume you have a Dockerfile in the current directory

# Step 1: Build the image
buildah bud -t my-test-image .

# Step 2: Create a container instance from the image
container_id=$(buildah from my-test-image)
buildah commit $container_id my-test-image-for-use
buildah container my-test-container

# Step 3: Install a package within the container instance
buildah run my-test-container yum -y install jq

# Step 4: Run a command that uses the installed package
buildah run my-test-container jq -V

# Step 5: Cleaning up
buildah rm my-test-container
buildah rmi my-test-image-for-use

```

In this example, we create a container named `my-test-container` as before, but now we also install the `jq` package within the container using another `buildah run` command. After that, we can use `jq` by executing `buildah run my-test-container jq -V`, demonstrating that the command works within the container context.

The key takeaway is that `buildah run` works on containers—instances that have been created from images, and given a name—and not directly on images. The `buildah from` command does the image to container conversion. Further, `buildah container` turns it into a manageable working container instance for us to work with.

For anyone wanting a deeper dive into the intricacies of `buildah` and container internals, I'd strongly recommend exploring the official `buildah` documentation on GitHub; it's comprehensive and continually updated. Also, “Container Security: Issues, Solutions, and the Road Ahead” by Junaid Farooq is a great resource to gain a thorough understanding of container technologies. Furthermore, understanding the concepts described in the OCI (Open Container Initiative) specifications helps in grasping the low-level details of container management. Finally, reading articles and papers focusing on immutable infrastructure are useful in understanding why these types of tools are so important.

Ultimately, the "container not known" error is a helpful reminder that when dealing with container build tools like `buildah`, we must clearly separate the steps of image creation from container instantiation. By ensuring we’re always referencing named containers when using `buildah run`, these kinds of errors can be easily resolved. Remembering these steps avoids a lot of head-scratching during the development cycle, ensuring your builds are consistently successful.
