---
title: "How do I use one docker image for multiple users with their system login $USER inside the docker container when the user runs the docker image?"
date: "2024-12-23"
id: "how-do-i-use-one-docker-image-for-multiple-users-with-their-system-login-user-inside-the-docker-container-when-the-user-runs-the-docker-image"
---

Alright,  Managing user identities within docker containers, especially when attempting to mirror host system users, is a recurring challenge. I've encountered this a few times, notably during a project where we were deploying individual development environments, each needing the specific user context of the developer using the container. It's not as straightforward as one might initially hope, but there are robust approaches we can leverage. The primary hurdle is that docker, by default, runs processes within a container under a single user, typically root, which creates a mismatch with the desired user-specific context.

The core issue stems from how docker's user namespaces function and interact with the host system. Each docker container typically operates within its own user namespace, isolated from the host's user namespace. This means that user IDs (UIDs) and group IDs (GIDs) inside the container are generally different from those on the host. Therefore, running processes as the host's `$USER` inside the container requires some careful orchestration. It's more than just passing environment variables; it requires actual UID/GID mapping or other methods that we'll examine here.

First, let's discuss the initial, naive approach and why it fails. One might think simply setting the `USER` environment variable inside the dockerfile, and then attempting to use that environment variable to, say, switch users using a `su` command within the entrypoint script, might solve the problem. However, `USER` is just a variable, not a command to actually change the effective user ID. The processes would still execute under root's jurisdiction unless explicitly altered. That's exactly the mistake I did the first time and learned it the hard way.

So, what are some viable approaches that actually achieve what we need? There are primarily three methods that I've personally found useful in various circumstances.

1. **Using `--user` Flag with UID/GID Mapping:**

   The cleanest method, particularly if you're dealing with known user IDs, involves utilizing the `--user` flag when starting the container. This allows you to specify which user (and optionally, group) the container's processes should run as. To dynamically use the host user, we need to ascertain their UID and GID on the host and pass them during the `docker run` operation.

   Here's how you could structure it:

   ```bash
   # On the host
   HOST_UID=$(id -u)
   HOST_GID=$(id -g)
   docker run --user $HOST_UID:$HOST_GID -it my_image /bin/bash
   ```

   In the dockerfile, you generally don't need any special user creation. The image should just provide the tools and application you intend to use.

   **Dockerfile (example):**
    ```dockerfile
    FROM ubuntu:latest
    RUN apt-get update && apt-get install -y --no-install-recommends bash
    CMD ["/bin/bash"]
    ```

   This approach relies on mapping the current host user's UID/GID to the container’s user. It's effective and doesn't require modifications within the image itself, making it flexible and easy to deploy. However, it needs manual adjustment whenever there's a change on the host side, making it ideal in development scenarios, but you must consider the implications in production environment. There are also potential issues if the UID/GID on the host clashes with a user inside the container which brings us to the second method.

2. **User Creation within the Dockerfile:**

   Another approach, useful when you don't want to rely on the host's UID and GID, is creating a user within the Dockerfile itself, and then executing the entry point as that user. This approach is necessary for ensuring a consistent and portable container environment, especially if the target user's uid is not present on the host.

   Here's how you would craft this:

   **Dockerfile (example):**
   ```dockerfile
   FROM ubuntu:latest

   ARG USER_ID=1000
   ARG GROUP_ID=1000

   RUN groupadd -g ${GROUP_ID} usergroup && \
       useradd -u ${USER_ID} -g usergroup -ms /bin/bash user

   USER user

   RUN apt-get update && apt-get install -y --no-install-recommends bash

   CMD ["/bin/bash"]
   ```
    And to run:
   ```bash
   docker run -it --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) my_image
   ```

   In this setup, we're creating a user named `user` with the provided `USER_ID` and `GROUP_ID` during the image build. The `USER user` instruction ensures the subsequent commands within the container and the initial process run with this user's context. The benefit here is it guarantees the user exists on the container, which removes a huge possibility of error. This is much better when used on production scenarios because you want predictable and portable containers.

3. **`gosu` or Similar Tools:**

   For scenarios where you need more granular control or dynamic switching, you can use tools like `gosu`. These are lightweight tools designed to execute commands under a specified user. The beauty here is that it doesn't rely on `sudo` which can complicate or bloat the image. These tools provide a safe way to temporarily drop privileges, which is particularly useful in complex entrypoints.

   **Dockerfile (example):**
    ```dockerfile
    FROM ubuntu:latest

    RUN apt-get update && apt-get install -y --no-install-recommends wget

    RUN wget -q -O /usr/local/bin/gosu "https://github.com/tianon/gosu/releases/download/1.16/gosu-amd64" && \
        chmod +x /usr/local/bin/gosu

    COPY entrypoint.sh /entrypoint.sh
    RUN chmod +x /entrypoint.sh

    ENTRYPOINT ["/entrypoint.sh"]

    ```
    And the `entrypoint.sh`:

    ```bash
    #!/bin/bash
    set -e
    HOST_UID=$(id -u)
    HOST_GID=$(id -g)

    gosu $HOST_UID:$HOST_GID "$@"
    ```
    Running the image:
   ```bash
   docker run -it my_image /bin/bash
   ```

   Here, the `entrypoint.sh` script dynamically captures the host UID/GID and uses gosu to launch the shell under the host user's context. The entry point can perform other set up tasks before launching the user process.

Now, picking the optimal solution depends on your specific needs. If you're in a development environment where you have control over the environment, the first method with the `--user` flag is straightforward and efficient. If you're aiming for portability and consistency, especially in production, creating a user inside the dockerfile, ideally through build-time arguments, would give a more dependable environment. When dealing with very specific and dynamic situations and also needing more granular control, a tool such as `gosu` adds flexibility and guarantees more control.

For further reading, I would recommend delving into the official docker documentation on user namespaces. It will offer a deep understanding of the inner workings. Also, "The Docker Book" by James Turnbull provides an excellent foundational understanding of docker concepts, including user management. In addition, diving into specific linux user system's documentation such as how user management works on Linux with `useradd` and `groupadd`, would provide a fundamental comprehension.

I’ve found that understanding how these underlying systems function is key to tackling the problem at hand. It requires careful orchestration and a good understanding of both docker and Linux user management, and hopefully, these methods can be a starting point for you.
