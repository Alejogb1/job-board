---
title: "How can I dynamically mount a directory to a VS Code Dev Container without hardcoding it in devcontainer.json?"
date: "2024-12-23"
id: "how-can-i-dynamically-mount-a-directory-to-a-vs-code-dev-container-without-hardcoding-it-in-devcontainerjson"
---

Alright,  I've bumped into this exact scenario more times than I care to remember, typically when working with projects that have highly variable data or configuration directories that shouldn't be baked into the image itself or checked into version control. We want flexibility, something that lets us adjust those mounts based on the specific environment we're working in, without needing to constantly modify the `devcontainer.json` file. The core of the issue lies in how dev containers handle mounting volumes: they read instructions primarily from `devcontainer.json`, which is not ideal when the source of the mount is dynamic.

My preferred strategy to address this involves leveraging environment variables, typically configured outside of the container's scope, along with some clever manipulation during container creation. It's a combination of techniques, but the result is a clean, maintainable setup. The crux of the matter is to have dev containers read environment variables that exist *outside* of the container's definition and use those to set up the mounts.

Let me give you an example from a past project where we were working on a large data analysis platform. We had terabytes of data spread across several storage solutions. Hardcoding each of those paths in `devcontainer.json` was a no-go, as each engineer needed to work with a specific subset of the overall dataset. Here’s how we solved it, and I'll illustrate this with code snippets.

**The core idea:** Inject environment variables during container creation and use them within `devcontainer.json`'s `mounts` array using variable substitution.

Here's the setup process:
1. **Set the Environment Variables:** Outside of the container, before you initiate the dev container, we will define variables that point to the directories to be mounted. I've seen this done using `.env` files that are read by the shell, set directly in the terminal, or managed by system-level settings. We would typically have scripts in the project repository that would do this setup automatically based on parameters provided by the engineers.
2. **Use Variable Substitution in `devcontainer.json`:** Inside the `devcontainer.json`, we’ll use string interpolation to reference these environment variables. This approach allows us to avoid hardcoding the path.
3. **Optional: Docker Compose and Custom Scripts:** Sometimes you might need to do more complex mounting or setup which is best delegated to a custom docker-compose file, that could be setup in `.devcontainer/docker-compose.yml` and then referenced in the `devcontainer.json`.

**Code Example 1: Basic environment variable mount**

Let's say we want to mount a directory called `my_data` on the host system to `/app/data` inside the container.

First, set your environment variable before starting the dev container:

```bash
export MY_DATA_DIR=/path/on/host/to/my_data
```

Then, in your `devcontainer.json`:

```json
{
    "name": "My Dev Container",
    "image": "mcr.microsoft.com/devcontainers/universal:2",
    "mounts": [
      "source=${env:MY_DATA_DIR},target=/app/data,type=bind,consistency=cached"
    ],
    "customizations": {
        "vscode": {
            "settings": {
                "terminal.integrated.defaultProfile.linux": "bash"
            }
        }
    }
}
```

In this example, `${env:MY_DATA_DIR}` instructs the dev container engine to substitute the value of the `MY_DATA_DIR` environment variable when setting up the mount. The `consistency=cached` part can improve performance if the source directory is accessed frequently. You might need to use `delegated` instead in some situations. Refer to the official docker volume mount documentation for best performance based on the specific use-case.

**Code Example 2: Multiple dynamic mounts**

Now, suppose you have multiple directories you want to mount dynamically. Here's how you would extend the previous example:

First set these env vars:

```bash
export DATA_DIRECTORY_1=/path/on/host/data1
export CONFIG_DIRECTORY=/path/on/host/config
```

And here's how we set it up in our devcontainer.json:

```json
{
    "name": "My Dev Container",
    "image": "mcr.microsoft.com/devcontainers/universal:2",
    "mounts": [
        "source=${env:DATA_DIRECTORY_1},target=/app/data1,type=bind,consistency=cached",
        "source=${env:CONFIG_DIRECTORY},target=/app/config,type=bind,consistency=cached"
    ],
    "customizations": {
        "vscode": {
            "settings": {
                "terminal.integrated.defaultProfile.linux": "bash"
            }
        }
    }
}
```

This will mount two separate host directories to two locations inside the container. Again, notice the use of `${env:VARIABLE_NAME}` for each mount. The approach can be extended as much as needed, scaling well to complex projects.

**Code Example 3: Using Docker Compose and an environment file**

For more complex setup, such as dynamically setting a specific user or other parameters during container start, we can leverage `docker-compose`. We can have the `devcontainer.json` reference a `docker-compose.yml` file that can handle dynamically setting up the volumes. We will again be using the environment variables but in a slightly different way to show the flexibility of the setup.

First, our `.devcontainer/docker-compose.yml`:

```yaml
version: '3.9'
services:
  devcontainer:
    image: mcr.microsoft.com/devcontainers/universal:2
    volumes:
      - type: bind
        source: ${DATA_DIR_COMPOSE}
        target: /app/data_compose
    environment:
      - USER_ID_COMPOSE=${USER_ID_COMPOSE}
    user: ${USER_ID_COMPOSE}:${USER_ID_COMPOSE}
```

Set the environment variables in your shell or in a `.env` file, making sure the shell you're using loads these variables. In the `.env` file, you'll have something like this:

```
DATA_DIR_COMPOSE=/path/on/host/compose_data
USER_ID_COMPOSE=1000
```

Then, in `devcontainer.json`:

```json
{
    "name": "My Dev Container with Docker Compose",
     "dockerComposeFile": ["docker-compose.yml"],
     "service": "devcontainer",
    "customizations": {
        "vscode": {
            "settings": {
                "terminal.integrated.defaultProfile.linux": "bash"
            }
        }
    }
}
```

In this setup, the `devcontainer.json` points to the `docker-compose.yml`, which in turn reads the environment variables (`DATA_DIR_COMPOSE`, `USER_ID_COMPOSE`) we have defined. This enables significantly more complex setups as needed. This might be needed if you want to set up different container user permissions or want to make a more sophisticated set up.

**Important considerations:**

1. **Security:** Be mindful of the environment variables you expose. Ensure you're not accidentally leaking sensitive data. Avoid setting up sensitive mounts, as that would lead to data leaks.
2. **Environment consistency:** When working with teams, ensure that everyone has a consistent environment setup by documenting the required environment variables. Scripts are really useful in automating this part of the setup.
3. **`.env` file usage:** If using `.env` files, make sure that `.env` is added to your `.gitignore` to avoid committing environment specific configurations into your repositories.

**Recommended Resources:**

1.  **Docker Documentation:** Specifically, the sections related to bind mounts and volumes: This is your go-to for understanding the fundamentals of how Docker manages data.
2.  **Microsoft Dev Containers Documentation:** Thoroughly reviewing the dev containers documentation on mounting and configurations will offer insights specific to the VS Code extension.
3.  **"Docker Deep Dive" by Nigel Poulton:** For a deeper understanding of Docker concepts and internal workings, this is an invaluable resource.
4.  **"The DevOps Handbook" by Gene Kim, Jez Humble, Patrick Debois, and John Willis:** While not strictly about docker it teaches about DevOps and infrastructure-as-code principles and can offer insights on the whole setup.

In summary, dynamically mounting directories to a VS Code Dev Container without hardcoding requires a strategic combination of environment variables, variable substitution, and if necessary, docker compose. It provides the flexibility we need to manage complex setups and avoids storing sensitive information within configuration files. With the techniques outlined, I’ve found this approach to be consistently effective and significantly improve my development workflows over the years.
