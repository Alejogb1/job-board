---
title: "How to let devs choose docker-compose.yml file in devcontainer.json?"
date: "2024-12-23"
id: "how-to-let-devs-choose-docker-composeyml-file-in-devcontainerjson"
---

Okay, let's tackle this. I've certainly been in the trenches with devcontainers and the nuances of customization, especially when it comes to managing different docker-compose setups for various development scenarios. The core issue, as you’ve framed it, is allowing developers to select a specific `docker-compose.yml` file directly through the `devcontainer.json` configuration. While there's no direct, built-in property to explicitly *choose* a file from a set of alternatives, there are several effective strategies we can implement to achieve this level of flexibility. I've used these, or variations of them, on projects ranging from small internal tools to complex microservices architectures.

The challenge arises from the fact that the `dockerComposeFile` property in `devcontainer.json` expects a direct file path or an array of file paths. It doesn't support, out of the box, something like a selection dropdown for different `docker-compose` files. However, we can use conditional logic, environment variables, and a clever bit of scripting to achieve the desired result.

Essentially, our goal is to inject the correct `dockerComposeFile` value dynamically based on some developer-defined condition, instead of hardcoding a specific file path within `devcontainer.json`. I generally avoid modifying the base `devcontainer.json` files because they are usually part of a shared workspace config. Instead, the solution involves creating a setup where the `devcontainer.json` reads an environment variable, and that environment variable points to our selected `docker-compose.yml` file. This environment variable could then be set before the container is built, via user input, for example.

Here's a detailed breakdown of how I'd approach this:

**1. Leverage Environment Variables:**

First, we'll add an environment variable within our `devcontainer.json`, which will determine which docker-compose file to use. We'll use `DEV_COMPOSE_FILE` as the key and allow a default value. This means we don't need to pass it initially, but we can if we want to override the default setup. Here's how our `devcontainer.json` might look:

```json
{
  "name": "My Dev Container",
  "dockerComposeFile": "${containerEnv:DEV_COMPOSE_FILE:-docker-compose.yml}",
  "service": "app",
  "workspaceFolder": "/workspace",
   //other configuration details ...
  "customizations": {
        "vscode": {
            // vscode customizations ..
        }
    }

}
```

In the snippet above, we use `${containerEnv:DEV_COMPOSE_FILE:-docker-compose.yml}`. This means: read the `DEV_COMPOSE_FILE` environment variable; if it's not defined, use `docker-compose.yml` as the default. This approach gives us a safe fall-back and doesn't break existing setups.

**2. Setup Script and Developer Choice:**

Now, we need a mechanism for the developer to select their desired `docker-compose` file. A simple script will serve this purpose. I've used bash for this example as it's quite common, but any scripting language will suffice. This script should execute before the `devcontainer` starts. To make this simple, I’ll name it `setup_env.sh`. This file will reside in the root of the repository alongside the `devcontainer.json` file. Here’s the `setup_env.sh` example.

```bash
#!/bin/bash

echo "Choose a docker-compose configuration:"
echo "1) Standard: docker-compose.yml"
echo "2) Test Environment: docker-compose.test.yml"
echo "3) Debug Environment: docker-compose.debug.yml"

read -p "Enter your choice (1-3): " choice

case $choice in
  1)
    export DEV_COMPOSE_FILE="docker-compose.yml"
    ;;
  2)
    export DEV_COMPOSE_FILE="docker-compose.test.yml"
    ;;
  3)
    export DEV_COMPOSE_FILE="docker-compose.debug.yml"
    ;;
  *)
    echo "Invalid choice. Using the default: docker-compose.yml"
    export DEV_COMPOSE_FILE="docker-compose.yml"
    ;;
esac

echo "Using docker-compose file: $DEV_COMPOSE_FILE"
```
This script presents the developer with a numbered list of `docker-compose` files. Based on their choice, the script sets the `DEV_COMPOSE_FILE` environment variable before the devcontainer is built.

**3. Integrating the Setup Script:**

Finally, we need to tell the `devcontainer` to execute the setup script *before* the container creation. We do this using the `onCreateCommand` or `postCreateCommand` in our `devcontainer.json`. I would use `onCreateCommand` to ensure the environment is set before the docker-compose is started.

```json
{
    "name": "My Dev Container",
  "dockerComposeFile": "${containerEnv:DEV_COMPOSE_FILE:-docker-compose.yml}",
  "service": "app",
  "workspaceFolder": "/workspace",
  "onCreateCommand": "bash ./setup_env.sh",
   //other configuration details ...
  "customizations": {
        "vscode": {
            // vscode customizations ..
        }
    }
}
```
By adding `"onCreateCommand": "bash ./setup_env.sh"`, we instruct devcontainer to execute the script before the container is built. This will set our desired `DEV_COMPOSE_FILE` environment variable for the container creation process.

**Practical Considerations and Further Reading:**

This solution is robust for multiple environments and user preferences. Here's a breakdown of why I favor this, and some potential enhancements.

*   **Flexibility:** Developers can choose their configuration with minimal effort. Adding more choices simply requires adding to the script and creating more `docker-compose.yml` files.
*   **Maintainability:** The `devcontainer.json` remains clean and doesn't require changes when adding or removing a `docker-compose` configuration.
*   **Extensibility:** This approach can be extended to support other configuration parameters, like a specific branch or other project-specific requirements.

For further study on Docker Compose, I highly recommend “*Docker in Practice*” by Ian Miell and Aidan Hobson Sayers, It offers an in-depth practical view of using Docker and Docker Compose effectively. Furthermore, understanding the finer details of devcontainers is aided greatly by reading the official *Microsoft documentation on devcontainers*, which is comprehensive and regularly updated.

There are additional strategies you might explore. For instance, in large development teams, I’ve seen use of different profiles or command line arguments used to set environment variables programmatically, depending on the tooling already used by the team. Also, if developers are using a particular build system you could expose a custom property in `devcontainer.json` and pass it as an environment variable using the build system’s tooling.

In summary, this environment variable-driven strategy, along with a simple setup script, provides a good balance between flexibility, maintainability, and usability. It is a pattern that has served me well on many projects, and I hope it can be of equal use to you.
