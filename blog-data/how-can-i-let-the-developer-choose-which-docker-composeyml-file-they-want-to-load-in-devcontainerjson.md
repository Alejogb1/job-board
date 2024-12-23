---
title: "How can I let the developer choose which docker-compose.yml file they want to load in devcontainer.json?"
date: "2024-12-23"
id: "how-can-i-let-the-developer-choose-which-docker-composeyml-file-they-want-to-load-in-devcontainerjson"
---

Okay, let’s tackle this. It’s a common challenge, and I’ve certainly been down that rabbit hole a few times, especially when dealing with complex microservices architectures. The core issue is providing developers with the flexibility to choose their docker-compose configuration within a devcontainer environment. You want to avoid a monolithic dev setup while keeping things manageable. Let's talk about how to achieve this practically, and I'll share some code examples to illustrate the approach.

Fundamentally, devcontainer.json gives us hooks that we can leverage. We’re going to exploit the `onCreateCommand` or `postCreateCommand` properties, coupled with a mechanism to allow the developer to specify their desired docker-compose file. We’ll sidestep the rigidity of hardcoding paths by introducing a variable or setting that developers can manipulate.

Let's break down the methods and look at some concrete ways you can achieve this.

**Method 1: Environment Variable with `onCreateCommand`**

One of the cleanest approaches is to utilize an environment variable, defined outside the devcontainer environment, that can be read by the container during its creation.

Here’s how it works in `devcontainer.json`:

```json
{
    "name": "Custom Compose Dev",
    "image": "mcr.microsoft.com/devcontainers/universal:2",
    "onCreateCommand": "if [ -n \"$DOCKER_COMPOSE_FILE\" ]; then docker compose -f $DOCKER_COMPOSE_FILE up --build -d; fi",
    "customizations": {
        "vscode": {
            "settings": {
                "terminal.integrated.defaultProfile.linux": "zsh"
            }
        }
    }
}
```

In this example, we are checking for the existence of an environment variable called `DOCKER_COMPOSE_FILE`. If this variable is set and not empty, we use it to launch the docker-compose setup.

**How to use this method:**

Before launching the devcontainer, the developer must set the `DOCKER_COMPOSE_FILE` environment variable. This can be done in their shell, for example, `export DOCKER_COMPOSE_FILE=./docker-compose.web.yml`. Then, when the devcontainer is built, the `onCreateCommand` will dynamically pick up the desired docker-compose file.

**Practical Scenario:** I used this extensively in a project where we had separate docker-compose files for web development, backend services, and data engineering pipelines. This setup made switching between those workflows extremely straightforward for my team.

**Method 2: Interactive Script with `postCreateCommand`**

This method gets a bit more interactive. We use a bash script that prompts the developer to select the docker-compose file they want.

Here's how the `devcontainer.json` and a hypothetical `select_compose.sh` script might look:

`devcontainer.json`:
```json
{
    "name": "Interactive Compose Dev",
    "image": "mcr.microsoft.com/devcontainers/universal:2",
	"postCreateCommand": "bash .devcontainer/select_compose.sh",
    "customizations": {
        "vscode": {
            "settings": {
                "terminal.integrated.defaultProfile.linux": "zsh"
            }
        }
    }
}

```

`select_compose.sh`:
```bash
#!/bin/bash

echo "Select docker-compose file:"
echo "1: docker-compose.web.yml"
echo "2: docker-compose.api.yml"
echo "3: docker-compose.all.yml"

read -p "Enter your choice (1, 2, or 3): " choice

case $choice in
    1) COMPOSE_FILE="./docker-compose.web.yml" ;;
    2) COMPOSE_FILE="./docker-compose.api.yml" ;;
    3) COMPOSE_FILE="./docker-compose.all.yml" ;;
    *) echo "Invalid choice."; exit 1 ;;
esac

docker compose -f $COMPOSE_FILE up --build -d

```

**How to use this method:**

Place `select_compose.sh` in `.devcontainer` directory. When the container is created, this script is executed, presenting a menu to the developer. After making the selection, the appropriate `docker compose up` command is executed.

**Practical Scenario:** In one project, where the team was relatively new to devcontainers, this interactive approach gave them a visual and more user-friendly mechanism for choosing their configuration. It made the process less intimidating for the team.

**Method 3: Configuration File with `onCreateCommand`**

Instead of relying on environmental variables or interactive prompts, you could have a simple configuration file within the project (e.g., `.devcontainer/dev.config`) that specifies the docker-compose file to be loaded.

`devcontainer.json`:
```json
{
    "name": "Config File Compose Dev",
    "image": "mcr.microsoft.com/devcontainers/universal:2",
	"onCreateCommand": "source .devcontainer/dev.config && docker compose -f $DOCKER_COMPOSE_FILE up --build -d",
    "customizations": {
        "vscode": {
            "settings": {
                "terminal.integrated.defaultProfile.linux": "zsh"
            }
        }
    }
}
```

`.devcontainer/dev.config`:
```bash
#!/bin/bash
DOCKER_COMPOSE_FILE="./docker-compose.web.yml"
```

**How to use this method:**

Create the `dev.config` file within the `.devcontainer` directory. Developers can modify the `DOCKER_COMPOSE_FILE` variable directly inside this file before building their container.

**Practical Scenario:** For one large monolithic project, we found this configuration file method beneficial for separating individual developer setups within the larger code structure. It provided a good balance of flexibility and ease of use.

**Important Considerations:**

*   **Error Handling:** Ensure your scripts handle errors gracefully. Check if the specified docker-compose files exist, and provide informative error messages to the developer.
*   **Security:** Be cautious when accepting arbitrary input from shell variables or file configurations. Always validate paths and file names.
*   **Documentation:** Clearly document the chosen method and how developers should use it. Providing comprehensive instructions is vital for onboarding and adoption.
*   **Version Control:** Keep these configuration files under version control. This ensures everyone is working with the correct set of configuration options.

**Resources:**

For further reading on devcontainers, I’d recommend:

*   **Microsoft's official documentation on devcontainers:** It’s regularly updated and offers the most comprehensive and up-to-date information. Specifically, look into the `devcontainer.json` specification and the various lifecycle commands.
*   **The Docker documentation on docker-compose:** A deep understanding of docker-compose's functionality and CLI options is crucial for implementing any of the strategies we discussed.
*   **"Effective DevOps: Building a Culture of Collaboration, Affinity, and Tooling at Scale" by Jennifer Davis and Ryn Daniels**: While not focused solely on devcontainers, this book offers broad perspectives and patterns that are applicable to this problem. It's a solid read for any developer seeking to understand how to manage and scale development workflows.

In my experience, picking the appropriate method depends on the project’s size, the team's expertise, and their preference for flexibility versus ease of use. These three options provide a good starting point for customizing your devcontainer setups and giving developers control over their environments. Remember to tailor these examples to your specific requirements and don't be afraid to explore what works best for your situation.
