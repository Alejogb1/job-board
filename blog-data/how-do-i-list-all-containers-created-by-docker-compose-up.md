---
title: "How do I list all containers created by `docker-compose up`?"
date: "2024-12-23"
id: "how-do-i-list-all-containers-created-by-docker-compose-up"
---

,  Funny enough, this is a question that takes me back a bit. I remember a particularly challenging deployment a few years ago where I needed to meticulously track down container lifecycles managed by docker-compose. It quickly became apparent that knowing *exactly* how to list the spawned containers was crucial. The 'docker ps' command alone isn’t sufficient; you need that added layer of understanding about the relationship docker-compose creates.

The core issue here is understanding that `docker-compose up` doesn’t simply launch containers with random names. It prefixes them, linking them directly back to the project and service definition within your `docker-compose.yml` file. This prefixing mechanism is key to correctly identifying containers belonging to a specific compose setup.

So, while `docker ps -a` will show all containers (running or stopped), it becomes unwieldy in a complex environment. We need to filter. That’s where `docker-compose ps` comes in, though with a slight twist. It won't give you *all* containers, especially if some are stopped. For a truly comprehensive view, you need to lean on docker’s underlying tools combined with some targeted filtering. The method I settled on, after some trial and error, involves leveraging the `docker ps --filter` option. This approach also lets you sidestep complexities that may arise when using a specific `compose project name` flag. Instead, we’ll filter by labels.

Here's the logic: each container created by `docker-compose` is tagged with a label, typically in the format `com.docker.compose.project`. The value for this label is derived from the directory containing your `docker-compose.yml` file, unless overridden by explicitly providing a `project_name` in your `docker-compose.yml` or through an environment variable. The goal, then, is to filter `docker ps` using this label.

Here are three practical ways to achieve this, along with code examples in Bash:

**Example 1: Using `docker ps` with label filtering (the most reliable method):**

This snippet is, in my opinion, the most direct and consistent way. It retrieves all containers, then uses `--filter` to narrow down to those with the `com.docker.compose.project` label matching the name of the parent directory of the compose file. The backticks allow us to execute the `basename` command to retrieve the directory name.

```bash
#!/bin/bash

project_name=$(basename "$(dirname "$PWD")") # gets the project directory name
docker ps -a --filter "label=com.docker.compose.project=$project_name"
```
This approach also handles cases where a specific `project_name` was assigned directly, as docker-compose itself will properly associate that with the label `com.docker.compose.project`.

**Example 2: Extracting just the container IDs for scripting (for automation):**

This expands on the first method, extracting only the container ids which are then printed one per line. This is particularly valuable for scripts requiring iterative actions on each container, or simply needing the container ids for any other docker commands.

```bash
#!/bin/bash

project_name=$(basename "$(dirname "$PWD")")
docker ps -a --filter "label=com.docker.compose.project=$project_name" --format "{{.ID}}"
```
The `--format "{{.ID}}"` extracts only the container IDs from the output, giving a clean, easy-to-parse list. Using format is incredibly versatile for parsing outputs of various docker commands, a skill well worth investing in.

**Example 3: Using `docker-compose ps -a` (less versatile, but convenient for live projects):**

While less versatile for scripting, using `docker-compose ps -a` when you are within your project’s directory will also show you all containers from the project regardless if they are running or not, providing an easy way to check for stopped containers specific to this compose setup. This method will use the currently active `docker-compose.yml` file or the default `docker-compose.yml` file in the directory to figure out which containers belong to it.
```bash
#!/bin/bash
docker-compose ps -a
```

This command directly communicates with `docker-compose` and its internal records. It can be helpful but is limited if you are operating outside the compose context or have a need for more flexibility.

**Why these solutions and not others?**

*   **Reliability:** Label filtering is more resilient to changes in docker-compose versions and naming conventions than trying to parse container names directly. Directly parsing strings can fail or create brittle scripts.
*   **Clarity:** Filtering by labels makes your intent explicit and is more readable.
*   **Flexibility:** The `docker ps` approach allows for further customization of the output with options like `--format` or additional filters.
*   **Consistency:** These methods work effectively whether containers are running or have been stopped.

**Things to Keep in Mind:**

*   **`project_name`:** Be aware of how `project_name` is determined within docker-compose: directory name by default, or explicit declaration in the `docker-compose.yml` file, or via environment variable.
*   **Context:** Always ensure you’re executing these commands within the correct directory or use the proper path in the `-f` flag, when using it.
*   **Environment:** The environment variables used by docker-compose can affect the project name, particularly if using a shared or CI/CD pipeline environment.

**Further Reading:**

To deepen your understanding, I'd suggest diving into the following resources:

*   **"Docker in Action"** by Jeff Nickoloff and Stephen Kuenzli: This book provides a comprehensive guide to Docker, including detailed explanations of its core concepts, such as labels, networking, and compose. It goes beyond basic tutorials and into the practical applications of running docker in real-world scenarios. It's a bit dated (last published in 2019), but still a highly valuable resource for getting a deep understanding of how docker works under the hood.
*   **"The Docker Book"** by James Turnbull: Another solid resource for a thorough understanding of docker, but it goes into less operational detail, and more into concepts than "Docker in Action".
*   **Docker Documentation:** The official documentation at `docs.docker.com` is absolutely essential. Pay close attention to the sections on `docker ps`, filtering, labels, and docker-compose.
*   **The `docker ps` manual page (`man docker ps`)**: For a complete reference, the manual page provides a formal description of all the command options and their precise effects. This can be invaluable for truly mastering the command.

In conclusion, while `docker-compose ps` offers a basic overview, leveraging `docker ps` with label filtering provides the most reliable and flexible way to list *all* containers associated with your docker-compose project, regardless of their state. The three examples above should provide a solid foundation for managing your docker-compose deployments effectively. Remember to keep practicing and exploring docker's functionalities, as hands-on experience truly deepens understanding.
