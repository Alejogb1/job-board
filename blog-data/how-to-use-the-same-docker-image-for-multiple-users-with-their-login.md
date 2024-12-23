---
title: "How to use the same docker image for multiple users with their login?"
date: "2024-12-23"
id: "how-to-use-the-same-docker-image-for-multiple-users-with-their-login"
---

, let's unpack this interesting challenge. I remember back at Stellar Dynamics Corp., we faced a very similar situation. We had multiple data scientists who needed isolated environments to run their experiments, but maintaining separate images for each user was proving a logistical nightmare. So, how do you effectively use a single docker image for multiple users while still keeping their data and sessions isolated? It’s achievable, and it largely hinges on understanding how docker, user management within containers, and volume mounting interact.

The core issue is that a docker image, by its nature, is a read-only template. Once you instantiate a container from that image, it operates within a self-contained environment. However, each user should ideally have their own persistent data storage and configuration. Therefore, the challenge is not about modifying the image itself, but rather how to leverage volumes and user context within a container lifecycle.

We essentially need to create the illusion of separate user environments while utilizing the same underlying image. There are a few well-established approaches. The method I found most robust at Stellar Dynamics involved a combination of user creation, data persistence using mounted volumes, and leveraging the container’s entrypoint to manage user-specific configurations.

Let’s break it down into three practical examples, each demonstrating a slightly different nuance:

**Example 1: Simple User Creation and Volume Mounting**

In this basic example, the container is launched with a volume mapping to a host directory for each user. Inside the container, a new user is created (or an existing one is used), and subsequent processes are run with that user's context.

```bash
# Dockerfile (Simplified)
FROM ubuntu:latest
RUN apt-get update && apt-get install -y sudo && apt-get clean && rm -rf /var/lib/apt/lists/*
# Create an 'entrypoint' script for more complex commands
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
```
```bash
# entrypoint.sh
#!/bin/bash
set -e

if [ "$1" = "user_specific_setup" ]; then
  user_id="$2"
  user_home="/home/${user_id}"
    if ! id -u "$user_id" &>/dev/null; then
       useradd -ms /bin/bash "$user_id"
       mkdir -p "$user_home"
    fi
  chown "$user_id":"$user_id" "$user_home"
  exec su "$user_id" -c "cd ${user_home}; bash"
else
  exec "$@"
fi
```
```bash
# How to run the container
docker run -d \
    -v /path/to/host/user1:/home/user1 \
    --name user1_container  \
    my_image user_specific_setup user1
docker run -d \
    -v /path/to/host/user2:/home/user2 \
    --name user2_container  \
    my_image user_specific_setup user2
```

Here, `user_specific_setup` is an argument passed to `entrypoint.sh`, which creates the user on first run and ensures the user owns their own directory. Every time a container is spun up with a specific user's volume, it will be owned by that user. This provides a basic level of user isolation. The docker run command maps `/path/to/host/user1` on the host to `/home/user1` inside the container, thereby giving user1’s container a persistent data volume while keeping the user-specific data separate.

**Example 2: Using Environment Variables and Dynamic User Switching**

This expands on example 1 by incorporating an environment variable to specify the user. This allows for more dynamic behavior and removes the need to hard-code the user at the command line, but it requires that user to exist already within the image, or that the user creation commands are added elsewhere.

```bash
# Dockerfile (Simplified - no user creation)
FROM ubuntu:latest
RUN apt-get update && apt-get install -y sudo && apt-get clean && rm -rf /var/lib/apt/lists/*
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
```

```bash
# entrypoint.sh
#!/bin/bash
set -e

user_id="${USER_ID:-default}"
user_home="/home/${user_id}"
exec su "$user_id" -c "cd ${user_home}; bash"
```

```bash
# How to run the container
docker run -d \
    -v /path/to/host/user1:/home/user1 \
    -e USER_ID=user1 \
    --name user1_container \
    my_image
docker run -d \
    -v /path/to/host/user2:/home/user2 \
    -e USER_ID=user2 \
    --name user2_container \
    my_image
```

This method utilizes the `USER_ID` environment variable to dynamically determine which user context to switch to within the container, improving flexibility. This approach assumes users like 'user1' and 'user2' are already created during the image build process or some other method before container launch. This method is particularly useful when the users exist within an authentication directory like LDAP.

**Example 3: Leveraging docker compose for easier management**

In this example, we will showcase how to use docker-compose to simplify creating and managing users. We extend the user-creation concept in example 1 by also adding a startup script to the Dockerfile.

```bash
# Dockerfile (Simplified)
FROM ubuntu:latest
RUN apt-get update && apt-get install -y sudo && apt-get clean && rm -rf /var/lib/apt/lists/*
COPY startup.sh /usr/local/bin/startup.sh
RUN chmod +x /usr/local/bin/startup.sh
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
```

```bash
# startup.sh
#!/bin/bash
set -e
user_id="$1"
user_home="/home/${user_id}"
if ! id -u "$user_id" &>/dev/null; then
   useradd -ms /bin/bash "$user_id"
   mkdir -p "$user_home"
fi
chown "$user_id":"$user_id" "$user_home"
```

```bash
# entrypoint.sh
#!/bin/bash
set -e

if [ "$1" = "user_specific_setup" ]; then
  user_id="$2"
  /usr/local/bin/startup.sh "$user_id"
  exec su "$user_id" -c "cd /home/${user_id}; bash"
else
  exec "$@"
fi
```
```yaml
# docker-compose.yml
version: '3.8'
services:
  user1_container:
    image: my_image
    volumes:
      - ./user1_data:/home/user1
    command: ["user_specific_setup", "user1"]
  user2_container:
    image: my_image
    volumes:
      - ./user2_data:/home/user2
    command: ["user_specific_setup", "user2"]
```

```bash
# How to run the containers with docker-compose
docker-compose up -d
```

This approach allows you to define multiple user-specific configurations in a declarative manner, making deployment and management of these containers easier, avoiding the need for long `docker run` commands. The `startup.sh` script handles user creation and file ownership.

**Important Considerations:**

*   **Security:** Proper user isolation is key. Ensure users cannot gain access to each other's volumes and processes. Refer to “Docker Security Best Practices” by Liz Rice and “Container Security: Principles, Practices, and Examples” by Vincent Scavetta for thorough guidance.
*   **User Management:** Depending on your environment, you might integrate with existing user management systems (LDAP, Active Directory) for authentication and authorization.  “Understanding Linux Kernel” by Daniel P. Bovet and Marco Cesati will be useful for understanding the low level details of Linux User management.
*   **Resource Limits:** When running containers for multiple users, carefully consider resource limits to prevent any one user from monopolizing server resources. Docker allows you to manage CPU and memory usage using flags on startup.
*   **Image Maintainability:** Keep the base image lean and well-maintained. Frequent updates help ensure security and stability.
*  **Persistence:** Always ensure proper volume management. Incorrect volume configurations can potentially lead to data loss or security issues. Always follow the official Docker documentation for the most up to date and secure practices.

In closing, using a single docker image for multiple users is achievable through proper use of volumes, user context, and a well-defined container entrypoint. The chosen approach needs to align with your specific requirements and constraints. I've found that starting simple, like example 1, then building on that with environment variables or compose, like examples 2 and 3, allows for better management and extensibility of your container deployments as the requirements evolve. Always prioritize security and thorough testing.
