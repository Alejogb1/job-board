---
title: "How do I use the environment from a docker-compose file with Portainer?"
date: "2024-12-23"
id: "how-do-i-use-the-environment-from-a-docker-compose-file-with-portainer"
---

Let’s address this directly, shall we? It’s a common scenario: you’ve defined your multi-container application using docker-compose, and now you’re looking to leverage Portainer's management capabilities without losing the environment variables you so carefully configured. I’ve dealt with this quite a few times, particularly back when we were migrating our monolithic application to microservices. The trick is understanding how Portainer interacts with existing docker setups, including docker-compose configurations, and what tools it offers for environment variable management.

The core challenge lies in the fact that docker-compose typically defines environments directly in the `docker-compose.yml` file, either explicitly or via environment files (`.env`). Portainer, on the other hand, has its own interface for container creation and management, which may not immediately recognize the environment definitions from your compose file. This can feel like a disconnect initially, but it's manageable once you understand the mechanics involved.

First off, Portainer doesn't directly import or interpret a `docker-compose.yml` in the way you might hope. It won’t magically create stacks or deploy services directly from it. The primary way to integrate a docker-compose setup within Portainer is through its "stacks" feature. Essentially, you'll be deploying the compose file through a Portainer-managed stack. This offers a method to retain the defined environments.

So, how does the environment get preserved? It relies on how Portainer interprets the compose file during stack creation. When you deploy a stack, Portainer takes your compose file, parses it, and then utilizes the docker API to create the containers and services based on your configurations. Importantly, it honors the environment variables specified, either directly within the `services` section of your docker-compose file or referenced using the `$VARIABLE` syntax together with `.env` files.

Let's walk through how this works with some examples.

**Example 1: Explicit Environment Variables in docker-compose.yml**

Suppose you have a simple web application using a docker-compose file like so:

```yaml
version: "3.8"
services:
  web:
    image: nginx:latest
    ports:
      - "8080:80"
    environment:
      - DEBUG_MODE=true
      - APP_VERSION=1.2.3
```

Here, `DEBUG_MODE` and `APP_VERSION` are directly defined as environment variables for the `web` service.

To deploy this within Portainer:

1.  **Navigate to Stacks:** In the Portainer interface, go to the "Stacks" section.
2.  **Add a Stack:** Click "Add stack".
3.  **Select Method:** Choose "Web editor" as the deployment method (you can also use a git repository or upload a docker-compose.yml file).
4.  **Paste Configuration:** Copy and paste the contents of the above `docker-compose.yml` directly into the web editor.
5.  **Deploy:** Click "Deploy the stack".

Portainer will then create the stack along with the web container, properly injecting the `DEBUG_MODE` and `APP_VERSION` environment variables into the running container. You can verify this by inspecting the container details in Portainer after the stack is deployed; you'll see those environment variables listed under the container's environment configuration.

**Example 2: Using .env files with docker-compose.yml**

Often, you’ll use `.env` files to manage configuration, keeping your docker-compose file cleaner. Consider this `.env` file:

```
DATABASE_URL=postgres://user:password@db:5432/mydb
API_KEY=some_secret_key
```

And the corresponding `docker-compose.yml`

```yaml
version: "3.8"
services:
  backend:
    image: my_backend_image:latest
    ports:
      - "8000:8000"
    env_file:
      - .env
```

Here, `backend` relies on the `.env` file for configuration.

To deploy this using Portainer, you must ensure the `.env` file is available in the same directory where the `docker-compose.yml` file will be used when the Portainer stack is created (for the “web editor” method, you'd ideally use a method other than copy and paste, such as uploading from a local computer or pulling from Git). It won't work if the file is available only on your machine. When using "Web Editor", you can use a workaround by pasting the content of your `.env` file in an environment variable using the `env` section.
Let’s modify our stack:

```yaml
version: "3.8"
services:
  backend:
    image: my_backend_image:latest
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgres://user:password@db:5432/mydb
      - API_KEY=some_secret_key

```

The steps to deploy this through Portainer are exactly as described in Example 1, except the content of the new stack is provided above. Portainer will parse the compose file, and the `backend` container will inherit the `DATABASE_URL` and `API_KEY` variables as intended.

**Example 3: Referencing External Environment Variables**

Sometimes, you may need to pass environment variables from your system into the containers. This is slightly more complex, because Portainer's stacks interface does not directly interpret shell environment variables like `$MY_VAR`, when we paste directly in the web editor. Let’s say you want to pass your current user id.

Your `docker-compose.yml` would look like this:

```yaml
version: "3.8"
services:
  worker:
    image: busybox
    command: sh -c "echo 'User ID: $USER_ID'"
    environment:
      - USER_ID=${USER_ID}
```

The `USER_ID` should be passed in. When using Portainer's stack, if you're using web editor, that value needs to be explicitly present. However, if you use a git repo instead, when you start the stack using the git method, you can pass values directly in the Portainer stack configuration UI.

To do it through web editor, again, you need to modify the stack:

```yaml
version: "3.8"
services:
  worker:
    image: busybox
    command: sh -c "echo 'User ID: $USER_ID'"
    environment:
      - USER_ID=1000
```

The deployment in Portainer will proceed normally, with the `worker` container having `USER_ID` set to `1000`.

It's critical to note that while you can edit environment variables directly within the Portainer UI *after* the stack is deployed, these changes won't persist if you redeploy the stack without updating the compose file. Portainer reads the docker-compose definition each time a stack is updated, so it will override any changes you made directly from the container UI. To have persistent changes, you will always need to update your original compose files or use the `env` section like illustrated above.

For further reading, I would recommend exploring the official docker-compose documentation. Specifically, the sections on environment variables and `.env` files. Also, consult the Portainer documentation related to stacks, as it will cover more advanced features and edge cases. For a general understanding of Docker and its underlying mechanisms, I highly suggest "Docker Deep Dive" by Nigel Poulton. It's a great resource that has helped many of us grasp the core concepts effectively.

In conclusion, while Portainer doesn't directly import docker-compose configurations, leveraging the "stacks" feature provides a method to deploy and manage your docker-compose setups while respecting environment variables. By ensuring that your `docker-compose.yml` and associated `.env` files are correctly used, Portainer will deploy your containers as expected, preserving your carefully crafted configurations. Remember that changes to environment variables should ideally be done in the original files for persistence and that Portainer will always default to the definitions present in the stack’s description, rather than directly reflecting container changes. This has been a common stumbling block during some of our projects and, hopefully, understanding this behaviour helps you avoid similar issues.
