---
title: "How do I use environment variables from docker-compose in Portainer?"
date: "2024-12-23"
id: "how-do-i-use-environment-variables-from-docker-compose-in-portainer"
---

Right, let's tackle this. It’s a common scenario, and one that I've encountered more than a few times in my days managing containerized applications. Getting environment variables from your `docker-compose.yml` into Portainer isn't as direct as you might initially think, but it's absolutely achievable and crucial for a dynamic setup. The core challenge is that docker-compose files aren't directly parsed by Portainer when deploying stacks. Portainer interprets the compose file itself, but the shell context from where `docker-compose up` runs is not inherently inherited.

My past projects, often involving complex microservices architectures, have forced me to refine my approach to this. Early on, we leaned too heavily on hard-coded configurations in our Portainer stack files, which, as you can guess, became a maintenance nightmare. We needed a solution that allowed us to leverage the environment variables defined in our docker-compose development process also in our more production-focused Portainer deployments. So, the goal is not to *use* the docker-compose file directly in portainer for this purpose, but to ensure the environment variables defined *in* our docker-compose file are *available* in our portainer deployments.

The most direct method involves explicitly defining environment variables in the Portainer stack deployment process, using variables that *mimic* what's in your compose file. There are two practical ways to implement this:

1.  **Explicitly Defining Variables in the Portainer Stack File:** This is essentially re-declaring your variables in the Portainer stack. It can feel repetitive, but it's straightforward and offers full control over which variables get passed and their values. The Portainer "stack file" (its equivalent of the docker-compose file) has a dedicated `environment` section for this purpose.

2.  **Using the `.env` file:** You can use a `.env` file which is an alternative to defining variables in the portainer stack file. Your docker-compose setup might already be using one, and the best practice would be to make this the source of truth for the environment variables that your docker containers are relying on.

Let me illustrate with a working code snippets.

**Snippet 1: Explicitly Defining Variables**

Suppose your `docker-compose.yml` looks something like this:

```yaml
version: "3.8"
services:
  web:
    image: nginx:latest
    ports:
      - "8080:80"
    environment:
      - VIRTUAL_HOST=my-app.example.com
      - HTTP_PORT=80
      - DATABASE_URL=postgresql://user:password@db:5432/mydb
    depends_on:
      - db
  db:
    image: postgres:latest
    environment:
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=password
      - POSTGRES_DB=mydb
```

In Portainer, when you create a stack, you'd translate the `environment` section into the Portainer deployment like this (this would be a simplified equivalent to a Portainer-ready stack deployment json):

```yaml
version: "3.8"
services:
  web:
    image: nginx:latest
    ports:
      - "8080:80"
    environment:
      VIRTUAL_HOST: my-app.example.com
      HTTP_PORT: 80
      DATABASE_URL: postgresql://user:password@db:5432/mydb
    depends_on:
      - db
  db:
    image: postgres:latest
    environment:
      POSTGRES_USER: user
      POSTGRES_PASSWORD: password
      POSTGRES_DB: mydb
```

This is very simple. We took the values from the docker-compose and included them in a portainer stack equivalent.

**Snippet 2: Using an .env File**

Often, you'll have an `.env` file for your compose setup:

```
VIRTUAL_HOST=my-app.example.com
HTTP_PORT=80
DATABASE_URL=postgresql://user:password@db:5432/mydb
POSTGRES_USER=user
POSTGRES_PASSWORD=password
POSTGRES_DB=mydb
```

Then, your `docker-compose.yml` would reference it like this:

```yaml
version: "3.8"
services:
  web:
    image: nginx:latest
    ports:
      - "8080:80"
    environment:
      - VIRTUAL_HOST
      - HTTP_PORT
      - DATABASE_URL
    depends_on:
      - db
  db:
    image: postgres:latest
    environment:
      - POSTGRES_USER
      - POSTGRES_PASSWORD
      - POSTGRES_DB
```

In Portainer, you can upload this `.env` file during stack creation and Portainer will use the variables defined in the `.env` file when it starts the containers. However, this approach requires the user to select the `.env` file upon stack creation, which is not ideal for a completely automated setup.

**Snippet 3: Combining .env file with Portainer UI Variable Overrides**

One way to make your .env variables accessible to your Portainer stacks, while also retaining some control at deploy-time is to use the Portainer UI. This involves two steps:

1.  Create a `.env` file with your variables
2.  Create a stack from your compose file, but do not pass the .env file (do not check "use .env file" box). The environment variables from the `docker-compose.yml` are then defined *in Portainer* as "stack variables" automatically.
3.  When you deploy the stack, Portainer will use the *names* of the environment variables defined in your compose file, but you can *override the values* with the Portainer UI.
4.  The benefit of this is that you don't have to copy and paste all of the values from your `.env` file into your portainer stack deployment file. You still can edit the variables when you deploy, but without having to search for all the variable names by copying and pasting values.

```
# Example docker-compose.yml
version: '3.8'
services:
  my_service:
    image: your-image:latest
    environment:
      - MY_VARIABLE
      - ANOTHER_VARIABLE
      - DATABASE_URI
    ports:
      - "8080:80"
```

```
# Example .env file
MY_VARIABLE=default_value
ANOTHER_VARIABLE=some_other_default
DATABASE_URI=postgres://user:pass@db:5432/some_db
```

When you deploy this in portainer (without specifying the .env file), the Portainer UI will give you an area to define the "stack variables" where you can override these defaults. This allows for more flexibility than using just the `.env` file or just the compose file.

**Considerations and Recommendations**

*   **Security:** Avoid hardcoding sensitive data (passwords, API keys) directly into environment variables or stack files. Explore using Docker secrets or environment variable management tools to handle secrets more securely. This is especially crucial in a production setting.
*   **Consistency:** Ensure consistency between your `docker-compose.yml`, `.env`, and Portainer configurations. Version control your `.env` files just like any other configuration artifact. While not usually something you'd commit in your main branch (consider a `.env.example` file for version control instead), keeping track of changes to the environment variables used in your local development can reduce deployment issues in Portainer.
*   **Automation:** Automate the process of updating your Portainer stacks. While it is not recommended to directly modify files in production environments, you should automate the *generation* of stack files through an infrastructure-as-code approach. Tools like Terraform or Ansible can help maintain consistent deployment configurations across environments.
*   **Documentation:** Clearly document which environment variables are expected by each service and where they are defined, or if there are any variable overrides that occur when the project is deployed in Portainer.

For further reading, I would recommend diving into the following resources:

*   **"Docker in Action"** by Jeff Nickoloff. It provides excellent insights into working with Docker and docker-compose.
*   **"The Twelve-Factor App"** (a methodology for building scalable, reliable web applications), which strongly recommends using environment variables for configuration.
*   **The official Docker documentation** on `docker-compose`, specifically the parts detailing environment variables and the `.env` file.

In summary, while not directly utilizing the `docker-compose.yml` environment section in Portainer, we can achieve similar outcomes using explicit definitions in Portainer's stack files, `.env` files (being careful with security implications), or a hybrid approach of Portainer stack variables with .env files. Careful planning and consistent practices are essential for avoiding issues down the line.
