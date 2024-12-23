---
title: "How can I use Portainer with docker-compose env variables?"
date: "2024-12-23"
id: "how-can-i-use-portainer-with-docker-compose-env-variables"
---

Let's tackle this one. Setting up Portainer to properly consume environment variables defined in your `docker-compose.yml` file, well, it's a scenario I've encountered more times than I care to recount, particularly back when I was leading the migration of our legacy services to containers. It’s a common stumbling block, but it’s a solvable one. The crux of the matter lies in understanding how Portainer interacts with Docker and Docker Compose, and where it expects to find these variables. The issue isn't that Portainer can't use them; it's more about how we need to explicitly tell it where to look.

Essentially, Portainer directly interacts with the Docker API, which in turn utilizes the Docker Compose configurations. However, Portainer doesn’t magically absorb environment variables defined solely within your `docker-compose.yml` file. Instead, these variables need to be either explicitly made available to the Docker environment or provided as part of the Portainer deployment. Let's break down the common methods and how to get them working.

The first, and often the most direct, approach is to use the `env_file` directive within your `docker-compose.yml`. This lets you define all your environment variables in a separate file, which can then be referenced in the compose file. This method works because when Docker Compose processes this, it injects the variables defined in the env file into the container's runtime environment. Portainer then interacts with the Docker API and therefore the container with these populated environment variables.

Here’s a simplified example:

First, create a `.env` file in the same directory as your `docker-compose.yml`:

```
# .env
DB_USER=my_db_user
DB_PASSWORD=my_secret_password
API_KEY=super_secret_api_key
```

Then modify your `docker-compose.yml`:

```yaml
version: '3.8'
services:
  my_app:
    image: my-application-image:latest
    ports:
      - "8080:8080"
    env_file:
      - ./.env
```

In this scenario, the `my_app` service will automatically have access to the `DB_USER`, `DB_PASSWORD`, and `API_KEY` variables when the container starts. Portainer, because it interacts with the Docker engine, will see these environment variables as part of the running container. This works because Docker Compose reads the `.env` file and injects the variables into the container’s environment, where Portainer then sees them.

A second, sometimes useful approach when working with variables specific to individual containers is to embed them directly within the `docker-compose.yml` using the `environment` directive. This works, and I've employed it when I needed a quick, less file-dependent approach, but it does come at a cost of potentially cluttering the compose file.

For example:

```yaml
version: '3.8'
services:
  my_app:
    image: my-application-image:latest
    ports:
      - "8080:8080"
    environment:
      DB_USER: my_db_user
      DB_PASSWORD: my_secret_password
      API_KEY: super_secret_api_key
```

Similar to the previous example, Portainer will display these variables as part of the container's environment, as they're defined within the service definition. Again, this works because the Docker engine passes these variables as part of the container creation process.

A third technique, which I’ve used extensively when dealing with environment variables that need to be accessible across multiple containers or even across different docker-compose setups is to configure variables as part of the host environment itself before starting the containers. This means setting them in your shell, or, if you're managing things programmatically, through other configuration mechanisms on your system or orchestration tool. When Docker is started from such an environment, it inherits those variables, effectively making them globally available to any containers you launch, assuming they are explicitly referenced in the `docker-compose.yml`.

In the `docker-compose.yml` this means referring to the host defined variables using the dollar sign syntax:

```yaml
version: '3.8'
services:
  my_app:
    image: my-application-image:latest
    ports:
      - "8080:8080"
    environment:
      DB_USER: ${DB_USER}
      DB_PASSWORD: ${DB_PASSWORD}
      API_KEY: ${API_KEY}
```

To make this work, you must define those variables in your shell before running `docker-compose up`:

```bash
export DB_USER=my_db_user
export DB_PASSWORD=my_secret_password
export API_KEY=super_secret_api_key
docker-compose up -d
```
By exporting these variables first, any service using the `environment` key with `${VARIABLE_NAME}` references will inherit those values at container startup. Portainer will correctly display these variables because, again, the variables are available within the docker runtime environment.

This last approach is crucial when managing secrets outside your source control or for sensitive information you don’t want directly in your Dockerfiles or compose files.

Regarding resources for more in-depth learning, I'd recommend diving into the official Docker documentation first. The sections covering `docker-compose` are essential, especially those detailing the `env_file` and `environment` directives. Another excellent resource is “Docker in Action” by Jeff Nickoloff, which provides a practical guide to working with Docker, including various environment configuration strategies. Also, exploring some of the official Docker blog posts and articles around security and environment management can be highly informative. For a deep dive on how Portainer works internally, and to properly understand how to use their API it is useful to refer to the official Portainer documentation.

In summary, Portainer doesn’t directly parse or consume environment variables from your `docker-compose.yml` without some explicit configuration, or without the variables having been injected into the docker runtime environment, whether via a .env file, embedded in the compose file, or inherited from the host environment. The key is to make these variables available to the Docker environment, which Portainer then interacts with. By using the techniques I’ve described, you should find a reliable way to use your Docker Compose defined environment variables seamlessly with Portainer.
