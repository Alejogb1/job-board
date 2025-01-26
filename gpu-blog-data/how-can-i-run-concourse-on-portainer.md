---
title: "How can I run Concourse on Portainer?"
date: "2025-01-26"
id: "how-can-i-run-concourse-on-portainer"
---

Concourse, a CI/CD system, and Portainer, a container management platform, can be effectively integrated to streamline deployment and management. The core challenge lies in orchestrating Concourse as a set of Docker containers within Portainer's environment, leveraging Portainer's visual interface for container lifecycle operations. Based on my experience migrating various internal build pipelines, this process demands careful configuration of Concourse's components and thoughtful resource allocation.

First, understand that Concourse operates on a multi-container model. It requires at minimum a `web` server, a `worker` agent, and often utilizes a PostgreSQL database for persistence. We'll configure these containers as distinct services within Portainer. This separation allows for individual scaling and easier maintenance. The critical part is ensuring each component is interconnected correctly.

Let's start with the database, a PostgreSQL instance. This will act as the persistent store for Concourse configurations and pipeline history. I typically prefer a dedicated database container rather than using a database running on the host system. This facilitates environment portability. Using Portainer's “Stacks” feature simplifies managing this group of containers. Below is a Docker Compose file, which can be directly used by Portainer’s Stacks:

```yaml
version: '3.8'
services:
  concourse-db:
    image: postgres:13
    environment:
      POSTGRES_USER: concourse
      POSTGRES_PASSWORD: your_strong_password
    volumes:
      - concourse-db-data:/var/lib/postgresql/data
    networks:
      - concourse-net

volumes:
  concourse-db-data:

networks:
  concourse-net:
    driver: bridge
```

Here, `concourse-db` defines the PostgreSQL container. The `image` tag specifies the PostgreSQL version. Environment variables set the database user and password. Replace `your_strong_password` with a secure password. A named volume, `concourse-db-data`, is used to persist database information across restarts. The container is placed on a dedicated network called `concourse-net` to ensure connectivity isolation. The database container is fundamental; without a properly configured database, the other Concourse components will not operate correctly.

Next, we'll configure the Concourse web server. It provides the user interface and serves as the central point of access for Concourse pipelines. Crucially, the web server requires an established database connection and a worker connection. Consider this Docker Compose fragment for the Concourse web component:

```yaml
  concourse-web:
    image: concourse/concourse:latest
    depends_on:
      - concourse-db
    environment:
      CONCOURSE_POSTGRES_DATABASE: concourse
      CONCOURSE_POSTGRES_USER: concourse
      CONCOURSE_POSTGRES_PASSWORD: your_strong_password
      CONCOURSE_POSTGRES_HOST: concourse-db
      CONCOURSE_BASIC_AUTH_USERNAME: admin
      CONCOURSE_BASIC_AUTH_PASSWORD: your_admin_password
      CONCOURSE_EXTERNAL_URL: http://your-public-ip:8080
      CONCOURSE_ADD_LOCAL_USER: admin:your_admin_password
    ports:
      - "8080:8080"
    networks:
      - concourse-net
    restart: unless-stopped

```

The `concourse-web` service uses the `concourse/concourse:latest` image. The `depends_on` directive instructs Docker Compose to start the database container before the web container. Note the environment variables; these configure the database connection, using the database hostname defined in our network and the login credentials. Replace both instance of `your_strong_password` and `your_admin_password` with secure values. `CONCOURSE_EXTERNAL_URL` needs to match the external URL users will use to access your Concourse instance, which we have set to your server's IP on port 8080. This is important because if the address is not reachable the interface will be partially unavailable. We expose port 8080 from the container to the host for direct access. Additionally the `restart: unless-stopped` will automatically restart the container if it fails unless explicitly stopped.

Finally, we configure the Concourse worker. Workers execute tasks in pipelines. Multiple workers can run concurrently to handle multiple build jobs. The configuration of a single worker is detailed below:

```yaml
  concourse-worker:
    image: concourse/concourse:latest
    depends_on:
      - concourse-web
    environment:
      CONCOURSE_WORK_DIR: /opt/concourse/worker
      CONCOURSE_TSA_HOST: concourse-web
      CONCOURSE_TSA_PORT: 2222
    volumes:
      - concourse-worker-work:/opt/concourse/worker
    networks:
      - concourse-net
    restart: unless-stopped

volumes:
  concourse-worker-work:
```

Here, `concourse-worker` also utilizes the `concourse/concourse:latest` image.  It depends on `concourse-web` to ensure the web server is available for worker registration. `CONCOURSE_WORK_DIR` defines the worker’s local directory for storing task data.  `CONCOURSE_TSA_HOST` is set to `concourse-web`, to connect to the web component and `CONCOURSE_TSA_PORT` is set to the port the webserver will listen to for worker connections (this is the standard Concourse worker registration port). The `concourse-worker-work` volume persists the working directories across container restarts.

Once these three services are defined within your `docker-compose.yml` file you can then upload this file to Portainer via the stacks tab. This approach leverages Docker Compose to orchestrate multiple containers with minimal complexity. Be sure the network used in the docker compose is consistent across all the services to enable container communication. The critical aspect to note is that the containers must be properly networked in order for worker registration to occur successfully.

After deploying the stack in Portainer, the Concourse web interface should be accessible through the configured `CONCOURSE_EXTERNAL_URL`. You can log in with the basic authentication credentials previously set up via environment variables.  From there, you can begin configuring pipelines.

Regarding resource allocation within Portainer, I suggest monitoring resource utilization of each container individually after deploying.  The PostgreSQL instance, especially in an environment with significant Concourse use, may require sufficient memory and disk space for sustained operation. I’ve found that initial allocation of 2GB of RAM for the database and web server containers and 1 GB for each worker is adequate for small to medium-sized deployments. You can then scale resource allocation up or down within Portainer based on actual usage to prevent bottlenecks.

When managing container updates, avoid blindly updating to the latest images, especially in production. When a new version becomes available, test the new version in a non-production environment by deploying a second stack to an isolated network with an adjusted exposed port. Observe its behaviour before upgrading your production instance. Upgrades can break compatibility, especially in Concourse.

For learning more about containerized systems and CI/CD pipelines I recommend exploring documentation resources such as Docker's official documentation. In addition, the official Concourse documentation will be essential for understanding specific Concourse concepts. Similarly the documentation provided by Portainer can provide guidance on managing deployed stacks and containers. These will help you to better manage your system and resolve common issues. By approaching the combination of Concourse and Portainer thoughtfully and by applying robust configuration strategies, you will find it can enhance your CI/CD workflow effectively.
