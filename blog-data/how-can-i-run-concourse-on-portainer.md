---
title: "How can I run Concourse on Portainer?"
date: "2024-12-23"
id: "how-can-i-run-concourse-on-portainer"
---

Alright,  It’s a scenario I've personally encountered a couple of times while setting up CI/CD pipelines in resource-constrained environments. You’re aiming to get Concourse, a powerful CI/CD system, running within Portainer, a container management platform. It’s a practical need, especially when you’re aiming for a more contained or managed Docker environment.

First, let’s clarify what we're dealing with. Portainer isn't an orchestrator itself like Kubernetes, so we won’t be using its features to schedule and scale Concourse like we would in a Kubernetes setup. Instead, we’re leveraging Portainer to manage the lifecycle of the containers needed for Concourse: the web node, the worker(s), and optionally, any supporting databases. Portainer provides the graphical interface to manage them as individual docker containers, which is exactly how we'll set them up. This will mean that the Concourse architecture itself remains standard, just that Portainer helps manage its containers.

The core challenge lies in configuration and networking. Concourse requires certain environment variables and specific network configurations to allow its components to talk to each other. Additionally, since we're using Portainer, the persistent storage configuration needs to be managed using named volumes to survive container restarts.

Here's the breakdown of my approach, along with some things I've learned through prior setups:

**Step 1: Defining the Concourse Environment**

Before jumping into Portainer, I like to sketch out a `docker-compose.yml` file. This helps me organize my thoughts, and it gives us an easily portable configuration if needed. Although we won't directly deploy this using docker-compose inside portainer, it's useful to have as a reference. We'll be replicating the configuration inside portainer instead.

```yaml
version: '3.7'
services:
  concourse-web:
    image: concourse/concourse:latest
    ports:
      - "8080:8080"
    environment:
      - CONCOURSE_EXTERNAL_URL=http://localhost:8080
      - CONCOURSE_BASIC_AUTH_USERNAME=user
      - CONCOURSE_BASIC_AUTH_PASSWORD=password
      - CONCOURSE_POSTGRES_USER=concourse
      - CONCOURSE_POSTGRES_PASSWORD=concourse
      - CONCOURSE_POSTGRES_DATABASE=concourse
    depends_on:
      - concourse-db
    volumes:
      - concourse-web-data:/concourse-web-data
  concourse-worker:
    image: concourse/concourse:latest
    depends_on:
      - concourse-web
    environment:
        - CONCOURSE_WORK_DIR=/concourse-work-dir
        - CONCOURSE_EXTERNAL_URL=http://concourse-web:8080
        - CONCOURSE_POSTGRES_USER=concourse
        - CONCOURSE_POSTGRES_PASSWORD=concourse
        - CONCOURSE_POSTGRES_DATABASE=concourse
    volumes:
      - concourse-worker-data:/concourse-work-dir
  concourse-db:
    image: postgres:15
    environment:
      - POSTGRES_USER=concourse
      - POSTGRES_PASSWORD=concourse
      - POSTGRES_DB=concourse
    volumes:
        - concourse-db-data:/var/lib/postgresql/data

volumes:
    concourse-web-data:
    concourse-worker-data:
    concourse-db-data:
```

Note the essential environmental variables, especially `CONCOURSE_EXTERNAL_URL`, `CONCOURSE_BASIC_AUTH_USERNAME`, and `CONCOURSE_BASIC_AUTH_PASSWORD`. You’ll also see the PostgreSQL settings. These configurations are very crucial; getting these wrong will lead to Concourse misbehaving. For a production environment, I highly recommend exploring the official Concourse documentation regarding securing the database and the Concourse web UI. Also, consider more sophisticated authentication methods for the web UI.

**Step 2: Setting Up Concourse in Portainer**

Now we’ll implement this in Portainer. I usually do this step-by-step, starting with the database:

1.  **Database (PostgreSQL):** Create a new container in Portainer using the `postgres:15` image. Under the 'volumes' tab, create a named volume called `concourse-db-data`, and mount it at `/var/lib/postgresql/data` inside the container. Go to the 'env' tab, and configure environment variables:
    *   `POSTGRES_USER`: set to `concourse`
    *   `POSTGRES_PASSWORD`: set to `concourse`
    *   `POSTGRES_DB`: set to `concourse`
    Then, deploy the container. Make sure the container is healthy after launching it.

2.  **Web Node:** Create another container using `concourse/concourse:latest` as the image. Under the 'ports' section, map host port `8080` to container port `8080`.
    *   Create a new volume `concourse-web-data` and mount it at `/concourse-web-data`.
    *   Under the 'env' tab, set these environment variables as shown in the `docker-compose.yaml` above:
        *   `CONCOURSE_EXTERNAL_URL`: `http://your-portainer-host-ip:8080` (replace with your actual host IP. In the example here I'll be using localhost)
        *   `CONCOURSE_BASIC_AUTH_USERNAME`: `user` (or your desired username)
        *   `CONCOURSE_BASIC_AUTH_PASSWORD`: `password` (or your desired password)
        *   `CONCOURSE_POSTGRES_USER`: `concourse`
        *   `CONCOURSE_POSTGRES_PASSWORD`: `concourse`
        *   `CONCOURSE_POSTGRES_DATABASE`: `concourse`
    Deploy the container. Note that this container uses the same environment variables that were used in the `docker-compose.yml` file.

3.  **Worker Node(s):** Now for the worker(s). Create one or more containers, also using `concourse/concourse:latest`.
    *   Create a volume `concourse-worker-data` and mount it at `/concourse-work-dir`.
    *   Under the 'env' tab, configure:
        *   `CONCOURSE_WORK_DIR`: `/concourse-work-dir`
        *   `CONCOURSE_EXTERNAL_URL`: `http://concourse-web:8080` (or the name of your web container in Portainer if you used a network configuration)
        *    `CONCOURSE_POSTGRES_USER`: `concourse`
        *   `CONCOURSE_POSTGRES_PASSWORD`: `concourse`
        *   `CONCOURSE_POSTGRES_DATABASE`: `concourse`
    Deploy the worker container. Note how we point the worker to the web container using the same `CONCOURSE_EXTERNAL_URL`.

**Step 3: Verification and Troubleshooting**

After deploying the containers, access the Concourse web UI via `http://your-portainer-host-ip:8080` (or `http://localhost:8080` in my example if you're testing locally) using the credentials you set. If everything is functioning correctly, you should see the Concourse dashboard.

Troubleshooting is a crucial step here. A few things I always check:

*   **Container Logs:** The first place to look for any issue is the logs of individual containers. Portainer allows you to view them directly. Look for database connection errors or issues with registering workers.
*   **Port Mapping:** Ensure the web UI port mapping is correct. If your host port mapping was messed up, you will not be able to access the UI.
*   **Environment Variables:** Triple-check your environment variables. Small typos are frequently the culprit.
*   **Volume Issues:** Check if volumes are correctly mounted. If persistent data is not saved, you might experience issues when restarting.

**Example Code Snippets**

Here are some snippets to illustrate my points further, but these are not runnable as such; rather, they are conceptualized examples of the Portainer configuration and container creation steps I’ve described above. The following code shows the equivalent actions described, but using a docker client instead of the web ui:

**Example 1: Creating the Web Container in Docker (equivalent Portainer Actions)**

```bash
docker run -d \
  --name concourse-web \
  -p 8080:8080 \
  -v concourse-web-data:/concourse-web-data \
  -e CONCOURSE_EXTERNAL_URL=http://localhost:8080 \
  -e CONCOURSE_BASIC_AUTH_USERNAME=user \
  -e CONCOURSE_BASIC_AUTH_PASSWORD=password \
  -e CONCOURSE_POSTGRES_USER=concourse \
  -e CONCOURSE_POSTGRES_PASSWORD=concourse \
  -e CONCOURSE_POSTGRES_DATABASE=concourse \
  concourse/concourse:latest
```

This shows the environment variables and port and volume mappings. The `docker run` command is directly equivalent to what you'd be setting up through the Portainer UI.

**Example 2: Creating the Worker Container in Docker (equivalent Portainer Actions)**

```bash
docker run -d \
  --name concourse-worker \
  -v concourse-worker-data:/concourse-work-dir \
  -e CONCOURSE_WORK_DIR=/concourse-work-dir \
  -e CONCOURSE_EXTERNAL_URL=http://concourse-web:8080 \
  -e CONCOURSE_POSTGRES_USER=concourse \
  -e CONCOURSE_POSTGRES_PASSWORD=concourse \
  -e CONCOURSE_POSTGRES_DATABASE=concourse \
  concourse/concourse:latest
```

Again, this translates directly to the settings you’d input in Portainer when creating a container. The `CONCOURSE_EXTERNAL_URL` is essential, pointing the worker back to the web UI container.

**Example 3: Creating the database container in Docker (equivalent Portainer Actions)**

```bash
docker run -d \
  --name concourse-db \
  -v concourse-db-data:/var/lib/postgresql/data \
  -e POSTGRES_USER=concourse \
  -e POSTGRES_PASSWORD=concourse \
  -e POSTGRES_DB=concourse \
  postgres:15
```

This shows how you would define the container using the `docker run` command line, and the way the volumes are mapped is important.

**Recommended Resources**

To delve deeper into these topics, I would suggest a few authoritative resources:

*   **The official Concourse documentation:** This is *the* primary source for everything related to Concourse. Look at the 'Deployment' section specifically.
*   **Docker’s official documentation:** Understanding Docker concepts is crucial. Especially container networking and volume management.
*   **'The Docker Book' by James Turnbull:** A very comprehensive book on Docker technology.
*   **'Container Security' by Liz Rice:** If security is a concern, this book is invaluable.

In my experience, this approach allows for a reliable Concourse setup within Portainer. Remember, meticulously check your configurations. A well-defined setup is half the battle won. Let me know if you have further specific questions; I’m happy to share more of what I’ve learned.
