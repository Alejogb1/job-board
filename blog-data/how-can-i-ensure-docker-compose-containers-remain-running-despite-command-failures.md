---
title: "How can I ensure Docker Compose containers remain running despite command failures?"
date: "2024-12-23"
id: "how-can-i-ensure-docker-compose-containers-remain-running-despite-command-failures"
---

Okay, let’s unpack this. I remember a particularly hairy incident a few years back where a seemingly benign update to an internal service kept crashing our whole dev environment because of a poorly handled dependency. Docker Compose, in its default configuration, isn’t particularly resilient to individual container failures. It will happily bring down the entire stack if one service goes belly up. This experience hammered home the importance of robust restart policies, and that's exactly what we need to discuss here.

The core issue is that Docker Compose, by default, treats any container exit as a reason to stop the service, and consequently the whole Compose setup if other services are dependent on the failing one. We need to explicitly tell it *how* to react to these non-zero exit codes, and that’s where `restart` policies come in. They dictate under what circumstances Docker should attempt to restart a container.

Let's first talk about the different options available. The `restart` policy in docker-compose has a few different settings we can utilize: `no`, `always`, `on-failure`, `unless-stopped`. The most basic is `no`, which, as you might guess, doesn't restart the container. We almost never want this. Next, we have `always`, which does exactly that—it tries to restart the container irrespective of its exit code. While this can be useful in very specific cases, it's generally too aggressive and can mask underlying issues. For example, if a container is crashing due to a configuration error, always restarting it would just generate endless restarts and a barrage of log messages.

Then there's `on-failure`, this is a more sensible choice for most use cases. It only restarts the container if it exits with a non-zero exit code, indicating an error. Crucially, `on-failure` can also accept a parameter: `on-failure:5`, for example, will retry a maximum of 5 times and stop afterward, preventing the container from getting stuck in a retry loop. The final option, `unless-stopped`, is similar to `always`, but it only restarts the container if the container hasn't been explicitly stopped manually. This is great for ensuring services are always running automatically even when the host restarts.

To implement this, you'll specify the `restart` policy within your `docker-compose.yml` file, specifically within the service definition. Let's look at a few examples:

**Example 1: Basic `on-failure` Restart Policy**

```yaml
version: "3.8"
services:
  web:
    image: nginx:latest
    ports:
      - "80:80"
    restart: on-failure
  app:
    image: my-app:latest
    restart: on-failure:3 # Retry up to 3 times on failure
    depends_on:
      - web
```

Here, both the `web` service running `nginx` and the `app` service running my custom application are set to restart automatically if they fail. The app container will only try three times before giving up. The key takeaway is that if any of the processes inside these containers exit with a non-zero status code, Docker will automatically try to spin it back up. Note the use of `depends_on` ensures the web service starts before the application, an important point which isn’t strictly related to the question but is often relevant.

**Example 2: Using `unless-stopped` for Persistent Services**

```yaml
version: "3.8"
services:
  database:
    image: postgres:latest
    restart: unless-stopped
    environment:
      POSTGRES_USER: admin
      POSTGRES_PASSWORD: password
      POSTGRES_DB: mydatabase
    volumes:
       - db_data:/var/lib/postgresql/data
volumes:
  db_data:
```

In this scenario, the `database` service using `postgres` is configured with `unless-stopped`. This is highly useful for persistent services such as databases. If the server restarts, this service will automatically restart, and as long as I don’t stop it manually, it will continue running. The volume mount for the data is essential here too, as it maintains the data's persistence across container restarts.

**Example 3: Combined Policies and Health Checks**

```yaml
version: "3.8"
services:
  api:
    image: my-api:latest
    restart: on-failure:5
    depends_on:
      database:
        condition: service_healthy
    healthcheck:
        test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
        interval: 30s
        timeout: 10s
        retries: 3
        start_period: 10s
  database:
    image: postgres:latest
    restart: unless-stopped
    environment:
      POSTGRES_USER: admin
      POSTGRES_PASSWORD: password
      POSTGRES_DB: mydatabase
    volumes:
       - db_data:/var/lib/postgresql/data
volumes:
  db_data:
```

This example introduces an essential addition – health checks. While a service may technically be "running," it might not be in a healthy state, for instance, if the application logic has failed, but the container is still up. The `healthcheck` block allows Docker to periodically check the status of the application running inside the container. The api container now restarts on failure up to 5 times and also depends on the `database` being healthy. If the health check fails, Docker will consider the service unhealthy. This is a crucial mechanism in conjunction with restart policies to ensure true application reliability. The combination of a `healthcheck` and a restart policy is a much more robust approach than just relying on `on-failure`. The database, meanwhile, still uses `unless-stopped` for maximum uptime and the `condition: service_healthy` ensures it's available before the API begins its work.

To further deepen your knowledge in this area, I highly recommend diving into *The Docker Book: Containerization is the New Virtualization* by James Turnbull. It offers a detailed overview of Docker fundamentals. In addition, for specific deep dives into Docker Compose, the official Docker documentation is invaluable and frequently updated. Specifically, you should review the section on `restart policies` and health checks in the compose specification. You will also benefit from researching advanced topics in container orchestration, such as Kubernetes, which addresses these issues at scale. The official Kubernetes documentation and books such as *Kubernetes in Action* by Marko Lukša are valuable resources.

Implementing restart policies isn’t about blindly restarting containers— it's about using a combination of these strategies along with health checks, to ensure our systems are resilient to failures, and as an experienced developer, I've found these techniques to be absolutely vital in creating truly robust and reliable applications, especially when working with distributed micro-services. They’ve saved me (and my team) from countless headaches over the years.
