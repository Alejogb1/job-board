---
title: "Why doesn't Docker+Airflow+Nginx reverse proxy work?"
date: "2024-12-23"
id: "why-doesnt-dockerairflownginx-reverse-proxy-work"
---

Okay, let's talk about why setting up a Dockerized Airflow environment behind an Nginx reverse proxy can sometimes feel like chasing a ghost. I’ve spent my share of late nights debugging exactly this scenario, and it usually boils down to a few key areas that are easy to overlook. It’s rarely a single, catastrophic error, but rather a confluence of configuration mishaps. Let's break it down.

First off, understand that this setup introduces complexity on several layers: the docker network, airflow's internal webserver, and the nginx proxy. Each one of these requires careful configuration, and when they don’t play nicely together, you'll likely end up with some variation of "connection refused" or "502 Bad Gateway" errors. It's not a fundamentally broken approach, far from it, but it requires meticulous attention to detail.

The core issue often isn’t with any *one* component being faulty, but rather, miscommunication between them, specifically around network reachability and URL handling. When we set up Airflow in docker, it's typically configured to bind to an internal network port inside its container, typically 8080. Now, when Nginx enters the picture, we want it to act as a gateway to this port. The problem arises when Nginx can’t reach the internal Airflow port, or when Airflow isn’t aware of the external URL it's supposed to be accessed from.

Let’s delve into the common culprits I’ve encountered and what I've done to address them.

**1. Docker Networking and Reachability Issues:**

The first place to check is your Docker network setup. If your Nginx container and Airflow container are not on the same network, they won't be able to talk to each other directly. Usually, I've found that using a user-defined bridge network is the most reliable approach. Default bridge networks can have unpredictable behavior, especially when you have more than a few containers.

Consider this scenario: you’ve spun up an Nginx container named `nginx-proxy` and an Airflow container named `airflow-app`, and they're on the default bridge network. Nginx is trying to connect to `localhost:8080`, which will never resolve to the airflow container’s internal address space. Instead, you need to use docker’s service discovery to allow containers to refer to other containers by name (assuming the containers are on the same user-defined network).

Here’s a snippet of how you might define a user-defined bridge network and then attach the containers to it using Docker Compose:

```yaml
version: '3.8'
services:
  airflow-app:
    image: apache/airflow:2.7.3-python3.11 # adjust as necessary
    ports:
      - "8080"
    networks:
      - airflow_net
    environment:
      - AIRFLOW__WEBSERVER__BASE_URL=http://airflow-app:8080
      - AIRFLOW__CORE__EXECUTOR=LocalExecutor
  nginx-proxy:
    image: nginx:latest
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/conf.d/default.conf
    networks:
      - airflow_net
    depends_on:
      - airflow-app

networks:
  airflow_net:
    driver: bridge

```
In this `docker-compose.yml` file, both the `airflow-app` and `nginx-proxy` containers are on the `airflow_net` network. Note how `AIRFLOW__WEBSERVER__BASE_URL` is now set to the internal name `airflow-app:8080`, which is used internally by Airflow to create links and handle redirects (more on this in point 3). With this, inside the `nginx-proxy` container, you can refer to the airflow app container by the service name `airflow-app`.

**2. Nginx Configuration Errors**

Next up, scrutinize your Nginx configuration. A misconfigured proxy_pass directive is a common source of frustration. You need to ensure that Nginx is correctly directing traffic to the correct internal port of your Airflow container. Specifically, be sure that the `proxy_set_header` directives are correctly configured so that the internal headers used by Airflow are not overwritten by the proxy. A minimal Nginx configuration might look something like this (stored in `nginx.conf`):
```nginx
server {
    listen 80;
    server_name localhost;

    location / {
        proxy_pass http://airflow-app:8080; #note: airflow-app resolves via docker service discovery

        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```
The key here is the `proxy_pass` directive: `http://airflow-app:8080`. Because both containers are on the same docker network, Nginx can use the service name of the airflow container to resolve the internal ip and port to route the traffic. The `proxy_set_header` directives are also crucial, as they forward the necessary information to the application behind the proxy. Omitting headers such as `Host` or `X-Forwarded-Proto` can cause Airflow to generate incorrect redirect links.

**3. Airflow Webserver Configuration and Base URL:**

The last piece of the puzzle, and arguably the most critical, is configuring the Airflow webserver. Even if your docker network and nginx configuration are perfect, if airflow is unaware of the external url being used to access it via the proxy, you will still have redirect issues. The `AIRFLOW__WEBSERVER__BASE_URL` setting is vital. If it’s not set or set incorrectly, the webserver might generate links that don't reflect the external URL, resulting in either endless redirects or inaccessible pages. The environment variable `AIRFLOW__WEBSERVER__BASE_URL`, as noted in the compose example, tells airflow what external URL to use for navigation within the UI. If you are exposing Airflow behind a different URL like `/airflow`, you would have to reflect that change here: `AIRFLOW__WEBSERVER__BASE_URL=http://localhost/airflow`.

Furthermore, if you have custom configurations in `airflow.cfg` that are not automatically getting picked up, that can introduce issues. I encountered such a scenario once when I was trying to implement custom authentication using a remote user provider. A seemingly unrelated setting in `airflow.cfg` that controlled the `SECRET_KEY` was not being applied correctly due to it being not properly formatted in the docker environment variable set up. This lead to the authentication mechanism failing silently. After inspecting the `airflow.cfg` that was being actively used by airflow, it became clear that all environment variables are converted to strings, and lists need to be converted from `['item1', 'item2']` to `'item1,item2'` to be correctly understood. This is particularly true when using settings where lists or tuples are expected.

Here’s an example of how to set the base url correctly using an environment variable:
```bash
docker run -d \
  -e AIRFLOW__WEBSERVER__BASE_URL=http://your-domain.com/airflow \
  -p 8080:8080 \
  --name airflow_app \
  apache/airflow:2.7.3-python3.11
```

Remember to adjust `http://your-domain.com/airflow` to match your actual domain and path if necessary.

**Concluding Thoughts and Further Reading:**

The complexities with Docker, Airflow, and Nginx together highlight the necessity for a deep understanding of each component's inner workings. Debugging issues often requires not just fixing the symptoms, but understanding the root cause of how these systems are interconnected. When dealing with reverse proxies, the devil is always in the detail of network and header configurations, and Airflow's URL handling is notoriously sensitive to getting these details exactly correct.

For a deeper understanding of Nginx’s reverse proxy capabilities, I recommend going through the official Nginx documentation. Specifically, focus on the sections regarding `proxy_pass`, `proxy_set_header`, and the different load balancing methods. In order to have a full understanding of how docker networking works, I would direct you to the official docker documentation on creating user defined networks using bridge mode, and docker's documentation on service discovery. For understanding Airflow’s configuration system and the myriad of available configurations options, refer to the Apache Airflow official documentation, especially the sections covering webserver configuration, including the various environment variables. Understanding how to interpret `airflow.cfg` and how environment variables interact with the configuration will prove very helpful. This should provide a solid foundation for troubleshooting any issues in your own setup.
