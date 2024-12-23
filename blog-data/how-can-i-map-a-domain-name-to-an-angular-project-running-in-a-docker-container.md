---
title: "How can I map a domain name to an Angular project running in a Docker container?"
date: "2024-12-23"
id: "how-can-i-map-a-domain-name-to-an-angular-project-running-in-a-docker-container"
---

Okay, let's tackle this. I've seen this scenario play out more times than I can count, and it always boils down to a few key components that need to play nice together. Mapping a domain name to an Angular application running inside a Docker container isn’t particularly complex, but it requires a good understanding of networking, web servers, and containerization. I remember back when I was first wrestling with this on a cloud-based application we were building, it felt like a magic trick. Over time, though, it’s just become a solid workflow.

Essentially, what we're aiming for is to route HTTP or HTTPS traffic coming into a server (or more specifically, to the public ip of the server) with a specific domain name to the particular Docker container that holds your angular application. There are a couple of ways to accomplish this, and the specific solution will depend on the environment you're deploying in, but the core principle is consistent: we need a reverse proxy.

First, let's discuss the layers at play. You've got your domain name, which resolves to a server's IP address via DNS. You then have the webserver, which typically runs on the host operating system, not within your docker container. This could be nginx, apache or others. We use this webserver as a *reverse proxy*, it forwards requests to specific containers based on the hostname (domain). Lastly, there's your Docker container housing the angular application. The application within the container is typically exposed on a particular port (e.g., 80 or 8080 within the container) which is then made available to the host (and thus the reverse proxy) via docker port mapping.

The most common and arguably most straightforward method is using `nginx` as the reverse proxy. `nginx` is a powerful, lightweight, and highly configurable web server that is perfect for this job. Here's how we generally set it up:

```nginx
# /etc/nginx/sites-available/my-angular-app.conf

server {
    listen 80; # Or 443 for HTTPS, config needed.
    server_name mydomain.com www.mydomain.com;

    location / {
        proxy_pass http://localhost:3000; # assuming container's port is mapped to 3000 on the host
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }
}
```

This configuration snippet, typically stored as a file like `/etc/nginx/sites-available/my-angular-app.conf`, directs all traffic arriving at `mydomain.com` or `www.mydomain.com` on port 80 to `http://localhost:3000`. Critically, the `proxy_set_header` directives are important for web sockets to function correctly if your angular application uses them. The `localhost:3000` implies that you've mapped a port within your Docker container to port 3000 on the host when starting the container.

Now, let's talk about the Docker container itself. Here's a basic example `docker-compose.yml` file that demonstrates this:

```yaml
# docker-compose.yml
version: '3.8'
services:
  angular-app:
    image: your-angular-image:latest  # your docker image
    ports:
      - "3000:80"  # maps host port 3000 to container port 80
    restart: always
```

In this `docker-compose.yml` example, we're specifying that the docker image named `your-angular-image:latest` (you'd replace this with the actual name of your angular docker image), expose port 80 within the container and map it to port 3000 on the host. The `restart: always` directive is useful to ensure that if the container crashes for some reason, docker will automatically attempt to restart it.

So, you launch the docker container using `docker-compose up -d` and ensure that the nginx config for your domain is enabled. Then your domain, `mydomain.com`, should point to your application in the docker container.

However, in more complex scenarios, especially with multiple applications on the same server, you might find yourself using a more robust reverse proxy setup, like traefik. Traefik is a popular cloud-native edge router, it can automatically detect services, and their network configuration (including docker containers and networks), making configurations more dynamic and less reliant on manual updates.

Here's how a traefik configuration using labels might look like:

```yaml
# docker-compose.yml for Traefik and your application
version: "3.8"
services:
  traefik:
    image: "traefik:v2.10"
    ports:
      - "80:80"
      - "443:443"
      - "8080:8080" # dashboard
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock:ro
      - ./traefik.yaml:/traefik.yaml
    command: --api.insecure=true --providers.docker=true --entrypoints.web.address=:80 --entrypoints.websecure.address=:443
  angular-app:
    image: your-angular-image:latest
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.my-angular-app.rule=Host(`mydomain.com`)"
      - "traefik.http.routers.my-angular-app.entrypoints=web"
      - "traefik.http.routers.my-angular-app.service=my-angular-app-service"
      - "traefik.http.services.my-angular-app-service.loadbalancer.server.port=80"
    restart: always

```

Here, the `docker-compose.yml` now includes traefik as a service, listening on standard http and https ports (80 and 443) and also exposing the traefik dashboard on port 8080. Most of the configuration of the proxying is done via *docker labels*.  In the `angular-app` service definition, we tell traefik that it should be proxying traffic destined for the host `mydomain.com` to this service. The service, named `my-angular-app-service` will then forward traffic to the container's port 80. The `traefik.yaml` file configures default settings for traefik, and it isn't necessary here if you are happy with the defaults, it can be omitted. This method uses container labels to dynamically configure the proxy on the fly.

Important things to consider:

* **DNS Records**: Make sure the A record for your domain (`mydomain.com`) points to the public IP address of your server.
* **HTTPS**: If you need HTTPS, you’ll need to configure your reverse proxy to handle SSL/TLS certificates. Certbot (with let’s encrypt) is a great option for automatically obtaining free certificates.
* **Firewalls**: Ensure that your server's firewall is configured to allow traffic on the ports you are using (80 and 443, or custom if used).
* **Docker Network:** By default, docker compose will create a bridge network and all the containers on that docker-compose file will be connected to this same network, which in most basic cases, is sufficient. If your services need to be on a separate docker network, the traefik and angular application container would need to be on the same docker network.
* **Application Port:** Be sure that your angular application is built to handle being served from the root URL (`/`). Configuration of the base URL is typically done within `angular.json` or programmatically using `APP_BASE_HREF` provider if the base url will be dynamically determined at runtime.

For a deeper understanding, I would recommend checking out *nginx's documentation* for its proxy module. Also, the official *Docker documentation* is invaluable, and reading the *Traefik documentation* will get you up to speed on more sophisticated cloud deployment scenarios.

This approach allows you to map multiple domain names to different applications all running on the same server by using various reverse proxy configurations, which is crucial as you scale your infrastructure. Remember, the key is to understand the flow of the request and how each component plays its role, from the user's browser to your application within the container.
