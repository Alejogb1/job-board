---
title: "Why is Laravel Sail, nginx, and a reverse proxy rendering localhost:8002 instead of site.xyz?"
date: "2025-01-30"
id: "why-is-laravel-sail-nginx-and-a-reverse"
---
The root cause of Laravel Sail, nginx, and a reverse proxy serving `localhost:8002` instead of `site.xyz` almost invariably stems from an incorrect configuration within the reverse proxy or a mismatch between the proxy's expected upstream and Sail's actual exposed port.  I've encountered this numerous times during development and deployment, primarily due to overlooking crucial details in the proxy's configuration file.

**1. Clear Explanation:**

Laravel Sail, by default, exposes the application's port 8000 within its Docker container.  This internal port is then mapped to a port on the host machine (often 8000, but configurable).  However,  serving a site externally necessitates a reverse proxy like nginx, which acts as an intermediary between the client (browser) and the application.  The proxy receives requests on port 80 (or 443 for HTTPS) and forwards them to the upstream server (Sail application) on the designated port.  Seeing `localhost:8002` implies the reverse proxy is either not properly configured to forward requests to the correct upstream, or the application is somehow exposing port 8002 instead of the expected port.  A common oversight is failing to specify the correct upstream address and port in the nginx configuration.  Another is assuming that simply mapping the port on the host machine will automatically make the application accessible externally without the proxy correctly routing requests.  Further, the reverse proxy might be misconfigured to handle different virtual hosts or domain names, resulting in the default behavior of exhibiting the upstream’s address.  Finally, a conflict in port mappings, especially within other Docker containers running simultaneously, can lead to this undesirable outcome.


**2. Code Examples with Commentary:**

**Example 1: Incorrect Nginx Configuration**

This example demonstrates a typical nginx configuration flaw where the `upstream` block is incorrectly defined, pointing to `localhost:8002` instead of the correctly mapped port on the host machine.  In my experience, this was often due to a typo or outdated configuration file after adjusting port mappings.

```nginx
server {
    listen 80;
    server_name site.xyz;

    location / {
        proxy_pass http://localhost:8002;  # INCORRECT - Should point to Sail's exposed port (e.g., localhost:8000)
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

**Corrected Configuration:**

```nginx
server {
    listen 80;
    server_name site.xyz;

    location / {
        proxy_pass http://127.0.0.1:8000;  # CORRECT - Assuming Sail's mapped port is 8000
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

**Example 2: Sail Configuration and Port Mapping Discrepancy**

This example highlights a potential mismatch between the port exposed by Sail and the port used in the nginx configuration. This often occurs if one modifies the Sail configuration (`docker-compose.yml`) without updating the reverse proxy configuration.

```yaml
# docker-compose.yml (incorrect port mapping)
version: "3.4"
services:
  app:
    build: ./
    ports:
      - "8002:8000" # INCORRECT - exposes 8002 on the host, requiring a change in nginx
    environment:
      - APP_ENV=local
```

In this case, the nginx configuration needs to be updated to reflect the port mapping done in `docker-compose.yml`.  The previously corrected nginx configuration would now be incorrect, and the `proxy_pass` directive should be changed to `http://127.0.0.1:8002`.  The ideal solution, however, is to maintain consistency:  Keep the internal port at 8000 and map that to 8000 on the host.


**Example 3:  Virtual Host Misconfiguration in Nginx**

This scenario shows an nginx configuration where the `server_name` directive is missing or incorrect, which prevents proper routing based on the domain name. This often happens during the initial setup or when switching between multiple domains hosted through the same nginx instance.

```nginx
# Incorrect virtual host configuration
server {
    listen 80;
    # server_name missing!
    location / {
        proxy_pass http://127.0.0.1:8000;
        # ... other directives ...
    }
}
```


Adding the correct `server_name` directive and properly configuring other relevant settings, such as `listen` and other virtual host configurations within the nginx configuration file, is crucial. A correct setup would resemble the corrected configuration provided in Example 1, but the `server_name` would be correctly defined as `site.xyz`.  Also, ensure that your DNS properly resolves `site.xyz` to the public IP address of your server.

**3. Resource Recommendations:**

The official nginx documentation; the official Docker documentation; the Laravel documentation, specifically sections detailing deployment and Docker integration; a book on Linux system administration; a guide to reverse proxy configuration best practices.


By systematically reviewing these configurations, paying close attention to the details of port mappings, upstream definitions in the proxy's configuration, and ensuring accurate virtual host setups, the issue of `localhost:8002` being displayed instead of `site.xyz` can be effectively resolved.  Remember to restart nginx after making any configuration changes.  Furthermore, consistently using the default port 8000 exposed by Sail unless you have a compelling reason to change it simplifies the process and reduces the likelihood of encountering such issues.  Finally, logging is your friend; checking the nginx error logs for any hints as to what went wrong is a crucial debugging step.
