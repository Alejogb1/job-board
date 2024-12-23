---
title: "How do I configure Nginx reverse proxy redirects for subdomains?"
date: "2024-12-16"
id: "how-do-i-configure-nginx-reverse-proxy-redirects-for-subdomains"
---

, let’s tackle this. I've spent more than a few late nights wrestling with Nginx configurations, particularly those involving reverse proxy setups for subdomains. It's a common scenario, and there are a few nuanced ways to approach it, so let’s dive in. The core concept is directing traffic arriving at specific subdomains to the appropriate backend services, which might be running on different ports or even different servers entirely.

Essentially, what we're doing is crafting Nginx server blocks that listen on specific hostnames (our subdomains) and then using the `proxy_pass` directive to route those requests to their intended destinations. This involves a combination of correctly matching the incoming `Host` header and then forwarding the request with the correct information. It’s not inherently complex, but precision is paramount; a small typo can lead to some head-scratching debugging sessions.

My experience stems from managing a microservices architecture, where several applications were each accessed through distinct subdomains: api.example.com, app.example.com, and static.example.com, for instance. Each required its own configuration, and it was crucial to maintain a clean, easily understandable structure.

Let's get to the practical part. The following configuration structure assumes that you're running Nginx on port 80 or 443 and that you have DNS records correctly pointing your subdomains to your server's IP address. The `server` blocks are where all the magic happens.

```nginx
server {
    listen 80;
    server_name api.example.com;

    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}

server {
    listen 80;
    server_name app.example.com;

    location / {
        proxy_pass http://127.0.0.1:3000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}

server {
    listen 80;
    server_name static.example.com;

    location / {
        proxy_pass http://127.0.0.1:9000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

In this basic setup, each `server` block listens on port 80 (you would ideally configure this for SSL using port 443 in a production environment). The `server_name` directive specifies the subdomain we're handling. The `location /` block within each `server` defines how all requests should be treated for that subdomain; in our case, we proxy all requests to a particular backend server based on the port number. The `proxy_pass` directive specifies where to forward the request. The `proxy_set_header` directives are equally important; they ensure that the correct `Host` header, the real IP of the client, and other necessary information are passed to the backend service. It is generally not recommended to blindly proxy the host header in untrusted networks since that can be used in an attack vector. These settings provide context to your application running behind the reverse proxy and allow it to behave correctly.

Now, let’s imagine a slightly more complex scenario involving a path-based routing alongside subdomain redirection. For example, maybe you have an ‘images’ folder on your static subdomain that needs to point to a specific backend.

```nginx
server {
    listen 80;
    server_name static.example.com;

    location / {
        proxy_pass http://127.0.0.1:9000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /images/ {
        proxy_pass http://127.0.0.1:9001; #images server
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

Here, anything coming in for `static.example.com` will be forwarded to the service listening on port 9000, but requests for `static.example.com/images/*` will specifically be routed to the server listening on port 9001. The order in which the `location` blocks appear is important; Nginx processes location directives in the order they're declared, so the more specific path `/images/` is evaluated before the general `/`.

Finally, let's consider an instance where we have different subdomains with different backends on a *different* machine altogether. This is very common in distributed systems.

```nginx
server {
    listen 80;
    server_name api.example.com;

    location / {
        proxy_pass http://192.168.1.100:8080; #different machine
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}

server {
    listen 80;
    server_name app.example.com;

    location / {
        proxy_pass http://192.168.1.101:3000; #different machine
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
         proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

In this scenario, the backend servers are not local to the Nginx machine itself. Here, we proxy to the specific IP addresses of our services that might reside on separate virtual machines or physical servers on the same network. Notice that we still retain the same `proxy_set_header` directives, which are essential for the proper functioning of the applications receiving the proxied traffic.

For anyone delving deeper into Nginx configuration, I'd strongly recommend the official Nginx documentation as the primary source of truth. Furthermore, the book "Nginx HTTP Server" by Jesse Davis offers a very thorough exploration of its features and capabilities, especially if you need a deeper understanding of the advanced directives and how to optimize performance. In addition, the book "High Performance Web Sites" by Steve Souders can shed a great deal of light on HTTP performance and how to optimize your websites when you configure the reverse proxy for them. Also, for a complete treatment of DNS, "DNS and BIND" by Cricket Liu and Paul Albitz is the most comprehensive resource available.

Remember, configuration is only one part of the equation. Understanding how Nginx handles requests, logging, and troubleshooting are equally crucial. Thorough testing is essential when making these changes, and reviewing the Nginx error logs should be the first stop whenever something goes wrong. I've had my share of head-scratching moments with seemingly minor misconfigurations, so careful attention to detail, and systematic testing, will save you considerable time in the long run. I hope this provides a helpful and concise framework for setting up your Nginx reverse proxy for subdomains.
