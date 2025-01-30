---
title: "Why is a subdomain from a different host experiencing issues with nginx?"
date: "2025-01-30"
id: "why-is-a-subdomain-from-a-different-host"
---
The frequent culprit when a subdomain hosted on a different server experiences issues with an Nginx-configured primary domain lies in misconfigurations of DNS, specifically related to how the primary domain's Nginx server handles requests directed to the subdomain. I've spent considerable time troubleshooting these scenarios, and the core problem is seldom with Nginx itself, but rather with the interplay of DNS records and Nginx server block configurations. Typically, the primary domain's server might receive the request for the subdomain but lacks the appropriate directives to process it or forward it correctly.

The central issue is that when a user attempts to access a subdomain, like `sub.example.com`, the initial step involves a DNS lookup. This lookup, if correctly configured, should resolve to the IP address of the server hosting the subdomain. However, if this is not the case, the request may be misdirected to the primary domain's server. The primary domain's Nginx, by default, will likely not have a server block (or virtual host) configured to handle requests for the subdomain. When this occurs, you might encounter common error responses such as "404 Not Found" or even a timeout. If the DNS for the subdomain does point to the correct server, the problem then lies within the misconfiguration of the subdomain's Nginx setup. Let’s focus on the common errors where the primary domain's server is intercepting traffic meant for the subdomain.

The first critical check is the DNS configuration for the subdomain. I always start by verifying the A record for `sub.example.com`. It should definitively point to the IP address of the server intended to host it. If that record erroneously points to the primary domain's server IP, all subsequent server block adjustments on either server will not resolve the issue. It becomes a matter of making the initial routing of the traffic correct. The second crucial piece of the puzzle is the server block within the Nginx configuration of the primary domain's server. By default, an Nginx server block is configured to serve content based on the `server_name` directive. If the server block for `example.com` lacks a `server_name` that includes `sub.example.com`, it will respond based on the `default_server` flag, or will attempt to match another `server_name`. The result is not what you want. A common mistake I’ve seen is assuming Nginx on the primary server will somehow automatically forward subdomain requests without explicit configuration.

The typical solution requires adjustments to both the DNS records and the primary domain server’s Nginx configuration. After ensuring the DNS points to the correct server or the appropriate load balancer, we need to instruct Nginx to either directly handle the subdomain’s request if both are on the same server or redirect it if the subdomain is elsewhere. In most instances, where the subdomain is on its own server, we'll configure Nginx on the primary domain’s server to forward the request to the appropriate server or a load balancer. This may involve either acting as a reverse proxy or using a simple redirect.
Here are three code examples demonstrating different Nginx configurations that can cause or resolve the described issue, focusing on a situation where `sub.example.com` is on a different server than `example.com`:

**Example 1: Misconfigured Primary Domain Server (Incorrect DNS)**

This configuration shows a scenario on the primary domain server that will not correctly handle a request for the subdomain. In this case, let’s assume, the DNS record for sub.example.com wrongly points to the IP address of the primary domain, `example.com`.

```nginx
server {
    listen 80;
    server_name example.com www.example.com;

    root /var/www/example.com/html;
    index index.html;

    location / {
        try_files $uri $uri/ =404;
    }
}
```

**Commentary:**

This is a standard Nginx server block. It listens for traffic on port 80 and responds to requests for `example.com` and `www.example.com`. If a request for `sub.example.com` is received by this server (because of incorrect DNS), it will not match the `server_name` directive. Based on how Nginx handles unmatched server_names, it will likely serve the default server block or use the first matching server block if default is not set, which could be this one. In either case, the request will be processed as if it were for the primary domain, resulting in the wrong content being served. This configuration isn't inherently bad; it's just not set up to handle subdomain requests when those requests shouldn't arrive at this server in the first place. The problem is in the DNS configuration directing subdomain traffic here.

**Example 2: Resolving the issue on the Primary Domain Server (Reverse Proxy)**

Assuming the subdomain DNS is now correctly configured, it may still be desirable to route some traffic to the subdomain from the primary domain. To achieve this, we can use a reverse proxy configuration on the primary domain's Nginx server:

```nginx
server {
    listen 80;
    server_name example.com www.example.com;

    root /var/www/example.com/html;
    index index.html;

    location / {
        try_files $uri $uri/ =404;
    }
}

server {
    listen 80;
    server_name sub.example.com;

    location / {
        proxy_pass http://<IP_OF_SUBDOMAIN_SERVER>; # Replace with the correct IP
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

**Commentary:**

Here, we’ve introduced a new server block. This server block listens on port 80 and specifically handles requests for `sub.example.com`. The key directive is `proxy_pass`. It sends all requests received on this server block to the IP address of the server hosting the subdomain. The `proxy_set_header` directives ensure that the necessary information, such as the original host and the client’s IP, are sent along with the request to the subdomain server.  This configuration will handle the traffic if you are hosting the subdomain on another server, and prefer the primary domain server to proxy the traffic. This assumes the subdomain’s server has its own correctly configured Nginx server listening on port 80.

**Example 3: Resolving the issue on the Primary Domain Server (Redirect)**

Alternatively, if the desire is for a simple redirect to the subdomain’s server, and you do not want to proxy traffic through the primary domain server, here is how we can configure Nginx:

```nginx
server {
    listen 80;
    server_name example.com www.example.com;

    root /var/www/example.com/html;
    index index.html;

    location / {
        try_files $uri $uri/ =404;
    }
}

server {
    listen 80;
    server_name sub.example.com;

    return 301 $scheme://<IP_OF_SUBDOMAIN_SERVER>$request_uri;
}
```

**Commentary:**

Again, we have added a new server block for handling `sub.example.com` requests. This configuration uses the `return 301` directive to issue a permanent redirect. When a request for `sub.example.com` arrives at this server, it will immediately redirect the user’s browser to the specified URL of the subdomain’s server. The $scheme variable ensures the redirect is done using the correct HTTP or HTTPS protocol. The $request_uri is needed to append all URI's so that a request to `sub.example.com/test` does not take the user to the main URL, but instead, `http://<IP_OF_SUBDOMAIN_SERVER>/test`. This approach eliminates the overhead of proxying traffic. This only makes sense if you intend to completely redirect traffic to the other server.

In summary, issues with a subdomain hosted on a different server typically arise from a combination of DNS misconfiguration and insufficient or incorrect Nginx server block definitions on the primary domain server. By meticulously verifying DNS records and implementing the correct server blocks – using either reverse proxies or redirects – these problems can be effectively addressed. The DNS records must be correct, and the primary domain server needs to be configured to either handle or redirect traffic. A reverse proxy setup provides increased control and flexibility at the expense of complexity, while a redirect provides a straightforward way to direct requests to their intended location. Carefully consider your needs when deciding on the proper approach.

For further learning, I recommend researching practical DNS management techniques using resources such as textbooks focusing on networking and domain name systems. Also, studying in-depth Nginx server block configurations via Nginx’s official documentation can provide valuable insight into configuring and troubleshooting these types of setups. Furthermore, many sites offer tutorials on Nginx server block configurations. Finally, practical hands-on experience is crucial to understanding the nuances of these systems, try setting up test servers and deliberately creating and fixing these sorts of configuration issues.
