---
title: "How can nginx handle wildcard subdomains as a reverse proxy?"
date: "2024-12-23"
id: "how-can-nginx-handle-wildcard-subdomains-as-a-reverse-proxy"
---

Alright, let's tackle this. It's a classic scenario, and one I've definitely spent a fair bit of time debugging over the years. Setting up nginx to gracefully handle wildcard subdomains as a reverse proxy isn't necessarily complex, but it does require a solid understanding of server blocks and regular expressions within nginx configuration. We're not simply mapping a single domain; we're dealing with a pattern, which needs precise handling to avoid unexpected routing issues.

Essentially, the goal here is to intercept requests coming into your server for any subdomain under a particular domain, and forward those requests to the appropriate backend service. This allows for dynamically scaling your services and providing a tailored experience based on the subdomain requested – say, `app1.example.com` goes to one container, while `app2.example.com` goes to another.

My experience with this started back during the early days of our cloud deployment. We were trying to implement a multi-tenant platform where each client had their own subdomain. Manual configuration of each subdomain was out of the question, so we dove deep into using `server_name` directives and regular expressions to achieve wildcard functionality.

The fundamental approach involves using the `server_name` directive with a regular expression, which allows nginx to capture a subdomain and use that captured value later in the configuration. This captured value can then be used, for example, to select the appropriate upstream service. Let's get into specifics with some examples.

**Example 1: Basic Wildcard Subdomain Routing**

This is our foundational configuration. It demonstrates how to catch any subdomain under `example.com` and direct all traffic to a single backend service, which is obviously not ideal for most use cases but good as a starting point.

```nginx
server {
    listen 80;
    server_name ~^(?<subdomain>.+)\.example\.com$;

    location / {
        proxy_pass http://backend_service;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

Here, the `server_name` directive uses a regular expression. The `~^` signifies that a regular expression follows, and the expression `(?<subdomain>.+)\.example\.com$` breaks down like this:
* `(?<subdomain>.+)` captures one or more characters (`.+`) before `.example.com` and stores it in a named capture group called `subdomain`.
* `\.` represents a literal period since a period has a special meaning in regular expressions, it needs to be escaped.
* `example\.com` matches the base domain name.
* `$` signifies the end of the string.

The `proxy_pass` directive forwards the request to the upstream service. You'll note the `proxy_set_header` directives as well; these are crucial for passing along the client's original host and IP to your backend.

**Example 2: Dynamic Routing Based on Subdomain**

Now let’s look at a more practical scenario: different subdomains pointing to different backend services. We'll achieve this with a combination of the capture group and a `map` directive.

```nginx
map $subdomain $backend {
    app1    http://app1_backend;
    app2    http://app2_backend;
    default http://default_backend;
}


server {
    listen 80;
    server_name ~^(?<subdomain>.+)\.example\.com$;

    location / {
        proxy_pass $backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

Here we've introduced a `map` block. This directive allows you to set a variable's value based on the value of another variable. In this case, it’s mapping the captured `$subdomain` to the corresponding backend service. If the `$subdomain` is `app1`, the `$backend` variable will be `http://app1_backend`, and so on. The `default` keyword ensures any non-matching subdomain is routed to the `default_backend`.

This approach significantly improves configurability and allows for easier expansion of your service landscape, something I can personally attest to being extremely valuable in high-growth environments.

**Example 3: Routing with Specific Paths**

Finally, let's get more granular. Perhaps you want to route specific paths within subdomains to different backends, which is a common requirement.

```nginx
map $subdomain $backend {
    app1    http://app1_backend;
    app2    http://app2_backend;
    default http://default_backend;
}

server {
    listen 80;
    server_name ~^(?<subdomain>.+)\.example\.com$;

    location /api/v1 {
      proxy_pass $backend/api/v1;
      proxy_set_header Host $host;
      proxy_set_header X-Real-IP $remote_addr;
    }


    location / {
        proxy_pass $backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

}
```

Now we've added a specific `location` block for `/api/v1`.  Any requests with that prefix within the subdomain will be routed to the corresponding backend service, with the path retained. This lets you handle api requests differently from regular page requests for instance.

**Crucial Considerations:**

While these examples provide a solid foundation, keep these important factors in mind:

*   **SSL/TLS:** You’ll almost certainly need to configure SSL certificates for your wildcard domain. Tools like Let's Encrypt with wildcard certificates (using DNS validation) can automate this process.
*   **DNS configuration:**  Ensure your DNS records are correctly set up to point all subdomains of `example.com` to the server running nginx.  This is usually an `A` record with `*.example.com` pointing to your server's IP address.
*   **Caching:** Properly configure nginx caching to reduce load on backend services.
*   **Error Handling:** Implement specific error pages and log rotation, as those things are not addressed here.

**Further Reading**

To really deepen your understanding beyond this explanation, I highly recommend these resources:

1.  **"Nginx HTTP Server" by Peter Ivanov**: This is a comprehensive guide that covers all aspects of nginx, from basic configuration to advanced tuning and module development.
2.  **"Mastering Nginx" by Dimitri Fedorov:** Provides a pragmatic view of how to utilize Nginx for both static content delivery and reverse proxy scenarios. Focuses on real world use cases and optimization.
3.  **Nginx Official Documentation:** The official documentation, available at *nginx.org*, should be your go-to reference for understanding any directive or feature within nginx.

In my experience, the approach of using `server_name` with regular expressions combined with `map` directives provides the most flexibility for implementing wildcard subdomain routing.  It allows for a highly configurable and maintainable setup. I’ve successfully deployed this pattern in various environments from personal projects to large-scale web applications. Remember to test your configuration thoroughly and always prioritize security. Let me know if you have any follow-up questions; I’m happy to share further insights if needed.
