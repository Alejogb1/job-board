---
title: "Why am I getting a 'too many redirects' error when accessing my subdomain through Nextcloud and a reverse proxy?"
date: "2024-12-23"
id: "why-am-i-getting-a-too-many-redirects-error-when-accessing-my-subdomain-through-nextcloud-and-a-reverse-proxy"
---

Alright, let's tackle this "too many redirects" issue. It’s a classic pain point when dealing with reverse proxies and applications like Nextcloud, especially when you’re managing subdomains. I've definitely seen this a few times during past projects—remember that project back in '18? Similar problem, took a while to pin down. The core issue, in my experience, lies in a miscommunication, often around the expected protocols and host headers, between your reverse proxy and your Nextcloud instance. It’s like trying to have a conversation with someone who speaks a slightly different dialect; both parties are speaking but not understanding each other correctly.

Fundamentally, the "too many redirects" error arises when a server continually redirects a client request back and forth, eventually exceeding a limit set by the client (typically the browser). In our context, this loop usually involves your reverse proxy and your Nextcloud application server. To understand why it occurs, let’s unpack the common configuration points.

First, we have the *reverse proxy*. This component, often something like Nginx or Apache, sits in front of your Nextcloud instance. It's designed to receive external requests on your public-facing subdomain and then forwards these requests to your internal Nextcloud server. The key here is that the proxy also relays information—headers, including the host header which tells the backend server the original requested domain.

Then we have *Nextcloud itself*. Nextcloud has its own configuration that dictates how it expects to be accessed. This involves settings related to trusted domains and protocols (http or https). When Nextcloud receives a request, it checks the host header. If the host header doesn’t match what it expects, it might redirect you to the correct domain according to its internal settings. This is where the potential redirect loop can begin.

Let's illustrate with a scenario. Suppose your subdomain is `cloud.example.com`, which points to your reverse proxy. The proxy, in turn, communicates with your Nextcloud instance running locally on `192.168.1.100:8080`. The problem typically arises if the reverse proxy sends requests to Nextcloud without properly modifying or passing on the correct host headers, especially if Nextcloud is not configured to accept these requests. Nextcloud sees a request with a host header it does not recognise, it redirects to what it thinks is the correct domain. This in turn might get picked up by the proxy and you are off to the races.

Now, let's dive into some code examples—or rather, configuration snippets—to show where things often go wrong and how to correct them. These are simplified, but illustrative.

**Example 1: Nginx Proxy Configuration (Common Misconfiguration)**

This snippet shows a typical *incorrect* Nginx configuration that could lead to redirect issues:

```nginx
server {
    listen 80;
    server_name cloud.example.com;

    location / {
        proxy_pass http://192.168.1.100:8080;
        proxy_set_header Host $host; # This is often the problematic line
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

The key mistake here often lies with the line `proxy_set_header Host $host;`. In this case, the header is passed through and not altered. What you often need to do is to tell Nextcloud the header that represents the intended url for that instance. The $host variable passes on the original header, which might then cause Nextcloud to redirect based on an internal setting of another name or address.

**Example 2: Nginx Proxy Configuration (Corrected)**

Here's the *corrected* Nginx configuration that is less likely to trigger redirect loops:

```nginx
server {
    listen 80;
    server_name cloud.example.com;

    location / {
        proxy_pass http://192.168.1.100:8080;
        proxy_set_header Host cloud.example.com; # Corrected header
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

The change here is replacing `proxy_set_header Host $host;` with `proxy_set_header Host cloud.example.com;`. We explicitly tell Nextcloud that the intended host name is `cloud.example.com`, regardless of the host header received by the proxy. This will work so long as `cloud.example.com` is a trusted domain within the Nextcloud configuration.

**Example 3: Nextcloud Config (config.php)**

Finally, a relevant portion of your Nextcloud `config.php` configuration file should include your trusted domain:

```php
<?php
$CONFIG = array (
  // ... other configurations ...
  'trusted_domains' =>
  array (
    0 => '192.168.1.100:8080', // Internal access may still be required
    1 => 'cloud.example.com',   // External facing access
  ),
  //... other configurations
   'overwrite.cli.url' => 'https://cloud.example.com',
);
```

The key part is within the `trusted_domains` array and setting the `'overwrite.cli.url'` parameter, where `cloud.example.com` is listed. This confirms to Nextcloud that `cloud.example.com` is an allowed host and it should not redirect requests from this domain.

This combination of correctly setting the `Host` header in your reverse proxy and adding your domain to the trusted domains list in your Nextcloud config will often eliminate the redirect loop issue.

**Key Takeaways and Further Reading**

The specific header adjustments you need in your reverse proxy configuration might vary slightly depending on the specific software you use and how your internal network is set up. Also note that if you are accessing via https, you will need to ensure the reverse proxy is also setup to handle this, and your nextcloud instance is configured to know it is being accessed via https. This is often through the X-Forwarded-Proto header.

Beyond these code snippets, understanding the fundamentals of the HTTP protocol, particularly the role of host headers and redirects, is crucial for debugging these kinds of problems. I’d highly recommend delving into "HTTP: The Definitive Guide" by David Gourley and Brian Totty. It’s a comprehensive resource that covers these concepts in detail. In addition, reading the official documentation for both your reverse proxy software (Nginx, Apache, etc.) and Nextcloud itself is also essential; they often detail specific configurations and requirements for common setups. I found reading through the Nginx documentation and Nextcloud Admin manual particularly helpful during my own previous experience. For a general overview, “Computer Networking: A Top-Down Approach” by James F. Kurose and Keith W. Ross provides a solid background in network communication, which is useful for understanding how these components interact at a lower level.

In summary, the "too many redirects" error is usually a consequence of a disagreement over the host header between your reverse proxy and your Nextcloud instance. By carefully setting the correct host header in the reverse proxy configuration and configuring your Nextcloud instance with the correct trusted domain, you can effectively resolve the problem and get your subdomain working as intended. It requires careful attention to configuration and a solid grasp of the underlying technology, but it's an important step in getting any production deployment running properly.
