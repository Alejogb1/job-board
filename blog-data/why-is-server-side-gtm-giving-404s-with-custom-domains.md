---
title: "Why is server-side GTM giving 404s with custom domains?"
date: "2024-12-23"
id: "why-is-server-side-gtm-giving-404s-with-custom-domains"
---

Okay, let’s tackle this one. I’ve seen this particular flavor of server-side google tag manager (gtm) 404 error with custom domains quite a few times, and it almost always boils down to a handful of core configuration issues. It's a maddening problem because the server setup *seems* correct initially, but a subtle misstep can throw the entire thing off. I remember back in '19 working with a client who was migrating their e-commerce platform—it was a textbook case of what *not* to do in terms of server-side configuration, and this specific issue was front and center.

The short version is that when you’re using a custom domain for your server-side gtm, your browser is actually making requests to *your* servers, not to a google-controlled subdomain. This means a few more pieces have to be correctly aligned: DNS, SSL, reverse proxy configuration, and of course, the gtm container itself. A 404 typically implies the server cannot find the resource being requested, in this context, it often means the request isn’t being routed correctly to the gtm container’s endpoint, or the container is not configured properly to handle the incoming request path.

Let’s go into detail on where things can go astray. The first critical area is DNS. You need to have an `a` record pointing your custom domain (e.g., `gtm.yourdomain.com`) to the IP address of your server running the gtm container. It may seem obvious, but a missing or incorrect `a` record is the most common culprit. More than once I've seen DNS propagated slowly, or worse, some internal dns servers weren't updated, so requests from certain networks would still fail.

Once the DNS is sorted, SSL is the next hurdle. Your custom domain needs a valid ssl certificate to serve over https. If you've only configured an http server, or have an expired/invalid certificate, browsers will often outright block requests, and might not explicitly present a 404, but may show connection errors which can lead to confusion. If you've generated your ssl cert yourself, ensure the proper chain certificates are also deployed with your certificate. Tools like `openssl s_client -connect gtm.yourdomain.com:443` (replace `gtm.yourdomain.com` with your actual domain) can help diagnose these certificate problems.

Now we get to the interesting part, the reverse proxy. In many setups, you won’t directly expose your gtm container endpoint to the outside world; instead, you use a reverse proxy like nginx or apache. This proxy handles ssl termination, and potentially other tasks, and routes requests to the gtm container itself. A misconfigured reverse proxy is perhaps the most frequent cause of 404 errors. Specifically, a reverse proxy needs a configuration to forward specific requests—typically any request to `gtm.yourdomain.com`—to the appropriate endpoint where the gtm container is listening, often `http://localhost:8080` (though your specific configuration might use a different port).

Here's an example snippet of an nginx configuration that would accomplish this, assuming your server-side gtm is running on port 8080:

```nginx
server {
    listen 443 ssl http2;
    server_name gtm.yourdomain.com;

    ssl_certificate /path/to/your/ssl.crt;
    ssl_certificate_key /path/to/your/ssl.key;

    location / {
        proxy_pass http://localhost:8080;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }
}
```

Key details to note here: the `server_name` should match your custom domain, `proxy_pass` routes the request to the application, and the `proxy_set_header` entries pass through critical information about the original request. The `ssl_*` lines point to the ssl certificate and private key used to decrypt the https traffic. Missing one of these crucial lines can lead to unexpected behavior, including 404s.

Another common cause is a mismatch between the incoming request *path* expected by the gtm server and the request *path* sent by the client. For example, a gtm container that expects requests to `/collect` might get an incoming request to the root of the domain `/`. Therefore the server will not find a resource mapped to the root path, resulting in a 404. Some misconfigurations might also involve incorrect request headers. So it's critical to observe the exact request the browser is making, using dev tools, to precisely determine what’s being sent from the browser. If you are seeing requests going to `/`, instead of say, `/collect`, this can indicate either a problem with the client-side gtm configuration, or the custom domain configuration in your server-side gtm container.

Here is a quick and dirty node.js application as an example of how a simple gtm-like server could handle incoming requests:

```javascript
const http = require('http');
const port = 8080;

const server = http.createServer((req, res) => {
    if (req.url === '/collect' && req.method === 'POST') {
        let body = '';
        req.on('data', (chunk) => {
            body += chunk;
        });
        req.on('end', () => {
            console.log('received:', body);
            res.statusCode = 200;
            res.setHeader('Content-Type', 'text/plain');
            res.end('data collected');
        });

    } else {
        res.statusCode = 404;
        res.setHeader('Content-Type', 'text/plain');
        res.end('not found');
    }
});

server.listen(port, () => {
    console.log(`server running on port ${port}`);
});

```
This basic example only handles requests to `/collect` with a POST method. Any other request path will result in a 404. It highlights the fundamental issue of route matching in a server. In the real world, the path processing in gtm's server-side container is much more involved, but the concept remains the same.

Finally, sometimes it’s a client-side configuration issue masquerading as a server problem. You need to check your client-side tag configurations and ensure the *transport url* field in the tag configuration is correctly set to your custom domain, not the default google-controlled one. This setting controls the actual destination of the http requests sent when your client-side tags trigger. If the configuration is not correct, you’ll be sending requests to the wrong endpoint and of course getting an error.

Let’s consider a third, more specific example, this time regarding handling authentication headers, as some setups require authentication to reach the gtm endpoint. Let's assume that your server expects a header named `x-api-key` with a valid key.

```nginx
server {
    listen 443 ssl http2;
    server_name gtm.yourdomain.com;

    ssl_certificate /path/to/your/ssl.crt;
    ssl_certificate_key /path/to/your/ssl.key;

    location / {
        proxy_pass http://localhost:8080;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header x-api-key $http_x_api_key; # pass through x-api-key header
        proxy_cache_bypass $http_upgrade;


        # Handle 403 errors (example of more complex setup)
        proxy_intercept_errors on;
        error_page 403 @forbidden;
    }

    location @forbidden {
        return 403 "invalid api key";
    }

}
```
Here, we've added `proxy_set_header x-api-key $http_x_api_key;` to pass through the authentication header. If your server-side gtm container is expecting that header and it's missing, it could be the source of a 404, although it may return other errors such as a 403. In complex configurations, you might need to modify and add or filter headers and other requests metadata using the reverse proxy or the application itself, depending on the context.

For further reading, I’d highly recommend checking out the documentation for nginx (if that's your proxy) which is exceptionally thorough. For a broader understanding of how http requests work, “http: the definitive guide” by David Gourley and Brian Totty is an invaluable resource. And, for a deeper dive into server architectures and networking, a good text on computer networks, like "computer networking: a top-down approach" by James Kurose and Keith Ross will provide the necessary fundamentals. I hope these insights are helpful.
