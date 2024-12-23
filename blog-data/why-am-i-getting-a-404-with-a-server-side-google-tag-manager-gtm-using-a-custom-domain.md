---
title: "Why am I getting a `404` with a Server-side Google Tag Manager (GTM) using a custom domain?"
date: "2024-12-23"
id: "why-am-i-getting-a-404-with-a-server-side-google-tag-manager-gtm-using-a-custom-domain"
---

Let’s unravel this `404` error you're experiencing with your server-side GTM setup using a custom domain. It's a common frustration, and from my experience tackling similar issues across several projects, I’ve come to appreciate the nuances of this configuration. Generally, a `404` error means the requested resource – in this case, the GTM container – couldn't be located at the specified address. With server-side GTM, this usually points to a misconfiguration either at the DNS level, the web server setup, or within GTM itself.

First, let's consider the DNS. When you set up a custom domain (let's say, `tags.yourdomain.com`) for your server-side GTM, that domain name needs to be correctly mapped to the server hosting the GTM container. In the past, I once spent a good half-day debugging a client's setup, only to find that the `CNAME` record was pointing to an old, decommissioned server. The telltale sign was the consistent `404` and the lack of any network activity beyond the initial request. A common mistake is creating an A record instead of a CNAME, or referencing the Google app engine domain directly instead of a proxy if that's how you've chosen to deploy server-side GTM. The fix here is straightforward: make sure your DNS provider has a `CNAME` record for `tags.yourdomain.com` pointing to the url your GTM server is listening on (e.g. your app engine address, if that's what you chose). It's advisable to test this lookup using tools like `dig` or `nslookup` from your terminal, confirming the correct mapping. If you get a 404 even when the DNS is pointing at your server, that signals an issue not at the domain resolution but rather at the server application level.

Next, we move to the server itself. The web server must be configured correctly to route traffic destined for the custom domain to the server-side GTM container application. This is often achieved through reverse proxies like Nginx or Apache or, if you're using Google App Engine, the app.yaml file can be used. Incorrect configurations here can manifest in various forms, including `404` errors. An example of this is the lack of `host` headers matching the server-side GTM domain, or routing to the wrong port. I recall an incident where, during a rush deployment, the port specified in the reverse proxy configuration didn't match the port on which the server-side GTM was listening; the consequence? a relentless barrage of 404's.

The third potential area for error lies within GTM itself. Your server-side container needs to be correctly configured with the appropriate server URL, so that the web client and the container are properly linked. Additionally, the server container requires a valid tagging server url to be configured. This url is generally the custom domain you're using, but it can also be your default Google app engine url, or a load-balancer url, depending on how you've configured your application. I've seen instances where an outdated GTM server container configuration was still referencing the default Google App Engine domain after the custom domain setup, leading to the container being unreachable through the custom domain resulting in a `404`.

To demonstrate these concepts, let's examine three practical scenarios with accompanying code snippets. These examples are simplified for clarity but represent real-world issues you might encounter.

**Example 1: Incorrect Nginx Reverse Proxy Configuration**

Suppose your Nginx configuration for `tags.yourdomain.com` is missing the necessary routing. Here's a simplified example of what *not* to do:

```nginx
server {
    listen 80;
    server_name tags.yourdomain.com;

    location / {
        # missing proxy_pass, resulting in no traffic forwarded to gtm app
        return 404;
    }
}
```

This configuration, while seemingly valid, will return a `404` because it doesn’t specify where the traffic should be forwarded. A fix for that could look like this:

```nginx
server {
    listen 80;
    server_name tags.yourdomain.com;

    location / {
        proxy_pass http://127.0.0.1:8080; # Assuming GTM is listening on port 8080
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }
}
```
The `proxy_pass` directive now correctly directs traffic to the server-side GTM application. Notice also the additional `proxy_set_header` lines that are necessary to pass host information to the server application, allowing it to handle the request properly.

**Example 2: Incorrect `app.yaml` Configuration in Google App Engine**

If you are using Google App Engine, your `app.yaml` must route traffic appropriately. A simple error in the url handler might lead to the 404. Here is an example of the *wrong* configuration:

```yaml
runtime: nodejs16
handlers:
- url: /.*
  static_files: build/index.html
  upload: build/index.html
```

Here, all urls are pointing to the index.html file in the build folder, which is wrong for server-side GTM. The correct `app.yaml` would look something like this:

```yaml
runtime: nodejs16
handlers:
- url: /
  script: auto
```
This configuration tells google app engine to route traffic to our node js application running in the container. Notice that we don't need to specify `index.js` since that is part of the default behaviour of Node app engine apps.

**Example 3: Mismatched Server URL in GTM**

Finally, a mismatch within GTM itself is also common. Suppose your GTM server-side container is still configured to use a default Google App Engine URL instead of your custom domain. This can be resolved in the server container's container settings. We do not have the ability to illustrate this in code snippets, but this can be readily verified by comparing your configured tagging server url in the server-side GTM container to the custom domain you have set up, or the load-balancer url if you are using one.

To dive deeper into these areas, I highly recommend exploring resources like “*High Performance Browser Networking*” by Ilya Grigorik for a thorough understanding of network protocols and DNS. For Nginx configurations, the official Nginx documentation is unparalleled. For Google App Engine specific errors, reviewing the Google Cloud documentation on App Engine deployments is essential. Also consider having a look at the server-side GTM documentation.

In summary, tackling a `404` error with a custom domain in server-side GTM requires a systematic approach. Start by verifying your DNS settings, then scrutinize your web server configurations (e.g. Nginx, Apache, or the `app.yaml` file in Google App Engine). Finally, check that the server-side GTM container is correctly configured with your custom domain, and make sure that there are no other misconfigurations within the Google Tag Manager console. These steps, along with thorough error logging and debugging, should help you resolve your `404` and get your server-side tagging up and running smoothly.
