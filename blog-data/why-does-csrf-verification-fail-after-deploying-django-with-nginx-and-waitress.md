---
title: "Why does CSRF verification fail after deploying Django with nginx and waitress?"
date: "2024-12-23"
id: "why-does-csrf-verification-fail-after-deploying-django-with-nginx-and-waitress"
---

Alright, let's tackle this. It's a familiar scenario, really. I've certainly seen this particular headache pop up more times than I care to count in the past, especially after transitioning from development servers to more robust production setups. The issue of csrf verification failing after deploying a Django application using nginx and waitress often boils down to a misunderstanding of how these components interact, specifically concerning cookie handling and request origins. It's rarely a fault in the core Django csrf implementation itself, but rather a configuration mismatch or oversight in the deployment pipeline.

Let's break it down from the ground up, and I'll share some specifics based on similar scenarios I've encountered. Initially, when you're working with Django's built-in development server, which typically runs on a specific address (like `127.0.0.1:8000`) and doesn't involve proxies or complex server setups, CSRF verification works smoothly. Django's CSRF protection primarily relies on a secret token, which is embedded in the HTML form (or included in request headers for AJAX calls) and stored as a cookie on the client's browser.

The problems typically arise when you introduce nginx and waitress. Waitress is your wsgi server, which is essentially listening on the backend for requests. Nginx, on the other hand, serves as a reverse proxy. This means client requests hit nginx first, and then nginx forwards them to waitress. The crucial part here is the `Host` header that nginx forwards. Django uses this header and the setting `ALLOWED_HOSTS` to ascertain if the incoming request is legitimate. If the request's `Host` header doesn't match anything listed in `ALLOWED_HOSTS`, Django won't set the CSRF cookie properly, or worse, it may reject the request entirely. This misalignment is often the culprit behind CSRF failures.

Further complicating this situation is that cookies operate under a domain and path scope. If Django issues the csrf cookie on a subdomain or a different port compared to your front-facing website's url, the browser simply won't send it back with the form post or ajax request, causing csrf verification to fail. If you have a load balancer that terminates SSL, then you can sometimes end up with the cookie having the wrong protocol set. In short, the browser has a security mechanism that it will not send cookies across different origins.

Here are some practical scenarios I've dealt with, along with the specific fixes I've used:

**Scenario 1: Incorrect `ALLOWED_HOSTS` setting:**

This is the most frequent issue I've seen. Django's default `ALLOWED_HOSTS` is empty, allowing any host, or `['*']`, which is not appropriate for production. When deploying, often times this is forgotten or misconfigured.

Let's say your domain name is `www.example.com`. Here's how to correct it:

```python
# settings.py

ALLOWED_HOSTS = ['www.example.com', 'example.com']
```
This configuration explicitly tells Django that requests from these domains are considered valid. If nginx is also handling other virtual hosts for subdomains, make sure these are also present in `ALLOWED_HOSTS`. For example, if you also have `api.example.com`, you would have it as follows:
```python
ALLOWED_HOSTS = ['www.example.com', 'example.com', 'api.example.com']
```

**Scenario 2: Proxy configurations and `X-Forwarded-Proto`:**

When nginx acts as a proxy, it’s crucial to correctly handle headers such as `X-Forwarded-Proto`. If your nginx config isn’t setting this, or is setting it incorrectly, Django may not recognize that the request was made using https, and may issue cookies with http protocol instead of https protocol. This can lead to browsers failing to send the cookie to the correct origin. Here's how you would handle this in `settings.py` and the nginx configuration.

```python
# settings.py
SECURE_PROXY_SSL_HEADER = ('HTTP_X_FORWARDED_PROTO', 'https')
```

In your nginx configuration, ensure you're forwarding the protocol correctly:

```nginx
server {
    listen 443 ssl;
    server_name www.example.com;

    # ... SSL certificates configurations ...

    location / {
        proxy_pass http://127.0.0.1:8080; # Or whatever port waitress listens on
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```
The `proxy_set_header X-Forwarded-Proto $scheme;` is key here, as it passes the protocol to the downstream application. This will signal to Django, specifically when using the `SECURE_PROXY_SSL_HEADER` configuration, that this request is to be treated as an SSL request, so it issues cookies with the https protocol instead of http.

**Scenario 3: Incorrect Cookie settings:**

Django allows granular control over cookie settings via settings such as `CSRF_COOKIE_DOMAIN`, `CSRF_COOKIE_SECURE`, and `CSRF_COOKIE_SAMESITE`. Incorrect values for these will cause csrf verification failures when deploying with nginx and waitress.

For instance, if your website runs on `www.example.com` but you’ve mistakenly set `CSRF_COOKIE_DOMAIN = 'example.com'`, this could cause CSRF verification issues, especially if the user has visited a different subdomain, as the browser might not include the cookie when submitting a form to the `www` subdomain. Here's an example of correctly setting the cookie settings:

```python
# settings.py

CSRF_COOKIE_DOMAIN = '.example.com'  # This includes both www.example.com and example.com if there is no subdomain.
CSRF_COOKIE_SECURE = True # only send cookie over https
CSRF_COOKIE_HTTPONLY = True # prevent javascript from accessing the cookie
CSRF_COOKIE_SAMESITE = 'Strict' # prevent cross-site request forgeries, ensure same site.

SESSION_COOKIE_DOMAIN = '.example.com'
SESSION_COOKIE_SECURE = True
SESSION_COOKIE_HTTPONLY = True
SESSION_COOKIE_SAMESITE = 'Strict'
```

When dealing with cross-subdomain scenarios, remember that if the `CSRF_COOKIE_DOMAIN` is not set, or set improperly, cookies might be specific to a subdomain, and thus not available to your application when posted from different origins. Setting it to `'.example.com'` makes the cookies available to all subdomains of `example.com` or example.com, but be cautious of this in multitenant environments. It is also important to set `SESSION_COOKIE_DOMAIN` if the session cookie is not functioning correctly.

For deeper reading on these concepts, I strongly suggest going through the official Django documentation on cross-site request forgery protection (`docs.djangoproject.com/en/stable/ref/csrf/`), which is very thorough. Also, reading the section about `SECURE_PROXY_SSL_HEADER` and related settings within Django's settings documentation is highly relevant (`docs.djangoproject.com/en/stable/ref/settings/`). Furthermore, understanding HTTP headers and cookies behavior is critical for web development, "HTTP: The Definitive Guide" by David Gourley and Brian Totty is an excellent resource. Understanding how nginx proxies requests, and its specific configurations for handling headers is also paramount, so the nginx official documentation is a must-read (`nginx.org/en/docs/`). Finally, delving into the specifics of the samesite attribute of cookies is helpful and you can find more info at the IETF specification page (`tools.ietf.org/html/draft-ietf-httpbis-rfc6265bis-02`).

In closing, the key to resolving csrf failures after deploying with nginx and waitress isn't to blame the framework, but rather to meticulously verify the interaction of your application, reverse proxy, and web server configurations, focusing on host matching, ssl handling, and correct cookie attributes. It's usually a subtle configuration detail that gets missed, but with a systematic approach, it's almost always resolvable.
