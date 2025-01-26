---
title: "Why does CSRF verification fail after deploying a Django application to nginx with Waitress?"
date: "2025-01-26"
id: "why-does-csrf-verification-fail-after-deploying-a-django-application-to-nginx-with-waitress"
---

The core reason for CSRF verification failures after deploying a Django application behind Nginx with Waitress stems from the interplay between how Django manages its CSRF token and how reverse proxies, such as Nginx, influence the request context. Specifically, the problem often surfaces due to discrepancies between the `X-Forwarded-Proto` header and Django's configuration. I've observed this pattern across several project deployments, and the fix generally involves ensuring Django accurately perceives whether a request was initiated over HTTPS or HTTP.

CSRF protection in Django relies on generating a secret token tied to a user's session and embedding it in forms or JavaScript interactions. When a form is submitted or an AJAX request is made, Django verifies this token against the user's session, preventing cross-site request forgery attacks. Crucially, the token generation and verification process consider the scheme of the request, i.e., HTTP or HTTPS, to prevent leakage and hijacking. When deploying behind Nginx, all requests pass through Nginx first before reaching the Waitress server hosting the Django application. Nginx, acting as a reverse proxy, often terminates SSL/TLS connections, passing the requests onward to Waitress over HTTP. This disconnect is where the problem manifests itself. Without explicit configuration, Waitress, and subsequently Django, receives an HTTP request regardless of the original scheme used by the client. If Django is expecting an HTTPS request, based on its settings, or cookies previously set, the CSRF token verification fails.

Let me illustrate with three code examples how configuration missteps can lead to this issue and how to correct them.

**Example 1: Default Django and Nginx Configuration - Resulting in Failure**

Here's a simplified scenario reflecting a common initial setup:

```python
# settings.py (Django)
SESSION_COOKIE_SECURE = True
CSRF_COOKIE_SECURE = True
```

```nginx
# nginx.conf (simplified)
server {
  listen 80;
  server_name example.com;

  location / {
    proxy_pass http://127.0.0.1:8000; # Waitress listening on 8000
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    # Notice absence of X-Forwarded-Proto
  }
}

```

In this setup, the Django application expects secure cookies due to `SESSION_COOKIE_SECURE` and `CSRF_COOKIE_SECURE`. The client accesses the application via HTTPS, but Nginx forwards the requests via HTTP to Waitress. Importantly, the `X-Forwarded-Proto` header is not being set. Waitress and Django subsequently treat the requests as initiated over HTTP, and therefore any previously set secure cookies are rejected, leading to inconsistent CSRF token values and failure upon verification. The secure cookie flag prevents the browser from including the cookies on the insecure connection from Nginx to Django.

**Example 2: Correct Configuration with `X-Forwarded-Proto`**

To rectify the issue, we must inform Django about the original request protocol. This is achieved by setting the `X-Forwarded-Proto` header in Nginx and instructing Django to utilize it:

```nginx
# nginx.conf (updated)
server {
  listen 80;
  server_name example.com;

  location / {
    proxy_pass http://127.0.0.1:8000;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-Proto $scheme;
  }
}
```

```python
# settings.py (Django)
SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")
SESSION_COOKIE_SECURE = True
CSRF_COOKIE_SECURE = True

```

Here, I've added `proxy_set_header X-Forwarded-Proto $scheme;` in the Nginx configuration. This header passes the original request’s scheme (either “http” or “https”) to the Waitress server. In Django’s `settings.py`, `SECURE_PROXY_SSL_HEADER` allows Django to trust the `X-Forwarded-Proto` header, enabling it to correctly interpret secure connections. By using this setting, Django treats the request as if it originated over HTTPS, and the secure cookies will be passed from the client to the Django application through the Nginx proxy. It's important to note the header and value names are case sensitive in the Django settings.

**Example 3: Alternative Configuration with `USE_X_FORWARDED_HOST`**

Another scenario involves ensuring that Django accurately uses the `Host` header provided by Nginx, particularly in deployments using load balancers or multiple proxy levels:

```nginx
# nginx.conf (simplified)
server {
  listen 80;
  server_name example.com;

  location / {
    proxy_pass http://127.0.0.1:8000;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-Proto $scheme;
  }
}
```

```python
# settings.py (Django)
SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")
USE_X_FORWARDED_HOST = True
SESSION_COOKIE_SECURE = True
CSRF_COOKIE_SECURE = True

```

In this setup, the key addition is `USE_X_FORWARDED_HOST = True` in Django's `settings.py`. By enabling this, Django explicitly trusts the `Host` header provided by the proxy (Nginx). This is beneficial in cases where the internal Django application may not directly know the public-facing hostname. While seemingly separate from the core CSRF issue, incorrect Host handling can create problems related to secure cookies and session management that further complicate CSRF verification in practice. It should be noted that `USE_X_FORWARDED_HOST` implies that you fully trust your proxy, since a malicious header value here could lead to an attack.

In summary, CSRF verification failures in this deployment context boil down to Django's inability to infer the original request scheme, typically HTTPS, due to an incorrect Nginx configuration or absence of appropriate headers like `X-Forwarded-Proto`. Ensuring Django accurately interprets requests as secure via appropriate configuration is paramount to correct functionality. Beyond the settings modifications, the `USE_X_FORWARDED_HOST` parameter may be required to support multi-level proxies or different port setups, by telling Django to trust the `Host` header of the incoming request.

For further reading on the concepts discussed above, I suggest consulting documentation on Django's security settings, Nginx's proxy module, and the specific deployment guides for Waitress. Delving into the RFC specifications regarding the `X-Forwarded-*` headers and their intended use within web architecture is also valuable. Understanding these pieces allows one to diagnose and address such problems effectively, as these concepts transcend beyond the Django and Waitress environments and can be applied in a wider variety of web development contexts.
