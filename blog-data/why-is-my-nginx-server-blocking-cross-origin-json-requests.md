---
title: "Why is my Nginx server blocking cross-origin JSON requests?"
date: "2024-12-23"
id: "why-is-my-nginx-server-blocking-cross-origin-json-requests"
---

Okay, let's tackle this one. I've seen this exact scenario play out more times than I care to remember, usually when a developer is first venturing into the complex world of cross-origin requests and nginx configuration. The frustrating part is that the error messages often don’t explicitly point to the problem. Typically, when you're encountering blocked cross-origin JSON requests with nginx, it’s not nginx itself inherently *blocking* the requests in the sense of denying access. It’s far more likely a configuration oversight where you're missing the necessary headers that allow the browser to accept the response. Think of it like this: nginx delivers the data, but the browser, following its same-origin policy, refuses to consume it without explicit permission.

The core issue lies in the browser’s enforcement of the same-origin policy, a security mechanism that prevents a malicious script on one origin from accessing resources on another origin. This policy defaults to blocking cross-origin requests unless the server explicitly allows them via headers. Therefore, when you make a JSON request from a domain different from where the JSON resource resides, the browser checks if the server has sent back specific cross-origin resource sharing (CORS) headers. If not, the request is blocked at the browser level, even though the server might be technically serving the data perfectly well.

My experience with this typically involves frontend teams banging their heads against a wall, convinced their fetch calls or xhr requests are flawed, while the backend team is equally adamant that the json is being returned properly. This back-and-forth often circles until the CORS headers are identified as the true culprit. This is a common scenario, and thankfully, the solution is often straightforward.

The key CORS headers are: `access-control-allow-origin`, `access-control-allow-methods`, `access-control-allow-headers`, and optionally, `access-control-expose-headers`. Let's break down each one:

*   `access-control-allow-origin`: This specifies which origins are allowed to access the resource. You can set it to a specific origin (`https://example.com`), a comma-separated list of origins, or, and this is what you'll often see in development, `*` to allow requests from any origin. While `*` is convenient for local testing, *never* use it in production. It opens you up to cross-site scripting (XSS) risks.
*   `access-control-allow-methods`: This lists which HTTP methods are allowed for cross-origin requests (e.g., `GET`, `POST`, `PUT`, `DELETE`). If you’re making a POST request and only `GET` is allowed, the browser will block the request.
*   `access-control-allow-headers`: This indicates which headers are allowed in the actual request. If the client is sending custom headers, these need to be explicitly declared in this response header.
*   `access-control-expose-headers`: This one is less frequently needed, it specifies which headers can be exposed to the client. By default, not all headers in a response are visible in the client, using this allows them to be.

Now, let's explore how to configure nginx to handle these headers. Here are three examples, ranging in complexity:

**Example 1: Basic CORS Configuration (Suitable for Development)**

This configuration uses `*` for `access-control-allow-origin` and allows all standard methods and common headers. *Note that you should replace `*` with specific origins for production environments*.

```nginx
server {
    listen 80;
    server_name example.com;

    location /api/ {
        add_header 'Access-Control-Allow-Origin' '*' always;
        add_header 'Access-Control-Allow-Methods' 'GET, POST, PUT, DELETE, OPTIONS' always;
        add_header 'Access-Control-Allow-Headers' 'DNT,User-Agent,X-Requested-With,If-Modified-Since,Cache-Control,Content-Type,Range' always;
        add_header 'Access-Control-Expose-Headers' 'Content-Length,Content-Range' always;


        # other configurations
        proxy_pass http://your_backend;
    }
}

```

**Example 2: Specific Allowed Origin and a Customized Header**

This one shows how to limit access to a single origin and allow a custom header named 'X-Custom-Header'.

```nginx
server {
    listen 80;
    server_name example.com;

    location /api/ {
        add_header 'Access-Control-Allow-Origin' 'https://frontend.example.com' always;
        add_header 'Access-Control-Allow-Methods' 'GET, POST, OPTIONS' always;
        add_header 'Access-Control-Allow-Headers' 'DNT,User-Agent,X-Requested-With,If-Modified-Since,Cache-Control,Content-Type,Range,X-Custom-Header' always;
        add_header 'Access-Control-Expose-Headers' 'Content-Length,Content-Range' always;

        # other configurations
        proxy_pass http://your_backend;
    }
}
```

**Example 3: Dynamic Origin Handling (More Advanced)**

This example uses nginx's map directive to dynamically set the `access-control-allow-origin` header based on the `origin` header of the incoming request.

```nginx
map $http_origin $cors_origin {
    default "";
    "https://frontend.example.com" "https://frontend.example.com";
    "https://another.example.com" "https://another.example.com";
}

server {
    listen 80;
    server_name example.com;

    location /api/ {
        if ($cors_origin = "") {
            return 403;
        }
        add_header 'Access-Control-Allow-Origin' $cors_origin always;
        add_header 'Access-Control-Allow-Methods' 'GET, POST, PUT, DELETE, OPTIONS' always;
        add_header 'Access-Control-Allow-Headers' 'DNT,User-Agent,X-Requested-With,If-Modified-Since,Cache-Control,Content-Type,Range' always;
        add_header 'Access-Control-Expose-Headers' 'Content-Length,Content-Range' always;

        # other configurations
        proxy_pass http://your_backend;
    }
}
```

In these snippets, I'm using `add_header ... always;` to ensure that the headers are sent in every response. The 'always' part is especially helpful because it ensures that if something fails upstream, the correct headers are still sent as well. When encountering CORS issues, make sure to double check the presence of these headers, especially during preflight requests which can be difficult to debug.

For further reading, I’d recommend looking at the official documentation of the w3c for CORS specification as well as reading through *High Performance Web Sites* by Steve Souders, which provides insight into several performance impacts in web development and some implications of these headers. Additionally, Mozilla’s developer network (MDN) has a comprehensive section dedicated to CORS that is helpful in understanding the underlying mechanisms. I also suggest that you take a look into the official nginx configuration documentation for more advanced use cases, such as `map` directive used in the last example. By understanding these principles and exploring the documentation, you'll be well-equipped to handle cross-origin issues when working with nginx. And frankly, once you’ve debugged it a few times, you’ll spot the missing headers a mile away. It just comes with experience.
