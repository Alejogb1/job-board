---
title: "How does a WordPress login cookie from a different origin handle CORS?"
date: "2024-12-23"
id: "how-does-a-wordpress-login-cookie-from-a-different-origin-handle-cors"
---

Let’s explore this. I recall a rather perplexing situation from a few years back. I was tasked with integrating a decoupled frontend application built in React with an existing WordPress backend, a fairly common setup these days. One of the hurdles we immediately encountered was managing user authentication across different domains; the WordPress site served content from `api.example.com`, while our React app resided on `app.example.com`. The core of the problem, as you might suspect, revolved around CORS and WordPress’s reliance on cookies for login authentication.

At first glance, it seems like a straightforward CORS issue, but the use of a login cookie adds a layer of complexity that needs careful consideration. Standard CORS headers, like `Access-Control-Allow-Origin`, primarily dictate whether a browser allows a script from one origin to access resources from another. This is generally effective for simple requests – say, fetching JSON data. However, when authentication is involved, particularly cookie-based authentication as is the case with WordPress, the browser’s behavior becomes more nuanced, driven by the need to prevent cross-site request forgery (CSRF) attacks.

The primary problem stems from the fact that cookies are, by default, domain-specific. A WordPress login cookie, conventionally named something like `wordpress_logged_in_[hash]`, is issued for the domain of the WordPress instance itself – in our example, `api.example.com`. Thus, when a user authenticates on `api.example.com`, a cookie is stored against that domain by the browser. When `app.example.com` subsequently attempts to access WordPress endpoints that require authentication, this cookie is *not* automatically sent with the request, and therefore, authorization fails. CORS, in its standard implementations, doesn't inherently solve the issue of sharing cookies across domains. This requires a specific setup, focusing not merely on the headers that govern data transfer, but the *credentials* themselves – a detail often overlooked.

Now, to make this work effectively, several crucial steps are needed, both on the server and client side. First, the WordPress backend needs to be configured to explicitly allow credentials (cookies) in the cross-origin request. This is achieved via the `Access-Control-Allow-Credentials` header. However, this header alone is not enough. When `Access-Control-Allow-Credentials` is included, the `Access-Control-Allow-Origin` header *cannot* use a wildcard (`*`). Instead, it must explicitly list the allowed origin, `app.example.com` in our case. Without these two in harmony, the browser will block any requests that carry cookies.

Here's the first code snippet, representing how you might configure your WordPress `.htaccess` file (or server config) to achieve this:

```apacheconf
<IfModule mod_headers.c>
    Header always set Access-Control-Allow-Origin "https://app.example.com"
    Header always set Access-Control-Allow-Credentials "true"
    Header always set Access-Control-Allow-Methods "POST, GET, OPTIONS, DELETE, PUT"
    Header always set Access-Control-Allow-Headers "Content-Type, Authorization, Accept, X-Requested-With, Cache-Control, Pragma"
</IfModule>

```

Note that the allowed methods and headers might vary depending on your specific application. The critical part is `Access-Control-Allow-Credentials` being set to `true`, and `Access-Control-Allow-Origin` being set specifically to the origin of your front-end application and *not* `*`. It's also important to know that preflight requests (OPTIONS) are also essential for cross origin requests that do not meet the simple request criteria.

Now, even with these server-side configurations, you might encounter issues if the WordPress cookie is not configured correctly. WordPress, by default, sets cookies with `SameSite=Lax`. `SameSite=Lax` and `SameSite=Strict` cookies are not sent cross origin by default. To enable cross origin access to cookies they must have the `SameSite=None` and secure attribute set. The secure attribute tells the browser that this cookie should only be sent over HTTPS. If your backend and frontend are both on HTTPS, this can be fixed. If either is not, this will *not* work.

Here’s the second code snippet, demonstrating how to configure your WordPress site using a PHP snippet, typically placed in your theme's `functions.php` file or a custom plugin to adjust cookie settings:

```php
<?php
function custom_cookie_settings($args) {
    $args['samesite'] = 'None';
    $args['secure'] = true;
    return $args;
}
add_filter('wp_session_cookie_args', 'custom_cookie_settings');
add_filter('auth_cookie_args', 'custom_cookie_settings');
add_filter('secure_auth_cookie_args', 'custom_cookie_settings');
```

This ensures that the cookies are sent with the `SameSite=None` and `Secure` attributes, making them available for cross-origin requests as long as the correct CORS headers are in place. It is important to remember that the `SameSite=None` attribute will *only* function correctly if the site is served over HTTPS, otherwise cookies will be rejected.

Finally, on the client-side (our React app), you’ll have to ensure that the `withCredentials` flag is set to `true` when making API requests via `fetch` or an equivalent HTTP client. This instructs the browser to include the cookies in the request. This is typically done as part of the request options.

Here’s an example of a JavaScript snippet using `fetch`:

```javascript
const fetchData = async () => {
  try {
    const response = await fetch('https://api.example.com/wp-json/my-endpoint', {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
      credentials: 'include', // important for sending cookies!
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    console.log(data);
  } catch (error) {
    console.error('Error fetching data:', error);
  }
};
```

Notice the `credentials: 'include'` option; without it, the cookie will not be sent with the request. This is crucial for maintaining authenticated sessions across different origins.

In summary, dealing with WordPress login cookies in a CORS context requires meticulous configuration of both the backend (WordPress server and cookie attributes) and the frontend application (request settings). Relying solely on standard CORS headers without considering the nuances of credentialed requests, particularly concerning `SameSite` cookie attributes, will not address the core issue of securely transferring authentication across origins. The combination of setting `Access-Control-Allow-Credentials: true` paired with an explicit origin in the `Access-Control-Allow-Origin` header, combined with setting cookies to `SameSite=None` and the secure flag and finally setting the `credentials: 'include'` flag in your API calls is vital to achieve a secure and functioning solution.

For a more comprehensive understanding, I highly recommend delving into resources such as the official documentation for the *CORS specification* itself and exploring the specifics within the *HTTP standards documents* related to cookie attributes. Additionally, reading relevant sections of *"Web Security: A Practitioner's Guide" by Bryan Sullivan* would prove beneficial. These sources provide in-depth coverage of the underlying technologies and best practices. They will equip you with a deeper knowledge beyond the surface-level implementations. Understanding these standards and their implications is paramount when building secure, robust applications that span multiple domains.
