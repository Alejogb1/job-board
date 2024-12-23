---
title: "How can cookies be shared between two specific subdomains?"
date: "2024-12-23"
id: "how-can-cookies-be-shared-between-two-specific-subdomains"
---

Okay, let's tackle this. Subdomain cookie sharing is a topic that, while seemingly straightforward, has nuances that can trip up even experienced developers. I’ve definitely been down that rabbit hole myself a few times, specifically when migrating a complex e-commerce platform a few years back. We had a main domain, `example.com`, and various subdomains like `shop.example.com`, `blog.example.com`, and `api.example.com`. Maintaining user sessions across these was a key requirement, and a naive approach would have led to a very frustrating user experience.

The crux of the issue lies in how browsers handle cookie domains. By default, a cookie set by `shop.example.com` is only visible to requests made to `shop.example.com` and not, say, to `blog.example.com`. This is due to the inherent security model of the web, preventing potential cross-site scripting (xss) vulnerabilities. The same-origin policy limits how scripts from one origin (protocol, domain, and port) can interact with resources from a different origin. Cookies fall under this domain restriction, so we need to explicitly configure them to be shared.

The solution, fundamentally, revolves around setting the `domain` attribute of the cookie when you're setting it on the server-side. Instead of letting it default to the specific subdomain, you explicitly set it to the base domain. This instructs the browser to share this cookie with requests to any subdomain of the specified domain.

Here’s a simple example, starting with a backend in Python using the `Flask` framework:

```python
from flask import Flask, make_response, request

app = Flask(__name__)

@app.route('/setcookie')
def set_cookie():
    response = make_response("Cookie set!")
    response.set_cookie('user_id', '12345', domain=".example.com", httponly=True, samesite='Lax') # note the leading dot
    return response

@app.route('/getcookie')
def get_cookie():
    user_id = request.cookies.get('user_id')
    return f"User ID: {user_id if user_id else 'Not found'}"

if __name__ == '__main__':
    app.run(debug=True, port=5000)
```

In this snippet, when the `/setcookie` endpoint is hit, we create a response and set a cookie named `user_id` with the value '12345'. Critically, we specify `domain=".example.com"`. Notice the leading dot. This dot is important. Without the dot, the cookie would be limited to `example.com`, not subdomains. `httponly=True` adds security against javascript access and `samesite='Lax'` offers protection against cross-site request forgery (csrf), but these are tangential to the main topic here and should be handled based on your application's needs. If you deploy this on your own machine, access `http://localhost:5000/setcookie`, then try to retrieve the cookie on any subdomain of `localhost` with your browser's developer tools, it will become clear that the cookie is correctly set with the `domain` attribute. You would need to use a tool that maps a hostname to `localhost` like `/etc/hosts`. To test this effectively, I suggest setting up `sub1.localhost` and `sub2.localhost` to point to 127.0.0.1 for testing purposes.

Here's how this might look in a Node.js application with Express:

```javascript
const express = require('express');
const cookieParser = require('cookie-parser');

const app = express();
const port = 3000;

app.use(cookieParser());

app.get('/setcookie', (req, res) => {
    res.cookie('user_id', '12345', { domain: '.example.com', httpOnly: true, sameSite: 'Lax'}); // note the leading dot
    res.send('Cookie set!');
});

app.get('/getcookie', (req, res) => {
  const userId = req.cookies.user_id;
  res.send(`User ID: ${userId ? userId : 'Not found'}`);
});

app.listen(port, () => {
  console.log(`Example app listening at http://localhost:${port}`);
});
```

Similarly, this example uses `cookie-parser` middleware for handling cookies, then sets a `user_id` cookie with the `domain: '.example.com'` option. Again, the dot is crucial for proper subdomain sharing and `httponly` and `samesite` are set for added security. These code examples can be readily deployed to understand the concepts discussed.

Finally, while server-side cookie setting is prevalent, it's worth illustrating a javascript example for cases where client-side control is also needed. This should typically be avoided for sensitive session information, however.

```javascript
// In your javascript, typically within a <script> tag

function setCookie(name, value, domain) {
  const expires = new Date();
  expires.setTime(expires.getTime() + (1 * 24 * 60 * 60 * 1000)); // Expires in 1 day
  document.cookie = `${name}=${value};expires=${expires.toUTCString()};domain=${domain};path=/;samesite=lax`;
}

function getCookie(name) {
    const nameEQ = name + "=";
    const ca = document.cookie.split(';');
    for(let i = 0; i < ca.length; i++) {
        let c = ca[i];
        while (c.charAt(0) === ' ') {
            c = c.substring(1, c.length);
        }
        if (c.indexOf(nameEQ) === 0) {
            return c.substring(nameEQ.length, c.length);
        }
    }
    return null;
}

// setting
setCookie('user_id', '12345', '.example.com'); // note the leading dot

//reading
const userId = getCookie('user_id')
console.log("User ID: " + (userId ? userId : "Not found"));
```

Here, the `setCookie` function takes the cookie name, value, and crucially, the `domain`, as arguments and sets the cookie on the `document`. This would need to be deployed on a webpage served from the domains you want to share cookies on. The `getCookie` function demonstrates how one might retrieve a cookie. It's imperative, when manipulating cookies on the client side that you are aware of potential security issues and mitigate them appropriately. This example can be tested by including it in the html served from an appropriate test server.

Several crucial points deserve attention here:

1.  **The leading dot in the domain**. It's not optional; it's vital. Without it, cookies are tied specifically to the subdomain setting them.
2.  **Domain specificity**. Be as specific as possible with the domain attribute. Setting it too broadly (e.g., just `.com` if you own a domain called `example.com`) can be a security risk.
3.  **Path Attribute:** The default path is `/` which means the cookie is valid for the entire domain.
4.  **HttpOnly Flag:** This is crucial for security. If you do not need javascript to access a cookie, set this flag to prevent malicious scripts from getting access.
5.  **SameSite attribute**: Setting this is also good practice and provides protection against cross-site request forgery. `Lax` or `Strict` offer different protection levels.
6.  **Secure attribute**: When your app is running over `https`, you should set this flag so cookies are only transmitted over `https` to protect against man-in-the-middle attacks.

From my experience, proper configuration of cookie domains often involves understanding the specific limitations of the frameworks and servers used. It's worthwhile to consult the documentation for each library or service you're utilizing when dealing with cookies. Some resources I'd suggest are RFC 6265, the HTTP State Management Mechanism specification, which is the authority on how cookies operate. Further, you should consult the documentation for your specific language and web framework such as flask's documentation or express's. Finally, a deep understanding of security best practices, specifically around cookie handling such as those in OWASP’s documentation, is essential for a robust and secure application.

In closing, subdomain cookie sharing is not overly complex once you grasp the domain attribute of the cookie. When you run into issues, start by triple-checking that dot and ensuring all code is serving the cookies with the appropriate attributes and understand the nuances with `httponly` and `samesite` for added security. That’s usually the crux of any issues I've encountered myself.
