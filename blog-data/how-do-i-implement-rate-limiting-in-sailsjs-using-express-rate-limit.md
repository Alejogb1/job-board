---
title: "How do I implement rate limiting in Sails.js using express-rate-limit?"
date: "2024-12-23"
id: "how-do-i-implement-rate-limiting-in-sailsjs-using-express-rate-limit"
---

Okay, let's tackle rate limiting in Sails.js with `express-rate-limit`. It’s a crucial aspect of securing any API, and I've certainly had my share of late nights debugging issues caused by rate limit vulnerabilities before implementing these strategies. I'm going to share my approach, focusing on practical steps and nuances I've learned along the way.

First, understand that Sails.js, being built on Express.js, readily integrates with standard Express middleware. This means we can leverage `express-rate-limit` directly without too much fuss. The core principle of rate limiting is simple: it restricts the number of requests a client can make within a specific timeframe. This helps prevent brute-force attacks, denial-of-service attempts, and mitigates the impact of misbehaving clients.

Before diving into code, we need to install the package:

```bash
npm install express-rate-limit --save
```

Now, let’s break down how I typically integrate this into a Sails application. In my experience, the most effective place to apply rate limiting is often within Sails' policies. Policies are middleware that get executed before your controllers, giving us a centralized point to control request traffic. I usually create a policy specifically for rate limiting, and here’s what it might look like, step-by-step, in `/api/policies/rateLimit.js`:

```javascript
// api/policies/rateLimit.js
const rateLimit = require('express-rate-limit');

module.exports = async function (req, res, proceed) {
  const limiter = rateLimit({
    windowMs: 15 * 60 * 1000, // 15 minutes
    max: 100, // limit each ip to 100 requests per windowMs
    message: 'Too many requests from this IP, please try again after 15 minutes',
    standardHeaders: true, // Return rate limit info in the `RateLimit-*` headers
    legacyHeaders: false, // Disable the `X-RateLimit-*` headers
    handler: (req, res, next, options) => {
      res.status(429).send(options.message);
    }
  });

  limiter(req, res, proceed);
};
```

This code snippet defines a policy using `express-rate-limit`. The `windowMs` parameter sets the timeframe for our limit (15 minutes here), and `max` defines the maximum allowed requests from a single IP during that timeframe. If the limit is exceeded, the client receives a 429 status code along with the provided `message`. I prefer using the `handler` function for customization and making sure I send the correct status code alongside the message. Setting `standardHeaders` allows for more standardized rate limiting information to be sent back to the client, which aids in debugging or client-side logic.

Now that we have our policy, we need to apply it. I'd typically configure it in `/config/policies.js`. Here's how you might do that, applying rate limiting to all requests:

```javascript
// config/policies.js
module.exports.policies = {
  '*': 'rateLimit', // Apply the rateLimit policy to all routes
  // Other policies can be specified as necessary
};
```

By setting `'*': 'rateLimit'`, the `rateLimit.js` policy is executed for every incoming request. That's a great starting point, but typically you'll want finer-grained control. It's not always the case that all routes need the same rate limiting configuration.

Let’s see how to customize it for specific routes. For example, let’s say we want to set stricter rate limits for authentication routes. I would create another policy, say `/api/policies/authRateLimit.js`, similar to the previous one but with different parameters:

```javascript
// api/policies/authRateLimit.js
const rateLimit = require('express-rate-limit');

module.exports = async function (req, res, proceed) {
  const authLimiter = rateLimit({
    windowMs: 60 * 60 * 1000, // 1 hour
    max: 5, // limit each ip to 5 requests per hour
    message: 'Too many login attempts from this IP, please try again after 1 hour',
    standardHeaders: true,
    legacyHeaders: false,
    handler: (req, res, next, options) => {
       res.status(429).send(options.message);
    }
  });

  authLimiter(req, res, proceed);
};
```

Here, I've set a tighter limit of 5 requests per hour. To use this specifically for authentication endpoints, update `/config/policies.js` like this:

```javascript
// config/policies.js
module.exports.policies = {
   'auth/login': 'authRateLimit',
   'auth/register': 'authRateLimit',
  '*': 'rateLimit', // Apply default rate limiting to all other routes

};
```

With this setup, the stricter `authRateLimit` policy is applied only to the `auth/login` and `auth/register` routes, and the general `rateLimit` policy applies to everything else. This illustrates how you can apply different rate limiting rules based on your application’s needs.

Now, it’s crucial to consider some practical considerations when configuring rate limiting. One common issue is identifying clients when behind a load balancer or proxy. In such cases, the incoming IP address (`req.ip`) is usually the load balancer's address, not the client’s. To mitigate this, you'll need to configure your Sails application to trust the necessary headers that might contain the original client IP, typically `X-Forwarded-For`. You can usually achieve this within the Express configuration options, which Sails exposes via `config/http.js`:

```javascript
// config/http.js
module.exports.http = {
  trustProxy: true, // Or configure a specific header like 'X-Forwarded-For'
};
```

Setting `trustProxy` to `true` is often a sufficient starting point, but you can configure it more granularly, based on your specific infrastructure and proxy setup. Check the Express documentation for `trust proxy settings` for more specific scenarios.

Another point to consider: these implementations use memory storage by default. For high-traffic applications or in distributed environments, it is often advised to switch to a more durable storage mechanism to persist rate limit data, such as Redis or Memcached. `express-rate-limit` supports these alternatives. Consult the `express-rate-limit` documentation and examples for information on how to set this up.

For a more in-depth dive into rate limiting techniques and best practices, I recommend reviewing the paper "Rate Limiting Strategies and Their Impact on Web Application Security" (hypothetical title but look for related work in academic databases such as ACM or IEEE) and also explore resources on the OWASP website related to API security. The book "API Security in Action" by Neil Madden (Manning Publications) provides a comprehensive look at many such topics. These resources provide a more formal and detailed understanding of the mechanics behind rate limiting, its variations, and its role in securing web applications.

In summary, while `express-rate-limit` is easy to integrate into Sails.js, a bit of nuanced configuration, careful attention to your specific infrastructure, and a good grasp of rate limiting principles are key to setting up effective API security. I've found that iterating on these basic concepts based on actual usage and monitoring usually provides the most practical and secure setup.
