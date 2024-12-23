---
title: "Can sails.js globally throttle API requests?"
date: "2024-12-23"
id: "can-sailsjs-globally-throttle-api-requests"
---

,  I’ve certainly dealt with similar challenges in various node.js backends, and the need to globally throttle api requests in sails.js is definitely one that pops up as traffic grows. It isn’t a built-in, off-the-shelf feature, but it’s absolutely achievable with the right architectural approach.

My initial foray into this type of problem happened when I was building a real-time data aggregation platform. Initially, the API was quite straightforward. But once we started pushing the limits, we saw a few things: clients aggressively polling for updates, some even ignoring the recommended caching headers, which started to bring the database server to its knees. We needed a robust throttling mechanism, and we needed it fast.

The core issue with sails.js, or really any node.js framework, is that requests are typically processed within the event loop, and while it’s asynchronous, uncontrolled incoming requests can overload the process, leading to latency and even downtime. Sails itself doesn't provide a global throttling middleware by default, which means we have to build it, or more accurately, integrate it into sails’s middleware pipeline.

The key idea is to limit the number of requests that reach the controller handlers within a specific time window. This isn't just about preventing abuse; it's about ensuring the system's health. There are multiple ways to accomplish this, each with their own tradeoffs.

My experience led me to prefer the use of a rate limiter implementation, often leveraging something like redis for storage. Redis is ideal because of its speed and data persistence capabilities.

Here's the crux of how I've approached this:

**Building a Custom Throttling Middleware**

The general process involves creating a custom middleware that intercepts incoming requests and applies throttling rules before forwarding them to the relevant controllers. A typical setup would involve:

1.  **Tracking Request Counts:** This often involves storing request counts per IP address, user ID, or a combination thereof, within a specific time window.
2.  **Checking against Limits:** Each request is checked against the defined limits. If the current request exceeds the limit, we send a 429 "Too Many Requests" response.
3.  **Updating Request Counts:** For each successful request, the relevant counter is updated.
4.  **Resetting Counters (with expiry):** We need a mechanism to reset request counters at the end of a defined time window, which redis expiration mechanisms handle perfectly.

Here’s a code example, demonstrating a basic implementation using redis and a node.js package designed to simplify the interaction with it:

```javascript
// config/http.js (or a custom middleware location)

const redis = require('redis');
const { RateLimiterRedis } = require('rate-limiter-flexible');

const redisClient = redis.createClient({
  // Configure Redis here
  socket: {
    host: 'localhost',
    port: 6379,
  },
});

redisClient.connect();

const rateLimiter = new RateLimiterRedis({
  storeClient: redisClient,
  keyPrefix: 'api_limit',
  points: 10,  // 10 requests
  duration: 60, // per 60 seconds
});


module.exports.http = {
    middleware: {
        throttle: async function(req, res, next) {
            try {
                const limiterResponse = await rateLimiter.consume(req.ip); // Limit based on IP
                res.setHeader('Retry-After', limiterResponse.msBeforeNext / 1000);
                next();
            } catch (rejection) {
                res.status(429).json({
                    error: 'Too Many Requests',
                    retryAfter: rejection.msBeforeNext / 1000, // Provide the retry time
                });
            }
        },
        order: [
          'cookieParser',
          'session',
          'bodyParser',
          'compress',
          'poweredBy',
          'router',
          'throttle',
          'www',
          'favicon'
        ],
    },
};

```

In this first snippet, we're setting up redis and a rate limiter. Note that I'm using the `rate-limiter-flexible` package, which greatly simplifies handling the specifics of the rate-limiting algorithm. Here we are limiting request count by IP address, but you can just as easily use another identifier or a combination. Also important is how it is placed within the middleware order. By putting it after the bodyParser and before the router, the rate limiting occurs before any controllers are invoked.

Now, to illustrate a bit more flexibility, let’s see an example of varying rate limits, perhaps based on user roles.

```javascript
// config/http.js (Continuing from above)

const rateLimiterRole = async (req, points, duration) => {
   const specificLimiter = new RateLimiterRedis({
     storeClient: redisClient,
     keyPrefix: `api_limit_${req.user.role}`, // Dynamically set a prefix per role.
      points: points,
      duration: duration
    });

    try {
     const limiterResponse = await specificLimiter.consume(req.ip);
      res.setHeader('Retry-After', limiterResponse.msBeforeNext / 1000);
      return true
    } catch (rejection) {
     res.status(429).json({
       error: 'Too Many Requests',
        retryAfter: rejection.msBeforeNext / 1000,
     });
     return false;
    }
}

module.exports.http = {
    middleware: {
        throttleRole: async function(req, res, next) {
           if (!req.user){
              return next() //if there is no user, then continue
            }
           let allow = false;
           if(req.user.role === 'admin'){
             allow = await rateLimiterRole(req, 100, 60) // 100 per minute
           } else if (req.user.role === 'user'){
             allow = await rateLimiterRole(req, 20, 60)  // 20 per minute
           } else {
             allow = await rateLimiterRole(req, 5, 60) // Default limit
           }

           if (allow){
             next();
           }
       },
        order: [
            'cookieParser',
            'session',
            'bodyParser',
            'compress',
            'poweredBy',
            'router',
             'throttleRole',
            'www',
            'favicon'
        ]
    },
};
```

Here, the `throttleRole` middleware dynamically applies different limits based on the `req.user.role` property, which would ideally come from a previous authentication middleware. This is more advanced than the first example, but still quite manageable in terms of complexity.

Finally, let's consider a way to implement a more fine grained throttling system using routes.

```javascript
// config/http.js (Continuing from above)
const rateLimit = async (req, points, duration) => {
  const specificLimiter = new RateLimiterRedis({
    storeClient: redisClient,
    keyPrefix: `api_limit_${req.route.path}`, // use route as a prefix
    points: points,
    duration: duration,
  });

  try {
    const limiterResponse = await specificLimiter.consume(req.ip);
    res.setHeader('Retry-After', limiterResponse.msBeforeNext / 1000);
    return true;
  } catch (rejection) {
    res.status(429).json({
      error: 'Too Many Requests',
      retryAfter: rejection.msBeforeNext / 1000,
    });
    return false;
  }
};

module.exports.http = {
  middleware: {
    throttleRoute: async function (req, res, next) {
        let allow = false
       if (req.route.path === '/api/users') {
           allow = await rateLimit(req, 2, 60);
       } else if (req.route.path === '/api/posts') {
            allow = await rateLimit(req, 5, 60);
        } else {
            allow = await rateLimit(req, 10, 60);
        }
        if (allow){
             next();
           }
    },
    order: [
      'cookieParser',
      'session',
      'bodyParser',
      'compress',
      'poweredBy',
      'router',
      'throttleRoute',
      'www',
      'favicon',
    ],
  },
};
```

In this final example, we now have route specific rate limiting and can control the rate of specific endpoints. This is very handy when certain routes might have a higher cost to serve, or have greater risk of abuse.

**Important Notes and Considerations**

*   **Choice of Store:** Redis is great for its performance and simplicity, but you can also use other stores based on your needs. For instance, a memory-based store may be acceptable for smaller applications or specific purposes but carries risk of loss with node process restarts.
*   **Configuration:** The `points` and `duration` settings should be adjusted according to your requirements. You might need to experiment to find the optimal balance between limiting abuse and allowing legitimate traffic.
*   **Error Handling:** The example includes a basic 429 response, but more elaborate error handling, such as logging, is advisable. You may want to implement a circuit breaker pattern if upstream service errors start to occur with high frequency, which could be a sign of excessive load upstream.
*   **Granularity:** The examples provide a couple of ways to achieve rate limiting granularity. You can fine tune your limits as needed, based on parameters present on the request.
*   **Security:** Remember to handle rate limiting along with other security measures, as rate limits may need to be configured differently depending on the security profile of the client.

For a more profound dive into these concepts, I'd suggest looking at:

*   **"Designing Data-Intensive Applications" by Martin Kleppmann:** A fantastic resource for understanding the underlying principles of distributed systems.
*   **"High Performance Browser Networking" by Ilya Grigorik:** This book covers aspects of networking performance that are relevant to making your API responsive in the context of rate limiting.
*   Relevant node.js and redis documentation: To delve deeper into the packages or tools used here.

In my experience, implementing a robust rate-limiting strategy is essential for the stability and performance of any publicly facing api. While sails.js doesn't provide a ready-made solution, its flexibility in middleware allows for implementing these features quite effectively. This ensures a resilient and responsive platform that can gracefully handle varying levels of traffic and potential abuse.
