---
title: "Why are load test response times so poor in a new Sail.js project?"
date: "2025-01-30"
id: "why-are-load-test-response-times-so-poor"
---
The observed degraded performance during load tests of a newly provisioned Sails.js application, despite seemingly adequate resource allocation, often stems from a confluence of factors related to default settings and the inherent asynchronous nature of Node.js. I've encountered similar situations numerous times, particularly when transitioning from development to a production-like environment. The core problem rarely lies in Sail.js itself, but rather in the unoptimized configuration and common pitfalls associated with handling concurrent requests.

First, understand that Sails.js, built on Express.js, operates in a single-threaded, event-driven environment. This means a single Node.js process handles all incoming requests, asynchronously, using an event loop. While highly efficient for many tasks, heavy processing within the event loop can block other operations, leading to significant latency and poor response times, particularly under load. The default Sails.js configuration is optimized for developer convenience and rapid prototyping, not for sustained high throughput. For instance, the default logger, while useful for debugging, can introduce substantial overhead when writing to disk during a load test. Likewise, the ORM layer, Waterline, can become a bottleneck if database connections are not properly managed or queries are inefficient. Furthermore, middleware processing, especially complex custom middleware or inefficient session management, can add significant latency.

To illustrate, consider the scenario where you have an API endpoint designed to retrieve data from a database. In a basic implementation, without explicit tuning, this could manifest as follows.

```javascript
// Example 1: Basic Data Retrieval (potentially problematic under load)
module.exports = {
  getData: async function(req, res) {
    try {
       const records = await User.find({ where: { status: 'active' }});
       return res.json(records);
    } catch (error) {
        return res.serverError(error);
    }
  }
};
```

This initial code appears innocuous, but under high load, it poses several potential issues. The `User.find` operation is an asynchronous function using Waterline. If the database is slow or if there are many concurrent requests, this single operation can hold up the event loop. Additionally, the default connection pooling for Waterline might not be sufficient, leading to connection timeouts and further delays. This results in increased response times and reduced throughput. Each request sits and waits on the response from the database. The default Sails configurations do not optimize for this.

To mitigate these challenges, the first step is usually to optimize the ORM interaction. A common mistake is not using database-level indices and optimized queries. Another, more fundamental, problem is simply the overhead involved in using the ORM in the first place. Sometimes, in high throughput systems, raw queries can have substantially better performance. The second consideration is logging. Default levels can cause very significant performance degradation. Below is an example that moves logging to only error conditions and also replaces the ORM query with a raw database query.

```javascript
// Example 2: Improved Data Retrieval with Raw Query and Reduced Logging
module.exports = {
  getData: async function(req, res) {
    try {
      const records = await sails.getDatastore()
              .sendNativeQuery("SELECT * FROM user WHERE status = $1", ['active']);
       return res.json(records.rows);

    } catch (error) {
      sails.log.error("Error retrieving data:", error);
      return res.serverError(error);
    }
  }
};
```

This second snippet addresses a few issues. We now use a raw query that circumvents a large amount of ORM overhead. We've switched to using `sails.log.error` in the catch block which means logging only occurs on error conditions. This will likely still be too slow under heavy load. We could even reduce this to a simple logging library and completely bypass `sails.log` altogether.

The most crucial optimization for sustained load, however, lies in scaling the Node.js application using process managers. The single-threaded nature of Node.js limits its ability to fully utilize multi-core processors. By deploying multiple instances of the application, each acting as its own event loop, we can distribute the load and significantly improve throughput. We can accomplish this using tools such as `pm2` or `cluster`. I have personally found `pm2` to be a good option for this. Below is an example of using the Node `cluster` module.

```javascript
// Example 3: Scaling with Node.js Cluster
const cluster = require('cluster');
const numCPUs = require('os').cpus().length;

if (cluster.isMaster) {
  console.log(`Master ${process.pid} is running`);

  for (let i = 0; i < numCPUs; i++) {
    cluster.fork();
  }

  cluster.on('exit', (worker, code, signal) => {
    console.log(`worker ${worker.process.pid} died`);
  });
} else {
  require('./app'); // Path to your Sails.js app entry point
  console.log(`Worker ${process.pid} started`);
}

```

This code spawns one worker process for each CPU core, distributing the workload across multiple event loops. Note that the actual Sails application needs to be in a file called `app.js` in this case. The example assumes the application does not need to maintain any kind of shared state. If that is not the case, then external systems, such as Redis, will be required for shared state across the different process. Also note this code does not restart failed processes, and, depending on needs, this should be handled. The benefits are clear, however: multiple instances of the application are available to handle incoming requests, increasing overall throughput. Furthermore, using this structure means any single crash will only affect one of the workers, while the others continue serving requests.

Beyond code adjustments, there are several other system-level considerations. The operating system’s resource limits, such as the maximum number of open files and network connections, should be adjusted appropriately for the expected load. Furthermore, using a reverse proxy like Nginx can improve performance by handling static file serving and offloading SSL termination from the Node.js process. These are just starting points to address the problem.

In summary, poor load test response times in a new Sails.js application are typically due to a combination of factors including inefficient database interactions, inadequate resource allocation, and reliance on default settings optimized for development. By moving to raw database queries when needed, implementing proper logging, scaling the application horizontally with a process manager and adjusting OS-level configurations, performance can be significantly improved.

For more in-depth information, consider consulting resources like "Node.js Design Patterns" for architectural best practices and "High Performance Node.js" for detailed optimization strategies. Also review documentation on your chosen database system to understand performance tuning in that context. Finally, explore documentation on process managers like `pm2` to enhance your understanding of scaling applications.
