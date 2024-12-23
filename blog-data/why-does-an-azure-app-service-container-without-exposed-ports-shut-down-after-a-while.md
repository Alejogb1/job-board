---
title: "Why does an Azure App Service container, without exposed ports, shut down after a while?"
date: "2024-12-23"
id: "why-does-an-azure-app-service-container-without-exposed-ports-shut-down-after-a-while"
---

Okay, let's tackle this. I've seen this behavior more times than I'd care to remember, and it’s almost always a variation on the same root causes. An Azure App Service container shutting down when no ports are explicitly exposed might seem counterintuitive at first. You'd think a container just sitting there, happily processing, wouldn't be a problem, right? Well, the Azure platform has a few built-in mechanisms, primarily designed for efficiency and resource management, that can lead to this. And these are the ones I've personally tripped over in a few of my deployments over the years.

The fundamental issue revolves around the health monitoring and the lifecycle management within the Azure App Service environment. Specifically, if your container doesn't respond to the platform's health probes, it's considered unhealthy, and the platform will aggressively shut it down to reclaim resources. Remember, the service is trying to run *efficiently* on shared infrastructure, and unhealthy containers consume resources without actively contributing.

Here's a breakdown of the most common culprits:

1. **Health Check Failures:** Azure uses HTTP probes to determine the health of your application. If you haven't configured a health endpoint or if your application isn't responding correctly to the probe (typically a 200 OK response), Azure will deem it unhealthy. Because there are no exposed ports, the app is never accessible to the health probe. This is the most frequent problem I encounter. It's not that the app is *broken* per se, it just isn't playing by the Azure platform's rules. The service expects to be able to verify the application is functioning properly and can’t do that without a port.

2. **Idle Timeout:** While technically separate from health checks, idle timeout behavior can manifest in similar ways. If your application isn't actively processing requests, and has no exposed port to listen to requests, or executing long running jobs and the web app believes its idle, Azure can aggressively shut down the application to free up resources, even if the container image itself has no issues. This usually applies more to situations where the app does have a exposed port, but a lack of activity can trigger this process. I've worked on apps before that have long running background processes that aren’t immediately obvious, but Azure still interprets a lack of network activity with its own logic for efficiency.

3. **Container Startup Issues:** If your application is taking an excessively long time to start within the container, the health probes may timeout before your application is ready, causing the same 'unhealthy' assessment and subsequent shutdown. Things like large databases, extensive static file builds or pre-processing scripts can delay the health checks enough to trigger the problem. This is less of a case of no ports exposed, but it's worth mentioning because the symptoms are the same, and often the *fix* is similar: optimize how the container starts, to get to a healthy state as quickly as possible.

Let's get into some code examples to illustrate these points. I’m assuming a basic Dockerfile setup here, building a simple node.js application, but the concepts apply generally.

**Example 1: Simple Health Check Endpoint**

This first snippet shows a very basic express server that exposes a `/health` endpoint. This is the crucial part: without this, Azure has no way of knowing your application is 'up'.

```javascript
const express = require('express');
const app = express();
const port = 3000;

app.get('/', (req, res) => {
  res.send('Hello World!');
});

app.get('/health', (req, res) => {
  res.sendStatus(200);
});

app.listen(port, () => {
  console.log(`App listening on port ${port}`);
});
```

In this example, I’ve defined a rudimentary health endpoint at `/health` that will respond with a 200 status code when accessed. This is the bare minimum Azure needs to confirm the app is running. Note that we are exposing port 3000 which can then be used in the settings of the Azure App Service. Without this specific endpoint and port, Azure has no way of knowing this app is available.

**Example 2: No health check, just a processor (BAD)**

This example shows a node application that processes something but does NOT expose a health check. This WILL fail in azure even though it may be doing exactly what you want it to be doing.

```javascript
const processData = async () => {
  console.log("Starting processing...")
  // Simulate processing data
  await new Promise(resolve => setTimeout(resolve, 30000));
  console.log("Processing complete.")
};

processData();

setInterval(() => {
  console.log("Still running...")
}, 10000)
```

In this example, the application just does work and outputs messages to the console. Without any exposed ports and a health check, Azure will eventually believe that the application has died and shut it down. It is not enough for the application to just run.

**Example 3: Modifying Startup to avoid health check issues**

Building on the first example, let's say your start-up sequence is lengthy (loading data, for example) this shows how to delay reporting healthy until everything is prepared.

```javascript
const express = require('express');
const app = express();
const port = 3000;
let isReady = false;

async function initializeApp() {
  console.log("Starting initialization...");
  // Simulate some lengthy startup process
  await new Promise(resolve => setTimeout(resolve, 10000));
  console.log("Initialization complete, app is now ready");
  isReady = true;
}

app.get('/', (req, res) => {
  res.send('Hello World!');
});

app.get('/health', (req, res) => {
    if(isReady){
      res.sendStatus(200);
    } else {
        res.sendStatus(503) // Service Unavailable
    }

});

initializeApp().then(() => {
  app.listen(port, () => {
    console.log(`App listening on port ${port}`);
  });
});
```

Here, the server will not respond healthy until a startup process is completed. During the initialization, a 503 status code will be returned when hitting the healthcheck which will tell Azure the application is not yet healthy, but it's *working* to get ready. Once the initialization is done, the health check will report 200. This can resolve a common issue that would cause the app to crash when its healthcheck is immediately hit but the application isn’t yet ready.

When dealing with Azure App Services and containers without exposed ports, the core principle to understand is that *it isn't just about whether your container is running, but also if it’s running 'correctly' in the context of the Azure platform*. Azure requires an active, demonstrable confirmation of health. It requires a signal that the container is *ready* to serve its intended function. Without that signal, the platform will assume the application is not functioning as expected.

To go deeper, I'd recommend getting into these resources:

*   **“Docker Deep Dive” by Nigel Poulton:** An excellent book for understanding container fundamentals, which is crucial for troubleshooting issues like this. Understanding the inner workings of Docker will allow you to develop your containers with the required processes to run in a cloud environment.

*   **Microsoft Azure documentation on App Service Health Checks:** It has extensive details on how Azure health probes work, how to configure them, and the troubleshooting steps you should try. The official docs are generally a great place to start when having these issues.

*   **"Kubernetes in Action" by Marko Lukša:** While focused on Kubernetes, understanding its concepts of liveness and readiness probes provides valuable context. The principles are similar to the Azure App Service health checks, and the understanding gained will broaden your approach to these types of problems.

Ultimately, these shutdowns aren't a bug; they're a result of the design that prioritizes efficient resource usage. Once you understand the underlying mechanisms at play, you can avoid running into these situations in the future. By incorporating these concepts into your development process, you will minimize issues when deploying. I hope this breakdown gives you the specific knowledge and approach needed to address the root cause.
