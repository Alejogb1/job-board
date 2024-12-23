---
title: "kubernetes http probe failed with statuscode 404?"
date: "2024-12-13"
id: "kubernetes-http-probe-failed-with-statuscode-404"
---

so you're hitting that classic Kubernetes liveness probe 404 right I've seen this one more times than I've had lukewarm coffee at 3 am believe me. It's a pain in the neck but usually it's something straightforward. I've been wrestling with Kubernetes since version 1.2 which was like trying to tame a wild badger with a rubber chicken but we got through it somehow and honestly this specific error feels like a rite of passage.

First off let's break down what's probably going on that 404 error code means the Kubernetes probe is hitting your application at the specified path and your application is saying "nope I don't know what you're talking about" Basically the probe which is a really simple http request is not finding anything at the endpoint you configured. It's not a Kubernetes problem exactly it's your app not playing nice at the path.

Usually this falls into a couple of common categories I've learned through my share of debugging sleepless nights.

**The Probe Path Mismatch**

This is the most obvious one and surprisingly common even with experience. You set up a probe like this in your pod manifest:

```yaml
    livenessProbe:
      httpGet:
        path: /healthz
        port: 8080
      initialDelaySeconds: 5
      periodSeconds: 10
```
Simple enough right Well you might have put /healthz in your kubernetes definition but is that *actually* the path your app is serving? Double check your application code. This is a human error situation. I had a whole microservice once that was actually using /health instead of /healthz because someone copy pasted some code and didn't pay attention or perhaps he did and he's just evil I don't know.

I have a special key in my notes "pay attention to your health endpoints" I know it seems too obvious to mention but sometimes when you have 20 different microservices and you are deploying every day things get overlooked its like having a memory of a goldfish.

**The Port Problem**

Sometimes it isn't even a path problem. It could be your port. Your container might actually be listening on port 8080 just like the example but does the container exposed it? Is your application really binding to that port inside the container? You also have to take into account networking policies between your pods and namespaces which could be an issue. It can get confusing. My experience with this is having to go back and double checking that the container port actually matches my application port. Its a silly mistake but its worth mentioning.

**The Application Not Ready**

Sometimes the application is booting up still. It's starting but hasn't reached the state where it responds to requests or your health check. A 404 in this scenario can be an application not being ready to handle requests. The fix? Adjust your `initialDelaySeconds` in your kubernetes definition. Give your application some time. You can also use readinessProbe to handle that situation.

Let me show you a typical kubernetes definition with both probes that I use and I've found that works in many scenarios:

```yaml
    livenessProbe:
      httpGet:
        path: /healthz
        port: 8080
      initialDelaySeconds: 15
      periodSeconds: 10
      failureThreshold: 3
    readinessProbe:
      httpGet:
        path: /readyz
        port: 8080
      initialDelaySeconds: 5
      periodSeconds: 5
      failureThreshold: 3
```

With this example we also have a readinessProbe that might be needed. Usually when there is an endpoint like /readyz for example that means that your application can receive requests and it's good to go. This example shows that we give the liveness probe more time to fail. Sometimes the application might need more time than 5 seconds to boot up. Its a matter of testing and using the right parameters.

Another thing to take into account is that sometimes the container probe fails because of network problems this is usually a more difficult problem to diagnose. I had a network policy issue one time that made the application look like it was failing when it was not. It was not fun it was like an endless pit of despair. I eventually find the problem after many cups of coffee and debugging.

**Troubleshooting Steps**

Here's how I usually approach debugging this problem:

1.  **Double-check the probe path:** Connect inside the pod and curl the health check path. See if it returns 200, 404 or anything else. Example with a simple curl command.

```bash
kubectl exec -it <pod-name> -n <namespace> -- curl http://localhost:8080/healthz
```

2.  **Verify application health:** Check the application logs to see if there are any errors during startup or problems with the health check itself. You're gonna need some good logs to check or if your application can log all requests this might give you an idea too.

3.  **Check network connectivity:** Make sure that there aren't any networking problems between the pod and the kubernetes node.

4.  **Check your kubernetes manifest:** Is everything correct? Are your ports correct? Is the path correct? The details are what matters. The devil is in the details. I've told myself this too many times to count and I still sometimes make the same errors. I'm not sure why I still do this.

5.  **Increase initialDelaySeconds:** It's a very effective way to see if the application needs more time.

**Code Example**

Lets say you have a simple Node.js application. A health check could look like this:

```javascript
const express = require('express');
const app = express();

app.get('/healthz', (req, res) => {
  res.status(200).send('OK');
});

app.get('/readyz', (req, res) => {
    //check if the database is running or any other requirement you need
    //if everything is ok respond 200
    //else 503
    res.status(200).send('Ready');

});

const port = 8080;
app.listen(port, () => {
  console.log(`Server is running on port ${port}`);
});
```

And a basic python example could be:

```python
from flask import Flask, Response

app = Flask(__name__)

@app.route('/healthz')
def healthz():
    return Response("OK", status=200)

@app.route('/readyz')
def readyz():
    #check database or anything needed here
    return Response("Ready", status=200)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
```

Both examples shows the same basic pattern an endpoint that responds 200 on /healthz and /readyz. If you try to curl or make a request to any other endpoint that is not /healthz or /readyz then you will get a 404 that is the nature of a http webserver.

If you are struggling with this kind of problem and need help beyond this basic troubleshooting check out the Kubernetes documentation its quite well written. Also look into "Programming Kubernetes" by Michael Hausenblas and Stefan Schimanski and "Kubernetes in Action" by Marko Luksa these are good books that goes deeper into the subject.

Remember this is usually something basic so don't panic it's not the end of the world. I once spent 3 hours debugging this issue only to find out I had forgotten to deploy the new version of the application. Sometimes you just need a break and come back to it later. It's better to laugh about it later than to cry about it during development. It's like that old tech joke I heard once: "There are 10 types of people in the world those who understand binary and those who don't" haha funny right?

Good luck with your kubernetes journey and may your probes always return 200. If not at least you know what to look for.
