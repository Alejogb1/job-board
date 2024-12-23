---
title: "How can I retrieve accuracy metrics from a deployed DataRobot model via API?"
date: "2024-12-23"
id: "how-can-i-retrieve-accuracy-metrics-from-a-deployed-datarobot-model-via-api"
---

Okay, let's tackle this. I've actually spent a fair bit of time interfacing with deployed DataRobot models, and retrieving accuracy metrics programmatically is certainly a common need. It's not always as straightforward as you might initially expect, especially if you're used to working primarily within the DataRobot UI itself. The crucial thing to understand is that DataRobot provides a robust api for this sort of thing, but knowing which endpoints to hit and how to interpret the results is key.

Essentially, what you're asking is how to get that same performance data you see in the DataRobot Leaderboard, but programmatically. You’re not going to get the same level of detail as a full leaderboard view, but rather, specific metrics from a deployment. We’ll primarily focus on using the DataRobot Python client to achieve this, as it offers a user-friendly way to interact with the api. Let's break down the process and then I’ll provide some code examples.

The primary route to access metrics is through the 'deployments' endpoint. Each deployment has associated performance data that's updated as new data passes through the model. However, these metrics aren’t stored directly on the deployment object. They're actually associated with what DataRobot calls 'prediction server instances.' Each instance corresponds to an actively running model capable of scoring data, and it's these server instances that maintain the actual metric history. It's an important distinction to make.

So, to retrieve the information you want, you’ll need a two-step approach: first, locate your deployment, then get details of its server instances. Finally, the actual metric information can be pulled from these instances. You'll be interested in 'actuals data' which represents the true results against which the model’s predictions were evaluated. This data, when available, lets DataRobot compute performance metrics.

Let's assume you've already installed and configured the `datarobot` python client and have an established connection to your DataRobot instance. Here’s how I would generally approach this, step-by-step, and the code I'd use.

**Step 1: Locate your Deployment**

The first step is to identify the deployment for which you need metrics. This can be done using the deployment id or its name. I prefer the ID since it is generally more reliable.

```python
import datarobot as dr
import os

# Assuming you've set environment variables for your DataRobot API Key and Endpoint

try:
    dr.Client(token = os.environ['DATAROBOT_API_TOKEN'], endpoint=os.environ['DATAROBOT_ENDPOINT'])
except Exception as e:
    print(f"Error connecting to datarobot: {e}")
    exit()

deployment_id = "YOUR_DEPLOYMENT_ID" #Replace this with your deployment ID
deployment = dr.Deployment.get(deployment_id)
print(f"Found deployment: {deployment.name} (ID: {deployment.id})")
```
*Code Snippet 1: Finding the Deployment*

In this snippet, we are retrieving the deployment object. Replace `"YOUR_DEPLOYMENT_ID"` with your deployment’s actual ID. Make sure the API token and endpoint are set correctly, I generally use environment variables for this and the connection check helps ensure the client is configured right. If the deployment is found, we'll output its name and id, just to be sure we have the correct one.

**Step 2: Retrieve Instance Metrics**

Once we have the deployment, we need to fetch its associated prediction server instances. Each instance will have associated metrics. Here’s the code to accomplish this and extract the actual metrics:
```python
import datarobot as dr
import os
from datetime import datetime, timedelta

# Same environment checks as above
try:
    dr.Client(token = os.environ['DATAROBOT_API_TOKEN'], endpoint=os.environ['DATAROBOT_ENDPOINT'])
except Exception as e:
    print(f"Error connecting to datarobot: {e}")
    exit()

deployment_id = "YOUR_DEPLOYMENT_ID" #Replace this with your deployment ID
deployment = dr.Deployment.get(deployment_id)

instance_metrics = {}

for instance in deployment.get_prediction_server_instances():
    print(f"Checking instance ID: {instance.id}")

    metrics = instance.get_metrics()
    if metrics:
        instance_metrics[instance.id] = {}
        for metric in metrics:

                instance_metrics[instance.id][metric.metric_name] = {
                    "value": metric.value,
                    "start_time": metric.start_time,
                    "end_time": metric.end_time
                }
    else:
        print(f"No metrics found for instance {instance.id}")

print(f"Retrieved metrics from all instances: {instance_metrics}")
```
*Code Snippet 2: Fetching Instance Metrics*

This section goes through the instances attached to the deployment. For each instance, `instance.get_metrics()` returns metric data that will be stored in `instance_metrics` dictionary. I include the start and end time of a metric for tracking purposes, and it's important to note that this is not just a single snapshot but a series of measurements over time, though metrics are typically rolled up over a time frame, usually 10 minutes by default. If a server has not yet produced metrics, that instance will be skipped and noted in the output.

**Step 3: Querying for Metrics over a Specific Time Period**

The above code gives you all historical metrics, which can be a lot. often, you only care about a recent period, which can be achieved via parameters on `get_metrics()`. Here's how to specify a time window.
```python
import datarobot as dr
import os
from datetime import datetime, timedelta

# Same environment checks as above
try:
    dr.Client(token = os.environ['DATAROBOT_API_TOKEN'], endpoint=os.environ['DATAROBOT_ENDPOINT'])
except Exception as e:
    print(f"Error connecting to datarobot: {e}")
    exit()

deployment_id = "YOUR_DEPLOYMENT_ID" #Replace this with your deployment ID
deployment = dr.Deployment.get(deployment_id)

instance_metrics_timespan = {}
# Define time range (for last day)
end_time = datetime.utcnow()
start_time = end_time - timedelta(days=1)

for instance in deployment.get_prediction_server_instances():
    print(f"Checking instance ID: {instance.id}")
    metrics = instance.get_metrics(start_time = start_time, end_time = end_time)
    if metrics:
        instance_metrics_timespan[instance.id] = {}
        for metric in metrics:
                instance_metrics_timespan[instance.id][metric.metric_name] = {
                    "value": metric.value,
                    "start_time": metric.start_time,
                    "end_time": metric.end_time
                }
    else:
        print(f"No metrics found for instance {instance.id}")

print(f"Retrieved metrics from all instances within specified time frame: {instance_metrics_timespan}")
```
*Code Snippet 3: Fetching Instance Metrics within a Timespan*

This snippet is quite similar to the last one, but this time we set a timeframe, one day in this example using datetime operations. We pass the `start_time` and `end_time` arguments to the `get_metrics()` method, which returns the available metrics within that range. You can modify this to change the timeframe you are interested in.

It's useful to note that you’re not limited to common metrics. Depending on the model and the evaluation process in DataRobot, there may be other, specialized metrics available such as custom metrics, for example. The `metric.metric_name` in the examples will tell you what is available, allowing you to filter what you need. Also, it’s worth noting that some metrics will be unavailable if actuals data isn’t being collected by the deployment configuration, usually because it is disabled.

For further reading, I’d recommend referring to the official DataRobot API documentation. Start with the section detailing ‘Deployments’ and pay close attention to the ‘prediction server instances’ subsection. Also, the DataRobot Python API client’s documentation is invaluable. DataRobot also produces various practical guides and best practices documents (often found on their resource pages) which outline more sophisticated methods for model monitoring. For a deeper dive into metric interpretation, the book *Statistical Methods for Machine Learning* by Jason Brownlee is quite informative, although it is not specific to DataRobot it is a great foundational text for understanding metrics, and the documentation around the metrics DataRobot provides is often good in detailing what they do.

That should give you a solid start for pulling accuracy metrics via the api. Remember to replace the `YOUR_DEPLOYMENT_ID` placeholder with the actual deployment ID and always check that your API token and endpoint are set correctly. Also, error handling is important in a production environment, so think about how you'll manage missing deployments or unexpected responses. Good luck with your metrics retrieval!
