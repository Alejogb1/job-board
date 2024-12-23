---
title: "family metrics scylla implementation?"
date: "2024-12-13"
id: "family-metrics-scylla-implementation"
---

 so family metrics scylla implementation right Been there done that let me tell you about the joy and slight despair I encountered when tackling this beast of a task First things first why family metrics Right so in Scylla context we are talking about metrics that are inherently grouped together right metrics that belong to the same “family” think CPU usage network traffic disk I/O these all have multiple variations or subtypes. Instead of managing them as totally separate singletons you’d want to group them and access them together this grouping can simplify things at a system level and your code too.

Why did I need this I used to work for this company and I was dealing with a pretty massive Scylla cluster back then huge number of nodes and data throughput and the monitoring was a complete mess trying to find the correct metrics in the monitoring systems was hell I had metrics coming at me from every direction and it felt like trying to find a specific grain of sand in the Sahara Desert. I started looking into how Scylla actually manages this internally and then started brainstorming solutions. It became pretty clear quickly that we would need a more structured approach.

The Scylla implementation as far as I saw it and I dug pretty deep was based on a tagging system every metric was tagged with its family and then those families are processed and exposed accordingly. Its all based in c++ and not exposed that much through the API but there are some exposed metrics that give us hints of how they do it so I thought I could use the same approach in my own monitoring stack. I mean it worked for them right it should for me too.

Now lets talk code because we are all developers here right nobody wants a history lesson all the time I'm gonna give you a very simplified example here using python. It could be easily adapted to any other language with a library that can push metrics like Prometheus or StatsD. I chose Python because I think is more user-friendly for a general audience.

First we need a structure to hold our metrics family data that is going to make our life easier and will allow to add more metrics of different types in the future I suggest something like this:

```python
class MetricFamily:
    def __init__(self, name):
        self.name = name
        self.metrics = {}

    def add_metric(self, metric_name, value, labels=None):
        self.metrics[metric_name] = {"value": value, "labels": labels or {}}

    def get_metric(self, metric_name):
        return self.metrics.get(metric_name)

    def get_all_metrics(self):
      return self.metrics
```
This class acts as our metric family container. It has methods for adding metrics fetching specific metrics and also fetching all the metrics which is going to be used in the next snippet.

Now for an example of actually using it in a monitoring system I used something like this. I was sending metrics to Prometheus at the time:

```python
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway
import time

registry = CollectorRegistry()

cpu_family = MetricFamily("cpu_usage")
network_family = MetricFamily("network_traffic")

def collect_metrics():
    # Simulating collecting actual metrics
    cpu_family.add_metric("user", 15.5)
    cpu_family.add_metric("system", 5.2, {"core":"1"})
    cpu_family.add_metric("idle", 75.3)
    network_family.add_metric("bytes_in", 12345)
    network_family.add_metric("bytes_out", 67890, {"interface":"eth0"})

    # Process and expose the metrics
    for family in [cpu_family, network_family]:
        for metric_name, metric_data in family.get_all_metrics().items():
           value = metric_data.get("value")
           labels = metric_data.get("labels")

           gauge = Gauge(f"{family.name}_{metric_name}", f"Description for {metric_name} metric from {family.name}", registry=registry, labelnames=labels.keys())
           gauge.labels(**labels).set(value)

    #Push to prometheus
    push_to_gateway("localhost:9091", job="my_app", registry=registry)

if __name__ == "__main__":
    while True:
        collect_metrics()
        time.sleep(5)
```

This code creates two metric families cpu usage and network traffic then it adds sample metrics to them and it iterates over each family and then each metric within that family creating a Prometheus gauge for each one and pushing the results to a Pushgateway. The labels were also implemented so you can add extra information about your metric. You could extend this quite easily to create families of metrics for anything else.

Now lets talk about Scylla specifically as you were asking the actual Scylla monitoring exposes many metrics related to disk I/O which are grouped as a family. You can see that by querying the `/metrics` endpoint they are exposed in Prometheus format as well so if you inspect the text output of that endpoint you will quickly see that is the case. Scylla doesn’t use this structure in a way that they are actual python classes but its easy to see how is done. They use tags and then expose those in a family grouping format.

Another example this one is going to show how to organize the data for metrics that can be represented as a histogram. This can happen with latency metrics. Think about request latency or read latency. Those can be aggregated as histograms. This example is also pushing to prometheus.

```python
from prometheus_client import CollectorRegistry, Histogram, push_to_gateway
import time
import random

registry = CollectorRegistry()

latency_family = MetricFamily("request_latency")

def collect_latency_metrics():
    # Simulating collecting actual metrics
    latencies = [random.random() for _ in range(100)] # generate random latency values
    latency_family.add_metric("request_latency",latencies, {"method":"GET"}) # Add latency as an array of values

    #Process and expose metrics
    for metric_name, metric_data in latency_family.get_all_metrics().items():
         values = metric_data.get("value") # array of values
         labels = metric_data.get("labels") # extra information about metric

         hist = Histogram(f"{latency_family.name}_{metric_name}", f"Description for {metric_name} metric from {latency_family.name}", registry=registry, labelnames=labels.keys())
         for v in values:
            hist.labels(**labels).observe(v)

    #Push to prometheus
    push_to_gateway("localhost:9091", job="my_app", registry=registry)

if __name__ == "__main__":
    while True:
        collect_latency_metrics()
        time.sleep(5)
```
This code creates a latency metric family that adds a list of latencies to the metric itself. Then when exposing the data the histogram is built and filled with the latency values. This is pretty common to see in metrics systems because this can give you a better representation of the actual behavior of the system than just an average or a max value. Also Prometheus can create nice graphs with histogram data.

so that’s pretty much it the idea is to group metrics of the same type into a family so your monitoring code is cleaner and its easy to keep track of what is being monitored. There isn't a single right way to do it and it will depend on what you are actually trying to achieve with your monitoring.

Now for a few tips from my past experiences here. When designing your metrics make sure you define labels wisely. Too many labels will increase cardinality and slow down your queries and too few labels will not give you enough data to actually troubleshoot anything. Its all about finding the right balance. Second thing try to use well-defined metric names so other people can understand what they mean and what they represent. We have all worked with code that uses strange metrics names and that is never fun to debug.

One time I spent almost a whole week troubleshooting a bug because of confusing metric names turns out the issue was not in the database but in a misconfiguration of the application server and if I had good metric names I would have known that in minutes and not in days. That's the joke that's the only one I am doing.

For resources I highly recommend looking into the original Prometheus documentation and also papers about time-series databases. Understanding the theoretical concepts behind metrics and time series will give you a better perspective on how to approach the practical side of things. Also familiarize yourself with the OpenTelemetry specification that is rapidly becoming an standard in metrics. This standard also defines and recommends how metrics should be structured and handled. You can also look for papers related to the monitoring systems that major companies are using there is a lot of knowledge that can be obtained and you can see how they tackled similar issues. A book like "Site Reliability Engineering" from google is a good place to start understanding monitoring in general.

 thats it. Let me know if there is anything else I can help you with.
