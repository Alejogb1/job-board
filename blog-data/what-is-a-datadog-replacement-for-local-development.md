---
title: "What is a Datadog replacement for local development?"
date: "2024-12-15"
id: "what-is-a-datadog-replacement-for-local-development"
---

alright, so you're looking for a datadog-like setup for your local dev environment, right? i've been down that rabbit hole a few times, let me tell you. datadog's great for production, no doubt, but when you're hacking away locally, it's often overkill and adds unnecessary complexity. i've always found that simplicity is the key for rapid prototyping and debugging local stuff.

i remember back in my early days, i was working on this microservice architecture for a toy project. i tried using datadog agents locally, and while it technically *worked*, it was just too much overhead. resource hogging, the constant configuration headaches, it was like trying to use a sledgehammer to crack a nut. that's when i started exploring lighter alternatives. i learned it’s better to have simpler tools when you are working locally.

so, what can you use? well, it really depends on what aspects of datadog you need to replicate. are you mainly after metrics, logs, tracing, or all of the above? for most local setups, you can achieve a lot with readily available open-source tools.

let's start with metrics. datadog lets you monitor various performance metrics of your applications. a good local replacement for that is prometheus. it’s an amazing time-series database and a monitoring system that's extremely popular in the devops world. it’s very easy to set up locally and scrape your app's metrics. your application can expose its metrics using the prometheus client libraries. i've always found the prometheus python client library really straightforward, here's a quick example of that:

```python
from prometheus_client import start_http_server, Summary, Counter
import random
import time

# create a counter to count how many requests
request_counter = Counter('my_app_requests_total', 'Total number of requests')

# create a summary to time how long requests take
request_latency = Summary('my_app_request_latency_seconds', 'Latency of requests')

def process_request():
    """simulate a request that takes time"""
    start = time.time()
    # pretend to work
    time.sleep(random.random() / 2)
    end = time.time()
    request_latency.observe(end - start)
    request_counter.inc()


if __name__ == '__main__':
    start_http_server(8000)
    while True:
        process_request()
        time.sleep(random.random() / 4)
```

then you just fire up prometheus with a config that points to your app's exposed metrics endpoint and voila, you have local metric monitoring, and that's what I personally use. if you're into cloud native environments, there are prometheus operator and helm charts for kubernetes that are really neat and i've played with them a lot lately.

now, for logging, datadog's log management is very robust. but for local dev, often, a simple log aggregator is all you need. i would recommend using loki or even docker logs depending on your environment setup. loki is a log aggregation system designed to be cost-effective and easily scalable. it is similar to prometheus but for logs, not metrics, they are sister projects. it does indexing for labels. loki works really well with promtail, it is a log shipper. let’s see a quick example on how to use promtail and loki via docker compose:

```yaml
version: "3.7"

networks:
  loki:

services:
  loki:
    image: grafana/loki:2.9.2
    ports:
      - "3100:3100"
    networks:
      - loki
    command: -config.file=/etc/loki/local-config.yaml

  promtail:
    image: grafana/promtail:2.9.2
    volumes:
      - ./promtail.yaml:/etc/promtail/promtail.yaml
      - /var/log:/var/log
    networks:
      - loki
    depends_on:
      - loki
    command: -config.file=/etc/promtail/promtail.yaml
```

in the `promtail.yaml` file, you’d configure the log sources you want to ship to loki and then you can query logs by using label selectors in loki's interface. i find this approach very flexible for local experiments. again, this is what i use for all of my personal projects.

and if you want to add distributed tracing, datadog does tracing very well. for local, you can easily use jaeger. it's another cloud native open source tool for distributed tracing. the jaeger agent and collector can be deployed locally and you can integrate it with your application. you'd usually do this with a tracing library that sends the tracing information to the jaeger agent. i have used this before in an erlang service and i can show you a simple example in python using the opentracing standard:

```python
from jaeger_client import Config
from opentracing import tracer
from time import sleep
import random


def initialize_tracer():
    config = Config(
        config={
            'sampler': {
                'type': 'const',
                'param': 1,
            },
            'logging': True,
        },
        service_name='my-local-app',
        validate=True,
    )

    return config.initialize_tracer()

def main():
    jaeger_tracer = initialize_tracer()

    with jaeger_tracer.start_span('process_request') as span:
      span.set_tag('event', 'request started')
      sleep(random.random() / 2)
      with jaeger_tracer.start_span('database_call', parent=span) as db_span:
          db_span.set_tag('query', 'select * from users')
          sleep(random.random() / 4)
          db_span.log_kv({"event":"query finished"})
      span.log_kv({"event": "request finished"})

if __name__ == '__main__':
    main()
```

you start the jaeger all-in-one instance via docker and then you can see the traces of your application in its interface. it's incredibly useful to understand the request flow and debug bottlenecks. honestly, i’ve spent days trying to understand what exactly went wrong in a service without distributed tracing. with it it’s really a game changer. i think that's a super power that people do not know they are missing.

now, the combination of prometheus, loki, and jaeger gives you pretty much all the features you need from datadog for local development. the setup might sound involved at first, but it's quite straightforward once you get a hang of it, and, believe me, the benefits outweigh the initial setup cost. there’s something about having your own observability tools running on your machine that feels incredibly powerful. you can tweak them however you like and dive into the deepest corners of your application with full control.

some might ask 'but what about datadog alerts and dashboards?'. well, locally, i generally find those to be less necessary. if you really need those, you can always configure prometheus to alert based on certain metrics, and grafana can visualise those metrics. it's a bit more work, but it’s doable if you have specific requirements. i remember one time, i spent hours staring at a dashboard trying to figure out why my service was lagging only to find out there was a typo in a config. that's why i prefer more verbose logging locally to fix those silly mistakes quickly. i guess that's a lesson learned the hard way.

for learning more about these tools, i highly suggest checking out the official prometheus documentation, the loki documentation, and the jaeger documentation, they are excellent. you might also want to look into "site reliability engineering" by betsy beyer, chris jones, jennifer petoff, niall richard murphy for a more in-depth look into this area. and for more conceptual learning i also recommend "distributed systems concepts and design" by george coulouris, jean dollimore, tim kindberg, and gordon blair. it's the bible of distributed systems. you will never look at microservices the same way.

in conclusion, you don't necessarily need datadog locally. a combination of prometheus, loki, and jaeger will likely get you much closer to what you really need, while keeping resource usage low and the configuration simple. and remember, sometimes simpler tools are better tools. don't add complexity just for the sake of it.
