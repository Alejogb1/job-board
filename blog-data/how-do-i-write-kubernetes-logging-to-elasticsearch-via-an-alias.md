---
title: "How do I write Kubernetes logging to Elasticsearch via an alias?"
date: "2024-12-16"
id: "how-do-i-write-kubernetes-logging-to-elasticsearch-via-an-alias"
---

Alright, let’s tackle this. Kubernetes logging to Elasticsearch, particularly via an alias, can initially seem complex, but it's a fairly manageable setup once you understand the underlying pieces. I've certainly spent my share of evenings debugging this very scenario back in the day, particularly when we were transitioning from a simpler logging stack. The core challenge usually revolves around ensuring the logs are not only routed correctly but are also easily queryable and maintainable over time. A direct index approach can get messy quite fast. Using aliases provides that much-needed abstraction.

The objective here is straightforward: instead of sending logs to a specific index, we route them to an alias. This alias can then point to one or more actual indices. This is powerful for several reasons. Primarily, it allows for easy index rollover (think daily, weekly, or monthly indices) without needing to change the logging configuration in your cluster. It also simplifies searching – you query the alias, and Elasticsearch takes care of the rest. It's like having a single, convenient entry point for your logs while the underlying data is actually organized in a way that promotes both performance and manageability.

My past experience with this involved a large-scale microservices architecture. We initially started by directing logs directly into indexes based on service names. As you can imagine, this approach became unwieldy as the number of services increased and we started to encounter search performance issues due to a few very large indexes. The transition to aliasing was our solution, and it worked wonders.

Let's break down how to get this set up, focusing on the practical aspects. We'll use a fairly standard logging stack involving Fluent Bit as our log collector, a custom configuration to direct logs to an alias, and Elasticsearch as our backend.

First, Fluent Bit needs to be configured to send logs to Elasticsearch. The key here is to not specify a static index but rather the alias name. Here’s a basic configuration snippet demonstrating this:

```yaml
[OUTPUT]
    Name            es
    Match           *
    Host            elasticsearch.service.namespace.svc.cluster.local
    Port            9200
    Index           log-alias
    Type            _doc
    Logstash_Format On
    Retry_Limit     3
```

Notice the `Index` parameter? We're using `log-alias` here, *not* a concrete index like `app-logs-2024-07-14`. This is crucial. Fluent Bit will now send all matched logs to this alias. Now, on the Elasticsearch side, you need to ensure that this alias points to a real index (or multiple indices, which is usually the case). This is done using the Elasticsearch api.

Second, we need to create the alias and point it to an actual index. For this, we would use the Elasticsearch API. Assuming you’ve configured basic auth to your ES cluster, or are using a k8s service to communicate with it, you'll use a tool like `curl` to interact with the Elasticsearch API. Below is an example of creating a single index and mapping the alias to it:

```bash
curl -X PUT \
  -H 'Content-Type: application/json' \
  -u 'user:password' \
  "https://elasticsearch.service.namespace.svc.cluster.local:9200/app-logs-2024-07-14" \
  -d '{
    "mappings": {
      "properties": {
        "@timestamp": { "type": "date" },
        "log": { "type": "text" },
        "kubernetes": { "type": "object" }
      }
    }
  }'

curl -X POST \
  -H 'Content-Type: application/json' \
  -u 'user:password' \
  "https://elasticsearch.service.namespace.svc.cluster.local:9200/_aliases" \
  -d '{
    "actions": [
      { "add": { "index": "app-logs-2024-07-14", "alias": "log-alias" } }
    ]
  }'
```

This example first creates an index named `app-logs-2024-07-14` with some basic mappings (timestamps, message, kubernetes metadata). It then adds this index to the alias `log-alias`. This way, when Fluent Bit sends data to `log-alias`, it's actually being written into this specific index. Remember that you will typically do this automatically, probably via a lifecycle management process, instead of doing it manually. The index name and mappings above are merely examples.

The real strength of alias usage comes when we need to do index rotation. For example, once we get to a new day, we can create a new index and update the alias to point to that new index. Here's a simplified illustration of that using `curl` again:

```bash
curl -X PUT \
  -H 'Content-Type: application/json' \
  -u 'user:password' \
  "https://elasticsearch.service.namespace.svc.cluster.local:9200/app-logs-2024-07-15" \
  -d '{
    "mappings": {
      "properties": {
        "@timestamp": { "type": "date" },
        "log": { "type": "text" },
        "kubernetes": { "type": "object" }
      }
    }
  }'


curl -X POST \
  -H 'Content-Type: application/json' \
  -u 'user:password' \
  "https://elasticsearch.service.namespace.svc.cluster.local:9200/_aliases" \
  -d '{
    "actions": [
      { "remove": { "index": "app-logs-2024-07-14", "alias": "log-alias" } },
      { "add": { "index": "app-logs-2024-07-15", "alias": "log-alias" } }
    ]
  }'
```

This snippet creates a new index `app-logs-2024-07-15`, then removes the old index `app-logs-2024-07-14` from the `log-alias` and adds the new index. Importantly, notice that this update process is an atomic operation from an alias perspective. There won’t be downtime or data loss from the logging client perspective because, from Fluent Bit's view, it's always sending to `log-alias`.

This three-step example, although simplified, demonstrates the core workflow. In practice, you would likely integrate this with a more automated lifecycle management process, such as using Curator or Index Lifecycle Management (ILM) policies within Elasticsearch itself.

For further reading, I would strongly recommend examining *Elasticsearch: The Definitive Guide* by Clinton Gormley and Zachary Tong. It provides an exhaustive look into Elasticsearch and explains concepts like aliases, mapping, and index management in depth. Furthermore, consider studying the official Elasticsearch documentation for the most up-to-date information on index lifecycle management policies and relevant apis. To fully understand the logging pipeline itself, the Fluent Bit documentation on outputs and configuration parameters is an invaluable resource.

In closing, logging to Elasticsearch via an alias offers considerable benefits in terms of scalability and maintainability. By using aliases, you abstract away the complexity of individual indices, which ultimately makes your logging infrastructure more robust and easier to manage. Remember, the devil is in the details, so careful planning and thorough understanding of the involved components (Fluent Bit and Elasticsearch) are key to a successful deployment.
