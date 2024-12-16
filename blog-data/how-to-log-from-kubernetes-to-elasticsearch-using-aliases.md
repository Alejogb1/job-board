---
title: "How to log from Kubernetes to ElasticSearch using aliases?"
date: "2024-12-16"
id: "how-to-log-from-kubernetes-to-elasticsearch-using-aliases"
---

Okay, let's tackle this. I've spent a fair amount of time wrangling Kubernetes and Elasticsearch together, and logging is one of those areas where things can quickly become… well, let’s just say *complex*. Managing indices with proper naming conventions and avoiding schema conflicts is crucial, and using aliases provides an elegant solution. It's not always straightforward, so let me walk you through how I typically approach this, based on real experiences.

The core idea here isn't merely sending logs; it's about managing those logs effectively within Elasticsearch. We’re aiming for a setup that's scalable, maintainable, and allows for easy querying. Instead of writing directly to concrete index names like `kubernetes-logs-2024-07-26`, we'll be logging to aliases such as `kubernetes-logs-write` and then subsequently querying using aliases like `kubernetes-logs-read`. This gives us that crucial abstraction layer we need for roll-overs and other index management tasks.

First, understand why aliases are so valuable. In a real-world Kubernetes environment, you likely have logs accumulating at a significant rate. If you continuously write to the same index, you encounter several issues: The index becomes huge and less performant; managing the schema or index settings becomes tedious; and rolling over indices based on size or time requires a whole bunch of manual changes. Aliases solve this. We can re-point the *write* alias to a new index periodically, while queries can always be routed through the *read* alias, encompassing all relevant indices. This decoupling simplifies index management significantly.

Let's examine how this is typically done in practice. We’ll be working with Fluentd (or Fluent Bit, depending on your specific setup), a popular log aggregator, and assuming Elasticsearch is our target.

**Configuration Example 1: Fluentd to Elasticsearch with Aliases**

Here's a snippet of a Fluentd configuration file, specifically the output section that handles Elasticsearch. This assumes your Fluentd input configuration is already configured to gather Kubernetes logs.

```yaml
<match kubernetes.**>
  @type elasticsearch
  host "${ENV['ELASTICSEARCH_HOST']}"
  port ${ENV['ELASTICSEARCH_PORT']}
  scheme https
  user "${ENV['ELASTICSEARCH_USER']}"
  password "${ENV['ELASTICSEARCH_PASSWORD']}"
  ssl_verify true
  logstash_format true
  logstash_prefix kubernetes-logs
  index_name kubernetes-logs-write
  <buffer>
    @type file
    path /var/log/fluentd/elasticsearch_buffer
    flush_interval 10s
    retry_max_interval 30
    retry_forever true
    chunk_limit_size 8m
    queue_limit_length 2048
  </buffer>
</match>
```

Notice the key line here: `index_name kubernetes-logs-write`. We are not specifying the concrete index name directly. This alias must already exist in Elasticsearch and configured to point to the currently active index for writing. On the Elasticsearch side, we would have something like:

```json
{
  "actions": [
    {
      "add": {
        "index": "kubernetes-logs-2024-07-26",
        "alias": "kubernetes-logs-write"
      }
    }
  ]
}

```
This indicates that `kubernetes-logs-write` points to the concrete index `kubernetes-logs-2024-07-26`. This is a simplified view; actual index creation and rollover logic will be handled separately, outside of Fluentd.

**Configuration Example 2: Elasticsearch Index Template and Rollover**

Now, let's see how we can ensure consistent mapping and rollover of these indices in Elasticsearch. We use a template to define index settings and mappings that will automatically apply when creating new concrete indices.

First, define a template to be applied to indices matching the pattern `kubernetes-logs-*`. Here’s an example Elasticsearch index template that defines the mappings for our `kubernetes` log entries:

```json
{
  "index_patterns": [
    "kubernetes-logs-*"
  ],
  "template": {
    "settings": {
      "index": {
        "number_of_shards": 3,
        "number_of_replicas": 1,
        "analysis": {
          "analyzer": {
            "lowercase_keyword": {
              "type": "keyword",
              "filter": [
                "lowercase"
              ]
            }
          }
        }
      }
    },
    "mappings": {
      "properties": {
         "container_name": { "type": "keyword" },
         "namespace": { "type": "keyword" },
         "pod_name": { "type": "keyword" },
         "log": { "type": "text" },
         "@timestamp": { "type": "date" }
        }
     }
  }
}
```

This template ensures that any new index matching the pattern `kubernetes-logs-*` will inherit these settings and mappings. Next, we would implement rollover. We will use an alias `kubernetes-logs-write` which always points to the current write index. We will set up index lifecycle management in Elasticsearch (ILM) to trigger index rollover at the desired time or based on size and make `kubernetes-logs-read` an alias which points to all the old indices. For example, the `kubernetes-logs-read` alias would be configured to include all concrete indices based on a matching pattern (`kubernetes-logs-*`), which means searching will access all indices. This ensures queries can retrieve data across all time ranges and not just the most recent index. This is typically configured outside Fluentd, within Elasticsearch.

**Configuration Example 3: Querying with Aliases**

Finally, when querying Elasticsearch, always use the *read* alias. This ensures your queries seamlessly access data across all relevant indices. Example query using `kubernetes-logs-read` alias in Kibana or Elasticsearch API:

```json
{
  "query": {
    "match": {
      "container_name": "my-container"
    }
  }
}

```

This query doesn't care which concrete index `my-container` is logged to. Because we are querying `kubernetes-logs-read` alias, Elasticsearch will search all the relevant indexes included in the alias.

Implementing this approach isn't a one-time thing; it requires ongoing maintenance and understanding. You’ll need to handle index lifecycle management (ILM) policies, monitor index sizes, and adapt based on log volume and query patterns. A robust ILM policy, which is beyond the scope of this response, is as essential as the configurations shown.

For further reading, I'd recommend diving into the Elasticsearch documentation, especially the sections on index templates, aliases, and index lifecycle management. The official *Elasticsearch: The Definitive Guide* (Clinton Gormley and Zachary Tong) is a good starting point, as is *Elasticsearch in Action* (Radu Gheorge, Matthew Lee Hinman, Roy Russo, and Peter Doolan). These resources go into considerably more detail about the concepts and configurations I’ve touched on. Understanding those principles is key to building a stable and scalable logging system with aliases. I've seen too many environments buckle under the weight of poorly managed logs; the approach I’ve detailed here, though requiring careful setup, prevents that scenario.
