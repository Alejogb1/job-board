---
title: "How can filebeat/elasticsearch performance be further optimized?"
date: "2025-01-30"
id: "how-can-filebeatelasticsearch-performance-be-further-optimized"
---
Filebeat, as a lightweight shipper, often becomes a performance bottleneck when dealing with high-throughput log ingestion. Optimizing its performance alongside Elasticsearch requires addressing several interconnected aspects, focusing on both Filebeat's configuration and Elasticsearch's ability to index the incoming data. I've personally encountered this challenge while managing a large-scale microservices deployment, where log volume spikes regularly and can overwhelm the ingestion pipeline. The key is to understand where the pressure points are and then apply targeted adjustments.

Firstly, examining Filebeat itself, several configuration parameters drastically impact performance. Specifically, the `spool_size` and `publish_async` settings within the `output.elasticsearch` section dictate how efficiently events are sent to Elasticsearch. The `spool_size` represents the maximum number of events Filebeat buffers in memory before flushing them to the output. A larger `spool_size` allows Filebeat to accumulate more events, leading to more efficient batch processing by Elasticsearch. However, excessively large values can strain Filebeat's memory resources, potentially causing it to crash or slow down due to excessive memory pressure. It is essential to correlate the `spool_size` with the overall available memory. The `publish_async` option, when set to `true`, allows Filebeat to continue processing events without waiting for acknowledgments from Elasticsearch for each batch, improving throughput, especially in cases of temporary network issues or Elasticsearch slowdowns. If set to false, it could lead to back pressure and slow down in a high-volume environment.

Furthermore, the behavior of Filebeat's processors, defined within the `processors` section of the configuration, significantly influences resource consumption. Some processor types, particularly those involving complex string manipulation like grok or dissect, can be computationally intensive. This is crucial to note; if you are doing some complex string parsing you will be using CPU cycles on the Filebeat box. Reducing the amount of processing burden can be achieved by moving more complex parsing logic to Elasticsearch Ingest Pipelines. Filebeat should act as a pure shipper as much as possible, focusing primarily on collecting and transmitting logs efficiently.

On the Elasticsearch side, optimizing index settings and cluster capacity are paramount. Incorrect index mappings, particularly dynamic mapping, can lead to inefficient indexing, impacting both indexing performance and disk space usage. Specifically, enabling dynamic mapping is very convenient, but it should be avoided in production environments for large amounts of data. It's more efficient to have carefully defined mappings ahead of time. Furthermore, improperly sized or configured shards affect how Elasticsearch distributes the indexing load. Too many small shards introduce unnecessary overhead, while too few large shards can lead to hotspots and inefficient utilization of cluster resources. Sizing the index shards and number is an art. It is best to look at the data in each index. It is commonly suggested that shards should be sized to 30-50GB but this depends upon the type of index.

Finally, disk I/O performance also is a major concern. Elasticsearch is very heavily dependent on disk I/O. Using Solid State Drives (SSDs) will considerably increase performance in large Elasticsearch deployments. If you are using spinning disks, then the performance can be heavily limited. If using a cloud provider, ensure that the type of disk you are using for the cluster is a SSD.

Here are three code examples demonstrating these concepts:

**Example 1: Filebeat `output.elasticsearch` configuration with optimizations.**

```yaml
output.elasticsearch:
  hosts: ["https://elasticsearch-node1:9200", "https://elasticsearch-node2:9200"]
  username: "elastic"
  password: "changeme"
  index: "filebeat-%{+yyyy.MM.dd}"
  spool_size: 2048 # Adjust as needed based on available memory and throughput
  publish_async: true
  compression_level: 3  # balance compression with CPU utilization, 3 is a good start.

```

*Commentary:* This snippet demonstrates optimized settings for the `output.elasticsearch` section. I've increased the `spool_size` for more efficient batching. Enabling `publish_async` significantly improves throughput by decoupling Filebeat from Elasticsearch acknowledgments. The `compression_level` can make a huge difference in bandwidth usage, and 3 is a good middle ground between CPU usage and network bandwidth. Note that compression can also increase the CPU usage on the Filebeat node so it may require testing to see if the improvement in bandwidth is worth the increase in CPU usage. This example uses HTTPS which is recommended. Remember to rotate and use real passwords in your deployment,

**Example 2: Filebeat processors, reducing processing complexity.**

```yaml
processors:
  - add_host_metadata: ~
  - add_cloud_metadata: ~
  #- dissect: # Commenting out complex processing, move this to an ingest pipeline.
   # tokenizer: "%{time} %{log_level} [%{module}] %{message}"
   # target_prefix: "log"
  - timestamp:
      field: "@timestamp"
      timezone: "UTC"
```

*Commentary:* This shows a simplified Filebeat processor configuration. Instead of processing complex log patterns with processors like `dissect`, I've commented it out and moved the parsing logic into an Ingest Pipeline in Elasticsearch. This shifts the processing burden away from Filebeat, allowing it to focus on shipping log data efficiently.  I have added an add_host_metadata and add_cloud_metadata processor that can give extra details to be used in searches. I have also added a timezone processor which forces all timestamps to UTC to avoid confusion in multiple timezones.

**Example 3: Elasticsearch index template with static mapping and shard settings.**

```json
{
  "index_patterns": ["filebeat-*"],
  "order": 1,
  "settings": {
    "index": {
      "number_of_shards": 3,
      "number_of_replicas": 1,
       "codec": "best_compression"
    }
  },
  "mappings": {
    "properties": {
      "@timestamp": {
        "type": "date"
      },
      "host": {
        "properties": {
          "name": {
             "type": "keyword"
           }
        }
      },
     "message": {
       "type": "text"
     }
    }
  }
}
```

*Commentary:* This defines a basic index template that will be applied to all indexes created with filebeat index patterns. It explicitly sets the `number_of_shards` to 3 and the `number_of_replicas` to 1. This example provides an optimized starting point; these settings will need further customization depending on the cluster size. The type of compression used can heavily affect the index size. `best_compression` will use the most compression, but it will cause the Elasticsearch cluster to use more resources. I have also explicitly defined some fields such as the `host.name` as a keyword which allows for quicker searching. I have also explicitly defined a `message` field as a text field. Note that using a text field may cause your indexes to be quite large. A text field will store the data in many forms in order for you to run full text queries such as matches or match phrase. If you don't need this full text searchability you may choose to make the `message` field a keyword field instead.

To further improve performance, I recommend investing time into understanding these aspects:

1.  **Elasticsearch Ingest Pipelines:** As mentioned, leverage Ingest Pipelines for heavy processing. They allow you to perform data transformation and enrichment on the Elasticsearch cluster itself, which is generally much more performant than handling it on each agent.

2.  **Shard Sizing:** Continuously monitor shard size and adjust the number of shards as needed. This will require monitoring the indices and determining if they are appropriately sized. Having too many small shards will lead to more resource usage on the cluster. However, too few large shards will lead to hotspots. The target size for a shard should be between 30-50GB.

3.  **Filebeat Monitoring:** Utilize the monitoring capabilities of Filebeat (through Metricbeat or other tools) to pinpoint performance bottlenecks within the agent itself, such as high CPU usage due to poorly performing regular expressions in processors.

4.  **Hardware and Resource Allocation:** Properly allocating resources, such as RAM and CPU, is imperative for both Filebeat and Elasticsearch. It is important to note that you should be monitoring the resources on the Filebeat server as well. Ensuring the resources are sufficient is important for optimal performance.

5.  **Network Performance:** A high bandwidth and low latency network is crucial for high throughput log ingestion. Ensure there are no network issues.

In my experience, carefully applying these optimizations, particularly focusing on moving processing load away from Filebeat and carefully configuring Elasticsearch indices, has yielded substantial improvements in log ingestion performance and overall stability, significantly minimizing data loss and system slowdowns. This is an iterative process; continuous monitoring and adjusting configurations based on observed system behavior is critical to maintain optimal performance.
