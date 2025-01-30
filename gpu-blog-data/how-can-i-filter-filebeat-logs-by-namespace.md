---
title: "How can I filter Filebeat logs by namespace?"
date: "2025-01-30"
id: "how-can-i-filter-filebeat-logs-by-namespace"
---
The capability to filter Filebeat logs based on the Kubernetes namespace is crucial for effective log management in containerized environments. Without precise filtering, log aggregation becomes overwhelming, hindering debugging and performance analysis. Having spent several years optimizing logging pipelines for large-scale deployments, I have found that leveraging Filebeat's processor functionality, specifically the `add_kubernetes_metadata` processor along with conditional filtering, is the most effective approach. This combination allows for fine-grained control over which log lines are forwarded based on their associated namespace.

The core concept revolves around enriching each log line with Kubernetes metadata before evaluating filtering conditions. The `add_kubernetes_metadata` processor, when properly configured, injects a wealth of information about the originating pod, including its namespace. This information is then made available for subsequent processing steps, such as conditional filtering. The key is understanding how to access this added metadata and craft filtering expressions using Filebeat's processing language. If the `add_kubernetes_metadata` processor cannot determine the namespace, the field will not be present, which is important to keep in mind.

Let’s break down the implementation using configuration snippets. First, consider the most basic setup for `add_kubernetes_metadata`. This processor should be defined within a `processors` array, which is typically nested within an input’s configuration.

```yaml
processors:
  - add_kubernetes_metadata:
      host: ${NODE_NAME}
```

In this basic example, I am utilizing an environment variable, `NODE_NAME`, to ensure Filebeat is targeting the correct Kubernetes node. This practice is important for multihost Kubernetes clusters. It tells `add_kubernetes_metadata` the local node to use when querying the Kubernetes API. Without this configuration, you’d rely on auto-discovery mechanisms which may not always accurately resolve the node. With this simple configuration, Filebeat will attempt to pull all the Kubernetes metadata from the local Kube API. Note that Filebeat requires appropriate Kubernetes permissions to do this.

Following the metadata enrichment, I would add a conditional processor to drop log lines that do not originate from a specified namespace.  Let’s assume that I want to only forward logs from the 'frontend' namespace. My enhanced processors configuration would then look like this:

```yaml
processors:
  - add_kubernetes_metadata:
      host: ${NODE_NAME}
  - drop_event:
      when:
        not:
          equals:
            kubernetes.namespace: "frontend"
```

Here, I introduce `drop_event` processor combined with a conditional `when`. The condition employs a `not` to negate the result of the `equals` operation. Thus, if the value associated with the key `kubernetes.namespace` is not “frontend”, the `drop_event` processor will discard the event before it reaches any output, preventing the logs from being forwarded to the configured output. Notice how the `add_kubernetes_metadata` processor has enriched the log line with the nested field `kubernetes.namespace`. This processor is case-sensitive so be aware of the casing in your namespace names.

However, strict matching to a single namespace might not be sufficient in many situations.  I frequently deal with scenarios where logs from several namespaces must be captured, for example, “frontend”,”backend” and “monitoring”. In such situations, using the `in` operator is more efficient than listing multiple `equals` conditions.

```yaml
processors:
  - add_kubernetes_metadata:
      host: ${NODE_NAME}
  - drop_event:
      when:
        not:
          in:
            kubernetes.namespace: ["frontend", "backend", "monitoring"]
```

The `in` operator checks if the value of `kubernetes.namespace` is present within the provided array, and it negates this result with `not`. The effect is logs are dropped if their namespace is not one of “frontend”, “backend”, or “monitoring”. This configuration is much easier to maintain and extend when adding more namespaces. It is important to remember to use an array of strings. If, for example, integer values were inadvertently passed in, the `in` operator would not match them against the string namespace value, resulting in all log lines being dropped.

Furthermore, the conditions can become complex. Imagine that, in addition to filtering on namespaces, I also wanted to filter on a label within the originating pod. Assuming a label named "app-tier" on the pod I could modify the condition. The `add_kubernetes_metadata` processor would make pod labels available in the `kubernetes.labels` field. The following conditional drop demonstrates the ability to filter on both namespace and label.

```yaml
processors:
  - add_kubernetes_metadata:
      host: ${NODE_NAME}
  - drop_event:
      when:
         or:
            - not:
                in:
                  kubernetes.namespace: ["frontend", "backend", "monitoring"]
            - not:
                equals:
                    kubernetes.labels.app-tier: "web"
```

Here, I’ve added an `or` operator with two nested conditions. The first condition continues to filter based on namespace as before, while the second condition requires the pod label `app-tier` to be `web`. This demonstrates how filtering can be applied using logical operators and can leverage various fields made available via the `add_kubernetes_metadata` processor.

In summary, precise filtering in Filebeat hinges upon: properly configuring the `add_kubernetes_metadata` processor to enrich log lines, using the nested data created by the processor as values to be used in conditions, and crafting conditions using the various operators such as `equals` and `in`. This approach minimizes resource utilization by reducing the volume of irrelevant logs, resulting in an efficient and manageable logging infrastructure. The result is that the volume of logs that reach the output is significantly reduced, which decreases costs and greatly simplifies log analysis, both critical in production environments.

For further exploration into Filebeat, I recommend reviewing the official Filebeat documentation provided by Elastic. Their section on processors is crucial, particularly regarding the `add_kubernetes_metadata` and conditional processor functionalities. Additionally, various community-driven forums and blog posts often contain practical examples and troubleshooting tips for common scenarios. Finally, an advanced understanding of the Elastic Common Schema (ECS) is invaluable for optimizing log analysis and querying in the broader Elastic stack. Understanding the expected field names and values improves your ability to write effective conditional processors.
