---
title: "How can I filter Filebeat logs by namespace?"
date: "2024-12-23"
id: "how-can-i-filter-filebeat-logs-by-namespace"
---

Alright, let's tackle this. Filtering Filebeat logs by namespace – it's a common requirement, particularly in containerized environments like Kubernetes, and I've definitely spent my share of time fine-tuning configurations for just this scenario. I remember one project in particular where we were dealing with a massive microservices architecture, and without proper namespace-based filtering, our logging pipelines would have been utterly overwhelmed. The key, as you might expect, lies within the Filebeat configuration, specifically its processing pipeline.

The crucial part is understanding that Filebeat processes events through a series of *processors*. These processors allow you to manipulate event data before it's sent to the output. We'll be leveraging a processor to identify and filter events based on the namespace information embedded within the log data.

Filebeat, by default, does not inherently understand Kubernetes namespaces. We need to instruct it where to find this information in our logs. Often, Kubernetes containers will write their logs including namespace data. This data could be in the json payload as metadata. Alternatively, if you have adopted something like the docker logging driver, this data may also be in the metadata associated with the docker log message.

The common approach relies on the *add_fields* and *drop_event* processors in Filebeat's configuration. We first use the add_fields processor to extract the namespace data, if needed and to add it as a top-level field. Then, we use a condition on the drop_event processor to filter for only the namespaces we're interested in. The add_fields processor allows us to extract data from the source of the log message, which might require parsing it according to the log format.

Let's walk through a few specific examples.

**Example 1: Namespace information in the json payload**

Suppose your log lines are in json format, and they contain a field like `kubernetes.namespace` within the json structure. Here's a sample Filebeat configuration snippet you might use:

```yaml
filebeat.inputs:
- type: container
  paths:
    - "/var/log/containers/*.log"
  processors:
    - decode_json_fields:
        fields: ["message"]
        target: "json_message"
        overwrite_keys: true
    - add_fields:
        target: "event"
        fields:
            namespace: "${json_message.kubernetes.namespace}"
    - drop_event:
        when:
          not:
            equals:
              event.namespace: "mynamespace"
```
In this example:

1. We configure filebeat to ingest logs using the `container` input type.
2. We first use a `decode_json_fields` processor to turn the `message` field, containing a json string, into a `json_message` field which becomes a top-level json object.
3. The `add_fields` processor adds a field named `namespace` to the `event` object by reading the `kubernetes.namespace` field from the decoded json. Here, we have assumed that your logs contain json structure within the log message and it has a nested json property called `kubernetes.namespace`.
4. The `drop_event` processor discards any events where the `event.namespace` field is not equal to “mynamespace”. This is the filtering mechanism at work. This processor does an explicit compare. It does not do substring matching, so your namespaces will need to match exactly for them to be ingested.

**Example 2: Namespace information in the docker log metadata**

Let's consider a different scenario, where the namespace is present within the docker metadata, not the json message itself. Here’s how you would handle this in filebeat's configuration:

```yaml
filebeat.inputs:
- type: container
  paths:
    - "/var/log/containers/*.log"
  processors:
    - add_fields:
        target: "event"
        fields:
            namespace: "${container.labels.io_kubernetes_pod_namespace}"
    - drop_event:
        when:
          not:
            or:
              - equals:
                  event.namespace: "mynamespace1"
              - equals:
                  event.namespace: "mynamespace2"
```

In this case:
1. Again, we set filebeat's input type to `container`.
2. The `add_fields` processor extracts the namespace from the `container.labels` which are docker labels, specifically from the label `io_kubernetes_pod_namespace` associated with the container. This assumes that you are using a logging driver that will attach metadata to the log. Most standard logging drivers do provide this capability.
3. Then the `drop_event` processor discards events if the namespace field in the `event` object is not equal to `mynamespace1` or `mynamespace2`. This example uses an `or` condition that will ingest log events from both `mynamespace1` and `mynamespace2`. This is useful if you are trying to ingest logs from multiple namespaces.

**Example 3: Namespace information with regex parsing**

Suppose your logs contain a namespace as a string within the message, but not as a field, and not within the docker metadata. Here’s how to handle this:

```yaml
filebeat.inputs:
- type: container
  paths:
    - "/var/log/containers/*.log"
  processors:
    - dissect:
        tokenizer: '%{} %{namespace} %{}'
        field: "message"
    - drop_event:
        when:
          not:
            contains:
              event.namespace: "my"
```

In this example:
1. We again define the `container` input.
2. The `dissect` processor uses a simple tokenizer pattern `%{} %{namespace} %{}` where `%{}` matches a number of characters that we are not interested in, and `%{namespace}` captures the next set of characters as the `namespace` field. This assumes that our logs are formatted in a way that the namespace always comes in the same position. There are more powerful processors that support a variety of regex expressions. We can also use the `regex` processor instead of dissect.
3. Finally, `drop_event` discards all events for which the namespace field does not *contain* the string `my`. In this example, the namespace field must have a string including the word `my` as part of the namespace. This filter is more flexible than using an explicit `equals` statement as it can match `mynamespace1` or `mynamespace2`.

Important things to consider:

*   **Performance:** Processor logic does consume resources. If your log volume is high, meticulously crafted regex expressions or complex parsing logic can have an impact. Start simple, test thoroughly, and only add complexity where it is essential. In my experience, the less sophisticated `dissect` approach is more efficient than a complex `regex` processor.
*   **Flexibility:** The examples provided have explicit namespace names. If you need more advanced logic, you might consider using the *script* processor, but be aware that writing and debugging these can increase configuration complexity. This would involve writing javascript and integrating it into the pipeline.
*   **Error Handling:** Ensure your configurations include some error handling. If the `add_fields` processor can't find a namespace it will likely generate a null value for the namespace field, which might impact your filtering logic. Make sure to set a reasonable default value.
*   **Testing:** Filebeat has excellent configuration testing tools. Before deploying these configurations to a live production environment, always test your configuration locally, ideally against representative sample logs. You can do this with `filebeat test config`, and `filebeat test output`.
*   **Upgrades:** Keep your version of filebeat up to date to ensure that you are taking advantage of the latest capabilities and bug fixes.

As for resources, I highly recommend diving into the official Filebeat documentation, which is quite comprehensive. "The Logstash Book" by Jordan Sissel also provides a lot of depth on log processing concepts, although it focuses on Logstash, much of the theory and many of the techniques are applicable to Filebeat. And, for a deep understanding of kubernetes logging practices, check out "Kubernetes in Action" by Marko Lukša.

Filtering logs by namespace with Filebeat is a powerful way to keep your logging pipeline sane. It does require some configuration, but the trade-offs in terms of efficiency and maintainability are well worth it. Just remember to iterate, test, and, as always, start with a clear understanding of what you're trying to accomplish.
