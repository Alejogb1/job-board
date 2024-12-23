---
title: "How do I process json logs with Grafana and Loki?"
date: "2024-12-23"
id: "how-do-i-process-json-logs-with-grafana-and-loki"
---

Let’s dive straight in. Processing json logs with Grafana and Loki, it's something I've spent a fair bit of time tackling, both in previous roles handling massive application backends and in my own infrastructure projects. The beauty of Loki is its label-based approach, but when you're dealing with unstructured json logs, you need a strategy to extract meaningful fields that you can then query and visualize within Grafana. It's not always as straightforward as it seems.

My first encounter with this particular challenge was while managing a microservices architecture where each service spat out logs in its own flavor of json. It was chaos, to say the least. We weren't utilizing structured logging practices then, which made the switch to Loki slightly bumpy initially. Thankfully, after a few iterations, we developed a robust methodology that's served me well since.

The fundamental issue is that Loki primarily indexes log lines based on labels, not the content within each line. To efficiently search, filter, and visualize your json data, you need to tell Loki *how* to interpret the json and turn specific values into labels or extract them into usable fields. This is primarily achieved through the `json` logfmt parser in Loki, which is incredibly powerful once you get the hang of it.

The key lies in configuring the promtail agent, which is responsible for shipping logs to Loki. Within your promtail configuration, you define the pipeline that describes how each log source should be processed. Let's look at the main parts you'll be using: the `json` stage and the `labels` stage to apply extracted labels, and `template` for dynamic label creation.

Consider a simplified scenario. Assume we're dealing with logs that look something like this:

```json
{"timestamp": "2024-01-20T10:00:00Z", "level": "INFO", "message": "User login successful", "user_id": 12345}
```

Here’s a basic promtail configuration snippet showcasing how to extract `level` as a label:

```yaml
scrape_configs:
  - job_name: json_logs
    static_configs:
      - targets:
          - localhost
        labels:
          job: my-json-app
          __path__: /var/log/my-app/*.log
    pipeline_stages:
      - json:
          expressions:
            level: level
      - labels:
          level:
```

In this example, the `json` stage tells promtail to parse the incoming log lines as json. The `expressions` section specifies that we want to extract the value of the "level" field, mapping it to a variable also named `level` (though you can choose different names if you prefer, on the left-hand side). Then, the `labels` stage takes the extracted `level` variable and converts it into a Loki label of the same name.

Now, you can query Loki using logql to filter by this label in Grafana. For instance, ` {job="my-json-app", level="ERROR"}` would retrieve all error messages associated with your json logs.

This is fairly basic, however. What if, for instance, you have varying types of logs within the same file, or you need more elaborate transformations? Let’s elevate it with a second example:

```json
{"type":"auth", "timestamp": "2024-01-20T10:00:00Z", "user": {"id": 12345, "username": "test_user"}, "event": "login_success"}
```
```yaml
scrape_configs:
  - job_name: complex_json_logs
    static_configs:
      - targets:
          - localhost
        labels:
          job: my-auth-app
          __path__: /var/log/auth-app/*.log
    pipeline_stages:
      - json:
          expressions:
            event_type: type
            user_id: user.id
            username: user.username
      - labels:
         event_type:
         user_id:
      - template:
          source: username
          template: "{{ . | truncate 10 }}"
          target: short_username
      - labels:
         short_username:
```

Here, we are extracting the `type` field and the nested fields `user.id` and `user.username`. The dot notation inside the `json` expressions handles nested structures elegantly. The first `labels` stage converts `event_type` and `user_id` into Loki labels directly. Then, we use a `template` stage. Here I’m demonstrating a practical real-world use case: we truncate the username field to its first 10 characters because lengthy usernames will result in high cardinality labels (which slow down Loki and aren’t practical). Then we extract this truncated version of username to the label `short_username`.

This underscores a critical point: choosing which fields to convert into labels requires careful consideration. Labels are indexed by Loki and significantly affect performance. High-cardinality labels (those with many unique values) should be avoided whenever feasible. In this case, we decided that user id is okay for labels but username is not, so we truncate it. It's about making intelligent tradeoffs.

Finally, let’s look at how you can use the `json` stage combined with `match` stages to filter based on log properties. This example also demonstrates how to keep original JSON as is, in the log lines that goes to Loki, while extracting and processing some JSON properties for labels.

```json
{"type":"db", "timestamp": "2024-01-20T10:00:00Z", "query": "SELECT * FROM users WHERE id=123", "duration": 0.05}
```

```yaml
scrape_configs:
  - job_name: filtered_json_logs
    static_configs:
      - targets:
          - localhost
        labels:
          job: my-db-app
          __path__: /var/log/db-app/*.log
    pipeline_stages:
      - match:
          selector: '{job="my-db-app"}'
          stages:
            - json:
                expressions:
                    query: query
                    duration: duration
            - labels:
                duration:
      - match:
          selector: '{job="my-db-app"}'
          stages:
            - json:
                expressions:
                    query: query
                    duration: duration
            - template:
                source: query
                template: "{{ . | regexReplaceAll `WHERE id=[0-9]+` `WHERE id=XXX` }}"
                target: redacted_query
            - labels:
                redacted_query:
```

In this example, the first `match` stage applies the json processing and label extraction, making duration available as a label. The second match statement keeps original query, also extracts duration, but now also sanitizes the query, using template engine regex feature, before extracting the `redacted_query` to a label. Here's the difference. The log entry pushed to Loki, will contain the original JSON, however, you can query by `redacted_query`, which contains `SELECT * FROM users WHERE id=XXX`, or by duration. This shows that your processing pipeline can be as complex and as detailed as you need it.

These examples should provide a solid starting point. For further study, I strongly recommend looking into the official Loki documentation, specifically the sections on promtail configuration and pipeline stages. "Site Reliability Engineering" by Betsy Beyer et al., and "The Logstash Book" by Jordan Sissel are excellent resources for gaining a broader understanding of log management concepts and best practices. There's also a very useful blog by Tom Wilkie (one of the maintainers for Loki project) if you search "Tom Wilkie Loki blogs," that goes deep into various aspects of Loki and best practices around it. Experimentation is also crucial – try different extraction techniques, play with templates, and monitor the impact on Loki performance. Start with small, focused adjustments and gradually build your pipeline. It's a rewarding process and makes managing logs so much more effective.
