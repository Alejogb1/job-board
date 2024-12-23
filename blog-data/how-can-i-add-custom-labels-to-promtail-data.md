---
title: "How can I add custom labels to Promtail data?"
date: "2024-12-23"
id: "how-can-i-add-custom-labels-to-promtail-data"
---

Alright, let's tackle custom labels in Promtail. It's something I’ve definitely navigated before, and it can be surprisingly nuanced depending on your setup. Back in '18, we were moving a monolith to a microservices architecture and, predictably, the logs became a distributed mess. That's where a deep understanding of relabeling in Promtail became critical. The out-of-the-box log lines simply weren't cutting it for effective querying and alerting. So, here’s the approach I used and what I’ve learned since:

Fundamentally, Promtail's relabeling mechanism is how you add, modify, or drop labels before log data gets shipped to Loki. You aren't directly altering the *content* of the log, but rather enriching the associated metadata. This is incredibly powerful for filtering, aggregation, and ultimately, getting value out of your logs. The core idea is to use a series of rules within your Promtail configuration to extract information from the log stream and assign them as labels.

At its most basic level, Promtail configuration files describe how it reads log files or streams. The `scrape_configs` section dictates *where* Promtail gets the logs, and the `relabel_configs` sections are where you specify the transformations. The rules are applied sequentially, so the order matters.

A typical `relabel_configs` block contains several key components: `source_labels`, `regex`, `target_label`, `replacement`, and `action`. Let’s break those down.

*   `source_labels`: A list of labels whose values will be concatenated and used as the input string for the regex. If this is omitted, it defaults to the log line itself (`__line__`).
*   `regex`: A regular expression pattern that is applied to the concatenated `source_labels` value.
*   `target_label`: The name of the label that will be created or modified.
*   `replacement`: The replacement string. Captured regex groups can be referenced using `"$1"`, `"$2"`, and so on. If the regex doesn't match anything, this is not applied, unless the `action` is `replace` (more on that in a bit).
*   `action`: Specifies how to handle the regex match. Common values are `replace`, `keep`, `drop`, `hashmod`, `labelmap`, `lowercase`, `uppercase`, `keepequal`, `dropequal`.

Now, let’s get to some practical examples.

**Example 1: Extracting a service name from the log line**

Imagine log lines that consistently include a service identifier at the beginning, like so: `[service-a] Request received...`. To extract `service-a` as a label, the config might look like this:

```yaml
scrape_configs:
  - job_name: system
    static_configs:
      - targets:
          - localhost
        labels:
          job: system
          __path__: /var/log/my_app.log
    relabel_configs:
      - source_labels: ['__line__']
        regex: '\[(.*?)\](.*)'
        target_label: 'service'
        replacement: '$1'
        action: replace
```

Here, the `regex` captures everything between the square brackets. The `replacement` pulls that captured group into the `service` label, and because action is set to `replace`, even if the match fails, no transformation happens on the original value. All of this would be applied to each incoming line for this particular scrape configuration.

**Example 2: Labeling based on the file path**

Let’s say you're running multiple applications, and each writes logs to a different subdirectory. You don't necessarily have a clear identifier in each log, but you know the path structure. You can add a label that uses the filename, but you need to avoid hard coding the directories. Let's assume file structure is `/var/log/app1/access.log`, `/var/log/app2/access.log`... and you need `app1` and `app2` respectively as labels.

```yaml
scrape_configs:
  - job_name: access_logs
    static_configs:
      - targets:
          - localhost
        labels:
          job: access
          __path__: /var/log/*/*.log
    relabel_configs:
      - source_labels: ['__path__']
        regex: '/var/log/(.*?)/.*\.log'
        target_label: 'app'
        replacement: '$1'
        action: replace
```

In this snippet, the `source_labels` now target the `__path__` label which contains the full path to the log file. The regex captures the directory between `/var/log/` and `/`, which effectively extracts the name of the application, and applies this to a label called `app`.

**Example 3: Dropping labels**

Sometimes you have a label you don't want. Let's assume your logs have a `kubernetes_namespace` label, but you don’t care about namespace specific data and want to drop this label:

```yaml
scrape_configs:
  - job_name: kubernetes_logs
    kubernetes_sd_configs:
        - role: pod
    relabel_configs:
      - source_labels: ['kubernetes_namespace']
        regex: '.*'
        action: drop
```

This is a more concise example using the `drop` action. The `source_labels` here refer to a label automatically generated by Promtail for kubernetes pod logs. We simply match the whole thing, and the `action: drop` ensures any line coming from kubernetes will drop the `kubernetes_namespace` label, regardless of its value.

As for learning more, I'd suggest getting intimately familiar with the official Loki documentation. They have a solid section dedicated to relabeling within Promtail. Beyond that, understanding regular expressions deeply is paramount – "Mastering Regular Expressions" by Jeffrey Friedl is considered a bible in that space. Also, exploring the Prometheus documentation (since Loki is heavily influenced by Prometheus) and its relabeling practices will give you a deeper context. Finally, looking into the details of the `regex` and `actions` options directly in the [Promtail Configuration documentation](https://grafana.com/docs/loki/latest/clients/promtail/configuration/) is extremely useful. It's really worth going through every possible `action`, since things like `keepequal` or `hashmod` can be invaluable in more complex configurations.

Remember that designing your labels properly at the outset greatly impacts the effectiveness of your log aggregation and the efficiency of your queries. Think about what information you'll want to query on, aggregate on, and alert on, and structure your labels accordingly. It's a process of iteration and refinement as you learn more about your data. It's not magic, it's about thoughtful planning and clear, actionable rules in your configuration.
