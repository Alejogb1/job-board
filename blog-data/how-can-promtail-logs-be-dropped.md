---
title: "How can promtail logs be dropped?"
date: "2024-12-23"
id: "how-can-promtail-logs-be-dropped"
---

Alright, let's tackle dropping logs in promtail. It’s something I’ve dealt with extensively in my career, particularly when managing large-scale systems where the sheer volume of logs can become overwhelming and, frankly, expensive to store. It's not just about saving resources; it’s about ensuring the logs you *do* keep are valuable and actionable. So, let’s get into the technical details and the practical approaches I've found most effective.

First, it’s crucial to understand *why* you might want to drop logs. The most common reason is noise. Application logs, especially verbose ones, can include a lot of repetitive messages, debug statements, or irrelevant information that doesn't contribute to monitoring, debugging, or security analysis. Another reason is resource consumption; ingesting, processing, and storing these logs all come with a cost. Finally, compliance and security may dictate the need to drop certain sensitive or personally identifiable information that should never be stored in log aggregation systems.

Promtail offers a few mechanisms to achieve this, primarily through its configuration file. The configuration centers around *pipelines*, which are essentially processing workflows applied to log entries before they are sent to Loki. These pipelines can contain various stages, including stages specifically designed for dropping logs based on matching criteria. I've found this design incredibly powerful for fine-tuning what is captured.

Now, let’s get to the practical methods. The most direct way to drop logs is using the `drop` stage. This stage takes a series of actions, and if any of those actions return true (meaning a match), the log entry is dropped. These actions typically include matching based on regular expressions applied to specific fields of the log entry. Let me illustrate with a few examples.

**Example 1: Dropping Debug Logs**

Let’s say your application logs debug information, and you only want to retain informational, warning, or error logs. In my past projects, I’ve often seen the log format as something like: `[TIMESTAMP] [LEVEL] [COMPONENT] MESSAGE`. Here’s a Promtail configuration snippet demonstrating how to drop log entries with the `debug` level:

```yaml
  - job_name: application-logs
    static_configs:
      - targets:
          - localhost
        labels:
          job: my-app
    pipeline_stages:
      - match:
          selector: '{job="my-app"}'
          stages:
            - regex:
                expression: "^(?P<time>\\S+) (?P<level>\\S+) (?P<component>\\S+) (?P<message>.+)$"
            - drop:
                source: level
                value: "debug"
```

Here's what's happening: First, `match` finds log entries from our job `my-app`. Then, a `regex` stage parses the log line, extracting named capture groups like `time`, `level`, `component`, and `message`. The crucial part is the `drop` stage that checks if the captured `level` is equal to the value `debug`. If the level is "debug" the entire log entry is discarded.

**Example 2: Dropping Logs Based on Specific Message Content**

Sometimes, you might need to drop entries based on specific keywords or patterns within the message itself. Suppose you have repetitive log messages related to transient network errors that you are already tracking through other monitoring systems. Consider a message like "Connection refused: Error code 111". Here's how to drop messages containing that phrase:

```yaml
  - job_name: application-logs
    static_configs:
      - targets:
          - localhost
        labels:
          job: my-app
    pipeline_stages:
      - match:
          selector: '{job="my-app"}'
          stages:
            - regex:
                expression: "^(?P<time>\\S+) (?P<level>\\S+) (?P<component>\\S+) (?P<message>.+)$"
            - drop:
                source: message
                value: "Connection refused: Error code 111"
```

This configuration is very similar to the first example, but here, the `drop` stage checks the content of the `message` field instead of the `level` field. If the message matches the specific text, the entry is dropped. This approach becomes very useful when dealing with known-issue log patterns that create significant noise, something I've had to deal with on numerous occasions.

**Example 3: Using Regex for More Flexible Matching**

The `drop` stage can also use regular expressions directly for more flexible pattern matching. Let's say you want to drop all messages that contain any trace level of logging, such as "trace1", "trace2" or "tracing" to remove an entire family of log entries. Here's a way to do this:

```yaml
  - job_name: application-logs
    static_configs:
      - targets:
          - localhost
        labels:
          job: my-app
    pipeline_stages:
      - match:
          selector: '{job="my-app"}'
          stages:
            - regex:
                expression: "^(?P<time>\\S+) (?P<level>\\S+) (?P<component>\\S+) (?P<message>.+)$"
            - drop:
                source: message
                regex: "trace\\d+|tracing"
```

In this scenario, the `regex` keyword within the `drop` stage lets us use a regular expression: `trace\\d+|tracing`. This regex will match messages containing “trace” followed by a digit, or the word "tracing." This flexibility in pattern matching is critical for addressing complicated logging patterns, and I’ve found it significantly reduces noise in many real-world scenarios.

It's important to note that `drop` stages are processed in the order they appear within the pipeline. So if you have multiple `drop` stages, the first one to match a condition will cause the entry to be dropped, and the subsequent ones will not be considered for that particular log line. This pipeline-style processing ensures consistent and predictable behavior.

When implementing these configurations, I always encourage thorough testing, both locally and in a staging environment before deploying to production. A mistake in a drop rule can unintentionally discard valuable logs. Also, regularly review and adjust drop rules as your system and logging evolve; what’s considered noise today may be critical information tomorrow.

For further learning, I would strongly recommend reading the official documentation on Promtail's pipeline stages, particularly the `match`, `regex`, and `drop` stages. Beyond that, “Effective Logging: A Guide for Developers” by Jeff Atwood offers great insight into what makes valuable logs, which indirectly helps in figuring out what logs should be dropped. In addition, a deep dive into the "Regular Expression Pocket Reference" by Tony Stubblebine can enhance your ability to craft very specific rules. Lastly, studying the Loki documentation concerning log ingestion and query performance can provide context as to why managing log volume is critically important.

Dropping logs effectively with Promtail isn't just about reducing resource usage; it's about streamlining your monitoring and logging infrastructure to be more efficient and insightful. The configuration is powerful and flexible, allowing you to fine-tune your logging to focus on what truly matters. It's a skill that I’ve found to be increasingly valuable when maintaining observability in complex systems.
