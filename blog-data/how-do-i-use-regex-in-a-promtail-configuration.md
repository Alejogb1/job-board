---
title: "How do I use regex in a promtail configuration?"
date: "2024-12-23"
id: "how-do-i-use-regex-in-a-promtail-configuration"
---

Alright, let's talk about regex within promtail configurations. I’ve spent a fair amount of time tweaking and debugging promtail over the years, particularly when handling logs with complex, non-uniform structures. It's one area where mastery can significantly improve your observability pipeline and reduce the headache of sifting through poorly parsed data. So, let me break down how I typically approach regex usage in this context, focusing on the practical side of things.

The core concept revolves around leveraging regex within promtail's pipeline stages to extract meaningful data from log lines. These extractions are then used to create labels, which are crucial for effective querying and analysis in Loki. Simply ingesting raw log lines without proper labeling is akin to dumping data into a black hole. You can get it in, sure, but retrieving it with any precision is another matter entirely.

Typically, we encounter two main use cases for regex in promtail: the first is using regex as part of the `match` stage for filtering specific log entries, and the second is within processing stages like `regex` and `unpack` to extract fields. I've found that a clear understanding of these use cases and how they're employed is vital.

Let's start with `match` stages. The `match` stage essentially filters incoming logs based on a regex pattern applied to the log line's `message` attribute. It allows us to direct different log streams through separate pipelines, thereby applying specific processing steps to targeted data. This is particularly useful when you're dealing with different logging formats or sources within a single application.

For example, suppose I've got a system spitting out logs that sometimes contain an “error” level marker like `[ERROR]`. If I only want to process logs with this specific marker and extract the error message, I would define a `match` block that effectively acts as a gatekeeper.

Here’s a snippet that filters only log lines containing an error prefix:

```yaml
pipeline_stages:
  - match:
      selector: '{job="my-app"}' # your existing selector
      stages:
      - regex:
          expression: '^\\[(ERROR)\\]\\s+(.*)$'
          source: message
      - labels:
          level: "error"
          error_message: $2
```

In this configuration, logs from `my-app` that contain the string `[ERROR]` followed by a space are routed into the next processing stages. The regex `^\\[(ERROR)\\]\\s+(.*)$` is used for this filtering. Importantly, even though the capturing groups are defined in the regex stage *after* this filter, the filter implicitly requires a match of the entire expression against the message. Without it, nothing would be passed along.

The core processing of extraction comes in the `regex` and `unpack` stages, which are fundamental in my experience. The `regex` stage is used to extract individual values from a log line based on specific patterns. Captured groups from your regex become accessible using `$1`, `$2`, etc., and can be assigned to labels. This is where the real power of regex in promtail comes into play, allowing you to transform unstructured logs into structured, queryable data.

Let's look at a slightly more complicated case. Suppose I have a log message like: `timestamp=2023-10-27T10:00:00Z level=INFO component=auth message="User logged in: user123"` and I want to extract timestamp, level, component, and the message body.

Here's how I would achieve that:

```yaml
pipeline_stages:
  - regex:
      expression: 'timestamp=(?P<timestamp>[^\\s]+)\\s+level=(?P<level>[^\\s]+)\\s+component=(?P<component>[^\\s]+)\\s+message="(?P<message>.*)"'
      source: message
  - labels:
      timestamp:
        value: "{{.timestamp}}"
      level:
        value: "{{.level}}"
      component:
        value: "{{.component}}"
      message:
        value: "{{.message}}"

```

Here, the named capture groups (using `(?P<name>...)`) are used to label the extracted parts. This approach is generally cleaner and easier to manage than positional captures, especially when the order of fields in your logs might vary. These named capture groups then get directly mapped to our labels.

Now, consider a situation with a JSON log. This is where `unpack` shines. Let’s say your log message is a JSON string: `{"timestamp": "2023-10-27T10:00:00Z", "level": "INFO", "component": "database", "message": "Connection established."}`. Using `unpack`, I can parse this JSON into individual fields and create labels:

```yaml
pipeline_stages:
  - json:
      expressions:
        message:
  - unpack:
      source: message
      json: true
  - labels:
      timestamp: "{{.timestamp}}"
      level: "{{.level}}"
      component: "{{.component}}"
      message: "{{.message}}"

```

Notice, we use the `json` stage to make sure the message is not being treated as a literal string. After which, we unpack the json structure. `unpack`, by default, expects a JSON object and then creates values that can be used as template variables in labels. The `json: true` configuration ensures that the source field is parsed as a JSON string.

One key lesson I've learned over time is the importance of testing your regex patterns. A simple typo in your regex can lead to silent errors, or even worse, completely incorrect data being ingested. Tools like regex101.com are invaluable for interactive testing. Always thoroughly validate your patterns against example log lines before deploying them. Also, pay close attention to how greedy your expressions are. `(.*)` might seem convenient, but it can have unintended consequences if you're not careful about the context. In those situations, I prefer using a negative character class `[^...]` that will explicitly not match until a character is found or the end of the string.

Furthermore, the interaction between the different stages is also key. Pay attention to what a stage transforms and what gets passed down to the next. This is especially true with the `template` stage if you are attempting to construct complex labels.

For further reference on these topics, I highly recommend spending some time with the official promtail documentation, specifically focusing on the pipeline stages. It is comprehensive and contains numerous examples. Also, “Mastering Regular Expressions” by Jeffrey Friedl is an excellent text for truly understanding regular expressions. It's a challenging read but well worth the investment. You'll also find "The Logstash Book" by Jordan Sissel (though Logstash focuses on processing logs, the regex section is invaluable and the book provides insights into the underlying principles that apply to all log processing) incredibly beneficial to get a holistic understanding of log parsing.

So, in essence, using regex in promtail configuration is a combination of understanding the stages involved, crafting efficient regex patterns, and carefully testing everything. It’s a foundational skill for effective log management.
