---
title: "How can a part of a log be removed in Loki?"
date: "2024-12-23"
id: "how-can-a-part-of-a-log-be-removed-in-loki"
---

Okay, let's address this. I've seen this exact challenge crop up countless times, particularly when dealing with sensitive data or overly verbose applications pumping logs into Loki. Removing specific parts of log lines within Loki isn't a direct "delete" operation like you might find with a database. Instead, it's more about filtering and reshaping the data before it's ingested or presented in queries. The key here is understanding how Loki handles logs and the tools at our disposal. I'll walk you through several approaches, drawing on past experiences battling noisy logging setups.

The primary way to "remove" parts of a log in Loki involves the combination of log processing within the ingestion pipeline and the filtering capabilities of LogQL (Loki's query language). When logs enter Loki, they pass through a configurable pipeline where we can manipulate them before storage. This pipeline uses a component called a `stage`, which contains actions. A key concept is to never truly "remove" anything permanently but rather mask or exclude it in a way that it is not visible downstream for analysis or querying.

Let’s break down three practical scenarios with code snippets:

**Scenario 1: Removing Sensitive Data with Regex Stages**

Imagine a scenario where we have logs containing potentially sensitive information, like API keys or email addresses, embedded within messages. In one project, I had an application generating log lines that included user emails within the message text. We obviously couldn’t store those without some form of sanitization. Using the `regex` stage, we can rewrite the log line before it even reaches Loki's storage. Consider the following prometheus-style config for loki, in its `loki.yaml`:

```yaml
scrape_configs:
  - job_name: 'my_app_logs'
    static_configs:
      - targets:
          - localhost:3000 # example
        labels:
          job: my_app
    pipeline_stages:
      - regex:
          expression: '(?P<email>[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})'
      - replace:
         source: "email"
         replace: "REDACTED"
```

What this configuration does, is first uses a regex expression `(?P<email>[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})` to search for email addresses, and capture it as a label named `email`. The next `replace` stage replaces the content of the `email` label, which is the found email, with the string "REDACTED". Importantly, it does *not* change the actual log line itself. To do that we need to use the `template` stage.

```yaml
scrape_configs:
  - job_name: 'my_app_logs'
    static_configs:
      - targets:
          - localhost:3000 # example
        labels:
          job: my_app
    pipeline_stages:
      - regex:
          expression: '(?P<email>[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})'
      - replace:
         source: "email"
         replace: "REDACTED"
      - template:
          source: 'log'
          template: '{{ replace .log ( .email | default "" ) "REDACTED" }}'
```

This updated snippet is an improvement because it now uses the `template` stage. This stage provides fine-grained control over the log message. Here, it takes the original log message using `.log`. It then tries to find the captured regex email using the label we created called `email`. It then replaces that found string with the string `REDACTED`. It uses a default value to prevent errors if a regex was not found. This way, the log stored in Loki has the email addresses already replaced with ‘REDACTED’. Note: This means that while querying with logql, email fields will not be present in the logs. We have therefore effectively removed that information from further analysis.

**Scenario 2: Removing Unnecessary Prefixes with a `Regex` Stage and a `template` stage:**

Let’s consider a situation where our application logs are prefixing each line with a useless timestamp or some other extraneous text. I once had a system where logs came in with a format like `"2023-10-27T10:00:00.000Z [INFO] - Log message content"`. We didn’t need the timestamp or the "[INFO]" part in our logs. Here's how we can clean those up at the ingestion stage, again using a regex, and then the template stage:

```yaml
scrape_configs:
  - job_name: 'my_app_logs'
    static_configs:
      - targets:
          - localhost:3000 # example
        labels:
          job: my_app
    pipeline_stages:
      - regex:
          expression: '^(?P<prefix>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}.\d{3}Z\s+\[[A-Z]+\]\s+-\s+)(?P<message>.+)$'
      - template:
          source: 'log'
          template: '{{ .message }}'
```

Here, the regex `^(?P<prefix>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}.\d{3}Z\s+\[[A-Z]+\]\s+-\s+)(?P<message>.+)$` captures the timestamp/prefix in a group called "prefix" and the remaining content in a group called "message." The `template` stage then takes the log line using the `.log` keyword, and replaces the entire contents with only the `message` group, effectively removing the prefix, and storing only the important message.

**Scenario 3: Filtering Log Lines Based on Content with LogQL**

Sometimes, you don't want to remove data from the log line itself, but instead want to exclude entire log lines from being part of your analysis. In that case, filtering is the way to go. This is accomplished through LogQL query language, and is useful when you don't want to change the log during ingestion, but when analyzing. LogQL provides powerful filtering capabilities to only show the logs you care about. Imagine needing to exclude all log lines that contain the word "debug". Here is an example of how to accomplish that:

```logql
{job="my_app"} |= "debug" |~ `(?i).*error.*`
```

This query uses a LogQL `|=` operator which performs a substring search which will only return logs which *contain* the word debug. We then pipe it to the `|~` operator, which executes a regex and returns logs which contain `error` in them. Note that the `(?i)` at the beginning of the regex performs a case-insensitive match. This allows you to focus on specific errors or information by filtering out the uninteresting data during query time, so the actual log data is unchanged.

These examples highlight how you can effectively "remove" parts of logs in Loki. It's crucial to remember that these techniques are fundamentally about filtering and transforming log data, either at ingestion time or query time, not about altering the underlying log files themselves. The choice between manipulating logs during ingestion with regex stages or filtering them at query time with LogQL depends on the specific use case. If you need to sanitize data or remove unnecessary prefixes from all logs, the former is generally preferable for efficiency and data security. If filtering is done on an ad-hoc basis, or in a very limited number of cases, LogQL is the faster method of filtering out specific information.

For further reading, I recommend exploring the official Loki documentation for more detailed examples and configurations. Additionally, "Effective Logging" by Peter Seibel is a solid resource for understanding log management principles. For the deep dive on prometheus configuration (which loki is related to), I would suggest "Prometheus: Up & Running: Monitoring the World with Time Series Data" by Brian Brazil. These will give you a more thorough technical understanding and can be invaluable tools when dealing with log management challenges.
