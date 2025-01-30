---
title: "How can containerized applications output logs?"
date: "2025-01-30"
id: "how-can-containerized-applications-output-logs"
---
Containerized applications, by their nature, operate in isolated environments, making traditional file-based logging less effective for centralized monitoring and debugging. Consequently, redirecting standard output (stdout) and standard error (stderr) streams, as well as utilizing dedicated logging drivers, becomes the primary approach for collecting and managing logs generated within containers. I've spent the last five years building microservices architectures, and experience has shown me the vital role effective log management plays in operational stability and incident resolution.

The fundamental principle here is that container runtimes, like Docker or containerd, capture everything written to stdout and stderr by the process running inside the container. These captured streams can then be routed to various destinations, allowing developers and operators to access, aggregate, and analyze application logs. This contrasts with the traditional model of writing logs to local disk files within the container, which are often ephemeral and inaccessible without directly accessing the container's file system.

The benefit of standard output and error redirection is its inherent simplicity. No specific logging library is required within the application itself beyond standard output or error writing functions. This avoids introducing unnecessary dependencies or making code changes to accommodate a specific logging framework. Most programming languages and frameworks offer straightforward methods for achieving this using features like `print()` or `console.log()` for stdout and `System.err.println()` or similar for stderr. Container runtimes automatically handle the routing of these streams.

However, relying solely on stdout and stderr has limitations. The logs, in their raw form, lack context like timestamps, log levels (e.g., debug, info, error), and the application component that generated the log entry. For small applications with limited concurrency, this might be adequate. However, for complex microservices, enriched logging metadata is crucial for effective troubleshooting.

This is where logging drivers come into play. These drivers provide mechanisms for the container runtime to route captured logs to external logging systems. Different drivers are available, each with its own characteristics and target use case. For instance, the `json-file` driver persists logs as JSON objects to the host's file system. While not ideal for production environments due to storage limitations, it's sufficient for local development. The `fluentd` driver, on the other hand, forwards logs to a centralized Fluentd instance, allowing for sophisticated log aggregation and processing. Cloud platforms often offer proprietary logging drivers that integrate seamlessly with their managed logging services.

It’s important to note that the application can also be configured to generate structured logs, such as JSON, that already contain the metadata required. While this involves application-level code modifications, it significantly simplifies processing downstream within the logging infrastructure. The container runtime and logging drivers can then utilize this structure directly instead of relying solely on basic log lines.

Let's look at three concrete examples.

**Example 1: Basic Output Redirection**

This example demonstrates how even a simple Python script uses stdout for logging, which is captured and routed by the container runtime.

```python
# app.py
import time

while True:
    print("Application is running")
    time.sleep(5)
```
To containerize this:
```dockerfile
# Dockerfile
FROM python:3.10-slim-buster
WORKDIR /app
COPY app.py .
CMD ["python", "app.py"]
```
Then build and run the image:
```bash
docker build -t my-python-app .
docker run my-python-app
```

In this case, the `print()` statement writes to stdout, and you can see the "Application is running" message every 5 seconds in the container logs using `docker logs <container_id>`. No specialized libraries are used; stdout redirection manages this. While basic, it establishes the foundation upon which more advanced logging practices are built.

**Example 2: Application Structured Logging (JSON)**

This example illustrates an application generating JSON formatted log entries to stdout to improve processing.

```python
# app_json.py
import time
import json
from datetime import datetime

def log(level, message):
  timestamp = datetime.utcnow().isoformat()
  log_entry = {
      "level": level,
      "message": message,
      "timestamp": timestamp,
      "component": "app_json"
  }
  print(json.dumps(log_entry))

while True:
  log("INFO", "Processing data")
  time.sleep(3)
```

The Dockerfile is very similar:

```dockerfile
# Dockerfile_json
FROM python:3.10-slim-buster
WORKDIR /app
COPY app_json.py .
CMD ["python", "app_json.py"]
```

And build and run:

```bash
docker build -t my-json-app -f Dockerfile_json .
docker run my-json-app
```

Now, each log entry is a JSON object, making it easier to parse and analyze with downstream tools like Fluentd or Elasticsearch. The crucial improvement is that instead of a basic string, we're outputting a structured, self-describing payload to stdout. The container runtime treats it identically – as an stdout stream – but the structured data greatly aids further processing.

**Example 3: Logging Driver Configuration**

While the previous examples show application behavior, this focuses on the container runtime configuration.  Let's assume we want to use the `fluentd` driver for the `my-json-app` we created in Example 2. Before running the container, we need a Fluentd instance running and listening on its forward port, usually 24224. The exact setup depends on your Fluentd configuration. We'll assume the Fluentd is listening at `fluentd_host:24224`.  Running the container requires specifying the logging driver and its options:

```bash
docker run \
--log-driver=fluentd \
--log-opt fluentd-address=fluentd_host:24224 \
--log-opt tag=my-json-app \
my-json-app
```

With this command, the container runtime routes all output from stdout to the Fluentd instance specified. This output will now appear in the centralized Fluentd log aggregation system, allowing further processing, indexing, and querying using tools like Elasticsearch and Kibana, based on the tag `my-json-app`. The command line arguments `--log-driver=fluentd` and the `--log-opt` entries configure the Docker daemon's logging behavior.

In conclusion, effective container logging relies primarily on redirecting stdout and stderr and employing logging drivers to route those streams to external systems. While simple stdout output provides the foundation, structured logging and logging drivers are essential for practical, real-world use cases. These examples show the different layers to address effective logging, from within the application to the container runtime level.

For additional learning, I highly recommend examining documentation from container runtimes, such as Docker's documentation on logging drivers. Additionally, research dedicated log aggregation tools like Elasticsearch, Fluentd, and Loki to understand how they utilize container log output. These tools often include their own setup instructions and best practice documentation for production environments. Understanding these tools in the context of container logging is crucial for building robust, observable systems. Finally, explore the specific logging features within your programming language or framework for application level logging strategies.
