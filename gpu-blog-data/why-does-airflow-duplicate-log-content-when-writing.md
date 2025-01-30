---
title: "Why does Airflow duplicate log content when writing to GCS?"
date: "2025-01-30"
id: "why-does-airflow-duplicate-log-content-when-writing"
---
Airflow's duplication of log content when writing to Google Cloud Storage (GCS) frequently stems from misconfiguration of the logging mechanism, specifically concerning the `logging_config` within the Airflow configuration.  My experience troubleshooting this issue across numerous large-scale data pipelines has shown that the root cause is almost always a combination of improperly configured handlers or unexpected interactions between the Airflow logging system and the underlying GCS library.

**1. Clear Explanation:**

Airflow's logging system relies on Python's `logging` module.  By default, this module writes logs to a local file.  However, to direct logs to GCS,  Airflow needs a custom handler.  This handler, typically a `google.cloud.logging.Client` or a similar GCS-specific logger, needs to be correctly integrated within Airflow's logging configuration.  The duplication issue arises when multiple handlers inadvertently write the same log messages.  This can occur due to several scenarios:

* **Multiple Handlers Writing to GCS:** The most common cause.  If both the default local file handler and a GCS handler are active and configured to log at the same level, each will receive and write the log messages, resulting in duplication. This often happens when a custom logging configuration is added without explicitly disabling or replacing the default handlers.

* **Handler Propagation:**  Python's logging system uses a hierarchical structure.  If a handler is added to a parent logger, its output will propagate down to all child loggers.  Without proper configuration, this can lead to the same log messages being written multiple times through different loggers and their associated GCS handlers.

* **Concurrent Loggers:**  In highly concurrent environments, multiple worker processes or executors might independently write logs to GCS, potentially leading to duplicated entries if they are not properly synchronized. While not directly causing duplication within a single log file, it can lead to the same message appearing across multiple log files in GCS.

* **GCS Client Configuration:** Issues within the GCS client's setup itself can lead to log write attempts failing and getting retried.  If these retries are not handled gracefully, or if they write to different locations, it can simulate log duplication.


**2. Code Examples with Commentary:**

**Example 1: Incorrect Configuration Leading to Duplication:**

```python
# airflow.cfg (incorrect)
[loggers]
keys=default, airflow

[handlers]
keys=consoleHandler, fileHandler, gcsHandler

[formatters]
keys=simpleFormatter

[logger_default]
level=DEBUG
handlers=consoleHandler, fileHandler, gcsHandler
qualname=default

[logger_airflow]
level=DEBUG
handlers=consoleHandler, fileHandler, gcsHandler
qualname=airflow

[handler_consoleHandler]
class=StreamHandler
level=DEBUG
formatter=simpleFormatter
args=(sys.stdout,)

[handler_fileHandler]
class=FileHandler
level=DEBUG
formatter=simpleFormatter
args=('/tmp/airflow.log',)

[handler_gcsHandler] # INCORRECT: Adding this without disabling others
class=airflow.providers.google.cloud.log.gcs_hook.GCSHook
level=DEBUG
formatter=simpleFormatter
args=("gs://my-bucket/airflow-logs",)

[formatter_simpleFormatter]
format=%(asctime)s - %(name)s - %(levelname)s - %(message)s
```

This configuration adds a GCS handler (`gcsHandler`) without removing or changing the default file handler (`fileHandler`). Both will write the same logs, leading to duplication.


**Example 2: Correct Configuration with Explicit Handler Removal:**

This example demonstrates how to correctly configure Airflow to only write to GCS, effectively avoiding duplication. Note that this requires custom configuration and potentially adjusting other parts of your Airflow setup depending on your chosen logging approach.

```python
# airflow.cfg (correct)
[loggers]
keys=default, airflow

[handlers]
keys=gcsHandler

[formatters]
keys=simpleFormatter

[logger_default]
level=DEBUG
handlers=gcsHandler
qualname=default

[logger_airflow]
level=DEBUG
handlers=gcsHandler
qualname=airflow

[handler_gcsHandler]
class=airflow.providers.google.cloud.log.gcs_hook.GCSHook
level=DEBUG
formatter=simpleFormatter
args=("gs://my-bucket/airflow-logs",)

[formatter_simpleFormatter]
format=%(asctime)s - %(name)s - %(levelname)s - %(message)s
```

This configuration explicitly removes the default handlers (`fileHandler`, `consoleHandler`) and relies solely on the `gcsHandler` for logging.


**Example 3: Using a Custom Logger with GCS Integration:**

This example showcases a more advanced approach, using a custom logger to write to GCS. This provides better control over logging behavior and facilitates more complex logging requirements.

```python
import logging
from google.cloud import logging as google_logging

# ... other imports

# Initialize Google Cloud Logging client
client = google_logging.Client()
logger = client.logger('airflow_custom_logger')

# Configure the logger for GCS output
handler = google_logging.handlers.CloudLoggingHandler(client)
handler.setLevel(logging.DEBUG)
logger.addHandler(handler)

# Log messages using the custom logger
logger.info("This message will go to GCS.")
logger.error("This error message will also go to GCS.")
```

This example leverages Google Cloud's logging library directly. The custom logger ensures that logs are directly streamed to GCS, bypassing Airflow's default handlers. This requires proper authentication and authorization with your Google Cloud project.


**3. Resource Recommendations:**

The official Airflow documentation on logging configuration.  Consult the documentation for your specific Airflow provider package (e.g., the Google Cloud Provider) for details on configuring GCS logging.  Study the Python `logging` module documentation for a thorough understanding of Python's logging mechanisms.  Familiarize yourself with the Google Cloud Logging documentation for best practices and advanced usage.  Review the documentation for any third-party logging libraries that you might be using in conjunction with Airflow.  A comprehensive understanding of these resources will significantly aid in resolving logging issues.  If you are deploying to a managed environment such as Google Cloud Composer, review their specific configurations and guidelines, as additional elements may be involved.
