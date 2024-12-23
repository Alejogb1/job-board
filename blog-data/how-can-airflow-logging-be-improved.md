---
title: "How can Airflow logging be improved?"
date: "2024-12-23"
id: "how-can-airflow-logging-be-improved"
---

Alright, let's unpack the question of improving Airflow logging. It's a topic I've spent a fair amount of time grappling with, particularly during a stint working with a large-scale data pipeline at *Acme Corp.* a few years back. We were ingesting terabytes of data daily, and pinpointing issues amidst the sheer volume of task executions became a real challenge. The default Airflow logging, while functional, often left us wanting more granular control and easier analysis.

So, what are the primary areas where we can make things better? I see three key aspects: granularity of logs, efficient log storage and retrieval, and finally, the ease of parsing and analysis.

Firstly, **granularity**. The out-of-the-box Airflow logging often provides a somewhat generic view of task execution. We get start and end times, and generally, a summary of success or failure. But what about the nitty-gritty details happening *inside* the task? That’s where custom logging becomes paramount. Relying solely on Airflow's task logs limits visibility. I prefer implementing specific logging within the tasks themselves to gain a much more detailed insight into the process.

For example, say we have a data transformation task. Instead of just seeing that the task succeeded or failed, we’d ideally want to log key information such as the number of rows processed, the values of specific parameters, or intermediate data checks.

Here's a simple Python snippet, using the standard `logging` library, demonstrating how to achieve this:

```python
import logging
import time

def transform_data(input_path, output_path):
  """Simulates a data transformation process and logs details."""

  log = logging.getLogger(__name__)
  log.info(f"Starting data transformation for input: {input_path}")
  start_time = time.time()

  # Simulate some data processing
  rows_processed = 1000
  log.info(f"Processed {rows_processed} rows of data.")

  end_time = time.time()
  duration = end_time - start_time
  log.info(f"Data transformation completed in {duration:.2f} seconds. Output saved to: {output_path}")

  return True # assuming success


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    transform_data("data/input.csv", "data/output.csv")
```

In this example, rather than only using the standard Airflow log, the `transform_data` function actively uses logging to record critical information related to the task. You'll notice how I explicitly configure a logger, and set the logging level. I strongly recommend setting the logging level at the function level within the specific dags. It's much more targeted than just a blanket setting for all tasks. When integrated into an Airflow task, you would need to configure the Airflow task appropriately to capture this logger output. You do this by properly setting the logging level, and the handlers to direct the logging appropriately.

The second area for improvement lies in **efficient log storage and retrieval**. As you scale, the default file system-based logging of Airflow can become problematic. Searching through potentially hundreds of log files can be incredibly time-consuming. I’ve found the most effective way to overcome this is by centralizing logs in a more searchable, scalable system such as Elasticsearch, Splunk, or even cloud-native alternatives like AWS CloudWatch Logs or Google Cloud Logging.

Here’s a snippet that outlines how to structure a custom logging handler, in this case, targeting an example log server. Note that this is an oversimplified view, but it illustrates the key concepts:

```python
import logging
import requests
import json

class RemoteLogHandler(logging.Handler):
    """A custom logging handler sending logs to a remote server."""

    def __init__(self, log_server_url):
      logging.Handler.__init__(self)
      self.log_server_url = log_server_url

    def emit(self, record):
        log_entry = {
            "time": self.formatTime(record),
            "level": record.levelname,
            "message": self.format(record),
            "task_id": record.__dict__.get('task_id'),
            "dag_id": record.__dict__.get('dag_id'),
            # other relevant data...
        }
        try:
          requests.post(self.log_server_url, data=json.dumps(log_entry), headers={'Content-Type': 'application/json'})
        except Exception as e:
          print(f"Error sending log: {e}") # Log locally in case of issues

if __name__ == '__main__':
  logging.basicConfig(level=logging.INFO)
  logger = logging.getLogger("remote_logger")
  handler = RemoteLogHandler("http://localhost:8000/log")
  formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
  handler.setFormatter(formatter)
  logger.addHandler(handler)

  logger.info("Example log entry", extra = {'task_id': 'test_task_1', 'dag_id': 'example_dag'})
```

This `RemoteLogHandler` provides a basic template. In an actual implementation, you would enhance the error handling, utilize more robust serialization for your log entries, and ensure secure communication with the log server. Crucially, you see how you can attach additional context, such as the `task_id` and `dag_id`, which makes searching and filtering in the log aggregation system much easier. The crucial point is that you can push logs to the server of your choosing rather than relying on file based outputs.

Finally, there's the issue of **ease of parsing and analysis**. Raw text logs are difficult to query, especially when spread across many files. Structured logging using JSON or similar formats makes querying logs significantly easier. Both the above examples have moved some steps towards structured logging, but we can take it further with more deliberate attention to the format. A more complete example might utilize a format that includes more metadata. This is something you would also want to consider alongside your decision on where to send your logs, as the ingestion pipelines for each system will have its own nuances.

For a more robust logging output, consider this revised snippet:

```python
import logging
import json
import datetime

class JsonFormatter(logging.Formatter):
    """A custom formatter that outputs JSON log entries."""
    def format(self, record):
      log_entry = {
            "timestamp": datetime.datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "task_id": getattr(record, 'task_id', None),
            "dag_id": getattr(record, 'dag_id', None),
            "module": record.module,
            "line_number": record.lineno,
            "function": record.funcName
            # add other contextual data as needed
      }
      return json.dumps(log_entry)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("json_logger")
    handler = logging.StreamHandler()
    formatter = JsonFormatter()
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logger.info("This is a structured log message", extra = {'task_id': 'sample_task', 'dag_id': 'sample_dag'})
```

Here, the `JsonFormatter` takes our logger data and converts it into a single line JSON string, which can be easily indexed, searched, and analyzed by systems like Elasticsearch. I've added more context, such as the timestamp, originating module, and line number of the log event. We've again added custom attributes using `extra` for the `info` call. It’s a small thing, but it can save countless hours when debugging in production environments.

For further reading on these concepts, I’d recommend looking into: "Effective Python" by Brett Slatkin for detailed insights into using python's logging modules more effectively. Also, familiarize yourself with the documentation for the logging system you choose (e.g. the official elastic documentation). For a deeper understanding of log aggregation and analysis, “Designing Data-Intensive Applications” by Martin Kleppmann is extremely useful, particularly chapters dealing with stream processing and distributed systems. And, of course, make sure you’ve a strong understanding of your specific systems’ documentation, be that Cloudwatch, GCP Logging, or your log server of choice.

By implementing these logging strategies – from adding detailed custom logs inside task executions to centralizing and structuring your log output – you gain a much more complete and actionable view of your Airflow environment. These changes can be the difference between hours of frustrating troubleshooting and a straightforward resolution, especially when dealing with complex data pipelines. The key is to not treat logging as an afterthought, but as a critical part of your pipeline development process.
