---
title: "How can Airflow scheduler logs be output to stdout or cloud storage?"
date: "2025-01-30"
id: "how-can-airflow-scheduler-logs-be-output-to"
---
The Apache Airflow scheduler's logging mechanism, while robust, defaults to writing logs to files within the Airflow instance's configured directory.  This presents challenges for monitoring in distributed environments and for integration with centralized logging services.  My experience managing large-scale Airflow deployments has consistently highlighted the need for more flexible logging solutions, prioritizing output to standard out (stdout) or cloud storage for improved observability and scalability.  Achieving this requires careful configuration and potentially custom logging handlers.

**1.  Explanation:**

Airflow leverages Python's `logging` module for its logging functionality.  The scheduler, a core component, generates a substantial volume of logs detailing task scheduling, execution, and any encountered errors. By default, these logs are written to files based on the `AIRFLOW_LOG_FOLDER` configuration setting. To redirect this output, we need to modify the logging configuration, either globally or specifically for the scheduler.  This involves replacing the default file handlers with handlers capable of writing to stdout or integrating with cloud storage solutions like Google Cloud Storage (GCS), Amazon S3, or Azure Blob Storage.  The key to this is understanding the hierarchical structure of Airflow's logging and how to programmatically intercept and redirect log messages at the appropriate level.  Directly modifying the Airflow source code is generally discouraged; configuration-based solutions are far preferable for maintainability and upgrade compatibility.  Crucially, appropriate log rotation and retention policies must be implemented irrespective of the chosen output method to prevent unbounded log growth.

**2. Code Examples:**

The following examples demonstrate different approaches to modifying Airflow’s logging behavior.  Note that these examples assume a basic understanding of Airflow's configuration and the Python `logging` module.  Error handling and more robust configurations would be necessary in production environments.

**Example 1:  Outputting Scheduler Logs to stdout:**

This example leverages a simple `StreamHandler` to redirect the scheduler’s logs to stdout. This approach is suitable for local development or debugging but isn't ideal for production deployments due to the potential volume of logs.

```python
import logging
import logging.config
import os

# Configure logging to output to stdout
logging_config = {
    'version': 1,
    'formatters': {
        'simple': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'simple',
            'level': 'DEBUG'  # Adjust logging level as needed
        }
    },
    'loggers': {
        'airflow.scheduler': {
            'handlers': ['console'],
            'level': 'DEBUG',  # Adjust logging level as needed
            'propagate': False
        }
    }
}

# Apply the configuration
logging.config.dictConfig(logging_config)

# (Rest of your Airflow code/scripts)
```


**Example 2:  Outputting Scheduler Logs to Google Cloud Storage (GCS):**

This example utilizes the `google.cloud.logging` library to send scheduler logs to Google Cloud Logging.  Prior to execution, you'll need to ensure the necessary Google Cloud credentials are configured and that the `google-cloud-logging` library is installed. This method offers scalability and centralized logging, preferred for production scenarios.

```python
import logging
import logging.config
from google.cloud import logging as google_logging

# Initialize Google Cloud Logging client
client = google_logging.Client()

# Configure logging to send logs to Google Cloud Logging
logging_config = {
    'version': 1,
    'formatters': {
        'simple': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        }
    },
    'handlers': {
        'gcs': {
            'class': 'google.cloud.logging.handlers.CloudLoggingHandler',
            'client': client,
            'name': 'airflow_scheduler', # Customize log name
            'formatter': 'simple',
            'level': 'DEBUG'  # Adjust logging level as needed
        }
    },
    'loggers': {
        'airflow.scheduler': {
            'handlers': ['gcs'],
            'level': 'DEBUG',  # Adjust logging level as needed
            'propagate': False
        }
    }
}

# Apply the configuration
logging.config.dictConfig(logging_config)

# (Rest of your Airflow code/scripts)

```

**Example 3:  Custom Logging Handler for Flexibility:**

For maximum control, a custom logging handler can be created to handle log output to any desired destination. This example demonstrates a basic framework;  it would need to be extended to accommodate specific cloud storage APIs.

```python
import logging
import logging.config

class CloudStorageHandler(logging.Handler):
    def __init__(self, storage_path):
        super().__init__()
        self.storage_path = storage_path

    def emit(self, record):
        # Implement logic to write log record to cloud storage
        # This requires specific implementation for the chosen cloud provider
        # ... (Code to interact with Cloud Storage API) ...
        log_message = self.format(record)
        try:
            #Simulate writing to cloud storage
            with open(os.path.join(self.storage_path, "airflow.log"), "a") as f:
                f.write(log_message + "\n")
            print(f"Log message written to {self.storage_path}")
        except Exception as e:
            print(f"Error writing log to storage: {e}")


# Configure logging to use the custom handler

logging_config = {
    'version': 1,
    'formatters': {
        'simple': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        }
    },
    'handlers': {
        'cloud_storage': {
            '()': 'CloudStorageHandler',
            'storage_path': '/tmp/airflowlogs', #replace with your desired path.
            'formatter': 'simple',
            'level': 'DEBUG'
        }
    },
    'loggers': {
        'airflow.scheduler': {
            'handlers': ['cloud_storage'],
            'level': 'DEBUG',
            'propagate': False
        }
    }
}

logging.config.dictConfig(logging_config)

#Rest of your Airflow code
```

**3. Resource Recommendations:**

The official Apache Airflow documentation.  Python's `logging` module documentation.  The documentation for your chosen cloud storage provider's client libraries (e.g., Google Cloud Storage, Amazon S3, Azure Blob Storage).  A comprehensive guide to Python logging best practices.  A book on advanced Python logging techniques.  A tutorial on configuring logging in large-scale Python applications.


Remember to adapt these examples to your specific environment and requirements.  Consider factors such as log volume, security, and the reliability of your chosen logging destination.  Thorough testing is crucial before deploying any changes to your production Airflow environment.  The approaches presented offer diverse strategies; the optimal solution will depend on your operational constraints and scalability needs.
