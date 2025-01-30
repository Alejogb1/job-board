---
title: "How to apply custom logging to a Python file executed by a BashOperator in Airflow's airflow_local_settings.py?"
date: "2025-01-30"
id: "how-to-apply-custom-logging-to-a-python"
---
Python's standard `logging` module offers granular control over output, and its integration within Airflow workflows executed via a `BashOperator` requires a specific approach due to the subprocess nature of the operator. The `airflow_local_settings.py` file, while enabling customizations, affects Airflow's core logging, not necessarily the subprocesses initiated by `BashOperator`. Therefore, to implement custom logging for a Python file invoked by `BashOperator`, one must configure logging *within the Python file itself*, leveraging its own configuration mechanism.

My experience working with distributed workflows highlighted that relying solely on Airflow’s logging for subprocesses can be insufficient for granular debugging. The `BashOperator` essentially executes a command in a separate shell, distinct from the Airflow worker's logging context. Capturing the desired output necessitates incorporating logging directly into the Python script being executed.

The core principle involves configuring the Python script to write logs to a location that's accessible and meaningful. This location can be: 1) a file, 2) the console, 3) a network socket, or 4) a combination thereof. While console logging is straightforward, it is generally less suitable for production workflows where persistent and manageable logs are required. File-based logging provides the benefit of storage and easier post-processing.

Here's how to configure logging in a Python script designed to be executed by an Airflow `BashOperator`:

```python
# example_script.py
import logging

# Configure basic logging
logging.basicConfig(
    filename='my_script.log', # Output log file
    level=logging.INFO, # Minimal level that's logged.
    format='%(asctime)s - %(levelname)s - %(message)s', # Formatting the log entries.
    datefmt='%Y-%m-%d %H:%M:%S'  # Formatting dates.
)

def process_data(data):
    logging.info(f"Received data: {data}")
    try:
        result = data * 2
        logging.info(f"Processed data: {result}")
        return result
    except Exception as e:
        logging.error(f"An error occurred during processing: {e}")
        return None


if __name__ == "__main__":
    my_data = 5
    output = process_data(my_data)

    if output is not None:
        logging.info(f"Final result: {output}")
    else:
        logging.warning("Processing encountered an issue.")


```

In this example, I’ve initialized the logging module with `logging.basicConfig()`. This basic setup directs all log messages of level `INFO` or higher (e.g., `WARNING`, `ERROR`, `CRITICAL`) to the file `my_script.log`. I have provided a date/time formatting and a basic message structure. Crucially, the use of `logging.info()`, `logging.error()`, and `logging.warning()` within the script enables me to track different events and issues. If this file is executed using a bash operator, those logs will not show up in Airflow logs directly but would be in the file 'my_script.log'.

Now, let's consider a more complex scenario involving log rotation to prevent a single log file from growing excessively:

```python
# rotating_log_example.py
import logging
from logging.handlers import RotatingFileHandler

# Configure rotating file handler
log_handler = RotatingFileHandler(
    filename='rotating_log.log',
    maxBytes=1024 * 1024, # 1 MB per log file
    backupCount=5 # Keep 5 backups
)
log_handler.setLevel(logging.DEBUG) # Log even more detailed information

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s') # Adding module name to format
log_handler.setFormatter(formatter)

logger = logging.getLogger('my_module')  # Custom logger for a given module
logger.addHandler(log_handler)
logger.setLevel(logging.DEBUG)

def perform_task(input):
    logger.debug(f"Beginning task with input: {input}")
    try:
       output =  input ** 2
       logger.info(f"Task successful, result: {output}")
       return output
    except Exception as err:
       logger.error(f"Task failed with error: {err}")
       return None


if __name__ == "__main__":
    task_input = 10
    result = perform_task(task_input)

    if result is not None:
        logger.info(f"Final value: {result}")
    else:
        logger.warning("Task resulted in an issue.")

```

This example shows the usage of `RotatingFileHandler` to manage log file size and rotation. Instead of just a single `basicConfig`, here, a `RotatingFileHandler` is initiated with specific size limits, and backup count, which can be tailored for diverse applications and log management. Note that `basicConfig` has been avoided here. A custom logger named 'my\_module' is used, and the handlers and level are set for the given logger. This also includes the module name in the log format, which can be helpful for debugging in larger projects.

Finally, consider a scenario where we want to log to both the console (for real-time viewing during development) and a file:

```python
# combined_logging_example.py
import logging

# Get the root logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Create file handler
file_handler = logging.FileHandler('combined_log.log')
file_handler.setLevel(logging.DEBUG) # Log more verbosely to file

# Create console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO) # Less verbosity to console

# Create formatters
file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
console_formatter = logging.Formatter('%(levelname)s: %(message)s')

# Assign formatters
file_handler.setFormatter(file_formatter)
console_handler.setFormatter(console_formatter)

# Add handlers
logger.addHandler(file_handler)
logger.addHandler(console_handler)


def perform_calculations(first_val, second_val):
    logger.info(f"Received values: {first_val}, {second_val}")
    try:
        result = first_val + second_val
        logger.info(f"Result : {result}")
        return result
    except Exception as ex:
        logger.error(f"Calculation failed with exception: {ex}")
        return None


if __name__ == "__main__":
    val_1 = 10
    val_2 = 20
    final_result = perform_calculations(val_1, val_2)

    if final_result is not None:
        logger.info(f"Calculated value: {final_result}")
    else:
         logger.warning("Calculation task encountered an issue.")
```

In this example, I configure the root logger to have both a `FileHandler` and a `StreamHandler`. This allows me to view log messages in the console during execution and also have a detailed log written to `combined_log.log`. Different log levels are set for each to provide more verbose logs to the file and less verbose logs to the console.  Two formatters are used to format each handler differently.

For additional information regarding this, I recommend consulting the official Python documentation on the `logging` module. Specifically, the sections on the various handlers (`FileHandler`, `RotatingFileHandler`, `StreamHandler`, etc.) and formatters are highly beneficial. Further investigation into the capabilities of `logging.config` may prove useful for complex configurations based on external files. Also, for comprehensive understanding, the Airflow documentation regarding operators is useful, including information on bash operators. I would additionally suggest exploring resources detailing best practices in production logging.
