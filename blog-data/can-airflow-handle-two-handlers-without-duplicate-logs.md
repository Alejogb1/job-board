---
title: "Can Airflow handle two handlers without duplicate logs?"
date: "2024-12-23"
id: "can-airflow-handle-two-handlers-without-duplicate-logs"
---

Right, let's tackle this question regarding Airflow and duplicate logging when using two handlers. It's a nuanced area, and something I've definitely encountered firsthand in my years managing complex workflow systems. When we talk about handlers in the context of Python's logging framework, we're essentially referring to destinations for your log records – things like the console, files, or remote services. Airflow, as you know, builds heavily upon this foundation.

The short answer is yes, Airflow *can* handle two handlers without duplicate logs, but it's not automatic, and it requires careful configuration. The crucial point here is understanding how Python's logging system works and how Airflow leverages it. By default, if both of your handlers are attached to the same logger *and* that logger's `propagate` attribute is set to `True`, you'll see duplicate entries. Let me break down the mechanics, and then we can look at some code.

The default logging setup in Airflow often results in the root logger propagating log messages to all of its handlers. In practical terms, the logging system operates through a hierarchy. If a child logger doesn’t have a particular handler defined, it will “propagate” the message up the hierarchy to its parent (or ancestor), all the way to the root logger if necessary. If the root logger has multiple handlers, the message gets duplicated. So, if we have both a `FileHandler` and a `StreamHandler` connected to the root logger, every message will be written to the file *and* to the console. That’s why we observe duplication when not careful.

Now, I recall this issue vividly from a previous project at a Fintech firm, where we had an auditing requirement. We needed both a central, searchable log repository (handled by something like Elasticsearch) *and* individual task logs stored locally. We had initial attempts where we just appended handlers without explicitly managing propagation, and the result was… well, a mess. Duplicate logs everywhere.

To avoid this, you need to configure your loggers and handlers in a very deliberate way, specifically targeting the propagation attribute. The main strategy is to make sure that your child loggers do *not* propagate to the root logger where this duplication takes place, or explicitly filter the log entries handled by each handler. We achieve this primarily by setting a logger's `propagate` attribute to `False`.

Let me show you some code examples to illustrate these concepts.

**Example 1: Basic Configuration with Duplicates**

This code shows the setup that causes duplication. Assume this code is executed in a Python script that Airflow is using (for example, in a custom operator).

```python
import logging

# Configure a file handler
file_handler = logging.FileHandler('my_app.log')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

# Configure a stream handler (console)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

# Get the root logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Add both handlers to the root logger
logger.addHandler(file_handler)
logger.addHandler(stream_handler)

# Log a message
logger.info("This log message will appear twice.")

```

If you run this, you'll see the "This log message will appear twice" in both the console and `my_app.log`. The root logger propagates the message to both its handlers.

**Example 2: Preventing Duplicates using `propagate`**

Here’s how to prevent duplication by using the propagate flag. Note we don't manipulate the global root logger, but instead configure our specific use case with its own separate logger.

```python
import logging

# Configure a file handler
file_handler = logging.FileHandler('my_app_specific.log')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

# Configure a stream handler (console)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

# Get a named logger
my_logger = logging.getLogger("my_app")
my_logger.setLevel(logging.INFO)

# Add handlers to this logger
my_logger.addHandler(file_handler)
my_logger.addHandler(stream_handler)

# Prevent propagation to parent loggers
my_logger.propagate = False

# Log a message
my_logger.info("This log message will appear twice.")

# Logging using the root logger for comparison
logger = logging.getLogger()
logger.info("This root message will only appear once")

```

Running this, you'll now notice "This log message will appear twice" appears as expected in both the file and the console, but the "This root message will only appear once" will appear only on the console (since it is using the root logger). If we changed the `my_logger.propagate = True`, we would see both messages appear twice as before. This isolates the messages that use `my_logger`. Notice we did *not* modify the root logger, we just prevented `my_logger` from passing on logs to the root handler.

**Example 3: Using Filters to Further Control Logging**

This example shows another common approach to preventing logging duplication, using the filter mechanism.

```python
import logging

class MyFilter(logging.Filter):
    def filter(self, record):
        return record.name == "my_app"

# Configure a file handler
file_handler = logging.FileHandler('my_app_specific_filtered.log')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s - %(name)s'))

# Configure a stream handler (console)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s - %(name)s'))

# Get a named logger
my_logger = logging.getLogger("my_app")
my_logger.setLevel(logging.INFO)


# Add filter to file handler
file_handler.addFilter(MyFilter())

# Add handlers to this logger, this time we propagate
my_logger.addHandler(file_handler)
my_logger.addHandler(stream_handler)


# Log a message
my_logger.info("This log message with my_app.")
logger = logging.getLogger()
logger.info("This message will only be logged to console")
```

In this instance, we explicitly define a filter. Only the messages coming from the logger called `my_app` will be output to the file, but both messages will be sent to the console (because the root logger's stream handler is still active). We are explicitly controlling the output of each handler using the filter mechanism. Notice we have set the root logger to info. If we had instead set it to error, the first call of logging.info would not appear in console.

So, in essence, for real-world Airflow environments, the key lies in controlling the logging hierarchy. You typically don't want to manipulate the root logger directly (unless you know exactly what you're doing). Instead, the best practice involves defining your custom loggers within your DAGs, custom operators, or plugins, and then carefully setting the `propagate` attribute to `False` where duplication is observed, or using filters, as you've seen in the examples.

For further reading, I would highly recommend thoroughly reviewing the Python logging module’s documentation. Understanding the underlying mechanisms is critical. In addition, “Expert Python Programming” by Tarek Ziadé provides a deep dive into advanced Python topics including logging, and it is useful as reference. Also, reading official Airflow documentation on logging is essential as well; it details how Airflow handles logging and how it fits into the overall Python logging framework. This combination of Python-level understanding and framework-specific knowledge is what's needed to manage complex logging requirements in an Airflow environment without inadvertently creating duplicate entries.
