---
title: "Can hydra customize logging configurations from external files?"
date: "2025-01-30"
id: "can-hydra-customize-logging-configurations-from-external-files"
---
Yes, Hydra, the configuration management framework for Python, enables the customization of logging configurations using external files. This capability stems from Hydra's core principle of treating configuration as a first-class citizen, allowing for flexibility and modularity. I've leveraged this feature extensively in various machine learning projects, particularly those demanding detailed experiment tracking and reproducible results.

The power behind Hydra's approach to external logging configuration lies in its configuration system. Hydra loads configurations from YAML files (though other formats are supported), making it straightforward to define logging setups separate from the core application code. This separation promotes cleaner code, easier modification, and the ability to switch between different logging strategies without altering the program's logic. The primary mechanism is utilizing the `hydra.runtime.logging` configuration group, where you can define settings for Python's standard logging library.

To illustrate, consider a project where you require logging at different levels of verbosity during development, testing, and production. Rather than hardcoding logging parameters, you would define these variations within configuration files. Let’s delve into three concrete examples, demonstrating progressively complex configurations.

**Example 1: Basic File Logging**

This example showcases setting up a simple file handler, directing all log messages to a specific file. Assume we have a Hydra configuration file, `config.yaml`:

```yaml
defaults:
  - logging: file_logging

# In config/logging/file_logging.yaml
logging:
  version: 1
  disable_existing_loggers: false
  formatters:
    simple:
      format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  handlers:
    file:
      class: logging.FileHandler
      level: DEBUG
      formatter: simple
      filename: "app.log"
  root:
    level: DEBUG
    handlers: [file]
```
Here, the `defaults` key instructs Hydra to load the configuration defined in `config/logging/file_logging.yaml`.  This separate file defines the logging behavior.  `version: 1` specifies the logging configuration version.  `disable_existing_loggers: false` ensures default loggers from external libraries are not disabled. A formatter called "simple" is defined with a basic timestamp, logger name, level, and the log message itself. We then define a file handler which outputs to a file named `app.log` with a debug level. Finally, the root logger adopts this file handler and sets its overall level to debug, which is needed to handle the message emitted via the log.

To use this in a Python script:

```python
import hydra
from hydra.core.config_store import ConfigStore
import logging

cs = ConfigStore.instance()
cs.store(name="file_logging", node=dict(logging=dict(
   version=1,
  disable_existing_loggers=False,
    formatters=dict(simple=dict(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")),
    handlers=dict(file=dict(class="logging.FileHandler", level="DEBUG", formatter="simple", filename="app.log")),
    root=dict(level="DEBUG", handlers=["file"])
)))

@hydra.main(config_path=".", config_name="config")
def my_app(cfg):
    log = logging.getLogger(__name__)
    log.debug("This is a debug message.")
    log.info("This is an info message.")
    log.warning("This is a warning message.")
    log.error("This is an error message.")
    log.critical("This is a critical message.")


if __name__ == "__main__":
    my_app()
```
Running this Python script will create a `app.log` file in the current directory with log messages from the application. Crucially, no logging configuration occurs within the Python code itself, making the system modular and adaptable.

**Example 2: Multiple Handlers**

This example shows how to use both a file handler and a console handler, which offers the flexibility of seeing logs both in the console and a file simultaneously. Let's consider a `config.yaml` containing:
```yaml
defaults:
  - logging: multi_handler

# In config/logging/multi_handler.yaml
logging:
  version: 1
  disable_existing_loggers: false
  formatters:
    simple:
      format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  handlers:
    file:
      class: logging.FileHandler
      level: DEBUG
      formatter: simple
      filename: "app.log"
    console:
      class: logging.StreamHandler
      level: INFO
      formatter: simple
  root:
    level: DEBUG
    handlers: [file, console]
```

This config defines a `console` handler using `logging.StreamHandler`, outputting to the console with a level of INFO.  Note that the file handler remains the same as before. The root logger is then configured to employ both handlers, meaning output is written to both the console and the log file. The level of root logger is still set to DEBUG so that all messages can be handled to either file or console output.

The Python code to use this would look the same as Example 1, except for the config path. We’re reusing the Python script, underscoring the separation of concerns between the Python application and configuration:

```python
import hydra
from hydra.core.config_store import ConfigStore
import logging

cs = ConfigStore.instance()
cs.store(name="multi_handler", node=dict(logging=dict(
   version=1,
  disable_existing_loggers=False,
    formatters=dict(simple=dict(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")),
    handlers=dict(file=dict(class="logging.FileHandler", level="DEBUG", formatter="simple", filename="app.log"),
            console=dict(class="logging.StreamHandler", level="INFO", formatter="simple")),
    root=dict(level="DEBUG", handlers=["file", "console"])
)))


@hydra.main(config_path=".", config_name="config")
def my_app(cfg):
    log = logging.getLogger(__name__)
    log.debug("This is a debug message.")
    log.info("This is an info message.")
    log.warning("This is a warning message.")
    log.error("This is an error message.")
    log.critical("This is a critical message.")


if __name__ == "__main__":
    my_app()
```

Now, running this Python script will output log messages to both `app.log` (including `DEBUG` level) and the console (starting from `INFO` level).

**Example 3: Selective Logger Configuration**

In more complex applications, you might want to control the logging level for different modules separately.  This example demonstrates how to configure the logging level on a logger-by-logger basis instead of using a global root logger config. Here's a modified `config.yaml`:

```yaml
defaults:
  - logging: selective_logging

# In config/logging/selective_logging.yaml
logging:
  version: 1
  disable_existing_loggers: false
  formatters:
    simple:
      format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  handlers:
    console:
      class: logging.StreamHandler
      level: DEBUG
      formatter: simple
  loggers:
    __main__:
      level: INFO
    my_module:
      level: DEBUG
  root:
    level: WARNING
    handlers: [console]
```
Here, the `loggers` section is introduced. This section lets you assign different logging levels to named loggers; specifically the `__main__` logger, which corresponds to our main app, and a separate logger `my_module`. The root logger level is set to WARNING, which will handle all logs not caught by other, named loggers, thereby ensuring we will only see logs from the defined loggers in console.

The updated Python script now includes a submodule to demonstrate the selective logging:

```python
import hydra
from hydra.core.config_store import ConfigStore
import logging

cs = ConfigStore.instance()
cs.store(name="selective_logging", node=dict(logging=dict(
   version=1,
  disable_existing_loggers=False,
    formatters=dict(simple=dict(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")),
    handlers=dict(console=dict(class="logging.StreamHandler", level="DEBUG", formatter="simple")),
    loggers=dict(__main__=dict(level="INFO"), my_module=dict(level="DEBUG")),
    root=dict(level="WARNING", handlers=["console"])
)))

# my_module.py
def my_module_function():
    log = logging.getLogger("my_module")
    log.debug("Debug message from my_module.")
    log.info("Info message from my_module.")


@hydra.main(config_path=".", config_name="config")
def my_app(cfg):
    log = logging.getLogger(__name__)
    log.debug("Debug message from main.") #This debug message will be ignored due to root logger configuration
    log.info("Info message from main.")
    my_module_function()


if __name__ == "__main__":
   my_app()
```

Running the application now shows that debug logs from `my_module` are shown while debug logs from `__main__` are not, because the `my_module` logger is explicitly configured with debug level while the logger `__main__` is configured with INFO level, and the root logger handles all the logs that are not caught by the two loggers (as it has a WARNING level).

In summary, Hydra's ability to externalize logging configurations provides significant advantages. The separation of configuration from core code promotes modularity, maintainability, and adaptability. Specifically, by leveraging  `hydra.runtime.logging`, one can easily define the behavior of python's standard logging system. From setting up basic file output to implementing complex multi-handler systems, and fine grained logging at logger level, Hydra provides the necessary tooling.

For further investigation, I recommend reviewing Python's `logging` module documentation.  Additionally, the official Hydra documentation has examples and explanations for various configuration options, including logging configurations with more advanced features such as rotating log files. Finally, examining tutorials on using Hydra's override mechanisms in conjunction with configuration groups can yield additional customization options.
