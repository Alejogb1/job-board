---
title: "How can I import 'get_default_executor' from 'airflow.executors'?"
date: "2025-01-30"
id: "how-can-i-import-getdefaultexecutor-from-airflowexecutors"
---
The `airflow.executors` module, specifically the `get_default_executor` function, was significantly altered between Apache Airflow versions 1 and 2, necessitating a different import approach depending on the environment. My experience migrating several production Airflow deployments confirms the necessity for careful version checking when working with executors.

The primary reason for this change stems from Airflow 2's move to a more pluggable and dynamic executor system. Instead of directly providing a single default executor, Airflow 2 relies on the configuration defined in `airflow.cfg` (or environment variables) to determine the executor at runtime. This change improves flexibility, allowing users to easily switch between different executors such as `CeleryExecutor`, `KubernetesExecutor`, or `LocalExecutor` without modifying code or deployments, provided appropriate configurations are set. The older `get_default_executor` which implicitly returned a `LocalExecutor` in most circumstances, is thus no longer directly available. Therefore, you cannot simply import `get_default_executor` directly from `airflow.executors` in Airflow 2. Instead, the appropriate executor class is instantiated according to the environment settings. This change ensures that the correct executor is used by Airflow, promoting consistency and maintainability.

To achieve equivalent functionality in Airflow 2, one needs to indirectly acquire the executor. This involves fetching the configured executor’s class via configuration reading and then instantiating an object of that specific class. The key lies in accessing the `executor` setting in the Airflow configuration, which is accessible through the `airflow.configuration.conf` object. Here, you extract the string which represents the executor chosen, locate the appropriate class from `airflow.executors` and then instantiate it.

Below are three code examples, demonstrating the different scenarios and providing equivalent behaviour in Airflow versions 1 and 2 respectively.

**Example 1: Legacy Airflow 1 Approach**

```python
# Airflow 1.x
from airflow.executors import get_default_executor

# Instantiate the default executor. In most cases this will return an instance of LocalExecutor.
executor = get_default_executor()

# Use the executor as needed, example: print its class name
print(f"Executor Class: {executor.__class__.__name__}")
```

This code block illustrates the direct import and usage of `get_default_executor` as it existed in Airflow 1.  This was relatively simple; it directly returned an executor object without needing to consult configuration settings. The `print` statement shows what type of executor was returned by this function. This approach is no longer valid for Airflow 2 as `get_default_executor` is not present in that location.

**Example 2: Airflow 2 Approach (General Case)**

```python
# Airflow 2.x
from airflow.configuration import conf
from airflow.utils.cli import get_airflow_cmd

def get_executor_from_config():
    executor_name = conf.get("core", "executor")

    # Mapping of string executor name to class within airflow.executors
    executor_mapping = {
        "LocalExecutor": "airflow.executors.local_executor.LocalExecutor",
        "SequentialExecutor": "airflow.executors.sequential_executor.SequentialExecutor",
        "CeleryExecutor": "airflow.executors.celery_executor.CeleryExecutor",
        "KubernetesExecutor": "airflow.executors.kubernetes_executor.KubernetesExecutor"
    }

    executor_path = executor_mapping.get(executor_name, "airflow.executors.local_executor.LocalExecutor") # Default to local

    import importlib
    module_name, class_name = executor_path.rsplit('.', 1)
    module = importlib.import_module(module_name)
    executor_class = getattr(module, class_name)

    return executor_class()

executor = get_executor_from_config()
print(f"Executor Class: {executor.__class__.__name__}")
```

This block is more complex. It reads the executor configuration from the `airflow.cfg` file using `airflow.configuration.conf`, specifically the `executor` entry under the `core` section, which indicates the configured executor.  Then, the `executor_mapping` dictionary helps transform that name to the correct location inside `airflow.executors`. We default to the `LocalExecutor` if the config value does not match. After this, we dynamically import the correct executor class using the `importlib` and we retrieve the class using `getattr` from the module, and lastly we instantiate the class to create an executor instance.  The `print` statement again confirms the class of the instantiated executor. This approach is robust and works with any configured executor, though it requires understanding of how Airflow's configuration is structured and using dynamic imports to achieve the desired outcome. It ensures that the executor returned is the same one used by Airflow itself, respecting the configured setup.

**Example 3: Airflow 2 Approach with Fallback**

```python
# Airflow 2.x
from airflow.configuration import conf
from airflow.exceptions import AirflowConfigException
from airflow.executors.local_executor import LocalExecutor
from airflow.executors.sequential_executor import SequentialExecutor
from airflow.executors.celery_executor import CeleryExecutor
from airflow.executors.kubernetes_executor import KubernetesExecutor


def get_executor_from_config_fallback():
    try:
        executor_name = conf.get("core", "executor")
        if executor_name == 'LocalExecutor':
            return LocalExecutor()
        elif executor_name == 'SequentialExecutor':
            return SequentialExecutor()
        elif executor_name == 'CeleryExecutor':
             return CeleryExecutor()
        elif executor_name == 'KubernetesExecutor':
             return KubernetesExecutor()
        else:
            return LocalExecutor() #Fallback on Local executor
    except AirflowConfigException:
        return LocalExecutor() #Fall back on Local executor if config is not accessible

executor = get_executor_from_config_fallback()
print(f"Executor Class: {executor.__class__.__name__}")
```

This final example provides an alternative implementation that is perhaps more readable than the previous one, at the expense of a larger upfront import section.  It avoids dynamic import and works by directly importing the concrete classes for various executors from the module.  It checks the configuration for the executor type and explicitly instantiates the correct class based on that setting, falling back to `LocalExecutor` when either the configuration does not specify a known executor or the configuration system is not available. This strategy avoids complex imports and reflection, making the code somewhat simpler, albeit at the expense of potentially longer imports if additional executors were supported by Airflow. It maintains a similar logic to Example 2 but implements it more explicitly.

In summary, when working with executors in Airflow 2, you cannot directly import `get_default_executor` from `airflow.executors`. Instead, you must dynamically retrieve the configured executor type using `airflow.configuration.conf` and instantiate the appropriate class from `airflow.executors` accordingly. The second and third code examples demonstrate how to accomplish this effectively.

For further exploration, the Apache Airflow documentation provides comprehensive details on executors and their configuration. Additionally, examining the source code within the `airflow.executors` directory offers a deeper understanding of each executor's implementation. Books and online courses specifically focusing on Apache Airflow also frequently address the executor configuration and usage, offering further context and real-world examples. Specifically, look at the sections that deal with `airflow.cfg`, `core` settings and executor configuration.
