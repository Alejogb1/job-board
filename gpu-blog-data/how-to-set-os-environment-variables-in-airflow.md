---
title: "How to set OS environment variables in Airflow?"
date: "2025-01-30"
id: "how-to-set-os-environment-variables-in-airflow"
---
Setting environment variables within Apache Airflow requires a nuanced understanding of its architecture and the various execution contexts.  The key fact to grasp is that Airflow's environment variable propagation isn't uniform across all operator types and execution environments.  My experience working on large-scale data pipelines has highlighted the need for a multi-pronged approach, tailored to the specific context.

**1. Understanding the Execution Contexts:**

Airflow's execution unfolds across several layers.  The scheduler orchestrates the DAGs, while individual tasks are executed within worker processes.  Environment variables set globally might not necessarily be available to all tasks depending on how they are invoked.  Furthermore, the distinction between a scheduler environment and an executor environment is crucial.  The scheduler runs continuously and manages the DAGs, while the executor (e.g., CeleryExecutor, LocalExecutor) is responsible for actually running the tasks. This separation means that simply setting an environment variable on your operating system may not suffice.


**2. Methods for Setting Environment Variables:**

There are primarily three reliable methods:

* **Airflow Configuration Files (airflow.cfg):** This approach is suitable for global settings that apply across all DAGs and tasks. Modifications to `airflow.cfg` require restarting the Airflow scheduler for the changes to take effect.  This is generally preferred for variables not specific to a single DAG or task. While less granular, it ensures consistency.

* **Environment Variable in `airflow.cfg`:**  Within the `airflow.cfg` file, specific environment variables can be set directly. This leverages the `[core]` section, using a key-value pair formatted as `env_variable = value`. This value is then accessible within your tasks.  This method is robust for constants needed across multiple DAGs.


* **DAG-Level Configuration using `default_args`:** For variables specific to a particular DAG, the `default_args` dictionary within the DAG definition provides more targeted control.  This allows for setting environment variables accessible by all tasks within that specific DAG.  It enhances organization and avoids global namespace pollution.  Changes here are effective immediately after the DAG is loaded by the scheduler. This approach provides a good balance between global and local scope.


**3. Code Examples:**

**Example 1: Setting environment variables in `airflow.cfg`:**

```python
# airflow.cfg (or equivalent configuration file)

[core]
env_variable_example = my_secret_value
another_env_variable = /path/to/resource
```

This approach populates the `os.environ` dictionary with `env_variable_example` and `another_env_variable`. Access them within your operators using `os.environ.get('env_variable_example')`.  Remember to restart the scheduler after any change to this file.


**Example 2: Setting environment variables via `default_args` in a DAG:**

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import os

with DAG(
    dag_id="dag_with_env_vars",
    start_date=datetime(2023, 1, 1),
    schedule=None,
    catchup=False,
    default_args={'env_vars': {'MY_DAG_VAR': 'dag_specific_value'}}
) as dag:
    task1 = PythonOperator(
        task_id="task1",
        python_callable=lambda: print(f"My DAG variable: {os.environ.get('MY_DAG_VAR')}")
    )

    task2 = PythonOperator(
        task_id="task2",
        python_callable=lambda: print(f"Another DAG variable: {os.environ.get('ANOTHER_DAG_VAR', 'not set')}")
    )

    task1 >> task2
```

This snippet demonstrates setting `MY_DAG_VAR` within the `default_args`.  Note that `os.environ` is still used within the task to access the values.  Crucially, other environment variables like `ANOTHER_DAG_VAR` are not present in this `default_args` section and therefore won't be populated for this DAG.


**Example 3:  Passing Environment Variables to Operators Directly (using `environment`):**

```python
from airflow import DAG
from airflow.providers.postgres.operators.postgres import PostgresOperator
from datetime import datetime

with DAG(
    dag_id="dag_with_operator_env_vars",
    start_date=datetime(2023, 1, 1),
    schedule=None,
    catchup=False,
) as dag:
    postgres_task = PostgresOperator(
        task_id="postgres_task",
        postgres_conn_id="my_postgres_conn",
        sql="SELECT * FROM my_table",
        environment={'DATABASE_URL': 'postgresql://user:password@host:port/database'}
    )
```

This illustrates setting environment variables specifically for a single operator. The `environment` parameter allows injecting environment variables directly into the operator's execution environment.  This is particularly useful for operators that require database credentials or other context-specific settings that should not be stored in the global configuration.  It is more specific but requires attention per operator.


**4. Resource Recommendations:**

I recommend reviewing the official Apache Airflow documentation, focusing on sections related to configuration and operator-specific parameters.  Examining example DAGs provided in the documentation and community repositories will greatly enhance your understanding.  Finally, consulting the documentation for your specific operators is essential, as their handling of environment variables may vary.  Pay attention to the specifics of your chosen executor as well.


**5. Additional Considerations:**

Security best practices dictate avoiding storing sensitive information directly within configuration files or DAG code. Consider using secrets management solutions integrated with Airflow to handle passwords and API keys securely.  This ensures that such sensitive values are not inadvertently committed to version control and reduces risk.  Furthermore, always validate the existence of environment variables before using them to handle cases where they might be missing. Using the `.get()` method with a default value will mitigate potential errors.  Testing your DAGs thoroughly is imperative to confirm that your environment variable setup functions as intended.



My experience suggests a combination of these approaches provides the greatest flexibility and maintainability.  For global constants, the `airflow.cfg` approach is suitable.  DAG-specific variables are best managed with `default_args`, and operator-level variables benefit from the operator's `environment` parameter.  This layered strategy reduces conflicts and improves the overall robustness of your Airflow pipelines.
