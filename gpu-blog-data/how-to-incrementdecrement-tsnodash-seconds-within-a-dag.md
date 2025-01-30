---
title: "How to increment/decrement ts_nodash seconds within a DAG?"
date: "2025-01-30"
id: "how-to-incrementdecrement-tsnodash-seconds-within-a-dag"
---
The fundamental challenge in manipulating `ts_nodash` seconds within an Apache Airflow DAG lies in the fact that `ts_nodash`, representing the execution date and time in YYYYMMDDTHHMMSS format, is a static string determined at DAG run time, not a modifiable integer. Directly incrementing or decrementing it as a numerical value is not the intended approach; instead, we must work with the underlying datetime object from which it's derived. I've encountered this situation repeatedly when needing to schedule tasks relative to the DAG's execution time or partition data based on time increments, often when interacting with downstream systems that utilize specific timestamp formats.

To effectively handle this, we need to leverage the Jinja templating capabilities and Python's datetime module accessible within the Airflow context. The core process involves obtaining the execution date as a datetime object, performing date/time arithmetic on it, and then formatting the resulting datetime back into the desired `ts_nodash` representation. This requires careful attention to timezones to maintain accuracy, especially in distributed systems. The `execution_date` variable, automatically injected into the template context, is key to initiating this manipulation.

Specifically, the `execution_date` comes as an aware datetime object, complete with timezone information, which makes it ideal for reliable calculations. Modifying it requires converting it to a timezone-aware object compatible with the datetime arithmetic library, performing the modifications, and then finally converting it back to the `ts_nodash` format.

Here’s how we can accomplish this through code.

**Code Example 1: Incrementing Seconds**

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from airflow.utils.timezone import utc
from airflow.decorators import task

default_args = {
    'owner': 'airflow',
    'start_date': utc.localize(datetime(2023, 1, 1)),
}

with DAG(
    dag_id='increment_ts_nodash_seconds',
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
    tags=['example']
) as dag:

    @task
    def extract_and_increment(execution_date):
        """
        Extracts execution date, increments by 60 seconds, and returns the modified ts_nodash string.
        """
        modified_date = execution_date + timedelta(seconds=60)
        return modified_date.strftime('%Y%m%dT%H%M%S')

    incremented_ts = extract_and_increment()

    @task
    def display_incremented_ts(incremented_ts):
        """
        Displays the modified ts_nodash string.
        """
        print(f"Incremented ts_nodash value: {incremented_ts}")

    display_incremented_ts(incremented_ts)
```

In this example, `extract_and_increment` receives the `execution_date` automatically from the Airflow context.  We then utilize `timedelta` to add 60 seconds. The `.strftime('%Y%m%dT%H%M%S')` then converts the modified `datetime` object back into the `ts_nodash` format. This illustrates a simple case of incrementing by one minute. It’s important to explicitly use `utc.localize` to define the start_date, to ensure the `execution_date` is timezone aware and avoid unexpected behavior related to timezones. Using `task` decorators is preferred here for easier integration with newer Airflow features. The `display_incremented_ts` task will print the new modified value.

**Code Example 2: Decrementing Seconds and Passing Between Tasks**

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from airflow.utils.timezone import utc
from airflow.decorators import task

default_args = {
    'owner': 'airflow',
    'start_date': utc.localize(datetime(2023, 1, 1)),
}

with DAG(
    dag_id='decrement_ts_nodash_seconds',
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
    tags=['example']
) as dag:

    @task
    def extract_and_decrement(execution_date):
        """
        Extracts execution date, decrements by 30 seconds, and returns the modified ts_nodash string.
        """
        modified_date = execution_date - timedelta(seconds=30)
        return modified_date.strftime('%Y%m%dT%H%M%S')

    decremented_ts = extract_and_decrement()

    @task
    def use_decremented_ts(decremented_ts):
      """
      Demonstrates how the decremented ts_nodash can be used in downstream tasks.
      """
      print(f"Decremented ts_nodash value for downstream usage: {decremented_ts}")

    use_decremented_ts(decremented_ts)

```

This example demonstrates decrementing the `execution_date` by 30 seconds. Similar to the previous example, the core logic is within the `extract_and_decrement` function. Critically, the output from that task is then passed to a subsequent task, `use_decremented_ts`, showing that the modified value can readily be used in downstream operations within the DAG. This highlights the ability to propagate these time-shifted values throughout the workflow, useful when data partitioning or system interactions are required with a time offset.

**Code Example 3: Incrementing Based on a Configuration Variable**

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from airflow.utils.timezone import utc
from airflow.decorators import task
from airflow.models import Variable

default_args = {
    'owner': 'airflow',
    'start_date': utc.localize(datetime(2023, 1, 1)),
}

with DAG(
    dag_id='increment_ts_nodash_config',
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
    tags=['example']
) as dag:

  seconds_to_add = int(Variable.get("seconds_offset", default_var=120))

  @task
  def extract_and_increment_configurable(execution_date, seconds_to_add):
    """
    Extracts execution date, increments by a configurable number of seconds, and returns modified ts_nodash string
    """
    modified_date = execution_date + timedelta(seconds=seconds_to_add)
    return modified_date.strftime('%Y%m%dT%H%M%S')

  incremented_ts_config = extract_and_increment_configurable(seconds_to_add=seconds_to_add)


  @task
  def display_incremented_ts_config(incremented_ts_config):
     """
     Displays the modified ts_nodash string based on config.
     """
     print(f"Configurable Incremented ts_nodash value: {incremented_ts_config}")

  display_incremented_ts_config(incremented_ts_config)
```

This example demonstrates how to dynamically modify the increment (or decrement) value using Airflow variables.  The `seconds_to_add` is retrieved using `Variable.get` which allows the offset to be configured externally without modifying the code. It defaults to 120 seconds if the variable isn’t set. Then, we pass this variable to the `extract_and_increment_configurable` function during its task call, making the shift amount configurable. This is valuable for situations where an offset needs to be changed based on environmental factors or scheduling requirements.

**Resource Recommendations**

For further understanding and development related to this topic, I recommend consulting the official Apache Airflow documentation, particularly the sections pertaining to Jinja templating, the `datetime` module, and task dependencies.  Additionally, exploring examples within the Airflow community repositories can expose various practical implementations. Further study of Python's datetime library and its timezone handling capabilities is highly advised. Reading material on working with DAG contexts and task decorators within the Airflow framework will improve overall proficiency. Lastly, gaining deeper knowledge of Airflow Variables and their use cases will add considerable flexibility and power to your workflows. These focused areas of study will greatly enhance the ability to manage complex temporal manipulation within Airflow DAGs and avoid common pitfalls related to incorrect usage of templating and timezone considerations.
