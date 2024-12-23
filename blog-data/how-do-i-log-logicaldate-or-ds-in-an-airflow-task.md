---
title: "How do I log 'logical_date' or 'ds' in an Airflow task?"
date: "2024-12-23"
id: "how-do-i-log-logicaldate-or-ds-in-an-airflow-task"
---

,  I’ve definitely had my share of encounters with date handling in Airflow, particularly when trying to ensure consistency across my DAGs, so I think I can shed some light on this. The question of accessing ‘logical_date’ or ‘ds’ (the execution date) within your Airflow tasks is fundamental for most workflows, especially those dealing with time-series data. It’s not just about getting the date; it’s about getting *the* date, the one Airflow uses to define that particular execution.

The magic here lies in Airflow's templating system. When you define a task within a DAG, Airflow makes certain context variables available to you through Jinja templating. The variables `ds` and `logical_date` are among these crucial context pieces.

The `ds` variable is a string representation of the execution date in the format 'YYYY-MM-DD'. It's very handy for file path construction or any situation where you need a simple string representation. On the other hand, `logical_date` represents the execution date as a datetime object, providing more flexibility for date manipulation if required. Internally, `ds` is actually derived from `logical_date`. Now, let's dive into how you'd typically use these in your tasks.

I remember struggling with this quite a bit when I was building a financial data pipeline a few years ago. We needed to accurately name our daily data files using the Airflow execution date and later when we wanted to compute specific date-based aggregates, these context variables were crucial.

Let's look at some code.

**Example 1: Using `ds` for file path creation:**

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

def create_data_file(**context):
    execution_date = context['ds']
    file_path = f"/data/raw/{execution_date}/data.csv"
    # Simulate writing to a file (in a real scenario, you'd have actual data writing)
    with open(file_path, 'w') as f:
        f.write(f"Data for {execution_date}")

with DAG(
    dag_id='example_ds_path',
    start_date=datetime(2023, 1, 1),
    schedule='@daily',
    catchup=False
) as dag:
    create_file_task = PythonOperator(
        task_id='create_data_file',
        python_callable=create_data_file
    )
```

In this first example, the `create_data_file` function receives the `ds` context variable from Airflow. We then use that string to construct a file path. When this task runs as part of the DAG, Airflow will execute this function and `ds` will contain the logical date of that particular execution. It’s straightforward and very common. The point here is that when you examine the output, you'll see a new folder created with the corresponding execution date. If the DAG ran for instance, on the 2023-01-02 it will create /data/raw/2023-01-02/data.csv.

**Example 2: Using `logical_date` for date manipulation:**

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

def calculate_previous_date(**context):
    execution_date = context['logical_date']
    previous_date = execution_date - timedelta(days=1)
    print(f"Logical Date: {execution_date}, Previous Date: {previous_date}")

with DAG(
    dag_id='example_logical_date_manipulation',
    start_date=datetime(2023, 1, 1),
    schedule='@daily',
    catchup=False
) as dag:
    calculate_date_task = PythonOperator(
        task_id='calculate_previous_date',
        python_callable=calculate_previous_date
    )
```

In the second scenario, we're demonstrating the utilization of `logical_date`. Notice how this one provides a full datetime object. We then use standard python date manipulation methods to derive the date from the previous day, which is exactly what you need when dealing with data roll ups or windowed computations. You can do much more complex date math using datetime objects if your pipeline requires it.

**Example 3: Accessing and Formatting date in templated bash commands:**

```python
from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

with DAG(
    dag_id='example_templated_bash',
    start_date=datetime(2023, 1, 1),
    schedule='@daily',
    catchup=False
) as dag:
    bash_task = BashOperator(
      task_id='echo_date',
      bash_command="echo 'Execution date is {{ ds }} and logical date is {{ logical_date }}'"
    )

```

This third example highlights another critical use-case. Often you will want to use these dates not in a PythonOperator but in other operators. Templating with `{{ ds }}` or `{{ logical_date }}` allows you to seamlessly use these dates within templated bash commands or other operators that take Jinja-templated arguments. Here the bash command will output the `ds` (execution date as YYYY-MM-DD string) as well as the full `logical_date`.

A note of caution – when using `{{ logical_date }}` within Bash commands or similar templated strings, its datetime representation will be passed as a string, so ensure any string manipulation you perform is well-defined. If you need a specific format, you can also leverage the Jinja date filter capabilities. For instance, you could do something like `{{ logical_date.strftime('%Y%m%d') }}` to format the `logical_date` into a string formatted as 'YYYYMMDD'.

From experience, this templating functionality is incredibly powerful and becomes essential in any moderately complex pipeline. It allows for consistent data processing and naming conventions, thereby aiding greatly in the maintainability of your workflows.

Beyond these basics, keep in mind that Airflow's backfill mechanism affects the behavior of these dates. When you backfill a DAG, each task run is associated with the date for which the data was *intended* to be processed, not the date when the task was actually executed. This is key to comprehend and test thoroughly.

For deepening your knowledge on Airflow's templating, I would highly recommend the official Airflow documentation; start with their section on "Jinja Templating" because it’s always the best resource. You could also look at the book “Data Pipelines with Apache Airflow” by Bas Harenslak and Julian Rutger and "Programming Apache Airflow" by Erik Romijn and Mark West. These are invaluable resources for understanding the intricacies of Airflow, including, of course, the powerful templating engine which is the key to the subject of this discussion. These materials will assist you not just with date handling, but with all of Airflow's major features. In addition to this you may also want to investigate how to use Macros in Airflow templates, which will give you further flexibility when generating strings based on dates.

In summary, accessing `ds` and `logical_date` within Airflow tasks is critical for ensuring your data pipelines are consistent and correct. By using Jinja templating, you can seamlessly integrate these date variables into your task definitions, enabling proper path construction, date manipulation, or whatever other needs might arise. Understanding this aspect is a cornerstone of effective workflow management using Airflow.
