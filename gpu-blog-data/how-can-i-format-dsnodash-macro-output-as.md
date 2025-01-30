---
title: "How can I format ds_nodash macro output as YYYY/MM in ariflow2?"
date: "2025-01-30"
id: "how-can-i-format-dsnodash-macro-output-as"
---
The `ds_nodash` macro in Apache Airflow 2, while providing a date string representation for task execution, natively outputs in the format `YYYYMMDD`. To obtain the desired `YYYY/MM` format, one must employ Jinja templating within the macro or utilize an alternative date formatting approach within the task's Python code. I’ve encountered this specific formatting challenge numerous times when designing data pipelines needing monthly partitioning or reporting structures.

The core issue arises from the inherent functionality of `ds_nodash` which delivers the execution date without any formatting capabilities beyond removing dashes. Airflow's templating engine, which leverages Jinja2, allows for string manipulation including parsing the date string and reformatting it. Alternatively, one can access the execution date as a datetime object and apply standard Python datetime formatting. I have found both approaches equally valid depending on the context and complexity of the task. Let’s examine how to achieve the `YYYY/MM` output using Jinja templating directly within an Airflow task definition and then review Pythonic manipulation within the task itself.

First, we can reformat the output directly in the task definition using a Jinja filter. The Jinja expression `{{ dag_run.logical_date.strftime('%Y/%m') }}` accesses the execution date as a datetime object, then formats it according to the format string passed to `strftime`. The `strftime` method offers a powerful and standard way to format date and time, with `%Y` signifying the four-digit year and `%m` the zero-padded month. The `dag_run` context variable is available during task execution and provides access to run-specific attributes, including the execution timestamp. This approach bypasses the use of `ds_nodash` entirely. Consider the following code snippet for an example of this method within a BashOperator:

```python
from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

with DAG(
    dag_id='formatted_date_example_1',
    start_date=datetime(2023, 1, 1),
    schedule_interval=None,
    catchup=False
) as dag:
    bash_task = BashOperator(
        task_id='bash_formatted_date',
        bash_command="echo 'Execution date: {{ dag_run.logical_date.strftime('%Y/%m') }}'"
    )
```

In this code, the `BashOperator` uses the specified Jinja expression within the `bash_command` to output the execution date in `YYYY/MM` format to the logs. No intermediate variable assignment is necessary. This method promotes readability, especially for basic formatting requirements. This has been my preferred method for straightforward date formatting in Bash-based tasks as it avoids the need for additional Python code.

Second, one can use Jinja string manipulation to process the output of `ds_nodash` instead of using the execution time directly. While not as concise as the previous example, it demonstrates an alternative when utilizing `ds_nodash` is a requirement. The `ds_nodash` output is accessed through `{{ ds_nodash }}` and then processed as a string. One can use Jinja’s `substring()` filter to extract the relevant parts of the string and assemble the final format. First we take the first 4 characters for the year; then take characters 4 and 5 for the month; then we combine this with the required slash. The code to perform this is shown here: `{{ ds_nodash[:4] }}/{{ ds_nodash[4:6] }}`. This method treats the output as a simple string and does not involve datetime object manipulation. This was my initial approach years ago, when the `strftime` method was less used in Jinja contexts. This example of `BashOperator` usage illustrates the concept:

```python
from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

with DAG(
    dag_id='formatted_date_example_2',
    start_date=datetime(2023, 1, 1),
    schedule_interval=None,
    catchup=False
) as dag:
    bash_task = BashOperator(
        task_id='bash_formatted_date',
        bash_command="echo 'Execution date: {{ ds_nodash[:4] }}/{{ ds_nodash[4:6] }}'"
    )
```

Here, the `BashOperator` task demonstrates that the `ds_nodash` macro output is accessed and then manipulated using string slicing directly within the Jinja template. This is a functional, if less elegant, method for achieving the `YYYY/MM` output.

Finally, consider formatting the date within a PythonOperator. Python gives us direct access to the execution time and we can format using native Python libraries. Here I’d extract the logical date and then apply `strftime` to it as was done previously with Jinja.  The advantage here is increased control and complexity of formatting, and the ability to process further date logic.

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

def format_date(**kwargs):
    execution_date = kwargs['logical_date']
    formatted_date = execution_date.strftime('%Y/%m')
    print(f"Formatted date in Python: {formatted_date}")

with DAG(
    dag_id='formatted_date_example_3',
    start_date=datetime(2023, 1, 1),
    schedule_interval=None,
    catchup=False
) as dag:
    python_task = PythonOperator(
        task_id='python_formatted_date',
        python_callable=format_date,
    )
```

In this code, the `PythonOperator` executes a function that receives task execution parameters. The `logical_date` parameter, which holds the execution timestamp as a datetime object, is passed into the function. This allows the Python code to apply formatting using `strftime` and print the resulting `YYYY/MM` string to the task logs. This approach separates date formatting from the Jinja context, often making complex logic easier to read and maintain.

In summary, three different approaches can be employed to format `ds_nodash` output as `YYYY/MM`. The first uses the `strftime` directly on the execution datetime object using the `dag_run` context. This method is concise and avoids intermediate string manipulation. The second method processes the output of `ds_nodash` as a string using string slicing, and is useful if there is a requirement to use `ds_nodash` explicitly. Finally, the third method is via a Python task that accesses the execution time and formats via standard Python date methods. This method offers the most control over date formatting. I typically prefer the first and third approaches in practice because of their clarity and reduced potential for introducing formatting issues from manipulating string indices in Jinja.

For additional resources, I recommend reviewing the Airflow documentation specifically sections related to macros and Jinja templating, as well as the Python datetime module documentation. Also consider exploring community resources on datetime manipulation techniques in Python, and best practices for date formatting in Jinja2 environments. Examining example DAGs within public Airflow repositories can also provide insight into how other users implement custom date formatting. These are my key go-to resources for advanced Airflow development.
