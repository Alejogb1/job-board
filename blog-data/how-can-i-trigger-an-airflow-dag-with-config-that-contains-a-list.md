---
title: "How can I trigger an Airflow DAG with config that contains a list?"
date: "2024-12-23"
id: "how-can-i-trigger-an-airflow-dag-with-config-that-contains-a-list"
---

, let's tackle this one. I've seen this come up more than a few times over the years, especially with the increasing complexity of data pipelines. Configuring Airflow dags using dynamic lists is definitely something you’ll run into when dealing with environments that scale or have varying input requirements. It's not as straightforward as passing a single value, but with a few techniques, it’s quite manageable. The issue is really about how to efficiently pass and interpret that list as part of your dag's configuration parameters.

Initially, back when I was working on a large-scale ETL system for a financial institution, we faced a very similar challenge. Our processing pipelines needed to adjust their behavior based on a variable set of trading instruments, sometimes daily, other times intraday. That meant passing lists of instrument IDs to Airflow.

The key challenge lies in ensuring that the list is correctly serialized and deserialized both when triggering the dag and when it’s accessed by individual tasks within the dag. Airflow’s `conf` parameter, which is passed during dag triggering, primarily handles json-serializable data. This means you can't just dump any python object into it and expect it to magically work.

So, here are some common and effective approaches I’ve found, along with concrete examples. I'll show you how to handle these lists and retrieve them correctly in your tasks.

**Method 1: Passing a JSON Encoded List**

This is probably the most straightforward way to pass a list. You serialize the list into a json string and send it through the `conf` parameter when triggering the dag. The receiving dag's task can then decode that json back to a python list.

Here's an example of how to define the dag, then how to trigger it, and finally how to access the list within a task:

```python
# example_dag_1.py
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import json

def process_list(**kwargs):
    dag_run = kwargs['dag_run']
    if dag_run.conf:
      try:
        instrument_list = json.loads(dag_run.conf.get('instruments', '[]'))
        print(f"Received instruments: {instrument_list}")
        # Perform task logic using the instrument list
        for instrument in instrument_list:
          print(f"Processing instrument: {instrument}")
      except json.JSONDecodeError:
        print("Invalid JSON format for the instrument list")
    else:
       print("No configuration data found.")

with DAG(
    dag_id='example_dag_with_list_1',
    schedule=None,
    start_date=datetime(2023, 1, 1),
    catchup=False
) as dag:
    process_instruments_task = PythonOperator(
        task_id='process_instruments',
        python_callable=process_list
    )
```
To trigger it, you would use the Airflow CLI or the API, something like this:
```bash
airflow dags trigger example_dag_with_list_1 -c '{"instruments": "[\"AAPL\", \"GOOG\", \"MSFT\"]"}'
```

In this example, the python task receives the string `["AAPL", "GOOG", "MSFT"]`, loads it using `json.loads()`, and can then process the list. This approach keeps things relatively clean, since you're using a standard data format. Remember that handling json decode errors is crucial, as malformed json will cause failures.

**Method 2: Passing Comma-Separated Strings**

Another method, slightly less robust but occasionally useful for simpler lists, is to pass a comma-separated string. This is useful if your data consists of basic string values, and you want to avoid json-encoding, making it simpler to pass from the command line. It requires more manual processing in the task.

Here's the second example dag:

```python
# example_dag_2.py
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

def process_instruments_csv(**kwargs):
    dag_run = kwargs['dag_run']
    if dag_run.conf:
        instruments_str = dag_run.conf.get('instruments', '')
        if instruments_str:
          instrument_list = instruments_str.split(',')
          print(f"Received instruments: {instrument_list}")
          for instrument in instrument_list:
              print(f"Processing instrument: {instrument.strip()}") # Using strip in case there are extra spaces.
        else:
            print("No instrument list provided")

    else:
       print("No configuration data found.")

with DAG(
    dag_id='example_dag_with_list_2',
    schedule=None,
    start_date=datetime(2023, 1, 1),
    catchup=False
) as dag:
    process_instruments_task = PythonOperator(
        task_id='process_instruments',
        python_callable=process_instruments_csv
    )
```
Here’s how you’d trigger it using the CLI:
```bash
airflow dags trigger example_dag_with_list_2 -c '{"instruments": "AAPL,GOOG,MSFT"}'
```
This method means you need to split the input string inside the task. Notice that I added `.strip()` to each element in the resulting list because spaces before or after the commas will make the list elements less predictable. You have to be particularly careful with the format when constructing the configuration from the CLI, as this method is less forgiving when extra spaces exist.

**Method 3: Using Jinja Templating with Airflow Variables**

This method offers more flexibility. You can predefine your list as a json-encoded string in an Airflow variable and then use jinja templating in your dag to access it. This works if your list is not strictly dynamic and can be centrally managed.

This is the third example dag:

```python
# example_dag_3.py
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.models import Variable
from datetime import datetime
import json

def process_instruments_var(**kwargs):
    instrument_list_str = Variable.get('instrument_list', default='[]')
    try:
      instrument_list = json.loads(instrument_list_str)
      print(f"Received instruments: {instrument_list}")
      for instrument in instrument_list:
          print(f"Processing instrument: {instrument}")
    except json.JSONDecodeError:
        print("Invalid JSON format for the instrument list")

with DAG(
    dag_id='example_dag_with_list_3',
    schedule=None,
    start_date=datetime(2023, 1, 1),
    catchup=False
) as dag:
    process_instruments_task = PythonOperator(
        task_id='process_instruments',
        python_callable=process_instruments_var
    )
```

Before running the DAG, you'd need to set the Airflow variable via the UI or the CLI:

```bash
airflow variables set instrument_list '["AAPL", "GOOG", "MSFT"]'
```

This approach keeps the configuration outside of the immediate dag execution. The `Variable.get` method will retrieve the contents of this variable.

**Considerations**

Regardless of which method you choose, always remember these points:

*   **Error Handling:** Implement robust error handling, particularly around json parsing and data conversion. Invalid input should never crash your tasks.

*   **Data Validation:** Always validate the input list before passing it further down the pipeline. Check for the expected type and content to avoid downstream errors.

*   **Security:** Be careful when storing sensitive information directly as Airflow variables. Consider using Airflow secrets backend instead.

For further reading and understanding, I would highly recommend:

1.  "Programming in Python 3: A Complete Introduction to the Python Language" by Mark Summerfield: For a comprehensive understanding of python’s handling of data structures and serialization with json, this book is excellent.

2.  The Official Airflow Documentation: Always refer to the official documentation for the latest features and best practices of the version you are using. Pay particular attention to how parameters are accessed within a dag context and the available functionalities of the `dag_run` object.

3.  “Effective Python: 90 Specific Ways to Write Better Python” by Brett Slatkin: For general guidance on writing clean and efficient Python code, particularly around variable usage and data validation.

In conclusion, handling lists as part of the Airflow dag configuration is achievable through various means. Each method has advantages and disadvantages, but the most critical aspect is to process and interpret the information reliably within your tasks. Understanding the nuances of serialization and deserialization is key to successful implementation. I’ve personally found that the first method, passing json-encoded strings, provides a good balance between robustness and simplicity for a majority of use-cases.
