---
title: "How can I use a PythonOperator's list output to iterate another operator in Airflow 2?"
date: "2024-12-23"
id: "how-can-i-use-a-pythonoperators-list-output-to-iterate-another-operator-in-airflow-2"
---

,  From a slightly different angle, instead of launching directly into a theoretical discussion, let's frame it around a scenario I encountered a few years back. We were processing a large dataset of image files. The initial stage involved using a pythonoperator to generate a list of file paths that needed further processing, and we needed to dynamically create tasks based on that list. If you've faced this sort of problem, you'll appreciate the need for efficient, flexible solutions within Airflow. So, how do we achieve this?

Fundamentally, the challenge is to transfer data— specifically, a list— outputted from one task to the next, and then use that data to generate dynamic tasks in Airflow 2.0 and beyond. The core lies in two primary mechanisms: XComs and task mapping. XComs allow tasks to communicate by pushing and pulling small amounts of data, and task mapping empowers us to dynamically expand tasks based on input from XComs. Let’s break this down.

First, let’s examine a basic example of how to utilize a pythonoperator to generate a list. The python function in your pythonoperator *must* return the list, making it available to be pushed to an xcom.

```python
from airflow.decorators import task
from airflow.models.dag import DAG
from airflow.utils.dates import days_ago
import pendulum

@task
def generate_file_list():
    """
    Generates a sample list of file paths. In a real-world scenario
    this function would interact with some storage system to get the data.
    """
    return [f"/data/file_{i}.txt" for i in range(5)]


with DAG(
    dag_id="dynamic_task_example_basic",
    start_date=days_ago(2),
    schedule=None,
    catchup=False,
    tags=['example'],
    ) as dag:
   file_list = generate_file_list()
   print(file_list) #This will print to the logs for the task.
```

In this snippet, our `generate_file_list` function creates a list of sample file paths. The crucial part is that it *returns* the list. Airflow automatically pushes this return value to XCom, using the task id as the xcom key. In your logs you will find this printed. This example does not proceed to map more tasks, but it illustrates the generation of the list, and the printing of the list.

Next, let’s consider how we can then use this list to dynamically create tasks via task mapping. Suppose we want to process each of the files in our generated list using another task called, “process_file”. Here's how we'd accomplish that:

```python
from airflow.decorators import task
from airflow.models.dag import DAG
from airflow.utils.dates import days_ago
import pendulum

@task
def generate_file_list():
    """
    Generates a sample list of file paths. In a real-world scenario
    this function would interact with some storage system to get the data.
    """
    return [f"/data/file_{i}.txt" for i in range(5)]

@task
def process_file(file_path):
    """
    Processes a file. In a real-world scenario this would
    perform an actual operation on the file.
    """
    print(f"Processing file: {file_path}")
    return f"processed: {file_path}"


with DAG(
    dag_id="dynamic_task_example_mapped",
    start_date=days_ago(2),
    schedule=None,
    catchup=False,
    tags=['example'],
    ) as dag:
   file_list = generate_file_list()
   process_files = process_file.override(task_id="process_file").expand(file_path=file_list)
```

In this improved code, `generate_file_list` produces the list as before, but the difference here is how we use the output. We define a separate task named `process_file` which will perform operations on a single file. The important part is `.expand(file_path=file_list)`. This tells Airflow that we will be using the previously created list (`file_list`, which contains the xcom reference automatically) to generate a mapped task with `file_path` parameters. Airflow will dynamically create a task for each item in the list, passing the current list element as the `file_path` argument to the `process_file` task. You will find that there will be 5 distinct task logs for the processing of each file in the original list.

Now, let’s consider a more complex scenario. Imagine our processing needs are a little more intricate. Let’s assume that for each file, we also need to know the location where its processed version should be saved. Perhaps the output path is related to the input path. In this case, we would pass two parameters to the expanded function. This demonstrates that you are not constrained to just one output parameter.

```python
from airflow.decorators import task
from airflow.models.dag import DAG
from airflow.utils.dates import days_ago
import pendulum

@task
def generate_file_info():
   """
   Generates file information (input paths and their associated output paths).
    In a real-world scenario this could be generated from database query, etc.
    """
   files = [f"/data/file_{i}.txt" for i in range(5)]
   output_paths = [f"/processed/file_{i}.txt" for i in range(5)]
   return files, output_paths

@task
def process_file(file_path, output_path):
   """
   Processes a file. In a real-world scenario this would
   perform an actual operation on the file.
   """
   print(f"Processing file: {file_path}, output to: {output_path}")
   return f"processed: {file_path} outputted to: {output_path}"


with DAG(
    dag_id="dynamic_task_example_multi_params",
    start_date=days_ago(2),
    schedule=None,
    catchup=False,
    tags=['example'],
    ) as dag:
    file_paths, output_paths = generate_file_info()
    process_files = process_file.override(task_id="process_file").expand(file_path=file_paths, output_path=output_paths)
```

In this final example, `generate_file_info` returns a tuple, which Airflow automatically splits and uses to map values to the appropriate variables. We are now expanding with two lists, each the same length. Airflow is intelligent enough to realize that it should zip them together, taking the *i*th value of the first list, along with the *i*th value of the second. Therefore, each expanded `process_file` will be provided with a specific `file_path` as well as a corresponding `output_path`. This greatly enhances the flexibility of the task mapping.

Important considerations are in order. First, the input list should not become excessively large, otherwise the task expansion could cause memory issues. If you need to process a very large list of files, you should break it down into multiple dag runs or use other more scalable architectures, like a task that triggers multiple other DAGs. This approach also limits the visibility of the current execution. Secondly, always remember that each expanded task needs a unique task id. If you want to override the task_id, you must use `.override(task_id='some_id')` before expanding the task. Finally, be cautious about pushing large amounts of data into XComs, as Airflow's database might not be optimized for massive data transfers through XComs. Consider pushing smaller metadata and using storage solutions for larger datasets if needed.

For further study, I recommend reading the Apache Airflow documentation thoroughly, particularly the sections on XComs and task mapping. I’d also recommend taking a look at the source code for `airflow.decorators` to gain a better understanding of task and task mapping implementations. The book, “Data Pipelines with Apache Airflow,” by Bas P. Harenslak and Julian Rutger has a great section on dynamic task creation using mapped tasks. For deeper dives into the underlying mechanisms, consider also looking at the Celery documentation, which will elucidate how the tasks are being distributed. Mastering these mechanisms will enable you to construct more resilient and efficient data pipelines using Airflow.
