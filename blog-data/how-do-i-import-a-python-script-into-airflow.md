---
title: "How do I import a Python script into Airflow?"
date: "2024-12-23"
id: "how-do-i-import-a-python-script-into-airflow"
---

Okay, let’s tackle this head-on. Importing a Python script into Apache Airflow might seem straightforward, but it's a process that benefits from a careful understanding of Airflow's execution environment and Python's module import system. Over the years, I've seen a variety of approaches to this, some more maintainable than others, and I've certainly had my share of debugging headaches related to this seemingly simple task. Let’s break it down.

The core challenge isn't really about *importing*; Python’s `import` statement is quite capable. The real issue stems from where Airflow is looking for your scripts when it's executing a task. Airflow tasks are run by workers, often in separate processes or even on separate machines. Therefore, your Python script needs to be discoverable in the environment where the worker executes the code. We can address this in several ways, and I'll outline the most common, practical, and maintainable strategies I’ve found work well.

First, consider the `PYTHONPATH` environment variable. This is probably the simplest starting point, though not always the most robust long-term solution for a large-scale deployment. When Python encounters an `import` statement, it searches specific directories, and `PYTHONPATH` allows you to add additional paths. Imagine you have a script, `my_script.py`, located in a custom directory `/opt/my_scripts`. You could set `PYTHONPATH` to include `/opt/my_scripts` in your Airflow environment's worker configuration. I've done this frequently with smaller setups; it's quick to get up and running, but it becomes cumbersome to manage with numerous directories and changes.

To illustrate this with a code snippet:

```python
# my_script.py
def my_function():
    print("This is my custom function.")

# Inside your Airflow DAG
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

def run_my_script():
  from my_script import my_function
  my_function()

with DAG(
    dag_id='import_using_pythonpath',
    start_date=datetime(2023, 1, 1),
    schedule=None,
    catchup=False
) as dag:
  task_import_script = PythonOperator(
    task_id='import_script',
    python_callable=run_my_script
  )
```

In this example, if you configure your Airflow worker's environment to include `/opt/my_scripts` in `PYTHONPATH` where `my_script.py` resides, the `run_my_script` function should execute successfully. Remember, this requires consistency across all workers and schedulers.

A second, and generally more manageable, approach is to treat your Python scripts as part of a Python package. This brings the benefits of Python's module system, and also helps to structure your code logically. Let’s say, for instance, you create a folder structure like this:

```
my_package/
├── __init__.py
└── scripts/
    ├── my_utils.py
    └── __init__.py
```
Inside `my_utils.py`, you might have a function:

```python
# my_package/scripts/my_utils.py
def another_function():
    print("This is another custom function from a package.")
```

Now, you need to ensure that `my_package` directory is discoverable by Python, just like we did previously with `PYTHONPATH`, but ideally, you'd install this as a Python package. Using `pip install -e .` from within `my_package` (with a `setup.py` file) will install this in "editable" mode and make it available on your Python path.

The `__init__.py` files in the directories tell Python they are packages. This allows you to then import `my_utils` within your Airflow DAG:

```python
# Airflow DAG
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from my_package.scripts import my_utils

def run_package_script():
  my_utils.another_function()

with DAG(
    dag_id='import_using_package',
    start_date=datetime(2023, 1, 1),
    schedule=None,
    catchup=False
) as dag:
  task_import_package = PythonOperator(
    task_id='import_package',
    python_callable=run_package_script
  )

```

This method greatly enhances maintainability, particularly as your project grows in complexity. By treating your custom code as a proper package, you benefit from all the structure Python provides for organizing code and dependencies, and you can distribute your code using standard Python tooling.

The third solution involves utilizing Airflow's plugins mechanism or its built-in support for custom operators or hooks. This is a more advanced approach, especially when you need to encapsulate complex logic or reusable components within your workflows. With custom plugins, you have greater flexibility in extending Airflow’s capabilities. The plugins directory should be specified in your airflow.cfg.

For example, suppose I want to create a custom operator for a specific process, let's say, a file processing operation. I might create a plugin file called `file_processor.py`:

```python
# airflow_plugins/file_processor.py
from airflow.models.baseoperator import BaseOperator
from airflow.utils.decorators import apply_defaults

class FileProcessorOperator(BaseOperator):
  template_fields = ['file_path']

  @apply_defaults
  def __init__(self, file_path, *args, **kwargs):
        super(FileProcessorOperator, self).__init__(*args, **kwargs)
        self.file_path = file_path

  def execute(self, context):
        print(f"Processing file: {self.file_path}")
        # Insert your file processing logic here

from airflow.plugins_manager import AirflowPlugin

class FileProcessorPlugin(AirflowPlugin):
    name = "file_processor"
    operators = [FileProcessorOperator]
```

And then within your DAG, you can use the operator:

```python
# Airflow DAG
from airflow import DAG
from datetime import datetime
from file_processor import FileProcessorOperator # Note that the plugin has to be in your airflow plugin directory

with DAG(
    dag_id='import_using_plugin',
    start_date=datetime(2023, 1, 1),
    schedule=None,
    catchup=False
) as dag:
  process_file_task = FileProcessorOperator(
    task_id='process_file',
    file_path='/path/to/my/file.txt'
  )
```

Using plugins or custom operators allows you to build more modular and specialized components for your Airflow pipelines. This is particularly beneficial for repeatable tasks or operations you frequently use in multiple DAGs.

Ultimately, the best method depends on the scope and complexity of your project. For a small number of scripts, adding them to the `PYTHONPATH` might suffice. However, for anything beyond the very simple, I strongly advise organizing your scripts into Python packages, which also gives you the possibility to distribute your code through PyPI, or similar package repositories. When you need to extend Airflow's core functionality with customized behaviors, using plugins is the logical next step.

For further in-depth understanding, I recommend exploring "Fluent Python" by Luciano Ramalho for Python module and package details; "Programming in Python 3" by Mark Summerfield as a comprehensive guide to the language; and for Airflow specific concepts, the official Airflow documentation is the primary resource, especially sections on plugins and worker configurations. Remember to regularly consult and follow the Airflow community best practices as they evolve. It's an iterative process, and staying up-to-date with best practices ensures your Airflow implementations remain robust and scalable.
