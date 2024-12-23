---
title: "How do I set PYTHONPATH in AWS MWAA?"
date: "2024-12-23"
id: "how-do-i-set-pythonpath-in-aws-mwaa"
---

Alright, let's talk about `PYTHONPATH` in the context of AWS Managed Workflows for Apache Airflow (MWAA). It’s a topic that often surfaces, and I recall a particular incident a few years back when a colleague was wrestling with a similar setup. We had a rather complex multi-package project, and getting MWAA to recognize our custom libraries felt… well, let's just say it involved a fair amount of troubleshooting. So, I'll share my experience and insights based on those lessons learned.

Essentially, `PYTHONPATH` is an environment variable that tells the Python interpreter where to look for module files when you `import` something. Normally, Python searches the standard library directories, plus the directory of the script being executed. But if your custom packages or modules aren't located in those default places, you need to inform Python using `PYTHONPATH`. In an MWAA environment, this becomes particularly important when your DAGs (Directed Acyclic Graphs) need to utilize custom operators, hooks, or utilities residing outside of the default paths. This can occur when working with a multi-module project that's not entirely installed as a Python package within the MWAA environment.

There isn't a single "correct" way to approach this, and it often depends on your specific use-case and how you're packaging and deploying your code. The key is to make sure your python environment within the MWAA execution environment has access to your custom packages. From my experience, you have three main options which are the most reliable.

**Option 1: Including packages in the 'requirements.txt'**

The preferred method is packaging your python code and then declaring these packages as dependencies in your 'requirements.txt' file and uploading that file during the MWAA environment creation or update. This method is usually the easiest and most straightforward for most cases.

Here's how it typically plays out. Let's assume we have a project with the following structure:

```
my_project/
├── my_package/
│   ├── __init__.py
│   ├── my_module.py
│   └── sub_package/
│        ├── __init__.py
│        └── utility.py
├── dags/
│   └── my_dag.py
└── requirements.txt
```

Inside `my_package/my_module.py`, we might have a simple class:

```python
# my_package/my_module.py
class MyClass:
    def greet(self, name):
        return f"Hello, {name}!"
```

Now, the key is to package this into an installable python package. Within your `my_package` directory you need to add a `setup.py` file:

```python
# my_package/setup.py
from setuptools import setup, find_packages

setup(
    name='my_package',
    version='0.1.0',
    packages=find_packages(),
)
```

Now we need to create the python package archive:

```bash
cd my_project
python my_package/setup.py sdist
```

This will create a `dist` directory, which will contain our package `.tar.gz` archive. This needs to be added to the `requirements.txt`. Let's also assume our project also uses `requests`:

```
# requirements.txt
requests
./dist/my_package-0.1.0.tar.gz
```

Now in the MWAA DAG, you can then use the custom package like this:

```python
# dags/my_dag.py
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from my_package.my_module import MyClass

def my_task():
    my_instance = MyClass()
    message = my_instance.greet("Airflow")
    print(message)

with DAG(
    dag_id='example_pythonpath_1',
    start_date=datetime(2023, 1, 1),
    schedule_interval=None,
    catchup=False,
) as dag:
    task1 = PythonOperator(
        task_id='my_task',
        python_callable=my_task,
    )

```

In this approach, the MWAA environment's Python interpreter knows exactly where to locate `my_package` because it's treated as a dependency, just like `requests`. This approach works well in most cases because it ensures the environment is configured correctly and that the code is consistently available during the DAG execution.

**Option 2: Using the `plugins` folder and adjusting sys.path**

Another option which was useful in our case when dealing with the complex multi-package setup I mentioned earlier, is placing your custom packages in the MWAA's plugins directory and then explicitly modifying the `sys.path` in your code. MWAA automatically adds the `plugins` directory to the Python path. I will note that while this was useful in my experience, this does mean that your code is outside of the normal dependencies paradigm for python packages, which can make maintenance harder, and this approach is less ideal than option 1.

In the S3 bucket connected to MWAA, you would upload your packages to the `/plugins` folder, while ensuring you keep your folder structure:

```
s3://my-mwaa-bucket/
└── plugins/
    └── my_package/
        ├── __init__.py
        ├── my_module.py
        └── sub_package/
            ├── __init__.py
            └── utility.py
```

Then, within your DAG or any Python callable within the DAG, you can adjust the system path:

```python
# dags/my_dag.py
import sys
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import os

# Determine the plugins folder path based on the environment variables, or revert to a reasonable default if not found.
plugins_path = os.environ.get("AIRFLOW_HOME", "/usr/local/airflow") + "/plugins"
sys.path.append(plugins_path)


from my_package.my_module import MyClass


def my_task():
    my_instance = MyClass()
    message = my_instance.greet("Airflow Plugins Method")
    print(message)


with DAG(
    dag_id='example_pythonpath_2',
    start_date=datetime(2023, 1, 1),
    schedule_interval=None,
    catchup=False,
) as dag:
    task1 = PythonOperator(
        task_id='my_task',
        python_callable=my_task,
    )
```

Here, the key lies in dynamically obtaining the path to the `/plugins` folder on the MWAA environment via the `AIRFLOW_HOME` environment variable, and then appending this path to `sys.path`. This makes the custom packages in that location discoverable by the Python interpreter. This works because MWAA automatically adds the plugin path to the available locations which are searched during a python import.

**Option 3: Customizing the `PYTHONPATH` environment variable via configuration**

Lastly, you can explicitly set the `PYTHONPATH` environment variable within the MWAA environment configuration. This method gives you the most flexibility but also requires more understanding of the MWAA setup and environment. In this scenario, you can add your custom packages to an S3 bucket, and during MWAA env creation or update, you can specify that location via an environmental variable override.

For instance, suppose your S3 bucket is structured like this:

```
s3://my-mwaa-bucket/
└── custom_libs/
    └── my_package/
        ├── __init__.py
        ├── my_module.py
        └── sub_package/
            ├── __init__.py
            └── utility.py
```

Then during the creation or update of the MWAA environment, you can add environment variables, within the 'Environment Configuration' settings under 'Environment Variables', with key `PYTHONPATH` and value `s3://my-mwaa-bucket/custom_libs` . Note that you will need to make sure that this S3 location has permissions to be read from the MWAA environment. You also need to ensure that any directories that should be included in the python path are added to the `PYTHONPATH`, and that `PYTHONPATH` already contains some default directories from MWAA, therefore make sure not to overwrite, but to append to the current value.
 Then within your DAG file, or custom package files you can import your custom modules:

```python
# dags/my_dag.py
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from my_package.my_module import MyClass


def my_task():
    my_instance = MyClass()
    message = my_instance.greet("Airflow ENV Var Method")
    print(message)


with DAG(
    dag_id='example_pythonpath_3',
    start_date=datetime(2023, 1, 1),
    schedule_interval=None,
    catchup=False,
) as dag:
    task1 = PythonOperator(
        task_id='my_task',
        python_callable=my_task,
    )
```

In this case, MWAA sets the `PYTHONPATH` variable during the initialization of the environment, causing the Python interpreter to search the defined path for modules. This method works, however, it also has a few issues, primarily, that the python package dependencies are not being installed through the proper method, and that this relies heavily on manually maintaining the environmental variable values, which might be less maintainable.

**Recommendations**

For most cases, I would recommend the first approach utilizing `requirements.txt` and packaging your code properly. It's the cleanest and easiest to maintain long-term. For cases when that approach is not applicable, the secondary approach utilizing the plugins directory is also useful. The third option, while workable, should be avoided unless there is a specific reason that cannot be covered by the first two approaches.

For further reading, I suggest exploring these resources:

*   **"Effective Python: 90 Specific Ways to Write Better Python" by Brett Slatkin:** Provides detailed insights into Python best practices, including package management, which helps understand why certain methods are better than others.
*   **"Python Cookbook" by David Beazley and Brian K. Jones:** A comprehensive resource for Python developers, which can help with creating robust and easily maintainable Python packages.
*   **The Apache Airflow documentation:** Pay close attention to the sections on environment setup, custom providers, and plugins, as they provide specific insights into configuring Airflow in a way that’s compatible with MWAA.

Setting up `PYTHONPATH` can sometimes be challenging, but taking the time to understand how Python locates modules, and the specific nuances of MWAA, will save you a lot of headaches. These examples should offer a solid foundation, and these recommendations will help you navigate the world of Python development with confidence.
