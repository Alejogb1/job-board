---
title: "Why am I getting ModuleNotFoundError with Airflow external scripts?"
date: "2024-12-23"
id: "why-am-i-getting-modulenotfounderror-with-airflow-external-scripts"
---

Okay, let's talk about that pesky `ModuleNotFoundError` you're encountering with Airflow external scripts. It's a frustration I've certainly felt myself, more than once, during my time wrestling with various DAG deployments. While the error message itself is straightforward enough, pinpointing the *why* usually requires a deeper dive into how Airflow manages its execution environment, especially when external scripts are involved. Believe me, I’ve spent a few late nights debugging very similar issues, and it often boils down to a mismatch between the environment where your DAGs are defined and where your external python scripts are executed.

Essentially, a `ModuleNotFoundError` means that the python interpreter executing your external script cannot locate a module that you're trying to import. This is almost always an issue with your `PYTHONPATH`, or the python environment itself not having access to the package. It's not that python doesn't know the package exists globally but rather it cannot find the specified one from your script, given how it’s executing from within the airflow context. When you're using Airflow, especially with its often complex deployment setups like using a dockerized airflow or with a distributed setup, the problem is compounded by the separation between where the DAGs are parsed and where your tasks are executed.

In a typical scenario, you'll have a DAG definition that calls out to an external script via operators like `BashOperator` or `PythonOperator`. When your DAG parses, airflow needs to know how to find that external script and any libraries the script may need to run. The DAG, including the path to your external script, is typically evaluated by the scheduler, which lives within the Airflow webserver or the scheduler itself, and then passed to the executor. The executor is then responsible for actually executing your script via a python process, or a bash process, on the worker or the server depending on how airflow is deployed. The `ModuleNotFoundError` tends to show up at *that* stage.

I’ve seen this manifest in several ways. For instance, early in my career, I had a situation with a dockerized Airflow setup where the custom python libraries we developed were available on the machine building our docker image, and during the airflow scheduler parse, but weren't actually copied into the container itself. The docker image would execute on the airflow workers, and obviously those libraries were unavailable to python. This was resulting in `ModuleNotFoundError` when the scripts attempted to import our custom libraries. Another instance was involving a distributed airflow system where the external python scripts and the libraries required were not deployed into the same environment, resulting in the remote worker unable to find the required libraries or the external script itself. Let’s dive into three code examples to demonstrate these points.

**Example 1: Incorrect PYTHONPATH Configuration**

Assume you have an external python script, `my_script.py`, in a directory named `scripts`, and that requires a module named `my_package`. Your folder structure is like this:

```
my_project/
├── dags/
│   └── my_dag.py
├── scripts/
│   └── my_script.py
└── my_package/
    └── __init__.py
    └── module_a.py
```

`my_script.py` imports this module and does something, such as:

```python
# scripts/my_script.py
from my_package import module_a

def main():
    print(f"Hello from module_a: {module_a.some_function()}")

if __name__ == '__main__':
    main()

```

Here `my_package/module_a.py` might contain a simple function:

```python
# my_package/module_a.py
def some_function():
    return "This is some function."
```

Now, if you try to execute `my_script.py` with the python operator without a correctly set up `PYTHONPATH`, Airflow won't know where to find `my_package`. Even if your virtual environment has `my_package` installed, the process executing your script may not have the same `PYTHONPATH`. Here is what a simplified DAG looks like:

```python
# dags/my_dag.py
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import os

def run_script():
    script_path = os.path.join(os.path.dirname(__file__), "../scripts/my_script.py")
    # Here we are explicitly setting the PYTHONPATH.
    # This is one solution to make sure the script knows where to find the module
    os.environ["PYTHONPATH"] = os.path.join(os.path.dirname(__file__), "../") # set the right paths

    # we are using a subprocess call here. Usually you would use a python operator in a python script that is included within the package, and run that instead.
    import subprocess
    result = subprocess.run(["python", script_path], capture_output=True, text=True)
    if result.returncode != 0:
       raise Exception(f"Script failed: {result.stderr}")
    print(result.stdout)

with DAG(
    dag_id="module_error_example_1",
    start_date=datetime(2023, 1, 1),
    schedule=None,
    catchup=False
) as dag:
    run_my_script = PythonOperator(
        task_id='run_my_script',
        python_callable=run_script,
    )
```
In this scenario, by explicitly setting the `PYTHONPATH` environment variable within the python operator’s `python_callable` to include the path of your `my_package`, we ensure that the python process executing `my_script.py` is able to find the custom package, even if the environment variable wasn't available to begin with. You can see how not doing this would cause an error because the external script would not know where to import `my_package` from, resulting in `ModuleNotFoundError`.

**Example 2: Issues with Virtual Environments**

Suppose you’re using a virtual environment for development, which is good practice, but the virtual environment activated in the scheduler is different from the one where the external script is being executed by your worker. This difference can occur, especially if Airflow workers are in their own environment, whether docker or on separate VMs.

Consider a slightly modified DAG, that expects to execute in the same virtual environment as the scheduler:

```python
# dags/my_dag_env.py
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import subprocess
import os

def run_script_env():
    script_path = os.path.join(os.path.dirname(__file__), "../scripts/my_script.py")

    result = subprocess.run(["python", script_path], capture_output=True, text=True)

    if result.returncode != 0:
       raise Exception(f"Script failed: {result.stderr}")
    print(result.stdout)

with DAG(
    dag_id="module_error_example_2",
    start_date=datetime(2023, 1, 1),
    schedule=None,
    catchup=False
) as dag:
    run_my_script = PythonOperator(
        task_id='run_my_script',
        python_callable=run_script_env,
    )
```

In this DAG, we expect the `python` command to run from an environment where `my_package` is installed. But if your worker's environment doesn't have the same requirements or if it's not activating your virtual environment correctly, it will result in a `ModuleNotFoundError`. In this instance, you need to make sure the worker either has access to the right virtual environment, or ensure that libraries are installed in the worker's environment.

**Example 3: Dockerized Airflow and Missing Libraries**

Let's say you're running Airflow in a Docker container, and you’re using the `BashOperator` to execute external python scripts. If you don’t build your Docker image carefully, the libraries your external scripts depend on might be missing in your container. For instance:

```python
# dags/my_dag_docker.py
from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime
import os

with DAG(
    dag_id="module_error_example_3",
    start_date=datetime(2023, 1, 1),
    schedule=None,
    catchup=False,
) as dag:
    run_my_script = BashOperator(
        task_id="run_my_script",
        bash_command=f"python {os.path.join(os.path.dirname(__file__), '../scripts/my_script.py')}",
    )

```

If your `Dockerfile` doesn't install `my_package` the bash command that Airflow executes will not have the right libraries installed to handle `my_package`. To resolve this you would need to add a line such as  `RUN pip install -e ./my_package` into the dockerfile (assuming your dockerfile was placed at the root of the project, or the appropriate location) so that the workers will have access to the libraries. This example shows that the dockerized environment must have the library included at build time of the docker container, or have the library available via mount points.

**Recommendations**

To avoid these pitfalls, pay close attention to the following:

1.  **Environment Consistency:** Ensure your scheduler and executor environments (including workers) have identical package requirements. Use a `requirements.txt` file and build it into your docker image or install the packages into the workers' environment. You may want to look into using `pip freeze > requirements.txt` to generate that list from your development environment, and to install those packages with `pip install -r requirements.txt`.
2.  **`PYTHONPATH`:** Always review how the `PYTHONPATH` environment variable is set up for both your scheduler and your workers. Explicitly setting it in your DAG using `os.environ["PYTHONPATH"]` like I did in the example above can help but you should ensure that this path is set to the *location where your script will actually be executed* in the case of distributed workers.
3.  **Docker Considerations:** If you’re using Docker, create your image such that it includes all necessary dependencies. Be mindful of build-time and runtime dependencies for the correct configuration. This includes not only your packages but also making sure the external scripts you intend to run are actually available in the container image itself at the right path.
4.  **Virtual Environment Management:** When using virtual environments, ensure the correct one is activated for each component of your Airflow deployment. If you are using separate worker environments, you should make sure they have the same dependencies.

For deeper understanding of package management, I highly recommend reading the Python Packaging User Guide, particularly the sections related to virtual environments and dependency specifications. A classic reference for python development practices is *Fluent Python* by Luciano Ramalho which provides great information about how import statements work in python. Understanding this will significantly increase the control you have over how python scripts are executed, especially in complex systems like Airflow. Finally, for a deeper understanding of airflow and how it functions on various deployment strategies, I would look into the Apache Airflow documentation. It’s comprehensive and has examples of common deployment scenarios.

In summary, the `ModuleNotFoundError` when executing external scripts in Airflow is almost always an environmental or packaging issue, often linked to the separation between where DAGs are parsed and where their tasks are executed. By controlling the `PYTHONPATH`, package dependencies, and virtual environment configurations, especially when using distributed or dockerized environments, you can avoid most `ModuleNotFoundError` issues and ensure that your workflows run without errors. You need to understand that you're responsible for setting up the environment where your external scripts execute, and when you do that correctly, these types of issues tend to disappear.
