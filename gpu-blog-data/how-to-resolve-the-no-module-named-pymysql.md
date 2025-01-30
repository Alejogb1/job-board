---
title: "How to resolve the 'No module named 'PyMySQL'' error in an Airflow DAG scheduler?"
date: "2025-01-30"
id: "how-to-resolve-the-no-module-named-pymysql"
---
When an Airflow DAG fails due to the "No module named 'PyMySQL'" error, it signifies that the necessary Python library, PyMySQL, required to interface with MySQL databases, is absent within the Python environment where the Airflow scheduler is operating. This is a common issue arising from isolated Python environments and dependency management within Airflow deployments. My experience migrating a complex ETL pipeline from a local development environment to a cloud-based Airflow instance taught me the critical importance of meticulous dependency tracking and the different ways it can fail in distributed systems. This issue typically does not stem from a coding flaw within the DAG itself, but rather an insufficient provisioning of the execution environment.

The error arises because Airflow DAGs are essentially Python code that is interpreted and executed by an Airflow scheduler and potentially also worker nodes. These components operate within specific Python environments. If a DAG uses functionality of a specific Python module like PyMySQL, that module must be installed in the environment accessible by these components. If, for example, your local development environment has PyMySQL installed, but the remote server where Airflow runs does not, you will encounter the "No module named 'PyMySQL'" error during DAG execution. The import statement in your Python code, such as `import pymysql`, cannot find the requisite module, causing the scheduler to fail to initiate the scheduled task. Addressing this is not just about installing the library; it's about ensuring the installation is within the correct environment and that it will be accessible to both the scheduler and the worker nodes if applicable.

There are several ways to resolve this issue, each with varying applicability based on the Airflow setup. A basic approach is to use pip, the standard Python package installer, to install `pymysql`. However, the devil lies in the details, particularly in ensuring that pip installs the package to the Python environment used by the Airflow components and not just within your user environment.

Below are three code examples with explanations that cover typical scenarios that cause this kind of problem, followed by the most straightforward method to resolve it. The first example showcases a basic DAG that would trigger this error, the second demonstrates a naive solution that might be insufficient, and the third provides a more robust approach suitable for cloud environments.

**Code Example 1: Triggering the Error**

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import pymysql # Intentional import that will cause the error if not present

def connect_to_mysql():
    try:
        connection = pymysql.connect(
            host="your_mysql_host",
            user="your_mysql_user",
            password="your_mysql_password",
            db="your_mysql_db"
        )
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            print(f"MySQL Connection Successful, query returned: {result}")

        connection.close()

    except Exception as e:
        print(f"MySQL Connection Failed: {e}")


with DAG(
    dag_id="mysql_test_dag",
    start_date=datetime(2023, 1, 1),
    schedule=None,
    catchup=False
) as dag:
    test_mysql_task = PythonOperator(
        task_id="test_mysql_connection",
        python_callable=connect_to_mysql
    )
```

This DAG, `mysql_test_dag`, attempts to connect to a MySQL database using PyMySQL. If PyMySQL is not installed in the Airflow environment, the scheduler would encounter the "No module named 'pymysql'" error during the initial parsing of the DAG. The error would occur *before* the task is actually run and will prevent it being scheduled. The traceback will pinpoint the line with the `import pymysql` statement.

**Code Example 2: Insufficient Solution using Pip in DAG**

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import subprocess # Intentionally using subprocess for poor practice

def install_and_connect():
    try:
        subprocess.check_call(["pip", "install", "pymysql"]) # Never do this, unless you are using a very specific setup with specific purpose

        import pymysql
        connection = pymysql.connect(
            host="your_mysql_host",
            user="your_mysql_user",
            password="your_mysql_password",
            db="your_mysql_db"
        )
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            print(f"MySQL Connection Successful, query returned: {result}")

        connection.close()

    except Exception as e:
        print(f"Error: {e}")



with DAG(
    dag_id="mysql_test_dag_install_pip",
    start_date=datetime(2023, 1, 1),
    schedule=None,
    catchup=False
) as dag:
    install_and_test_task = PythonOperator(
        task_id="install_and_test_mysql",
        python_callable=install_and_connect
    )

```

This approach is flawed as installing packages with `subprocess.check_call` inside a DAG is generally not recommended. While this might seem to address the problem at first glance, it introduces several potential issues. Firstly, it will only install the library for the *specific task execution* in which it runs. It might also fail, because the user running the subprocess might not have the permission to install the package. If the package is installed successfully, it may be installed into the Python environment of the task itself, and *not* the Python environment of the scheduler. This means that if the DAG imports `pymysql` at the module level (as shown in the first example), it will fail with the same "No module named 'pymysql'" error. Finally, this approach makes dependency management very hard to track.

**Code Example 3: Correcting the issue using Airflow's `requirements.txt`**

This method does not involve any changes to the DAG code. This is because the correct solution is *not* related to changes to the DAG code. The problem is with the configuration of Airflow.

The solution to this problem is to ensure that `pymysql` (or any other dependencies) is added to a `requirements.txt` file and that the environment for Airflow is configured to use this file. In some installations of Airflow, this is done by creating a `requirements.txt` file in the same directory as the DAGs, and updating the configuration to read this file. Here is the content of `requirements.txt`:

```
pymysql
```

The exact method by which this `requirements.txt` is implemented depends on the implementation of the Airflow deployment. For example, when deploying Airflow using Docker, you might need to rebuild the docker image with this requirement. In cloud offerings, there are typically configuration options to specify such dependency files.

After updating the environment with the package requirement, the DAG will execute as intended. This solution is the *correct* solution to the original question.

In summary, the "No module named 'PyMySQL'" error arises due to missing dependencies within the Python environment used by the Airflow scheduler. Directly modifying your DAG is not recommended; instead, focusing on providing the correct packages during the environment setup is required. Correcting the dependencies also needs to account for how the system works - the `requirements.txt` file is the intended way to supply dependencies to Airflow, and not through a subprocess command within the DAG.

To further improve your understanding and management of Python dependencies within Airflow, I recommend studying the documentation provided for your specific deployment type (e.g., Kubernetes, Docker, or cloud provider implementations). Familiarize yourself with concepts like virtual environments, container image builds, and the role of configuration files. A deep understanding of these topics not only solves the described error, but makes your deployments more robust and easier to manage. Further reading on the usage of `requirements.txt` within Python projects, the differences between module-level and task-level imports, and best practices for setting up a reliable execution environment would be helpful. Furthermore, exploring tools to aid dependency management like Poetry and pip-tools can add greater control to the process of creating and maintaining robust development and production environments.
