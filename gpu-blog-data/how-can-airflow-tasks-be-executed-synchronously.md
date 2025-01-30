---
title: "How can Airflow tasks be executed synchronously?"
date: "2025-01-30"
id: "how-can-airflow-tasks-be-executed-synchronously"
---
Airflow, by default, orchestrates tasks asynchronously, leveraging its scheduler to queue and execute them based on defined dependencies. This inherent asynchrony is vital for its scalability and efficient handling of complex workflows. However, specific situations arise where synchronous execution of tasks within an Airflow DAG (Directed Acyclic Graph) becomes necessary, albeit generally considered an anti-pattern due to its potential to introduce bottlenecks. I've encountered this directly in scenarios involving tight coupling with external systems where the latency of one operation dictates the start of the next, precluding parallel execution. Achieving synchronous execution requires a deliberate approach that effectively overrides Airflow's default behavior.

The fundamental mechanism to force synchronous task execution involves using mechanisms that block further task progression until the current task has fully completed.  Airflow's `BashOperator` and similar execution operators, by nature, execute a single command and complete. This means a subsequent operator that uses a dependency (`>>` or `set_downstream`) will not execute until the preceding operator has finished. For synchronous execution within a single processing unit, this is already implicitly achieved. However, when the task needs to wait for an arbitrary process to complete, the blocking needs to be explicit within the task itself.

The approach relies on a few core methods:

1.  **Blocking Operators:** The simplest approach involves operators that inherently block until a process finishes. For instance, the `BashOperator` will complete only when the shell command finishes, as will a custom `PythonOperator` if its internal code does not spawn threads or processes and includes blocking operations like awaiting a response.
2. **External Sensors:**  These provide the means to monitor external conditions and only proceed after a particular criteria has been met. For example, if you want to ensure a file is present on a remote server before processing, a sensor should be used.  While this does not control synchronous execution within the task itself, it makes the entire workflow synchronous by preventing the next task from starting prematurely.
3. **Custom Blocking within Operators:**  By modifying the functionality within custom Python operators, we can have operators block execution until a specific condition is met. This is the most flexible, but also carries the most complexity.

Let's illustrate these principles with examples:

**Example 1: Synchronous Execution with `BashOperator`**

```python
from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

with DAG(
    dag_id="synchronous_bash_example",
    start_date=datetime(2023, 1, 1),
    schedule=None,
    catchup=False,
) as dag:
    task_1 = BashOperator(
        task_id="create_directory",
        bash_command="mkdir -p /tmp/airflow_sync_example",
    )

    task_2 = BashOperator(
        task_id="create_file",
        bash_command="touch /tmp/airflow_sync_example/example.txt",
    )

    task_3 = BashOperator(
        task_id="write_to_file",
        bash_command='echo "Hello, world!" > /tmp/airflow_sync_example/example.txt',
    )

    task_1 >> task_2 >> task_3
```

In this example, `task_1`, `task_2`, and `task_3` will execute sequentially, with each task starting only after the preceding task has completed. The `BashOperator` blocks until the shell command is executed which forces the synchronous behavior. While not a particularly interesting example, it demonstrates that operators naturally complete synchronously within a single process.  Each subsequent task relies on the output of the previous one existing (a folder, a file), so that if they were run in parallel, errors would occur.

**Example 2:  Synchronous Workflow with External Sensors**

```python
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.sensors.filesystem import FileSensor
from datetime import datetime

with DAG(
    dag_id="synchronous_sensor_example",
    start_date=datetime(2023, 1, 1),
    schedule=None,
    catchup=False,
) as dag:
    generate_data_task = BashOperator(
        task_id="generate_data",
        bash_command="touch /tmp/data_file.txt && sleep 10",
    )

    wait_for_file_task = FileSensor(
        task_id="wait_for_file",
        filepath="/tmp/data_file.txt",
    )

    process_data_task = BashOperator(
        task_id="process_data",
        bash_command="cat /tmp/data_file.txt",
    )

    generate_data_task >> wait_for_file_task >> process_data_task
```

Here, `generate_data_task` creates a file.  The `wait_for_file_task` is a sensor that waits until that file appears, preventing the execution of `process_data_task` until the file is present. This effectively makes the workflow synchronous with respect to the existence of a file, although the `generate_data_task` and the `process_data_task`  each behave asynchronously by completing and releasing the execution slot once the Bash commands finish, they are part of a synchronously controlled workflow, preventing the processing task from starting before the generate task finishes.

**Example 3:  Custom Synchronous Logic with a `PythonOperator`**

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from time import sleep
from datetime import datetime

def process_with_synchronous_wait(wait_time):
    print(f"Starting long operation, will sleep for {wait_time} seconds")
    sleep(wait_time)
    print("Long operation completed.")

with DAG(
    dag_id="synchronous_python_example",
    start_date=datetime(2023, 1, 1),
    schedule=None,
    catchup=False,
) as dag:
    task_1 = PythonOperator(
        task_id="long_operation_1",
        python_callable=process_with_synchronous_wait,
        op_kwargs={"wait_time": 5},
    )

    task_2 = PythonOperator(
       task_id="long_operation_2",
       python_callable=process_with_synchronous_wait,
       op_kwargs={"wait_time": 3},
    )

    task_1 >> task_2
```

In this final example, the `PythonOperator` executes the `process_with_synchronous_wait` function, which uses `sleep` to simulate a process that needs to finish before the next task can start.  Again, although each python task itself executes in an independent process and reports completion, the `>>` operator guarantees `task_2` will not be executed until after `task_1` is completed, forming a synchronous workflow. The blocking operation is within the called function (the simulated sleep), but it demonstrates the principle of using synchronous operations within a Python task to control workflow timing.

When employing synchronous task execution, it is important to consider that it is counter to Airflow's best design practices. While necessary in certain situations, it introduces dependencies on time-consuming operations that can lead to inefficiencies, reduce throughput and result in potential bottlenecks in your workflow. It also requires careful consideration of resource usage, since task slots will remain blocked until completion, potentially limiting the amount of work that can be executed in parallel if many workflows are configured in this fashion.

For anyone looking to deepen their understanding, I recommend exploring the Airflow documentation, particularly the pages on operators, sensors and DAG scheduling. The Airflow tutorials available in the official documentation and the community examples on Github are a valuable resources that offer practical examples. Additionally, a deep dive into the architecture of Airflow's scheduler is useful when troubleshooting or optimizing complex workflows involving synchronous behaviors. Finally, researching common workflow design patterns will help guide the development of robust solutions, avoiding the over-reliance on synchronous patterns.
