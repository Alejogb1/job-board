---
title: "How can airflow tasks iterating over a list be run sequentially?"
date: "2024-12-23"
id: "how-can-airflow-tasks-iterating-over-a-list-be-run-sequentially"
---

Let's dive right into this. I recall a project a few years back where we needed to process a series of user-uploaded files, each requiring a distinct set of transformation steps. Initially, we naively threw all tasks into the dag, hoping airflow’s parallelism would just ‘handle’ it, which, predictably, led to a chaotic mess of resource contention and failed tasks. It quickly became clear that simply iterating over a list and generating tasks was insufficient; we needed controlled sequential processing. The fundamental issue with directly iterating and creating tasks within a dag, especially when dealing with shared resources or dependencies, is that airflow inherently tries to parallelize them, unless told otherwise.

The core challenge is translating a conceptual sequence of operations over a list into a properly structured airflow dag. While simply iterating and defining tasks seems straightforward, airflow's scheduler treats each task as independent by default. To enforce sequential behavior, we need to use airflow's mechanism for task dependency management. We must explicitly link the tasks together.

One common approach revolves around using the `set_downstream` or `>>` operator to create dependencies manually within a loop. This can be effective for relatively short lists, but it quickly becomes cumbersome and less manageable as the list grows. This method, while functional, can often lead to difficult-to-read code and debug; it doesn't scale well in terms of complexity management.

A more robust solution often involves combining task grouping with a task that serves as a 'starter' and a 'finisher' for each iteration. This allows us to achieve a controlled sequential flow without making the DAG structure too complex or difficult to understand. The essence lies in linking the 'starter' task to the actual processing tasks for a specific item in the list, then linking all processing tasks to the 'finisher'. This way, the processing of an item in the list won't begin until the previous one has completely finished.

Let's solidify this with a few code examples. Assume we have a list of filenames that need processing. We will keep it simple for these examples, but this logic could extend to more complex tasks.

**Example 1: Basic Sequential Processing with `set_downstream`**

This demonstrates a straightforward but less scalable implementation using the `set_downstream` method directly.

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

def process_file(filename):
    print(f"Processing {filename}")

with DAG(
    dag_id="sequential_processing_basic",
    start_date=datetime(2023, 1, 1),
    schedule_interval=None,
    catchup=False
) as dag:

    filenames = ["file1.txt", "file2.txt", "file3.txt"]
    tasks = []
    for filename in filenames:
        task = PythonOperator(
            task_id=f"process_{filename.replace('.', '_')}",
            python_callable=process_file,
            op_kwargs={'filename': filename},
        )
        tasks.append(task)

    for i in range(len(tasks) - 1):
        tasks[i].set_downstream(tasks[i+1])

```

Here, we iterate through filenames, create a task for each, and then explicitly define downstream dependencies to force sequential execution. Note that `filename.replace('.', '_')` is used as task IDs cannot contain periods. This approach works for simple use cases but doesn't scale well with many items.

**Example 2: Sequential Processing with Starter and Finisher Tasks**

This method uses an initial and final task per filename to better encapsulate sequential logic.

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.trigger_rule import TriggerRule
from datetime import datetime

def start_process(filename):
    print(f"Starting processing for: {filename}")

def end_process(filename):
    print(f"Finished processing for: {filename}")

def process_file(filename):
    print(f"Processing {filename}")

with DAG(
    dag_id="sequential_processing_starter_finisher",
    start_date=datetime(2023, 1, 1),
    schedule_interval=None,
    catchup=False
) as dag:
    filenames = ["file1.txt", "file2.txt", "file3.txt"]
    last_finisher_task = None

    for filename in filenames:
      start_task = PythonOperator(
          task_id=f"start_processing_{filename.replace('.', '_')}",
          python_callable=start_process,
          op_kwargs={'filename': filename}
      )

      process_task = PythonOperator(
          task_id=f"process_{filename.replace('.', '_')}",
          python_callable=process_file,
          op_kwargs={'filename': filename}
      )

      finish_task = PythonOperator(
            task_id=f"finish_processing_{filename.replace('.', '_')}",
            python_callable=end_process,
            trigger_rule=TriggerRule.ALL_DONE,
            op_kwargs={'filename': filename}
      )

      if last_finisher_task:
        last_finisher_task >> start_task
      else:
        last_finisher_task=start_task

      start_task >> process_task >> finish_task
      last_finisher_task=finish_task
```

In this example, each file processing sequence is wrapped in start and finish tasks. We maintain the last `finish_task`, setting it as the upstream task of next starting one; we start the processing of file 'n' only after the processing of the file 'n-1' finishes. This provides clearer dependency management.

**Example 3: Using an Airflow SubDAG for Sequential Iteration**

Finally, using a `SubDAG` provides a more encapsulated solution for repeating tasks sequentially. Note that sub-dags are considered an anti-pattern by many in the airflow community, primarily due to issues with concurrency and debugging. However, they can sometimes be useful and are included for completeness.

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.subdag import SubDagOperator
from datetime import datetime
from airflow.utils.trigger_rule import TriggerRule

def process_file_subdag(filename, parent_dag_id, schedule_interval):

    subdag_id = f"{parent_dag_id}.subdag_{filename.replace('.', '_')}"

    with DAG(
       dag_id=subdag_id,
        start_date=datetime(2023, 1, 1),
        schedule_interval=schedule_interval,
        catchup=False,
    ) as subdag:

       def start_process(filename):
           print(f"Subdag Starting processing for: {filename}")

       def process_file(filename):
           print(f"Subdag Processing {filename}")

       def end_process(filename):
           print(f"Subdag Finished processing for: {filename}")

       start_task = PythonOperator(
           task_id=f"start_processing_{filename.replace('.', '_')}",
           python_callable=start_process,
           op_kwargs={'filename': filename}
        )

       process_task = PythonOperator(
           task_id=f"process_{filename.replace('.', '_')}",
           python_callable=process_file,
           op_kwargs={'filename': filename}
        )

       finish_task = PythonOperator(
           task_id=f"finish_processing_{filename.replace('.', '_')}",
           python_callable=end_process,
           trigger_rule=TriggerRule.ALL_DONE,
           op_kwargs={'filename': filename}
        )
       start_task >> process_task >> finish_task
    return subdag


with DAG(
    dag_id="sequential_processing_subdag",
    start_date=datetime(2023, 1, 1),
    schedule_interval=None,
    catchup=False
) as dag:

    filenames = ["file1.txt", "file2.txt", "file3.txt"]
    last_subdag_task = None

    for filename in filenames:

        subdag_task = SubDagOperator(
           task_id=f"subdag_wrapper_{filename.replace('.', '_')}",
           subdag=process_file_subdag(filename, dag.dag_id, dag.schedule_interval),
        )
        if last_subdag_task:
           last_subdag_task >> subdag_task
        last_subdag_task=subdag_task
```

This method encapsulates the sequential task execution into a sub-dag. This approach is more modular; however, as mentioned before, is not recommended for newer projects. The subdag is created by the helper function `process_file_subdag` which then is linked together using the `>>` operator.

In each of these cases, the core concept remains – defining explicit dependencies using `set_downstream` (or its equivalent operator) and ensuring a 'start-process-finish' structure. The specific approach often depends on the size of your list and the complexity of each processing step. For understanding Airflow in-depth I would suggest reviewing the official apache airflow documentation thoroughly. For the architectural implications of these different approaches, the book "Designing Data-Intensive Applications" by Martin Kleppmann offers useful insights, although it does not focus specifically on airflow. Also, I would point you towards the Google SRE book, which gives guidance on how to manage complex systems, useful when thinking about dag creation and maintenance. Finally, a deep dive into the Airflow source code is highly recommended. Remember, effective airflow development comes not just from understanding the mechanisms, but from choosing the correct one for the task at hand.
