---
title: "How can parallel tasks be sequenced in an Airflow DAG?"
date: "2025-01-30"
id: "how-can-parallel-tasks-be-sequenced-in-an"
---
Implementing sequential dependencies within concurrently executing tasks is a common, yet nuanced, challenge when building Apache Airflow DAGs. Unlike linear task flows where one task completes before the next starts, true parallel processing involves multiple tasks running simultaneously, often leading to the need for careful orchestration of sub-sequences within this parallel activity. Essentially, achieving ordered execution within a parallel context requires careful use of Airflow's task dependencies and sometimes the use of special operators.

Airflow's core design promotes parallel task execution. However, simply using the `>>` or `set_upstream` methods will establish a global sequential dependency within the DAG; it won't facilitate sub-sequences within parallel branches. To sequence tasks within a parallel context, we need to understand Airflow's task dependencies are ultimately interpreted as constraints within the scheduler's task execution logic. They are not a rigid pipeline but rather a series of preconditions that need to be met before a task can execute. This understanding is key to orchestrating parallelism while maintaining fine-grained control over execution order.

My experience building data processing pipelines has led to several practical approaches. The most straightforward is to leverage Airflow's ability to declare dependencies between tasks regardless of their position within a DAG structure. For instance, imagine a scenario where we need to ingest data from multiple sources (A, B, and C) in parallel. Following ingestion, each dataset requires distinct, sequenced transformation steps, before finally merging in a final aggregation task. The ingest tasks run truly in parallel, but their individual transformations must maintain their own linear sequence.

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import time

def ingest_data(source):
    print(f"Ingesting data from {source}")
    time.sleep(2)

def transform_data(source, step):
    print(f"Transforming {source} - Step: {step}")
    time.sleep(1)


def aggregate_data():
     print("Aggregating transformed data")


with DAG(
    dag_id='parallel_sequence_example',
    start_date=datetime(2023, 1, 1),
    schedule=None,
    catchup=False
) as dag:

    ingest_a = PythonOperator(task_id='ingest_a', python_callable=lambda: ingest_data('A'))
    ingest_b = PythonOperator(task_id='ingest_b', python_callable=lambda: ingest_data('B'))
    ingest_c = PythonOperator(task_id='ingest_c', python_callable=lambda: ingest_data('C'))

    transform_a1 = PythonOperator(task_id='transform_a1', python_callable=lambda: transform_data('A',1))
    transform_a2 = PythonOperator(task_id='transform_a2', python_callable=lambda: transform_data('A', 2))

    transform_b1 = PythonOperator(task_id='transform_b1', python_callable=lambda: transform_data('B', 1))
    transform_b2 = PythonOperator(task_id='transform_b2', python_callable=lambda: transform_data('B', 2))

    transform_c1 = PythonOperator(task_id='transform_c1', python_callable=lambda: transform_data('C', 1))
    transform_c2 = PythonOperator(task_id='transform_c2', python_callable=lambda: transform_data('C', 2))

    aggregate = PythonOperator(task_id='aggregate_data', python_callable=aggregate_data)

    #Parallel ingest tasks
    [ingest_a, ingest_b, ingest_c] 

    #Sequencing transformations within parallel branches
    ingest_a >> transform_a1 >> transform_a2
    ingest_b >> transform_b1 >> transform_b2
    ingest_c >> transform_c1 >> transform_c2

    #Aggregation depends on all transformation sequences.
    [transform_a2, transform_b2, transform_c2] >> aggregate
```

In this first example, `ingest_a`, `ingest_b`, and `ingest_c` are independent and can execute in parallel. However, each is followed by a sequence of two transformation steps. The aggregation task will only run after the final transformation tasks associated with each ingestion have completed. This demonstrates the ability to define parallel initial tasks and impose a sequential order on downstream tasks on each branch individually.

A more advanced approach when working with numerous or dynamic tasks is to leverage Airflow's task groups, combined with the ability to declare "dummy" tasks, that only serve as dependency anchors. This strategy keeps the DAG more organized and scalable. In practice, I've used this strategy to control the start and end of groups of sub-tasks within larger DAGs.

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.task_group import TaskGroup
from datetime import datetime
import time

def ingest_data(source):
    print(f"Ingesting data from {source}")
    time.sleep(2)

def transform_data(source, step):
    print(f"Transforming {source} - Step: {step}")
    time.sleep(1)


def aggregate_data():
     print("Aggregating transformed data")


with DAG(
    dag_id='parallel_sequence_taskgroup',
    start_date=datetime(2023, 1, 1),
    schedule=None,
    catchup=False
) as dag:
    with TaskGroup("ingest_tasks") as ingest_tasks:
        ingest_a = PythonOperator(task_id='ingest_a', python_callable=lambda: ingest_data('A'))
        ingest_b = PythonOperator(task_id='ingest_b', python_callable=lambda: ingest_data('B'))
        ingest_c = PythonOperator(task_id='ingest_c', python_callable=lambda: ingest_data('C'))

    with TaskGroup("transform_tasks_a") as transform_tasks_a:
        transform_a1 = PythonOperator(task_id='transform_a1', python_callable=lambda: transform_data('A',1))
        transform_a2 = PythonOperator(task_id='transform_a2', python_callable=lambda: transform_data('A', 2))
        ingest_tasks >> transform_a1 >> transform_a2

    with TaskGroup("transform_tasks_b") as transform_tasks_b:
        transform_b1 = PythonOperator(task_id='transform_b1', python_callable=lambda: transform_data('B', 1))
        transform_b2 = PythonOperator(task_id='transform_b2', python_callable=lambda: transform_data('B', 2))
        ingest_tasks >> transform_b1 >> transform_b2

    with TaskGroup("transform_tasks_c") as transform_tasks_c:
         transform_c1 = PythonOperator(task_id='transform_c1', python_callable=lambda: transform_data('C', 1))
         transform_c2 = PythonOperator(task_id='transform_c2', python_callable=lambda: transform_data('C', 2))
         ingest_tasks >> transform_c1 >> transform_c2

    aggregate = PythonOperator(task_id='aggregate_data', python_callable=aggregate_data)


    [transform_tasks_a,transform_tasks_b,transform_tasks_c] >> aggregate
```

In this example, we organize the ingestion tasks in a task group. The transformations are within their respective task groups where the dependencies ensure the correct sequential order within each group and each is dependent on all the ingest tasks in the previous task group. The aggregate task runs after all the transformations have completed using the dependency between task groups. The visual clarity this provides becomes especially beneficial in complex DAGs.

Finally, a more specific, albeit less common, use case where sequencing is controlled within parallel tasks arises when using a task that is capable of operating on a collection of items, while also needing to control what happens within its scope. For example, this may involve using a task that internally iterates through a collection to achieve a desired sequence of actions on each item. Consider a PythonOperator that processes a list of files. While the main task operates in a single parallel context, it's possible to have internal logic ensure that, for example, each file is processed in a specific order. The sequential processing is achieved within a single parallel task instead of relying on Airflow task dependencies.

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import time

def process_files(file_list):
    for file in file_list:
        time.sleep(1)
        print(f"processing file {file}")
    print("file processing done")


with DAG(
    dag_id='parallel_sequence_single_task',
    start_date=datetime(2023, 1, 1),
    schedule=None,
    catchup=False
) as dag:
    files_to_process = ['file_1.txt', 'file_2.txt', 'file_3.txt', 'file_4.txt']
    process_task = PythonOperator(
        task_id="process_files",
        python_callable=process_files,
        op_kwargs={'file_list': files_to_process},
    )

```

In this example, the `process_files` function will receive the list of files as an argument and process each file in the order listed by iterating through `file_list`. The task itself runs in a parallel context within the DAG, but the operations it performs are sequenced internally based on the python logic of the function. This pattern is suitable for operations where the sequence matters within the task, but not necessarily across different Airflow task instances.

When implementing this logic, I’ve found it beneficial to refer to the official Apache Airflow documentation for detailed information on operators, dependencies, and task groups. Additionally, studying example DAGs provided in Airflow tutorials or the source code repositories of various Airflow providers provides further insights into building maintainable DAGs. Finally, various educational websites provide practical examples and use-cases which are immensely useful for practical application of such concepts. These sources offer comprehensive explanations and best practices that are invaluable when constructing robust and scalable data pipelines.
