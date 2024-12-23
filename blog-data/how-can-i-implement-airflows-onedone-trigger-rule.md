---
title: "How can I implement Airflow's one_done trigger rule?"
date: "2024-12-23"
id: "how-can-i-implement-airflows-onedone-trigger-rule"
---

Alright, let's talk about `one_done` trigger rules in Apache Airflow. It's a situation I've encountered more than once, especially when dealing with complex branching workflows and needing precise control over downstream tasks. Rather than immediately diving into code, let's first clarify what `one_done` actually implies in the context of Airflow, and why you'd even reach for it.

Essentially, the `one_done` trigger rule means that a downstream task should only be triggered once _at least one_ of its upstream tasks has successfully completed. This contrasts with Airflow's default trigger rule of `all_success`, which demands all upstream tasks succeed. The subtlety lies in its behavior; the success of *any* upstream task is sufficient. It’s not “the first to complete,” rather it’s satisfied the moment a single success is recorded amongst the upstream dependencies, regardless of the success or failure of any other parents. This isn't a common requirement in many standard workflows, which is precisely why it can be overlooked. I initially brushed it off myself until I was dealing with some data-processing pipelines that had a fault-tolerant ingest mechanism.

The particular project involved ingesting data from multiple sources, and any *single* successful source was sufficient to kick off the downstream transformations; it didn't matter if some sources failed that day. Using `all_success`, the transformations would have been perpetually stalled, leading to unnecessary operational overhead. I was tasked to find a solution, and that's where I became much better acquainted with `one_done`. This allowed me to ensure that even when one source pipeline failed, the system would still process data and, thus, meet our internal service level agreements.

The way you implement this in Airflow is straightforward; it's done within the definition of your dag when declaring a task, or directly in the task's setup. Let’s look at some examples and dissect them.

**Example 1: Basic Implementation**

Let’s establish the base case. Here's a minimal DAG snippet showcasing how you'd use the `one_done` trigger rule. It assumes we have three upstream tasks – `extract_a`, `extract_b`, and `extract_c` – and a downstream task – `transform_data`. We will simulate their execution with `BashOperator`. This is fairly standard practice.

```python
from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

with DAG(
    dag_id='one_done_example_1',
    start_date=datetime(2023, 1, 1),
    schedule=None,
    catchup=False
) as dag:

    extract_a = BashOperator(
        task_id='extract_a',
        bash_command='sleep 5 && echo "Extraction A complete"'
    )

    extract_b = BashOperator(
        task_id='extract_b',
        bash_command='sleep 2 && echo "Extraction B complete"'
    )

    extract_c = BashOperator(
        task_id='extract_c',
        bash_command='sleep 10 && echo "Extraction C complete"'
    )


    transform_data = BashOperator(
        task_id='transform_data',
        bash_command='echo "Transforming data"',
        trigger_rule='one_done'
    )

    [extract_a, extract_b, extract_c] >> transform_data
```

In this setup, even if `extract_c` takes a long time or fails, the `transform_data` task will trigger as soon as either `extract_a` or `extract_b` succeeds. It is this property that is so powerful. Without specifying the trigger rule, `transform_data` would wait indefinitely for all extractions to complete successfully, even if one is sufficient.

**Example 2: Handling Failures and One_Success**

Now, let's consider a more practical example with one potential failing extraction. In this case, let’s also introduce another trigger rule `one_success` for comparison and demonstrate a common misinterpretation that may occur. We will demonstrate how `one_success` and `one_done` behave differently.

```python
from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

with DAG(
    dag_id='one_done_example_2',
    start_date=datetime(2023, 1, 1),
    schedule=None,
    catchup=False
) as dag:

    extract_a = BashOperator(
        task_id='extract_a',
        bash_command='sleep 2 && echo "Extraction A complete"'
    )

    extract_b = BashOperator(
        task_id='extract_b',
        bash_command='exit 1'  # Intentionally fail
    )

    extract_c = BashOperator(
        task_id='extract_c',
        bash_command='sleep 5 && echo "Extraction C complete"'
    )


    transform_data_one_done = BashOperator(
        task_id='transform_data_one_done',
        bash_command='echo "Transforming data using one_done"',
        trigger_rule='one_done'
    )


    transform_data_one_success = BashOperator(
        task_id='transform_data_one_success',
        bash_command='echo "Transforming data using one_success"',
        trigger_rule='one_success'
    )

    [extract_a, extract_b, extract_c] >> transform_data_one_done
    [extract_a, extract_b, extract_c] >> transform_data_one_success
```

Here, `extract_b` is designed to fail. With `one_done`, as soon as `extract_a` (or `extract_c`) succeeds, the `transform_data_one_done` will proceed. Crucially, `transform_data_one_success` will *not* trigger because it explicitly requires one successful outcome. One failure is still a completed state so it will eventually satisfy `one_done` however `one_success` requires at least one success. In many cases, this differentiation is vital for fine-tuning workflow dependencies based on specific tolerance levels. This distinction often trips people up and is a practical consideration when using either trigger rule.

**Example 3: Integration with a Branching Operation**

Now, let’s consider a more involved example: a branching situation. Often, in scenarios using dynamic workflows, `one_done` can reduce unnecessary resource consumption and complexity. Here, we have a branching condition based on an external system response.

```python
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.utils.trigger_rule import TriggerRule
from datetime import datetime
import random

def simulate_external_check(**kwargs):
    """Simulates an external system check with random results."""
    if random.choice([True, False]):
        kwargs['ti'].xcom_push(key='check_result', value='success')
    else:
        kwargs['ti'].xcom_push(key='check_result', value='failure')


def branch_task(**kwargs):
   if kwargs['ti'].xcom_pull(key='check_result', task_ids='external_check_task') == 'success':
        return 'branch_success'
   else:
        return 'branch_failure'


with DAG(
    dag_id='one_done_example_3',
    start_date=datetime(2023, 1, 1),
    schedule=None,
    catchup=False
) as dag:

    external_check_task = PythonOperator(
      task_id = 'external_check_task',
      python_callable=simulate_external_check
    )

    branch_decision_task = PythonOperator(
        task_id = 'branch_decision_task',
        python_callable = branch_task
    )

    branch_success = BashOperator(
        task_id='branch_success',
        bash_command='echo "Branch success path"',
    )

    branch_failure = BashOperator(
        task_id='branch_failure',
        bash_command='echo "Branch failure path"',
    )


    post_branch_task = BashOperator(
        task_id='post_branch_task',
        bash_command='echo "Post branch processing"',
        trigger_rule=TriggerRule.ONE_DONE
    )

    external_check_task >> branch_decision_task
    branch_decision_task >> [branch_success, branch_failure]
    [branch_success, branch_failure] >> post_branch_task
```

The critical aspect here is the `post_branch_task`, which has a `one_done` trigger. This ensures that regardless of which branch is chosen (either `branch_success` or `branch_failure`), the post-processing task will proceed once *any* branch is completed. Without this, the task would only trigger if *all* branches succeeded, which is unnecessary here. In a branching system, it's common to have one path that would successfully execute and the other that will always end with a failed state. It's important to keep this in mind when building more complex systems.

To dive deeper into workflow management, specifically within Airflow, I'd recommend looking at "Data Pipelines with Apache Airflow" by Bas Harenslak and Julian de Ruiter. For the core concepts of distributed systems and task coordination, "Distributed Systems: Concepts and Design" by George Coulouris, Jean Dollimore, and Tim Kindberg is a good resource, though not directly Airflow specific, it provides solid background knowledge.

In summary, `one_done` offers a highly specific solution for situations that require a downstream task to proceed as long as *at least one* of its upstream tasks has completed. This contrasts sharply with other trigger rules such as `all_success`, or even `one_success`. While `one_done` is not an everyday need, knowing when and how to use it can reduce unnecessary resource consumption and simplify complex DAG layouts. It's crucial, however, to carefully assess the logic of the overall system before incorporating it, as the subtle differences in trigger rules can drastically change the behaviour of a DAG.
