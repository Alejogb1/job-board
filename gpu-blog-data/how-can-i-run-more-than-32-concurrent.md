---
title: "How can I run more than 32 concurrent tasks in Apache Airflow?"
date: "2025-01-30"
id: "how-can-i-run-more-than-32-concurrent"
---
The inherent limitation of 32 concurrent tasks in Apache Airflow isn't a hardcoded restriction; rather, it's a consequence of poorly configured resource allocation or a misunderstanding of scheduler parallelism.  My experience debugging similar issues across numerous large-scale data pipelines has consistently pointed to the interaction between the Airflow scheduler, executor type, and underlying worker resources.  Therefore, exceeding the apparent 32-task concurrency limit necessitates a multi-pronged approach addressing these components.


**1.  Understanding the Parallelism Bottleneck:**

The perceived 32-task limit stems primarily from the `parallelism` setting within the Airflow scheduler configuration (`airflow.cfg`). This setting defines the maximum number of tasks the scheduler will attempt to run concurrently *across all worker instances*.  It's crucial to distinguish this from the number of tasks a *single* worker can handle.  If you have only one worker, this setting effectively limits concurrency to 32.  Furthermore, this limit applies across all DAGs running concurrently. A single DAG attempting to launch more than 32 tasks simultaneously will still be constrained by this parameter.  Therefore, increasing parallelism alone might not suffice if other resource limitations exist.


**2.  Executor Selection and Configuration:**

Airflow's executor type profoundly impacts concurrency management.  The `SequentialExecutor` executes tasks one by one, eliminating parallelism entirely.  The `LocalExecutor` runs tasks within the Airflow scheduler process, making it prone to single-process limitations.  To surpass the 32-task constraint, the `CeleryExecutor` or `KubernetesExecutor` are far more suitable.

* **CeleryExecutor:** This distributes tasks across multiple worker processes managed by a Celery message queue.  Its concurrency is controlled by the number of Celery worker instances and their available resources (CPU, memory).  Increasing the number of worker instances directly increases the potential for concurrent task execution, easily surpassing 32.  Efficient resource allocation is paramount; over-provisioning leads to idle workers, under-provisioning creates bottlenecks.

* **KubernetesExecutor:** This leverages Kubernetes for task execution, offering superior scalability and resource management.  Each task runs in its own Kubernetes Pod, ensuring isolation and efficient resource utilization.  The number of concurrent tasks is determined by the available Kubernetes resources (nodes, CPU, memory) and the pod specifications defined within your Airflow configuration.  Auto-scaling Kubernetes clusters adapt dynamically to workload demands, providing a highly scalable solution.


**3. Code Examples and Commentary:**

The following examples illustrate how to configure these executors and manage parallelism within your Airflow environment.  Note that these snippets are illustrative and require adaptation based on your specific environment and configuration.

**Example 1: CeleryExecutor Configuration (airflow.cfg):**

```python
[celery]
celery_app = airflow.executors.celery_executor.app
worker_concurrency = 100  # Adjust based on resource availability
```

This configuration utilizes the `CeleryExecutor` and sets the `worker_concurrency` to 100. This means each Celery worker process can handle up to 100 tasks concurrently.  Running multiple Celery worker instances allows for thousands of concurrent tasks.  Remember to adjust this value based on your hardware resources and to monitor worker performance closely to avoid overloading.


**Example 2: KubernetesExecutor Configuration (airflow.cfg):**

```python
[kubernetes]
executor = KubernetesExecutor
namespace = airflow  # Or your Kubernetes namespace
kube_config = /path/to/your/kube_config  # Path to Kubernetes config file
pod_override = {"resources": {"requests": {"cpu": "1", "memory": "2Gi"}, "limits": {"cpu": "2", "memory": "4Gi"}}} # Resource requests and limits
```

Here, the `KubernetesExecutor` is chosen.  The `pod_override` section is crucial for specifying the resource requirements for each task pod.  This prevents resource starvation, ensuring that each task receives sufficient resources.  Adjusting `requests` and `limits` allows for fine-grained control over resource allocation. This significantly enhances scalability compared to the limitations of the `LocalExecutor` or even `CeleryExecutor` without proper monitoring and resource management.


**Example 3: DAG with Parallelism Control:**

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

with DAG(
    dag_id='parallel_tasks',
    start_date=days_ago(1),
    schedule_interval=None,
    tags=['parallelism'],
) as dag:

    task1 = PythonOperator(
        task_id='task_1',
        python_callable=lambda: print('Task 1 executed'),
    )
    task2 = PythonOperator(
        task_id='task_2',
        python_callable=lambda: print('Task 2 executed'),
    )
    task3 = PythonOperator(
        task_id='task_3',
        python_callable=lambda: print('Task 3 executed'),
    )

    [task1, task2, task3] >> PythonOperator(task_id="final_task", python_callable=lambda: print("All tasks completed"))
```

This DAG demonstrates a simple task dependency. The key to parallelism here isn't within the DAG itself but rather the underlying executor.  The `CeleryExecutor` or `KubernetesExecutor` will handle the concurrent execution of `task1`, `task2`, and `task3` according to the available resources and the executor's configuration.  Adding more tasks to this DAG will still respect the overall scheduler parallelism and worker concurrency limits.


**4. Resource Recommendations:**

To successfully handle a high volume of concurrent tasks, consider these resources:

*   Thorough monitoring of your Airflow scheduler, workers, and underlying infrastructure (e.g., CPU utilization, memory usage, network I/O).
*   A robust message queue (for CeleryExecutor).  RabbitMQ or Redis are common choices.
*   A sufficiently sized Kubernetes cluster (for KubernetesExecutor) with auto-scaling enabled.
*   Comprehensive logging to facilitate debugging and performance analysis.
*   Careful consideration of task dependencies to avoid unnecessary serializations. Optimization of task duration and resource consumption.



By correctly configuring the Airflow scheduler's parallelism, selecting an appropriate executor (CeleryExecutor or KubernetesExecutor), and adequately provisioning resources, you can effectively handle significantly more than 32 concurrent tasks.  The key takeaway is that the perceived limit is an artifact of resource constraints rather than a fundamental limitation of Airflow itself.  Understanding this distinction is essential for building scalable and robust data pipelines.
