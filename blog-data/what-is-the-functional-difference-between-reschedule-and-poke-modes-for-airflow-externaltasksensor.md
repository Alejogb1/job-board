---
title: "What is the functional difference between 'reschedule' and 'poke' modes for Airflow ExternalTaskSensor?"
date: "2024-12-23"
id: "what-is-the-functional-difference-between-reschedule-and-poke-modes-for-airflow-externaltasksensor"
---

, let's break down the functional distinctions between “reschedule” and “poke” modes within Airflow's `ExternalTaskSensor`, something I’ve seen tripped up developers more than once, and I’ve had to troubleshoot myself in production scenarios. It’s not always immediately clear from the documentation, and the subtle differences in behavior can lead to unexpected outcomes if not properly understood.

The core purpose of an `ExternalTaskSensor` is to monitor the execution status of a task in a *different* dag, and to conditionally proceed based on that state. Think of it as a dependency mechanism that stretches beyond the confines of a single DAG. It's a powerful feature for coordinating complex, multi-dag workflows. The question comes down to *how* the sensor checks for that dependent task's status. This is where 'reschedule' and 'poke' come into play.

**"Poke" Mode: The Active Poller**

When an `ExternalTaskSensor` operates in 'poke' mode, it behaves like an active poller. Essentially, the sensor executes its `poke` method at a defined interval (controlled by the `poke_interval` parameter). Each time this `poke` method is executed, the sensor queries the Airflow metadata database to determine if the target task in the external DAG has reached a success state. If the target task hasn't succeeded, the sensor simply "pokes" again at the next interval, effectively holding its place in line.

This mode is generally resource-efficient in terms of worker slot consumption, as it doesn’t tie up a worker for extended periods. The worker essentially does a quick check and then goes back to the queue, awaiting the next poke interval. However, 'poke' mode can be less responsive, as there's a delay determined by `poke_interval` between the target task succeeding and the sensor task proceeding. Furthermore, if there are a large number of sensors in 'poke' mode, this constant polling could increase load on the metadata database.

Here's a straightforward example:

```python
from airflow import DAG
from airflow.sensors.external_task import ExternalTaskSensor
from airflow.utils.dates import days_ago
from datetime import timedelta

with DAG(
    dag_id='sensor_dag_poke',
    start_date=days_ago(2),
    schedule_interval=None,
    catchup=False,
    tags=['example'],
) as dag_poke:
    sensor_task = ExternalTaskSensor(
        task_id='wait_for_external_task',
        external_dag_id='target_dag',  # DAG containing the task to be monitored
        external_task_id='target_task', # ID of the target task within target_dag
        poke_interval=30,  # Check every 30 seconds
        mode='poke',
    )
```

In this code snippet, the `sensor_task` will, every 30 seconds, check if the task `target_task` within the DAG `target_dag` has completed successfully. Note that we have explicitly set `mode='poke'` and defined a `poke_interval`.

**"Reschedule" Mode: The Deferral Mechanism**

In contrast, 'reschedule' mode operates using a deferral mechanism. When the `ExternalTaskSensor` in this mode executes, it also first checks the target task’s status. However, if the target task hasn't succeeded, instead of waiting, the sensor *releases* its worker slot back to the pool and is rescheduled at a predefined future point, as defined in your scheduler configuration.

This mode is more efficient from a database perspective and can lead to more immediate reactions to the external task succeeding, since the sensor isn't periodically polling. There is less continuous database interaction compared to 'poke', and the scheduler itself is managing the re-activation of the sensor. However, it does consume a bit more scheduler resources since each reschedule action has to be handled by the scheduler.

Here’s a corresponding example using 'reschedule':

```python
from airflow import DAG
from airflow.sensors.external_task import ExternalTaskSensor
from airflow.utils.dates import days_ago
from datetime import timedelta

with DAG(
    dag_id='sensor_dag_reschedule',
    start_date=days_ago(2),
    schedule_interval=None,
    catchup=False,
    tags=['example'],
) as dag_reschedule:
    sensor_task_reschedule = ExternalTaskSensor(
        task_id='wait_for_external_task_reschedule',
        external_dag_id='target_dag',
        external_task_id='target_task',
        mode='reschedule',
    )
```

Here, we've set `mode='reschedule'`, which means the sensor will defer its execution until the scheduler reschedules it (based on Airflow's reschedule configuration) if the target task is not successful. Notice that there is no `poke_interval` defined as it’s not relevant in this case.

**Key Differences Summarized**

Let’s make a table to explicitly highlight the crucial differences:

| Feature          | "Poke" Mode                                   | "Reschedule" Mode                              |
|-------------------|-----------------------------------------------|-----------------------------------------------|
| **Mechanism**      | Active polling of task state               | Deferral and reschedule by scheduler              |
| **Worker Resource**| Short, repeated worker usage                  | Worker freed until rescheduled                  |
| **Database Load**  | Higher, due to constant polling              | Lower, less frequent queries                  |
| **Responsiveness**| Can be delayed by `poke_interval`              | Potentially faster due to deferral              |
| **Configurable Interval**| yes (through `poke_interval`)             | No (controlled by the scheduler configuration)|
| **Ideal Scenarios** | When short polling intervals are acceptable, and deferrals may lead to high scheduling load.| When immediate reactions are beneficial, and deferring resources is preferred.|

**Practical Considerations: A Case From My Experience**

In one particular project, we were dealing with a multi-stage data pipeline where one DAG was responsible for ingesting raw data and another DAG handled complex transformations. We initially used `ExternalTaskSensor` in 'poke' mode with a short `poke_interval` to check for the successful completion of the ingestion DAG before kicking off transformations. This worked initially, but as we scaled the number of these sensor tasks, we started noticing performance degradations on our metadata database and our scheduler. Eventually, we moved those critical sensor tasks to 'reschedule' mode, which decreased database and scheduler load drastically. This change significantly improved our overall pipeline performance.

However, there was another case where a very long running external task needed monitoring. Switching to 'reschedule' for such an infrequent event was actually wasteful. Every time it rescheduled, it put extra pressure on the scheduler. In this case, 'poke' mode with a long `poke_interval` turned out to be the better option. This allowed us to monitor without overly impacting the scheduler.

The key is to understand the specific context and resource constraints of your pipeline. If the target task is typically fast or requires immediate response, 'reschedule' may be the better choice. If the target task could take considerable time to complete, a less aggressive polling strategy provided by 'poke' can be more efficient. Also consider the overall scale of your Airflow deployment and its ability to handle rescheduling overhead.

**Further Reading**

For a deeper understanding, I highly recommend reviewing the following:

*   **The official Apache Airflow documentation:** Specifically the sections dedicated to `ExternalTaskSensor`, deferrable operators, and scheduler internals.
*   **"Data Pipelines with Apache Airflow" by Bas P. Harenslak and Julian Rutger Van Der Mee:** This provides a comprehensive guide to Airflow and contains helpful discussions on sensor implementations.
*   **Apache Airflow Improvement Proposals (AIPs):** These are excellent sources to understand design decisions and trade-offs made within the platform and can clarify the rationale behind these features. Look for AIPs related to deferrable operators and sensor mechanisms.
*   **"Designing Data-Intensive Applications" by Martin Kleppmann:** Although not solely focused on Airflow, this book provides solid foundations on data consistency and distributed systems principles, which are relevant for comprehending the architectural context of these sensor behaviors.

Choosing between 'reschedule' and 'poke' is not just about preference; it's about understanding the nuances of each mode and selecting the most appropriate approach for your specific use case. In the long run, making an informed decision can save you considerable debugging time and prevent unforeseen performance bottlenecks.
