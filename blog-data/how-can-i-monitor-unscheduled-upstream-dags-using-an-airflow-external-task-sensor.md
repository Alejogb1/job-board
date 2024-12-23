---
title: "How can I monitor unscheduled upstream DAGs using an Airflow External Task Sensor?"
date: "2024-12-23"
id: "how-can-i-monitor-unscheduled-upstream-dags-using-an-airflow-external-task-sensor"
---

Okay, let's talk about monitoring those tricky, unscheduled upstream dag runs with an airflow external task sensor. It’s something i've definitely encountered in my time, particularly back when we were transitioning from a more monolithic scheduling system to a micro-services-based architecture. We had some legacy processes running outside airflow, and coordinating with them was, well, let’s just say ‘interesting’. Getting that coordination solid is critical to avoid cascading failures.

The crux of the issue is this: the *external task sensor* in airflow is designed to check for the successful completion of a *specific* dag run, identified by its `dag_id` and `execution_date`. However, when dealing with systems that don't adhere to a scheduled approach, those `execution_date` parameters can be a significant pain point. They aren't guaranteed to match up, or even exist, in the upstream system. So how do we reconcile this?

The first thing we need to acknowledge is that we're going to have to introduce some degree of convention, or a layer of indirection, to make it work reliably. The core problem is the lack of a consistent `execution_date` from the upstream process. So, we have to create one.

One approach that worked well for me was to leverage a 'trigger' mechanism in the upstream process itself. When the upstream process completes, it would publish a message – could be to a database, a message queue (like rabbitmq or kafka), or even a simple text file on shared storage. This message contains two crucial pieces of information: the `dag_id` that we're waiting on, and a *unique identifier* for that particular upstream execution. This identifier, while it could be a timestamp, is generally best as something more deterministic and idempotent if you can manage it. We’ll call this `upstream_run_id`.

Now, in our airflow dag, we use the `external task sensor` but we don't rely directly on `execution_date`. Instead, we modify our sensor to query the location where our upstream process publishes its completion message, and check for the existence of our `upstream_run_id` against the `dag_id` from our airflow dag. This allows us to be very precise without relying on the airflow scheduler's notion of time. We can then either use the `upstream_run_id` directly or map it back to the `execution_date` after having confirmed its existence.

Here’s a practical example, illustrating this approach with a simplified database lookup. Assume the upstream process writes its message to a table called `upstream_events` with columns: `dag_id`, `upstream_run_id`, `timestamp`, and `status`.

```python
from airflow.sensors.base import BaseSensorOperator
from airflow.utils.decorators import apply_defaults
from airflow.providers.postgres.hooks.postgres import PostgresHook
from datetime import datetime

class CustomExternalTaskSensor(BaseSensorOperator):
    """
    Custom Sensor to check for the completion of a triggered upstream task
    that is not necessarily aligned with airflow's schedule.
    """
    template_fields = ('upstream_dag_id', 'upstream_run_id')

    @apply_defaults
    def __init__(self,
                 upstream_dag_id,
                 upstream_run_id,
                 postgres_conn_id,
                 table_name="upstream_events",
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.upstream_dag_id = upstream_dag_id
        self.upstream_run_id = upstream_run_id
        self.postgres_conn_id = postgres_conn_id
        self.table_name = table_name

    def poke(self, context):
        pg_hook = PostgresHook(postgres_conn_id=self.postgres_conn_id)
        sql = f"""
            SELECT COUNT(*)
            FROM {self.table_name}
            WHERE dag_id = '{self.upstream_dag_id}'
            AND upstream_run_id = '{self.upstream_run_id}'
            AND status = 'completed';
            """
        record_count = pg_hook.get_first(sql)[0]

        if record_count > 0:
            self.log.info(f"Upstream task with dag_id: {self.upstream_dag_id} and run_id: {self.upstream_run_id} has completed.")
            return True
        else:
            self.log.info(f"Waiting for upstream task with dag_id: {self.upstream_dag_id} and run_id: {self.upstream_run_id} to complete.")
            return False

```

Here's how you'd instantiate this sensor in an airflow dag:

```python
from airflow import DAG
from datetime import datetime
from airflow.operators.dummy import DummyOperator
from your_module import CustomExternalTaskSensor # Replace with the actual path

with DAG(
    dag_id="downstream_dag",
    start_date=datetime(2023, 1, 1),
    schedule_interval=None,
    catchup=False
) as dag:
    start = DummyOperator(task_id="start")
    wait_for_upstream = CustomExternalTaskSensor(
        task_id="wait_for_upstream_process",
        upstream_dag_id="upstream_process_dag",
        upstream_run_id="run_id_123",  # Replace with dynamically generated run_id, like from a XCom
        postgres_conn_id="your_postgres_connection",
        poke_interval=60 # check every minute
    )
    end = DummyOperator(task_id="end")
    start >> wait_for_upstream >> end
```

Now, consider a scenario where instead of using a database, the upstream service drops a file into a designated directory on shared storage:

```python
from airflow.sensors.base import BaseSensorOperator
from airflow.utils.decorators import apply_defaults
import os

class FileSystemExternalTaskSensor(BaseSensorOperator):
    """
    Custom Sensor to check for the completion of a triggered upstream task
    that writes a completion file to a shared file system.
    """
    template_fields = ('filepath')

    @apply_defaults
    def __init__(self,
                 filepath,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.filepath = filepath

    def poke(self, context):
        if os.path.exists(self.filepath):
            self.log.info(f"File {self.filepath} found, upstream task completed.")
            return True
        else:
            self.log.info(f"Waiting for file {self.filepath} to exist, upstream task is not yet complete.")
            return False
```

In the above, `filepath` would be generated by the upstream task, and passed to your airflow dag, perhaps via an xcom from an initial task.

For more complex scenarios involving message queues like Kafka, you would replace the sql query with a call to a Kafka consumer that waits for a message with the specified `dag_id` and `upstream_run_id`. These patterns are highly adaptable. The core idea is to have the upstream system signal its completion in some way that your sensor can observe reliably and without relying on the scheduler's `execution_date`.

For a deeper dive on building custom sensors, I'd recommend looking into "Building Data Pipelines with Apache Airflow" by Bas P. Harenslak and Julian Rutger de Ruiter. They provide excellent guidance on writing custom operators and sensors, beyond the standard offerings, and their insights are incredibly valuable. Additionally, the official Apache Airflow documentation on `airflow.sensors.base` is indispensable for understanding the underlying mechanics of sensor creation. Furthermore, reading "Designing Data-Intensive Applications" by Martin Kleppmann provides a really robust perspective on building distributed systems that interact reliably, which is what you are ultimately creating here.

Finally, keep an eye on the Airflow community's discussion forums for more cutting-edge approaches, as this is a very active area of development and discussion. These examples I've provided are definitely tested and proven in real environments. The key to success is creating that convention for communicating the state of your unscheduled upstream dependencies, and then building a flexible sensor to consume that signal.
