---
title: "Can external DAGs within a specified date range be executed?"
date: "2025-01-30"
id: "can-external-dags-within-a-specified-date-range"
---
External DAG execution, specifically within a defined date range, introduces a crucial layer of complexity when coordinating workflows across different systems or Airflow environments. This isn't about simply triggering a DAG; it's about a precise temporal synchronization of activity where one DAG's execution, running on one system, is predicated on the historical or intended execution context of another DAG residing elsewhere. My experience working with distributed Airflow deployments in a hybrid cloud environment has highlighted the challenges and required considerations for such scenarios.

The core issue resides in the inherent separation of state and scheduling between these DAGs. Airflow, as a platform, relies on its internal database to track DAG runs, task instances, and relevant metadata. When dealing with external DAGs, this shared state is absent. This means we can't simply rely on Airflow's native sensors or mechanisms to directly trigger an external DAG based on its past execution times. Instead, we must devise methods to extract relevant information, effectively translating the external DAG's execution history or intended schedule into a trigger condition within the current DAG.

The challenge is further exacerbated by the variety of potential external systems hosting the target DAG. This could range from another Airflow installation (perhaps on a separate Kubernetes cluster) to completely different orchestration platforms, or even a batch processing system running independently. Each of these scenarios dictates a different strategy for capturing the required execution data.

To illustrate this concept, consider these three scenarios each requiring a unique approach.

**Scenario 1: External Airflow DAG on a Different Cluster**

In this scenario, we are dealing with a different Airflow instance that is accessible via its API. The primary task is to retrieve run details for the external DAG from that API and then use that information to determine whether to trigger a downstream workflow in our local DAG.

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
import requests
import json
from datetime import datetime, timedelta

DEFAULT_DATE_TIME_FORMAT = "%Y-%m-%dT%H:%M:%S%z"

def check_external_dag_run(external_dag_id, target_date_start, target_date_end, api_endpoint):
    """
    Fetches external DAG runs from Airflow API within specified date range,
    and if runs exist then returns the success state of any runs.
    """
    url = f"{api_endpoint}/dags/{external_dag_id}/dagRuns"
    params = {
        'start_date': target_date_start.isoformat(),
        'end_date': target_date_end.isoformat(),
    }

    response = requests.get(url, params=params, auth=('username', 'password')) # Replace with your auth method
    response.raise_for_status()
    dag_runs_json = response.json()
    dag_runs = dag_runs_json.get('dag_runs',[])

    if not dag_runs:
        print(f"No runs found for {external_dag_id} within the specified range.")
        return False

    success = any(run['state']=='success' for run in dag_runs)
    return success

def trigger_downstream_workflow(**kwargs):
    """
    Triggers a dummy task to represent our downstream workflow.
    """
    print("Downstream workflow triggered.")


with DAG(
    dag_id='external_dag_dependency',
    schedule=None,
    start_date=days_ago(2),
    catchup=False,
    tags=['example'],
) as dag:

    target_start_date = datetime.now() - timedelta(days=1)
    target_end_date = datetime.now()

    check_dag_run = PythonOperator(
        task_id='check_external_dag_run',
        python_callable=check_external_dag_run,
        op_kwargs={
           'external_dag_id': 'external_dag_name', #Replace with the actual DAG ID
           'target_date_start': target_start_date,
           'target_date_end': target_end_date,
           'api_endpoint': 'http://external_airflow_host:8080/api/v1', # Replace with your external Airflow API endpoint
        }
    )
    
    trigger_downstream = PythonOperator(
        task_id='trigger_downstream',
        python_callable=trigger_downstream_workflow,
        trigger_rule='one_success' # will only be run if previous returned true and exited without error
        
    )
    check_dag_run >> trigger_downstream
```

*   **Explanation:** The `check_external_dag_run` function retrieves a list of dag runs for the specificed dag and range and then determines if one or more of the DAGs have succeeded within the specified time range. This function depends on the target Airflow API providing appropriate access controls. The `trigger_downstream_workflow` function represents downstream execution. The key here is dynamically defining the desired date range within the DAG, and querying the external system within that range.
*   **Important note:** I've used placeholder authentication which must be replaced. Handling authentication for an external API is a critical aspect in production environments.

**Scenario 2: External Batch Process with Timestamped Logs**

Here, the "external DAG" is a batch process that generates log files with timestamps. The approach involves parsing these logs to determine if the process completed within the defined date range. The log file location is assumed to be available to the current Airflow environment.

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
import re
from datetime import datetime, timedelta

def check_batch_log(log_file_path, target_date_start, target_date_end):
    """
    Parses a log file and checks if the batch process completed successfully within a specified time range.
    """
    try:
        with open(log_file_path, 'r') as f:
            log_content = f.read()
    except FileNotFoundError:
        print(f"Log file not found: {log_file_path}")
        return False

    timestamp_pattern = re.compile(r'(\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2}).*Completed Successfully', re.MULTILINE)
    matches = timestamp_pattern.findall(log_content)

    if not matches:
        print("No successful completion timestamp found in the log.")
        return False

    for timestamp_str in matches:
        timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
        if target_date_start <= timestamp <= target_date_end:
            print("Batch process completed successfully within specified range.")
            return True

    print("No matching completion time within the specified range.")
    return False

def trigger_downstream_workflow(**kwargs):
    """
    Triggers a dummy task to represent our downstream workflow.
    """
    print("Downstream workflow triggered.")


with DAG(
    dag_id='batch_process_dependency',
    schedule=None,
    start_date=days_ago(2),
    catchup=False,
    tags=['example'],
) as dag:

    target_start_date = datetime.now() - timedelta(days=1)
    target_end_date = datetime.now()


    check_log = PythonOperator(
        task_id='check_batch_log',
        python_callable=check_batch_log,
        op_kwargs={
            'log_file_path': '/path/to/batch.log', # Replace with your actual log file path
            'target_date_start': target_start_date,
            'target_date_end': target_end_date,
        }
    )
    
    trigger_downstream = PythonOperator(
        task_id='trigger_downstream',
        python_callable=trigger_downstream_workflow,
        trigger_rule='one_success'
    )
    
    check_log >> trigger_downstream

```

*   **Explanation:**  The `check_batch_log` function opens and reads the provided log file, and then employs a regex to search for completion timestamps. It parses the timestamps and verifies that at least one such completion exists within the specified date range. This is a very basic example, real world logs will most likely have more complexity. The regex and date format should be adjusted to match the specific log file.
*   **Caution:** This log parsing strategy is sensitive to log file format changes. Any changes in the external process's logging format could break this dependency. A robust solution could involve more structured log data or a separate metadata file.

**Scenario 3: External Orchestration Platform Using a Shared Database**

Imagine an external system that stores its execution timestamps in a database shared with the Airflow instance. In this case, querying that shared database allows us to synchronize execution.

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta
import os
import logging
def check_external_db_status(target_date_start, target_date_end, db_connection_string):
    """
    Connects to a database, and queries for records in between start and end dates. 
    Returns true if one or more entries are found.
    """
    engine = create_engine(db_connection_string)
    sql_query = text(
        """
        SELECT COUNT(*)
        FROM external_system_jobs
        WHERE start_time >= :start_date AND end_time <= :end_date
    """
    )

    try:
        with engine.connect() as connection:
            result = connection.execute(sql_query, start_date=target_date_start, end_date=target_date_end).scalar()
        if result > 0:
            return True
    except Exception as e:
            logging.error(f"Database error: {e}")
    return False
    
def trigger_downstream_workflow(**kwargs):
    """
    Triggers a dummy task to represent our downstream workflow.
    """
    print("Downstream workflow triggered.")


with DAG(
    dag_id='external_db_dependency',
    schedule=None,
    start_date=days_ago(2),
    catchup=False,
    tags=['example'],
) as dag:
    target_start_date = datetime.now() - timedelta(days=1)
    target_end_date = datetime.now()

    check_db = PythonOperator(
        task_id='check_external_db',
        python_callable=check_external_db_status,
        op_kwargs={
            'target_date_start': target_start_date,
            'target_date_end': target_end_date,
            'db_connection_string': os.getenv('DB_CONNECTION_STRING', 'postgresql://user:password@host:port/database'),
            
        }
    )

    trigger_downstream = PythonOperator(
        task_id='trigger_downstream',
        python_callable=trigger_downstream_workflow,
        trigger_rule='one_success'
    )
    
    check_db >> trigger_downstream
```

*   **Explanation:** The `check_external_db_status` function connects to a shared database (using SQLAlchemy) and performs a query to locate records with start and end time within the provided date range. The query will require adapting to the exact table schema. The assumption is that the connection string is stored as an environment variable.
*   **Security:** Database connection credentials should always be managed using secure methods such as environment variables or secrets managers. Hard coding such credentials is a serious security risk.

**Resource Recommendations:**

For further investigation into advanced Airflow patterns I suggest exploring resources related to custom operators and sensors, data serialization formats (such as JSON or protobuf for exchanging metadata), and best practices for distributed system monitoring and observability. Articles on building custom integrations and API usage in Python can also be beneficial. While the specific technology varies, the core pattern involves establishing a means of extracting necessary historical execution information and translating this into a condition for downstream tasks.
