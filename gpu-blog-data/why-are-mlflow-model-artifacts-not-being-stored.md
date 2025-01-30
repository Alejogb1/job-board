---
title: "Why are MLflow model artifacts not being stored during Airflow DAG execution, preventing experiment details retrieval?"
date: "2025-01-30"
id: "why-are-mlflow-model-artifacts-not-being-stored"
---
In my experience, the absence of MLflow model artifacts after an Airflow DAG run, preventing experiment detail retrieval, typically stems from a disconnect between the MLflow tracking URI configured within the DAG's Python tasks and the central MLflow tracking server or designated storage location. This disconnect manifests as the DAG code correctly executing MLflow functions for model logging, but the artifacts are not physically stored where MLflow expects them to be found.

The primary reason this happens is that Airflow tasks run in separate, often isolated environments from the process launching the DAG execution. When MLflow logging functions are invoked inside an Airflow task, they default to either a local file-based backend (typically `./mlruns`) or rely on environment variables. These local file systems are ephemeral within the context of a task’s execution environment. Since the Airflow task does not have shared access to where the experiment’s overall tracking is occurring, the logged artifacts never persist in the centralized location, even when the tracking URI is apparently set. The tracking URI is used to *send* logging data to the location, but the *storage* location is a different consideration. A similar issue exists with environmental variable configuration; even if the `MLFLOW_TRACKING_URI` is set correctly in the Airflow worker's environment, artifacts are by default stored on the worker's local filesystem.

To correctly track and persist artifacts within an Airflow DAG execution, the storage backend, as opposed to merely the tracking URI, must be configured such that both the MLflow client and any underlying artifact storage can reach it consistently. This can be achieved through specific environment configurations for the tasks, careful usage of the `MLflowClient`, or by explicitly specifying a location using the correct URI format.

Let’s illustrate this with a few code examples.

**Example 1: Incorrect Artifact Storage**

This DAG snippet showcases a common mistake: configuring only the `MLFLOW_TRACKING_URI`, leading to artifact loss.

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from mlflow import log_metric, log_param, log_artifact
from mlflow.tracking import MlflowClient
import time
from datetime import datetime

def train_model():
    client = MlflowClient(tracking_uri="http://your_mlflow_server:5000")
    experiment_name = "Airflow_Experiment"
    try:
        experiment_id = client.create_experiment(experiment_name)
    except:
        experiment_id = client.get_experiment_by_name(experiment_name).experiment_id

    with client.start_run(experiment_id=experiment_id) as run:
        log_param("learning_rate", 0.01)
        log_metric("accuracy", 0.95)

        with open("model.txt", "w") as f:
            f.write("Fake model weights.")
        log_artifact("model.txt")


with DAG(
    dag_id='mlflow_incorrect_artifact_storage',
    start_date=datetime(2023, 1, 1),
    schedule=None,
    catchup=False
) as dag:
    train_task = PythonOperator(
        task_id='train_model_task',
        python_callable=train_model
    )
```

Here, the `MLflowClient` is instantiated with a `tracking_uri`. When `log_artifact("model.txt")` is called, MLflow will log the path of the `model.txt` file to the specified URI. Critically, the *artifact storage* is not explicitly configured, so MLflow will by default attempt to store the file in a local `mlruns` folder in the filesystem context of the Airflow worker executing this task. This `mlruns` folder will be ephemeral, not shared with the central MLflow tracking server, leading to artifact loss. The experiment and run metadata would be correctly logged and viewable in the MLflow UI, but the model artifacts would be missing.

**Example 2: Explicit Artifact Storage with `MLFLOW_ARTIFACT_URI`**

This example demonstrates how to explicitly specify the artifact storage location, ensuring artifact persistence.

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from mlflow import log_metric, log_param, log_artifact
from mlflow.tracking import MlflowClient
import time
import os
from datetime import datetime

def train_model_explicit_storage():
    client = MlflowClient(tracking_uri="http://your_mlflow_server:5000")
    experiment_name = "Airflow_Experiment_Explicit"
    try:
        experiment_id = client.create_experiment(experiment_name)
    except:
        experiment_id = client.get_experiment_by_name(experiment_name).experiment_id
    
    artifact_uri_prefix = "s3://your-s3-bucket/mlflow_artifacts/"
    
    with client.start_run(experiment_id=experiment_id) as run:
        log_param("learning_rate", 0.01)
        log_metric("accuracy", 0.95)

        with open("model.txt", "w") as f:
            f.write("Fake model weights.")
        
        os.environ["MLFLOW_ARTIFACT_URI"] = artifact_uri_prefix
        log_artifact("model.txt")


with DAG(
    dag_id='mlflow_explicit_artifact_storage',
    start_date=datetime(2023, 1, 1),
    schedule=None,
    catchup=False
) as dag:
    train_task = PythonOperator(
        task_id='train_model_task',
        python_callable=train_model_explicit_storage
    )
```

In this corrected code, we use the environment variable `MLFLOW_ARTIFACT_URI` to inform MLflow where to store the artifacts. Here, it’s set to an S3 bucket. This ensures that even though the task runs in an isolated environment, it stores artifacts in a durable, accessible location, which the tracking server will then be able to reference correctly, allowing for proper retrieval. Using `MLFLOW_ARTIFACT_URI` is preferable to having Airflow use the default local `mlruns` folder on worker nodes.

**Example 3: Using the `MlflowClient.log_artifact` method's artifact_path argument**

This example provides an alternative, more direct way of specifying the artifact storage location, and avoids using environment variables.

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from mlflow import log_metric, log_param
from mlflow.tracking import MlflowClient
import time
from datetime import datetime
import os
import tempfile

def train_model_artifact_path():
    client = MlflowClient(tracking_uri="http://your_mlflow_server:5000")
    experiment_name = "Airflow_Experiment_Artifact_Path"
    try:
        experiment_id = client.create_experiment(experiment_name)
    except:
        experiment_id = client.get_experiment_by_name(experiment_name).experiment_id

    with client.start_run(experiment_id=experiment_id) as run:
        log_param("learning_rate", 0.01)
        log_metric("accuracy", 0.95)

        with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_file:
             temp_file.write("Fake model weights.")
             temp_file_name = temp_file.name

        client.log_artifact(run_id=run.info.run_id, local_path=temp_file_name, artifact_path="model.txt")
        os.remove(temp_file_name)


with DAG(
    dag_id='mlflow_artifact_path',
    start_date=datetime(2023, 1, 1),
    schedule=None,
    catchup=False
) as dag:
    train_task = PythonOperator(
        task_id='train_model_task',
        python_callable=train_model_artifact_path
    )
```

Here, we directly utilize the `MlflowClient`’s `log_artifact` method and specify an `artifact_path`. This `artifact_path` within the call designates the logical sub-directory, or relative path, within the specified artifact storage location where the `local_path` should be saved. Again, the model artifacts are reliably stored, and can be retrieved when needed, irrespective of the ephemeral nature of Airflow task execution environments. We use a temporary file in this example to not pollute the current workspace, but that is not required for `log_artifact` to function.

These examples demonstrate that simply defining the tracking URI is insufficient for proper artifact storage within Airflow. The key is to understand the difference between the tracking server address and the actual location where artifacts are stored. When using MLflow within an Airflow DAG, explicit artifact storage configuration is necessary for long-term data persistence and experiment reproducibility.

For further research, the official MLflow documentation is a crucial resource. I would also recommend examining Airflow's documentation regarding task configuration and execution environments. Finally, familiarize yourself with the specific cloud storage documentation, such as AWS S3, Azure Blob Storage, or Google Cloud Storage, if you are using cloud-based artifact storage, which is likely the case in a production environment. Understanding how those systems integrate with MLflow is critical for successful artifact management within Airflow.
