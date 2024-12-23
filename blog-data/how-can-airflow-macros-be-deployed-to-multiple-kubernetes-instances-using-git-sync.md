---
title: "How can airflow macros be deployed to multiple Kubernetes instances using Git sync?"
date: "2024-12-23"
id: "how-can-airflow-macros-be-deployed-to-multiple-kubernetes-instances-using-git-sync"
---

Let’s explore that scenario. I remember a project a few years back, deploying a suite of data pipelines across a distributed team, where we faced precisely this challenge: managing airflow macros across several independent Kubernetes clusters. We needed a way to keep these macro definitions consistent and automatically updated across all deployments. Simply put, hand-rolling updates was unsustainable; we needed a robust, automated solution. Git sync proved to be the key.

The core issue revolves around the need for *idempotency* and *consistency* in our infrastructure-as-code (IaC) practices. Airflow macros, essentially python code used to extend the templating capabilities of the Airflow DAG files, need to be deployed consistently across all execution environments. A discrepancy in macro definitions could lead to unpredictable behavior, or even critical failures. Now, while several methods exist to handle config management in kubernetes, such as ConfigMaps or Helm charts with complex templating, using Git sync for macro deployment offers a more straightforward and version-controlled approach.

Here's how we implemented it, and I’ll break down each step with practical code examples. We essentially created a separate Git repository dedicated exclusively to airflow macros. This separation allowed us to manage the macro code separately from the DAG repository. This is an important distinction to prevent coupling and potential cascading changes. This repository contained the python files representing our macros, typically organized into a clear directory structure.

First, consider a basic macro file named `custom_macros.py`:

```python
# custom_macros.py

from airflow.utils.dates import days_ago
from datetime import datetime, timedelta

def calculate_start_date(offset_days):
    """Calculates a start date based on a given number of offset days"""
    return days_ago(offset_days)

def format_date(date_obj, format_str="%Y-%m-%d"):
     """Formats a datetime object into a specific string format"""
     return date_obj.strftime(format_str)
```

This file defines two simple macros: `calculate_start_date` and `format_date`. These functions perform common tasks, illustrating what macros typically achieve within Airflow. This file lives inside our Git repository, let's say in `/macros/custom_macros.py`.

Now, on each Kubernetes cluster running Airflow, we deployed a `git-sync` sidecar container. The `git-sync` tool is lightweight and designed to keep a local directory synchronized with a remote Git repository. We configured the `git-sync` container within the airflow worker pod definitions, ensuring that this container pulled the macro repository. Here's an example of a snippet from a kubernetes pod definition:

```yaml
# worker-pod.yaml (snippet)
    spec:
      containers:
        - name: airflow-worker
          image: apache/airflow:2.8.0  #example Airflow image
          # ... other configuration for worker container
        - name: git-sync
          image: k8s.gcr.io/git-sync/git-sync:v3.6.0
          volumeMounts:
            - name: macros-volume
              mountPath: /git-sync-macros
          env:
            - name: GIT_SYNC_REPO
              value: "https://your-git-repo/airflow-macros.git"
            - name: GIT_SYNC_BRANCH
              value: "main"
            - name: GIT_SYNC_DEST
              value: "/git-sync-macros"
            - name: GIT_SYNC_ONE_TIME
              value: "false"  # Use 'true' if only one sync needed
            - name: GIT_SYNC_PERIOD
              value: "30" # Interval in seconds

      volumes:
        - name: macros-volume
          emptyDir: {}
```

In this configuration, the `git-sync` container, running within the same pod as the airflow worker, clones the specified Git repository to the `/git-sync-macros` directory inside a shared volume named `macros-volume`. This container is configured to continuously sync using `GIT_SYNC_PERIOD` which is in seconds. This means that every 30 seconds, if there is a change in Git it is synced into this shared volume.

Finally, within the Airflow configuration itself, we needed to instruct Airflow to load the macro definitions from this synchronized directory. This was accomplished by modifying the `airflow.cfg` configuration file within our Airflow deployment. The necessary setting is within the `[core]` section:

```
# airflow.cfg (snippet)

[core]
plugins_folder = /opt/airflow/plugins
dags_folder = /opt/airflow/dags
# add the macros directory to python path (using volume mount)
python_path = /opt/airflow/dags,/opt/airflow/plugins,/git-sync-macros
```

By adding `/git-sync-macros` to the `python_path` , Airflow is now capable of finding and loading the macro definitions defined in `custom_macros.py`. This ensures any changes to the Git repository will automatically be picked up by Airflow on the next DAG run or task execution (after the git-sync interval has passed). Airflow will automatically load the custom macro definitions into its template context.

To use these macros in your DAG, the DAG should now be able to access the macros as follows:

```python
# example_dag.py
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from custom_macros import calculate_start_date, format_date

with DAG(
   dag_id='example_macro_dag',
   schedule=None,
   start_date=datetime(2023,1,1),
   catchup=False,
   tags=['example']
) as dag:

    def print_formatted_date(**kwargs):
       start_date = kwargs['ti'].start_date
       formatted_date = format_date(start_date)
       print(f"The formatted start date is: {formatted_date}")

    def print_calculated_date():
      calculated_date = calculate_start_date(10)
      print(f"The calculated date is : {calculated_date}")

    print_date_task = PythonOperator(
        task_id="print_formatted_date",
        python_callable = print_formatted_date
    )

    calculate_date_task = PythonOperator(
         task_id="print_calculated_date",
        python_callable = print_calculated_date
    )

    calculate_date_task >> print_date_task
```

This DAG demonstrates how to use the `calculate_start_date` and `format_date` macros within a DAG definition. Crucially, these functions aren't being defined directly in the DAG, but are pulled from externalized macro files located within the synced repository.

In terms of resources, I’d recommend diving into the documentation for the Kubernetes `git-sync` sidecar container on the Kubernetes official site for a deep understanding of configuration options. Also, I found “Kubernetes in Action” by Marko Lukša to be extremely helpful for grasping the nuances of kubernetes deployments and sidecar patterns. To understand Airflow more thoroughly, the official Apache Airflow documentation is crucial, especially the sections on templating and configuration. For more advanced macro use-cases and patterns, “Fluent Python” by Luciano Ramalho provides a solid grounding in Python itself, allowing one to create robust and maintainable macros.

This method offers several advantages: version control of macro definitions, easy rollbacks, a clear separation of concerns between macro logic and DAG definitions, and automatic deployment of updates across all Kubernetes instances. This significantly reduced complexity for my team and ultimately made our system more robust and easier to manage. This architecture also supports further enhancements, such as utilizing a dedicated macro library and extending the macro functionality as needed. This experience showed that a well-defined strategy combining Git sync and proper Airflow configurations can successfully achieve the deployment of Airflow macros across multiple Kubernetes clusters, addressing the complexities of our distributed data pipeline environment.
