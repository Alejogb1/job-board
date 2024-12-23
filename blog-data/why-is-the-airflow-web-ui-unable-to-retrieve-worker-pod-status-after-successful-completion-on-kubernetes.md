---
title: "Why is the Airflow web UI unable to retrieve worker pod status after successful completion on Kubernetes?"
date: "2024-12-23"
id: "why-is-the-airflow-web-ui-unable-to-retrieve-worker-pod-status-after-successful-completion-on-kubernetes"
---

Okay, let’s unpack this—it’s a recurring issue that I’ve seen surface in many production environments. The situation where your Airflow web UI shows successful task completions but struggles to reflect the final status of worker pods on Kubernetes is definitely frustrating, and it usually points to a few common underlying causes. This isn't necessarily a bug in Airflow itself, but rather a mismatch in expectations or a configuration oversight, often involving the interplay between the Kubernetes executor, the Airflow scheduler, and Kubernetes APIs.

From my experience, the problem boils down to the ephemeral nature of worker pods created by the Kubernetes executor. These pods aren't designed to linger after task completion; they’re meant to execute, report back, and then gracefully exit. The Airflow scheduler monitors the state of these tasks via the Kubernetes API, but the web UI’s understanding of this is not always directly tied to the raw API events after the pod itself has been terminated. This separation of reporting and pod lifetime is crucial.

Specifically, the Kubernetes executor typically functions like this: when a task is scheduled, Airflow generates a pod specification, submits it to Kubernetes, and tracks its progress. Upon completion, the pod sends a signal or exit code back to the Airflow scheduler. The scheduler then updates the task’s status within the Airflow metadata database. Importantly, after the scheduler notes task completion, the Kubernetes pod, from Airflow's perspective, becomes largely irrelevant. It's Kubernetes that then reaps or terminates the pod after its job has finished. The Airflow web UI, however, tends to rely on information within the Airflow database and cached state rather than live Kubernetes pod status lookups for completed tasks.

The key element here is the 'eventual consistency' model that exists between Kubernetes and Airflow. Airflow's scheduler relies on Kubernetes API events to capture pod status changes, but these changes aren't always immediately reflected in the UI. There could be a brief period where the pod has finished but the web UI still shows a pending state (or, more commonly in this scenario, doesn't show pod status at all).

One thing I often see overlooked is the impact of the Kubernetes API polling interval in the `airflow.cfg` configuration file, specifically these parameters: `kubernetes_pod_watch_interval` and `kubernetes_worker_pods_polling_interval`. If these intervals are set too high, especially when tasks run quickly, the scheduler might miss the pod’s final state changes before it's cleaned up. The scheduler gets the final task success/failure result, but the details about the pod from Kubernetes can become stale or not be fetched by Airflow at all. This can lead to discrepancies in the UI.

Another factor comes into play when looking at the configuration of your Kubernetes cluster itself, specifically with respect to cluster resource availability. If there is significant resource contention on the cluster where the worker pods are being scheduled, or perhaps the Kubernetes scheduler is under strain, the pod termination might occur without fully being reflected in the Airflow internal state. This is less of an Airflow issue directly, but an infrastructure consideration.

To illustrate this, consider the following scenarios, with accompanying code examples:

**Scenario 1: The standard case – pod completes, status might not immediately update in the UI**

This showcases the primary issue where the pod status isn't readily available post-completion in the Airflow UI:

```python
from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

with DAG(
    dag_id='kubernetes_pod_status_example_1',
    start_date=datetime(2023, 1, 1),
    schedule_interval=None,
    catchup=False,
) as dag:
    task_a = BashOperator(
        task_id='simple_bash_task',
        bash_command='sleep 5 && echo "Task completed successfully"'
    )
```

In this trivial DAG, the bash operator runs inside a Kubernetes pod. Once completed, the Airflow scheduler knows the task succeeded, but the specific details of the pod (if the pod has already been cleaned up quickly by Kubernetes) will not show up consistently in the web UI. The key thing is that after the command finishes successfully, the pod becomes irrelevant to Airflow as the task outcome has already been recorded. Airflow does not maintain a live, continuous connection to pod status beyond this point (by design).

**Scenario 2: Adjusting polling intervals:**

To address the potential for missed events, it may be beneficial to adjust the polling intervals. However, excessively short polling intervals can increase the load on the Kubernetes API. This requires a balancing act to obtain a timely view while minimizing overhead.

```python
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago

from datetime import timedelta

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'retries': 0,
    'retry_delay': timedelta(minutes=1),
}

with DAG(
    dag_id='kubernetes_pod_status_example_2',
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
) as dag:
    task_a = BashOperator(
        task_id='simple_bash_task',
        bash_command='sleep 3 && echo "Task completed successfully"',
        env={
            'AIRFLOW__KUBERNETES__POD_WATCH_INTERVAL': '10', # explicitly setting the env
            'AIRFLOW__KUBERNETES__WORKER_PODS_POLLING_INTERVAL': '10'
        }
    )
```

In this case, we are explicitly injecting env vars into the task that directly influence Airflow's behaviour, specifically how frequently it polls Kubernetes, using the same variables that are found in the `airflow.cfg` file. Lowering these polling values may lead to the Airflow Scheduler picking up the Kubernetes events quicker and updating the UI more reliably, but at the cost of more calls to the Kubernetes API.

**Scenario 3: Resource constraints and cluster instability**

While Airflow does not directly control this, problems can manifest within the UI if worker nodes or the scheduler itself is resource constrained. This does not have a code example, but rather is a configuration issue. An overly loaded cluster may lead to pods being terminated quickly before Airflow's scheduler can obtain relevant status information. Resolving this requires monitoring your K8s cluster and ensuring resources are not overcommitted. Monitoring pod exit codes directly within Kubernetes can often pinpoint if there are K8s-related resource issues.

**Recommendations and Further Learning**

Instead of relying solely on the UI, actively query your Kubernetes cluster directly, using `kubectl get pods --all-namespaces`, and monitor the pod lifecycle using `kubectl describe pod <pod_name>`. This gives you a more direct, low-level view of what's happening within the K8s environment, independent of Airflow's interpretation.

For further reading, I'd highly recommend the official Apache Airflow documentation focusing on the Kubernetes executor. Also, Kubernetes documentation on pod lifecycle management will provide a robust understanding of the underlying mechanisms. Additionally, the paper "Distributed Computing Patterns" by J. Dean and S. Ghemawat (of Google fame) provides fundamental insights into distributed systems principles that apply to this scenario. A solid foundation in these areas will make tackling such issues more straightforward. For deep dives into Kubernetes, “Kubernetes in Action” by Marko Luksa is also an excellent resource.

In conclusion, the disconnect between Airflow’s UI and Kubernetes pod status after completion arises from the ephemeral nature of Kubernetes worker pods, along with the potential for missed API events, the eventual consistency model, and the scheduler’s polling behaviour. Addressing this issue involves a good understanding of the interactions between Airflow and Kubernetes, coupled with careful configuration and monitoring of both environments. It’s definitely a nuanced area and, with some experience, becomes a more manageable issue.
