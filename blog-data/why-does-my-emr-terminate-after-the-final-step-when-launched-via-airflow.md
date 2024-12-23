---
title: "Why does my EMR terminate after the final step when launched via Airflow?"
date: "2024-12-23"
id: "why-does-my-emr-terminate-after-the-final-step-when-launched-via-airflow"
---

, let's tackle this EMR termination puzzle after your final Airflow step. This is something I've seen, and debugged, more times than I care to count, and usually, it boils down to a combination of understanding how EMR lifecycles work and the specifics of your Airflow configuration. It's rarely a single 'aha!' moment, but rather a methodical process of elimination.

From my experience, often when an EMR cluster launched via Airflow unexpectedly terminates immediately after its final step, it points to a problem with the cluster's perceived purpose or a lack of instruction for it to stick around. Let’s break it down: the core issue is that EMR, by design, is quite ephemeral. If it doesn't have a reason to stay active – a job to run or a directive to keep itself going – it will shut down to conserve resources. Airflow, as an orchestrator, needs to specifically instruct EMR on its desired state, and sometimes the interplay isn't as intuitive as we'd like.

The most common reason for this behavior is that you’re launching the EMR cluster via the `EmrCreateJobFlowOperator` in Airflow. This operator is great for creating a cluster, running a set of steps, and then, by default, terminating it. It follows the “fire-and-forget” pattern. After that final step you defined within the operator, EMR sees no more work, and if no termination policy is configured for persistent behavior, it shuts down. In short, if your operator is constructed to create an EMR cluster, execute jobs, and then terminate; then that's exactly what it will do by default.

Here are the primary suspects we need to investigate, and I'll illustrate each with code snippets:

**1. The `KeepAlive` setting within Airflow:**

Many people miss the fact that a simple Boolean value often governs the persistence of their cluster. The Airflow `EmrCreateJobFlowOperator` (or similar operators from libraries like `airflow-providers-amazon`) often accepts a configuration setting called `keep_alive_when_no_steps`. If this value is not explicitly set to `True`, the default behavior is termination after the final job step. Let's examine what that looks like in the context of an operator definition within an Airflow DAG:

```python
from airflow import DAG
from airflow.providers.amazon.aws.operators.emr_create_job_flow import EmrCreateJobFlowOperator
from datetime import datetime

with DAG(
    dag_id='emr_keep_alive_example_1',
    start_date=datetime(2023, 1, 1),
    schedule_interval=None,
    catchup=False,
) as dag:
    create_emr_cluster = EmrCreateJobFlowOperator(
        task_id='create_emr_cluster',
        job_flow_overrides={
            "Name": "MyKeepAliveCluster",
            "ReleaseLabel": "emr-6.15.0", # Example
            "Applications": [{"Name": "Hadoop"}, {"Name": "Spark"}], # Example apps
            "Instances": {
                "InstanceGroups": [
                    {
                        "Name": "Master",
                        "Market": "ON_DEMAND",
                        "InstanceRole": "MASTER",
                        "InstanceType": "m5.xlarge",
                        "InstanceCount": 1,
                    },
                    {
                        "Name": "Core",
                        "Market": "ON_DEMAND",
                        "InstanceRole": "CORE",
                        "InstanceType": "m5.xlarge",
                        "InstanceCount": 2,
                    },
                ],
            },
        },
        keep_alive_when_no_steps=True, # The important part
    )
```

In this code, I explicitly set `keep_alive_when_no_steps=True`. This tells the cluster not to terminate after the jobs are complete. If this is not defined, or explicitly set to `False`, your cluster will vanish immediately after its work is completed. I can’t stress how common this oversight is.

**2. The absence of subsequent steps:**

Sometimes the EMR cluster is configured to stay alive, but you aren't doing anything useful. Even with `keep_alive_when_no_steps = True`, the cluster can sometimes idle too long and eventually terminate by AWS’s internal mechanisms or because its resources are no longer needed. While not directly causing instant termination after the final *defined* step, this can be easily confused for it. We need to ensure something is keeping the cluster busy if long-term persistence is required. I experienced a situation where I had a transient cluster which had a single step and terminated, so I implemented some additional 'no-op' steps to keep it alive, which is a bad pattern, but does prove the point.

```python
from airflow import DAG
from airflow.providers.amazon.aws.operators.emr_create_job_flow import EmrCreateJobFlowOperator
from airflow.providers.amazon.aws.operators.emr_add_steps import EmrAddStepsOperator
from airflow.providers.amazon.aws.sensors.emr_step import EmrStepSensor
from datetime import datetime

with DAG(
    dag_id='emr_keep_alive_example_2',
    start_date=datetime(2023, 1, 1),
    schedule_interval=None,
    catchup=False,
) as dag:
    create_emr_cluster = EmrCreateJobFlowOperator(
        task_id='create_emr_cluster',
        job_flow_overrides={
            "Name": "MyKeepAliveCluster",
            "ReleaseLabel": "emr-6.15.0", # Example
            "Applications": [{"Name": "Hadoop"}, {"Name": "Spark"}],
             "Instances": {
                "InstanceGroups": [
                    {
                        "Name": "Master",
                        "Market": "ON_DEMAND",
                        "InstanceRole": "MASTER",
                        "InstanceType": "m5.xlarge",
                        "InstanceCount": 1,
                    },
                    {
                        "Name": "Core",
                        "Market": "ON_DEMAND",
                        "InstanceRole": "CORE",
                        "InstanceType": "m5.xlarge",
                        "InstanceCount": 2,
                    },
                ],
             },
        },
        keep_alive_when_no_steps=True,
    )


    step_adder = EmrAddStepsOperator(
        task_id="add_steps",
        job_flow_id="{{ task_instance.xcom_pull(task_ids='create_emr_cluster', key='return_value') }}",
        steps=[
           {
                "Name": "Placeholder_step",
                "ActionOnFailure": "CONTINUE",
                "HadoopJarStep": {
                    "Jar": "command-runner.jar",
                    "Args": ["echo", "This_is_a_placeholder_step"]
                    },
                }
        ]
    )


    step_sensor = EmrStepSensor(
        task_id="wait_for_steps",
        job_flow_id="{{ task_instance.xcom_pull(task_ids='create_emr_cluster', key='return_value') }}",
        step_id="{{ task_instance.xcom_pull(task_ids='add_steps', key='return_value')[0] }}",
        target_states=["COMPLETED"]
        )

    create_emr_cluster >> step_adder >> step_sensor
```

Here, I've added a simple 'placeholder' step that just echoes a message, acting as a minimal piece of work to keep the cluster active. If you need a persistent cluster, you have to ensure there is ongoing activity, or at least some kind of persistent process.

**3. Misconfigured Termination Policies:**

EMR clusters also have internal termination policies. You might not set these through Airflow directly; these are often default EMR settings. If your cluster is set to terminate after a certain idle period or because of low utilization, it could shut down even if you've set `keep_alive_when_no_steps`. These are set through EMR itself as part of the cluster configuration. They're less common as the root cause but worth verifying. Check in the AWS console for the particular cluster configuration.

I will show how we might mitigate this by adding a custom termination policy, by setting the `TerminationProtected` property to true, to prevent manual termination of the cluster.

```python
from airflow import DAG
from airflow.providers.amazon.aws.operators.emr_create_job_flow import EmrCreateJobFlowOperator
from datetime import datetime

with DAG(
    dag_id='emr_keep_alive_example_3',
    start_date=datetime(2023, 1, 1),
    schedule_interval=None,
    catchup=False,
) as dag:
    create_emr_cluster = EmrCreateJobFlowOperator(
        task_id='create_emr_cluster',
        job_flow_overrides={
            "Name": "MyKeepAliveCluster",
            "ReleaseLabel": "emr-6.15.0",  # Example
            "Applications": [{"Name": "Hadoop"}, {"Name": "Spark"}],  # Example apps
            "Instances": {
                "InstanceGroups": [
                    {
                        "Name": "Master",
                        "Market": "ON_DEMAND",
                        "InstanceRole": "MASTER",
                        "InstanceType": "m5.xlarge",
                        "InstanceCount": 1,
                    },
                    {
                        "Name": "Core",
                        "Market": "ON_DEMAND",
                        "InstanceRole": "CORE",
                        "InstanceType": "m5.xlarge",
                        "InstanceCount": 2,
                    },
                ],
            },
             "TerminationProtected": True, # The important part
        },
        keep_alive_when_no_steps=True,
    )
```

Here, `TerminationProtected=True` will prevent manual termination, which may be desirable. You would also typically configure specific termination policies, such as the default ones set by EMR, via similar configuration options.

**Recommendations**

For deeper understanding, I highly recommend reviewing the official AWS EMR documentation, focusing on cluster lifecycles and termination policies. Specifically, pay close attention to the sections on transient versus persistent clusters and the available configuration options when creating an EMR cluster. The 'Amazon EMR Management Guide' is an excellent resource on this. I would also recommend reading the Airflow provider specific documentation for the Amazon providers in the Airflow documentation. Also, examining well-written example DAGs within the community's git repositories can provide invaluable insight.

In summary, EMR's lifecycle is governed by explicit instructions. The default is often termination, and if your Airflow DAG is not explicitly configured to change this, it will behave accordingly. Thoroughly check your `keep_alive_when_no_steps` setting, make sure your cluster has subsequent steps or a persistent process for cases of long term needs, and verify cluster termination policies as well as termination protection are configured to meet your requirements. Addressing these three main areas will almost certainly fix most instances of premature EMR termination.
