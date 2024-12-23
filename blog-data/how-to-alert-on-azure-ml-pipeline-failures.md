---
title: "How to alert on Azure ML pipeline failures?"
date: "2024-12-23"
id: "how-to-alert-on-azure-ml-pipeline-failures"
---

Alright, let's tackle the question of alerting on Azure Machine Learning pipeline failures. This is something I’ve spent a good bit of time refining over the years, having dealt with a few production meltdowns because a seemingly robust pipeline silently crumbled. It's more nuanced than simply checking for a failed job; it requires a holistic approach, encompassing multiple layers of monitoring and notification.

First, let's break down the problem. An Azure ML pipeline, in essence, is a series of interconnected steps, each potentially executing code, transformations, or machine learning models. A failure in any one of these steps could cascade and halt the entire pipeline. So, relying solely on a general "pipeline failed" notification is often inadequate. We need to know *which* step failed, why it failed, and perhaps even have the relevant logs readily available.

My journey started with the naive approach – simply checking the pipeline run status via the Azure ML SDK. While this works, it’s reactive, not proactive. I’d discover failures hours later, often when the downstream systems dependent on the pipeline had already started exhibiting issues. This was far from ideal, and it pushed me towards leveraging Azure Monitor.

Azure Monitor offers a more comprehensive way to track not only pipeline runs but also the resource utilization of the underlying compute. Specifically, I found three key features indispensable for robust alerting: *Metrics*, *Logs*, and *Activity Log*. Metrics provide numerical data about resource performance, logs capture event details and diagnostic information, and the activity log tracks changes at the resource level.

For basic alerting, Metrics are a great starting point. You can create alerts based on specific metrics emitted by the Azure Machine Learning service. For instance, you can monitor the 'PipelineRunStatus' metric and set up an alert if the run status becomes 'Failed'. However, for fine-grained alerts, the logs are where we find the true value.

Let's consider a scenario where a data validation step within the pipeline consistently fails due to malformed input. Here’s a Python snippet illustrating how we’d check for specific errors within logs and potentially raise an alert:

```python
from azure.identity import DefaultAzureCredential
from azure.monitor.query import LogsQueryClient
import json
import os

def query_pipeline_logs(resource_group, workspace_name, pipeline_run_id, query_time_range_hours=1):
    credential = DefaultAzureCredential()
    logs_client = LogsQueryClient(credential)

    query = f"""
    let dt = now() - {query_time_range_hours}h;
    AzureDiagnostics
    | where TimeGenerated >= dt
    | where ResourceProvider == "MICROSOFT.MACHINELEARNINGSERVICES"
    | where ResourceGroup == "{resource_group}"
    | where WorkspaceName == "{workspace_name}"
    | where OperationName == "Microsoft.MachineLearningServices/pipelineRuns/update"
    | where ResultDescription  contains "PipelineRunId('{pipeline_run_id}')"
    | extend ParsedMessage = parse_json(ResultDescription)
    | extend Stage = tostring(ParsedMessage.StageName)
    | extend Status = tostring(ParsedMessage.Status)
    | extend ErrorMessage = tostring(ParsedMessage.ErrorMessage)
    | where Status == "Failed"
    | project TimeGenerated, Stage, Status, ErrorMessage
    """

    response = logs_client.query(workspace_name, query)

    if response.tables:
        for row in response.tables[0].rows:
            time, stage, status, error_message = row
            print(f"Time: {time}, Stage: {stage}, Status: {status}, Error: {error_message}")
            # Raise alert logic here using your chosen mechanism

    return response.tables # returning the tables for further use if needed


if __name__ == '__main__':

    resource_group = "your_resource_group_name" #replace with your resource group
    workspace_name = "your_workspace_name" #replace with your workspace name
    pipeline_run_id = "your_pipeline_run_id"  #replace with your pipeline run id

    query_pipeline_logs(resource_group, workspace_name, pipeline_run_id, query_time_range_hours=2)

```

This code snippet pulls log entries related to a specific pipeline run, parses the json error message from the 'ResultDescription' field, and filters for failed stages. We can then extract relevant information like the timestamp, the failing stage, and the error message. In a real-world scenario, the comment `# Raise alert logic here using your chosen mechanism` would be replaced with code that triggers a notification mechanism (e.g., an email, SMS, or a message to a monitoring platform).

Another powerful tool is the Azure Activity Log. This log tracks management plane operations and offers insight into configuration changes. For example, if a compute cluster required for a pipeline is deleted, it won't necessarily cause a 'pipeline failed' metric, but it will show up in the Activity Log. Here's another short python example demonstrating extracting activity logs:

```python
from azure.identity import DefaultAzureCredential
from azure.monitor.query import ActivityLogQueryClient
import os

def query_activity_logs(resource_group, query_time_range_hours=1):
    credential = DefaultAzureCredential()
    activity_log_client = ActivityLogQueryClient(credential)

    query = f"""
    AzureActivity
    | where TimeGenerated > ago({query_time_range_hours}h)
    | where ResourceGroup == "{resource_group}"
    | where OperationNameValue  contains "Microsoft.MachineLearningServices/computes"
    | where OperationNameValue contains "delete"
     | project TimeGenerated, ResourceGroup, OperationNameValue, Caller, Status
    """


    response = activity_log_client.query(query)


    if response.tables:
        for row in response.tables[0].rows:
            time, resource_group, operation, caller, status = row
            print(f"Time: {time}, Resource Group: {resource_group}, Operation: {operation}, Caller: {caller}, Status: {status}")

    return response.tables  # returning the tables for further use if needed


if __name__ == '__main__':

    resource_group = "your_resource_group_name" #replace with your resource group
    query_activity_logs(resource_group, query_time_range_hours=2)
```

This snippet queries activity logs, filtering for compute operations, and focusing on deletions. This is useful to spot instances where the compute resources for the pipelines are modified. As before, the output can be connected to alerting mechanisms.

Now, beyond basic failure detection, it is also beneficial to setup alerting on data drift. When your input data changes, it might lead to model performance degradation or pipeline instability, and this should be something that is monitored as part of the overall pipeline. This can be done by registering a metric in a training step, which is then monitored via Azure Monitor Metrics, as shown in the snippet below:

```python
from azure.identity import DefaultAzureCredential
from azure.monitor.query import MetricsQueryClient
import os
from datetime import timedelta, datetime

def query_metric_logs(resource_id, metric_name, query_time_range_hours=1):
    credential = DefaultAzureCredential()
    metric_client = MetricsQueryClient(credential)

    end_time = datetime.utcnow()
    start_time = end_time - timedelta(hours=query_time_range_hours)

    response = metric_client.query(
            resource_ids=[resource_id],
            metric_names=[metric_name],
            timespan=(start_time, end_time)
        )
    if response.timeseries:
         for timeseries in response.timeseries:
             for data_point in timeseries.data:
                 if data_point.total is not None: #data_point.average or data_point.minimum could also be useful here
                     print(f"Time: {data_point.time_stamp}, Metric Value: {data_point.total}")
                 # add your logic to raise alerts based on data drift here

if __name__ == '__main__':
    resource_id = "/subscriptions/your_subscription_id/resourceGroups/your_resource_group_name/providers/Microsoft.MachineLearningServices/workspaces/your_workspace_name/computes/your_compute_cluster_name" #replace with your resource id
    metric_name = "your_metric_name" #replace with your metric name
    query_metric_logs(resource_id, metric_name, query_time_range_hours=2)

```

This snippet allows to pull metric data for a given resource and metric name within a time range. In practice, you'd have a training step output the drift metric to azure ML and then you could monitor it using this function, triggering alerts when this metric exceeds a certain threshold.

For further reading and a more in-depth understanding of this topic, I would strongly recommend these resources. Firstly, *Microsoft Azure Documentation* itself is incredibly comprehensive, specifically the sections on Azure Monitor, Azure Machine Learning, and the Python SDK. For a more theoretical perspective, consider “*Designing Data-Intensive Applications*” by Martin Kleppmann. It’s not specific to Azure, but it covers the principles of building robust and reliable systems, which is entirely relevant here. The book *Cloud Native Patterns* by Cornelia Davis can also be insightful, giving a holistic perspective on building distributed applications in the cloud. Lastly, for practical and hands on experience check out the official Azure samples documentation and the Azure SDK documentation.

In conclusion, alerting on Azure ML pipeline failures goes beyond rudimentary checks. By leveraging Azure Monitor’s metrics, logs, and the activity log, and having a thoughtful design to catch various failure modes, we can build pipelines that are not just functional but also resilient. The examples provided, while simple, show a clear and systematic approach, which has been essential to my practice in preventing system failures and identifying problematic areas swiftly.
