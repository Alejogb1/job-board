---
title: "How do I retrieve a pipeline run's status within a Vertex AI component?"
date: "2024-12-23"
id: "how-do-i-retrieve-a-pipeline-runs-status-within-a-vertex-ai-component"
---

Alright, let's tackle retrieving pipeline run status from within a Vertex ai component. I've bumped into this particular challenge more than once in the past, and it’s a pretty common requirement when dealing with more complex workflows. Initially, during a project involving automated model retraining, I needed to trigger subsequent steps based on the success or failure of a training pipeline. The core of the issue boils down to how you access metadata concerning your pipeline execution from within a component, which, by its very nature, operates in isolation.

The challenge arises because components are designed to be self-contained. They don’t inherently know about the pipeline they're part of, including its current status. The solution, therefore, involves leveraging Vertex AI’s metadata store and the client library to query this information. This approach ensures that components remain modular and decouples them from specific pipeline runs, thus making the solution reusable across different pipeline executions.

Essentially, within your component, you need to perform two key actions: first, identify the current pipeline run using the available metadata and second, query the Vertex ai api to retrieve the execution status of that particular run.

Let's dive into the technical particulars.

**Identifying the Current Pipeline Run**

When your component is running, Vertex AI injects metadata that includes the current execution id. You can capture this id within your python-based component. The crucial environment variable to look for is `AIP_RUN_ID`. This variable holds the unique identifier for the currently executing pipeline run.

**Retrieving the Pipeline Run Status**

Once you have the run id, you can use the Vertex ai python client library to query the api for the run status. Here's where things become a bit more nuanced. We'll need to initialize the client, construct the correct api path, and handle various possible states of the pipeline execution, such as 'running,' 'succeeded,' 'failed,' 'cancelling' and others.

Here's a python code snippet exemplifying the process:

```python
import os
from google.cloud import aiplatform

def get_pipeline_run_status():
    """Retrieves the status of the current Vertex AI pipeline run."""
    run_id = os.environ.get("AIP_RUN_ID")
    if not run_id:
      raise ValueError("AIP_RUN_ID environment variable not found. Not running within a Vertex AI pipeline?")

    project = os.environ.get("PROJECT_ID") # Make sure PROJECT_ID is set
    location = os.environ.get("REGION") # Make sure REGION is set

    aiplatform.init(project=project, location=location)

    pipeline_job = aiplatform.PipelineJob.get(resource_name=f"projects/{project}/locations/{location}/pipelineJobs/{run_id}")

    return pipeline_job.state.name

if __name__ == '__main__':
    try:
      pipeline_status = get_pipeline_run_status()
      print(f"Current Pipeline Run Status: {pipeline_status}")
      # Add logic to proceed based on status
    except ValueError as e:
        print(f"Error retrieving pipeline status: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
```

This snippet captures `AIP_RUN_ID`, fetches the pipeline job details using the Vertex ai api, and returns the `pipeline_job.state.name`. The `state` field is an enum that can take values that reflect the pipeline status, such as `PIPELINE_STATE_SUCCEEDED`, `PIPELINE_STATE_FAILED` or `PIPELINE_STATE_RUNNING`. I've added some error handling to make sure this works even if some expected environment variables are not set.

Now, let’s add more complexity. In cases where you're dealing with sub-pipelines, the approach is fundamentally similar but requires extracting the root pipeline id. This happens particularly often with nested kfp pipelines. While the `AIP_RUN_ID` within a component residing in a sub-pipeline will still provide the id for that particular nested execution, we may want the root parent pipeline's status instead. To do so, we’d need to access the `AIP_PARENT_RUN_ID` environment variable when available, and check the status of that pipeline. If that is not available, it will simply default to the current run's status.

Here is the updated snippet, covering this scenario:

```python
import os
from google.cloud import aiplatform

def get_pipeline_run_status():
    """Retrieves the status of the current or parent Vertex AI pipeline run."""
    run_id = os.environ.get("AIP_RUN_ID")
    parent_run_id = os.environ.get("AIP_PARENT_RUN_ID")
    project = os.environ.get("PROJECT_ID")
    location = os.environ.get("REGION")

    if not run_id:
        raise ValueError("AIP_RUN_ID environment variable not found. Not running within a Vertex AI pipeline?")


    aiplatform.init(project=project, location=location)

    target_run_id = parent_run_id if parent_run_id else run_id
    pipeline_job = aiplatform.PipelineJob.get(resource_name=f"projects/{project}/locations/{location}/pipelineJobs/{target_run_id}")
    return pipeline_job.state.name


if __name__ == '__main__':
  try:
    pipeline_status = get_pipeline_run_status()
    print(f"Current (or Parent) Pipeline Run Status: {pipeline_status}")
        # Add logic to proceed based on status
  except ValueError as e:
     print(f"Error retrieving pipeline status: {e}")
  except Exception as e:
    print(f"An unexpected error occurred: {e}")
```

Finally, let's add another layer of sophistication. Often, we may want to not only query for the current pipeline status but potentially wait for it to reach a specific state, such as 'succeeded,' before proceeding with the component’s logic. To achieve this, we can introduce a simple polling mechanism using the `time` library.

Here’s the final, more robust example:

```python
import os
import time
from google.cloud import aiplatform

def wait_for_pipeline_completion(target_status="PIPELINE_STATE_SUCCEEDED", timeout_seconds=3600):
    """Waits for the current pipeline run to reach a target state."""
    run_id = os.environ.get("AIP_RUN_ID")
    parent_run_id = os.environ.get("AIP_PARENT_RUN_ID")
    project = os.environ.get("PROJECT_ID")
    location = os.environ.get("REGION")

    if not run_id:
        raise ValueError("AIP_RUN_ID environment variable not found. Not running within a Vertex AI pipeline?")


    aiplatform.init(project=project, location=location)

    target_run_id = parent_run_id if parent_run_id else run_id


    start_time = time.time()
    while True:
        pipeline_job = aiplatform.PipelineJob.get(resource_name=f"projects/{project}/locations/{location}/pipelineJobs/{target_run_id}")
        current_status = pipeline_job.state.name

        if current_status == target_status:
            return True  # Target status reached

        if current_status == "PIPELINE_STATE_FAILED" or current_status == "PIPELINE_STATE_CANCELLED":
           raise RuntimeError(f"Pipeline failed or was cancelled, current state: {current_status}")


        elapsed_time = time.time() - start_time
        if elapsed_time > timeout_seconds:
            raise TimeoutError(f"Timeout waiting for pipeline completion, current state: {current_status}")


        time.sleep(60)  # Polling interval of 1 minute


if __name__ == '__main__':
   try:
        wait_for_pipeline_completion()
        print("Pipeline completed successfully.")
        # Proceed with subsequent tasks.
   except (ValueError, TimeoutError, RuntimeError) as e:
      print(f"Error waiting for pipeline completion: {e}")
   except Exception as e:
        print(f"An unexpected error occurred: {e}")

```

This code defines a `wait_for_pipeline_completion` function, that takes target status and timeout as input. It continuously polls the pipeline state, returning when the target state is reached, failing when it's reached failure or cancellation states, or timing out after a predefined period. It handles exceptions and prints the corresponding error message.

For deeper insight into the Vertex ai python client, the official documentation is invaluable. I would also recommend looking at the "Designing Machine Learning Systems" book by Chip Huyen which provides excellent guidelines for structured pipeline design. Google’s Vertex AI documentation also details the different possible values the pipeline state enum can hold. These resources should provide all the necessary information to work with pipeline statuses in Vertex AI effectively. This approach gives you the ability to adapt your components dynamically based on the pipeline's execution state, which adds a lot of flexibility and fault tolerance to complex ML workflows.
