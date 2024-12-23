---
title: "How do I parameterize Azure ML pipeline endpoint names?"
date: "2024-12-16"
id: "how-do-i-parameterize-azure-ml-pipeline-endpoint-names"
---

Alright,  Parameterizing Azure ML pipeline endpoint names, it's a detail that can easily become a pain point, especially as your projects scale and require different deployment environments or frequent updates. I've certainly been down this road; I recall a project some time back involving a complex model deployment process where hardcoding endpoint names quickly spiraled into an unmanageable mess. The key is to understand that these endpoints aren’t just static strings; they're essentially service identifiers that need to adapt to the dynamics of your development and deployment workflows.

The straightforward approach, that many begin with, is simply to embed the endpoint name directly in your pipeline definition or deployment scripts. This works fine initially, but it's brittle. If you need to switch to a staging environment, or if a naming convention changes, you’ll be editing code in multiple places, making it error-prone. So, let’s look at a more robust way, leveraging Azure ML's inherent capabilities and a bit of scripting.

First, the best practice is to parameterize this in your pipeline submission or deployment scripts. Azure ML supports parameters for pipelines, allowing you to dynamically inject values, including endpoint names. This approach enhances maintainability and facilitates environment-specific deployments. I've found that combining this with configuration files greatly simplifies things. Think of it like providing external settings to your pipelines, rather than baking everything in.

Here's how you can structure it using Python, which I've found to be most flexible for these tasks:

**Code Snippet 1: Defining Pipeline Parameters and Deploying**

```python
from azure.ai.ml import MLClient, command, Input, Output, pipeline
from azure.identity import DefaultAzureCredential
import os
import json

# Load configurations from a separate json file
def load_config(config_file_path):
    with open(config_file_path, 'r') as f:
        return json.load(f)

# Define the pipeline
def create_training_pipeline(input_data, output_path, compute_target, pipeline_name):
    training_step = command(
        code = "./src",
        command = "python train.py --data_path ${{inputs.input_data}} --output_path ${{outputs.output_path}}",
        inputs = { "input_data": input_data },
        outputs = { "output_path": output_path },
        environment="azureml-sklearn-0.24-ubuntu18.04-py37-cpu:1",
        compute=compute_target
    )

    pipeline_job = pipeline(
        name=pipeline_name,
        jobs={"train_step": training_step}
    )

    return pipeline_job

def deploy_endpoint(ml_client, pipeline_job, endpoint_name, deployment_name):
    # This is where you'd normally register a model, but this is skipped for simplicity
    # The main idea here is the endpoint_name
    # you will pass to the deploy code

    # Normally we deploy a model but for the sake of this example
    # we will just submit a pipeline
    submitted_pipeline_job = ml_client.jobs.create_or_update(pipeline_job)

    endpoint = ml_client.online_endpoints.begin_create_or_update(
        name=endpoint_name,
        auth_mode="key"
    ).result()

    deployment = ml_client.online_deployments.begin_create_or_update(
        name=deployment_name,
        endpoint_name=endpoint_name,
        model=None, # Replace with your model if deploying a model endpoint.
        instance_type="Standard_DS3_v2",
        instance_count=1
        ).result()

    return submitted_pipeline_job

if __name__ == "__main__":
    config = load_config("./config.json")

    credential = DefaultAzureCredential()
    ml_client = MLClient(credential, config['subscription_id'], config['resource_group'], config['workspace_name'])

    input_data = Input(path=config['input_data'], type="uri_folder")
    output_path = Output(path=config['output_path'], type="uri_folder")
    compute_target = config['compute_target']
    pipeline_name = config['pipeline_name']
    endpoint_name = config['endpoint_name']
    deployment_name = config['deployment_name']

    pipeline_job = create_training_pipeline(input_data, output_path, compute_target, pipeline_name)

    submitted_pipeline_job = deploy_endpoint(ml_client, pipeline_job, endpoint_name, deployment_name)
    print(f"Pipeline job submitted and Endpoint '{endpoint_name}' with deployment '{deployment_name}' initialized, see job at: {submitted_pipeline_job.studio_url}")

```

**config.json:**

```json
{
  "subscription_id": "<your-subscription-id>",
  "resource_group": "<your-resource-group>",
  "workspace_name": "<your-workspace-name>",
  "input_data": "https://azureopendatastorage.blob.core.windows.net/mnist/mnist.pkl",
  "output_path": "azureml://datastores/workspaceblobstore/outputs/",
  "compute_target": "cpu-cluster",
  "pipeline_name": "my_training_pipeline",
  "endpoint_name": "my-parameterized-endpoint",
  "deployment_name": "my-parameterized-deployment"
}
```

In this example, `endpoint_name` and `deployment_name` are loaded from the `config.json`. This setup is flexible, letting you change these names without altering your main Python script.

Now, let’s illustrate a scenario where your deployment process might involve multiple stages, like a pre-prod and prod. For that, you'd adjust your config and potentially introduce environment variables.

**Code Snippet 2: Environment-Aware Configuration**

```python
import os
import json

def load_environment_config(config_file_path, env_name):
    with open(config_file_path, 'r') as f:
        config = json.load(f)

    if env_name not in config['environments']:
        raise ValueError(f"Environment '{env_name}' not found in config file.")

    env_config = config['environments'][env_name]
    merged_config = {**config['defaults'], **env_config} # Merge default and env-specific settings

    return merged_config

if __name__ == "__main__":
    environment = os.environ.get("DEPLOYMENT_ENV", "dev")
    config_file = "./config_env.json"

    config = load_environment_config(config_file, environment)
    print(f"Using config for {environment} environment: {config}")

    # Further usage with the `config` loaded as before using previous code
    # credential = DefaultAzureCredential()
    # ml_client = MLClient(credential, config['subscription_id'], config['resource_group'], config['workspace_name'])
    # ... etc
```

**config_env.json:**

```json
{
  "defaults": {
     "subscription_id": "<your-subscription-id>",
    "resource_group": "<your-resource-group>",
    "workspace_name": "<your-workspace-name>",
    "input_data": "https://azureopendatastorage.blob.core.windows.net/mnist/mnist.pkl",
    "output_path": "azureml://datastores/workspaceblobstore/outputs/",
    "compute_target": "cpu-cluster",
    "pipeline_name": "my_training_pipeline"

  },
  "environments": {
    "dev": {
      "endpoint_name": "dev-endpoint",
      "deployment_name": "dev-deployment"
    },
    "staging": {
        "endpoint_name": "staging-endpoint",
        "deployment_name": "staging-deployment"
    },
    "prod": {
        "endpoint_name": "prod-endpoint",
        "deployment_name": "prod-deployment"
    }
  }
}
```

In this version, we’re loading configuration based on an environment variable, `DEPLOYMENT_ENV`. If not set, it defaults to "dev." You can run the script as `DEPLOYMENT_ENV=prod python your_script.py` to switch between environments without altering the script itself.

The third code piece focuses more directly on passing the name as part of the pipeline definition, which is crucial when dealing with dynamic pipeline structures.

**Code Snippet 3: Parameterized Endpoint Directly in Pipeline Definition**

```python
from azure.ai.ml import MLClient, command, Input, Output, pipeline
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import PipelineParameter
import os
import json

# Load configurations from a separate json file
def load_config(config_file_path):
    with open(config_file_path, 'r') as f:
        return json.load(f)

# Define the pipeline
def create_training_pipeline(input_data, output_path, compute_target, pipeline_name, endpoint_name):
    training_step = command(
        code = "./src",
        command = "python train.py --data_path ${{inputs.input_data}} --output_path ${{outputs.output_path}}",
        inputs = { "input_data": input_data },
        outputs = { "output_path": output_path },
        environment="azureml-sklearn-0.24-ubuntu18.04-py37-cpu:1",
        compute=compute_target
    )

    pipeline_job = pipeline(
        name=pipeline_name,
        jobs={"train_step": training_step},
        params={"endpoint_name": PipelineParameter(type='string')}
    )

    return pipeline_job

def deploy_endpoint(ml_client, pipeline_job, endpoint_name, deployment_name):
    submitted_pipeline_job = ml_client.jobs.create_or_update(pipeline_job, params={"endpoint_name":endpoint_name})

    endpoint = ml_client.online_endpoints.begin_create_or_update(
        name=endpoint_name,
        auth_mode="key"
    ).result()

    deployment = ml_client.online_deployments.begin_create_or_update(
        name=deployment_name,
        endpoint_name=endpoint_name,
        model=None,
        instance_type="Standard_DS3_v2",
        instance_count=1
        ).result()

    return submitted_pipeline_job

if __name__ == "__main__":
    config = load_config("./config.json")

    credential = DefaultAzureCredential()
    ml_client = MLClient(credential, config['subscription_id'], config['resource_group'], config['workspace_name'])

    input_data = Input(path=config['input_data'], type="uri_folder")
    output_path = Output(path=config['output_path'], type="uri_folder")
    compute_target = config['compute_target']
    pipeline_name = config['pipeline_name']
    endpoint_name = config['endpoint_name']
    deployment_name = config['deployment_name']

    pipeline_job = create_training_pipeline(input_data, output_path, compute_target, pipeline_name, endpoint_name)

    submitted_pipeline_job = deploy_endpoint(ml_client, pipeline_job, endpoint_name, deployment_name)
    print(f"Pipeline job submitted and Endpoint '{endpoint_name}' with deployment '{deployment_name}' initialized, see job at: {submitted_pipeline_job.studio_url}")

```

The main change here is that we are explicitly defining the endpoint as a `PipelineParameter`, this allows us to pass it when we create the pipeline. It can be used in the pipeline jobs as parameters for other components, if needed, or as shown here, just passed as parameters when submitting the pipeline.

For a deeper dive, I'd highly recommend reviewing the Azure Machine Learning documentation directly, specifically the sections on pipeline parameters and endpoints. Furthermore, consider reading "Designing Data-Intensive Applications" by Martin Kleppmann. While not exclusively about Azure, it offers a comprehensive understanding of designing systems where configurability and parameterization are key, and helps greatly with establishing good development patterns. "Continuous Delivery" by Jez Humble and David Farley is also a highly relevant text, as it provides strategies for managing deployment pipelines with a high degree of automation. Lastly, for those using python, ensure you have a good understanding of how to use the `json`, `os`, and `argparse` libraries which can handle configurations such as those shown in the examples.

In conclusion, parameterizing your endpoint names is not merely about avoiding code changes, it’s a critical aspect of building maintainable, scalable, and reliable machine learning systems. It allows your deployments to adapt without requiring manual alterations, aligning perfectly with modern development best practices. From personal experience, putting in the effort to parameterize will save a ton of time in the long run, so well worth the effort.
