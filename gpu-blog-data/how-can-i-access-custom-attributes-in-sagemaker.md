---
title: "How can I access custom attributes in SageMaker inference jobs using the Python SDK?"
date: "2025-01-30"
id: "how-can-i-access-custom-attributes-in-sagemaker"
---
I've frequently encountered the need to incorporate custom attributes into SageMaker inference jobs, typically to manage model deployments more granularly or to pass through specific user context. The SageMaker Python SDK provides a mechanism for this through the `EndpointConfig` and `CreateModel` API calls. These attributes, however, are not directly exposed as readily accessible fields within the prediction request. Instead, they are embedded in the SageMaker infrastructure and require a specific method to retrieve them from within the inference container code.

The core of the solution lies in understanding that custom attributes are stored as tags associated with the endpoint configuration. While the `sagemaker` SDK itself does not surface these tags in the prediction context, the underlying AWS API does. To access them, I've had success utilizing the `boto3` library directly from within the inference container, querying the `SageMaker` service for endpoint descriptions and extracting the relevant tags. This is a slight divergence from the typical `sagemaker` SDK workflow, requiring the incorporation of AWS low-level API interactions. I've found that this method, while slightly more complex, provides the flexibility required for real-world production scenarios involving detailed metadata tracking and management.

The first step involves configuring the endpoint such that it carries the necessary custom tags. This is generally done during endpoint creation and update operations. These tags are key-value pairs associated with the Endpoint Configuration resource. When creating an endpoint, the `sagemaker.endpoint.EndpointConfig` object is used. When updating the endpoint, a similar structure is employed when updating an endpoint configuration resource.

Once an endpoint is running, your inference code deployed within the model container can access these tags through `boto3`.  I use the following process: first, obtain the currently running endpoint name. This information is provided as an environment variable (`SAGEMAKER_ENDPOINT_NAME`) within the container. Next, the `boto3` SageMaker client is initialized and used to query the endpoint configuration for the target endpoint name. Finally, the returned description contains the tags I set during endpoint creation.

Here is an example illustrating how to create an endpoint with custom tags using the `sagemaker` library.  This assumes the existence of an estimator that has been successfully deployed:

```python
import sagemaker
from sagemaker.model import Model
from sagemaker.session import Session

# Assume 'my_estimator' is an already trained sagemaker.estimator.Estimator object
# and you have its model object 'my_model'

session = sagemaker.Session()

custom_tags = [
    {'Key': 'ModelVersion', 'Value': 'v1.2'},
    {'Key': 'DeploymentStage', 'Value': 'production'},
    {'Key': 'CreatedBy', 'Value': 'user123'}
]


#Create Endpoint Configuration
endpoint_config_name = "my-endpoint-config"

from sagemaker.config import EndpointConfig
endpoint_config = EndpointConfig(
        endpoint_config_name=endpoint_config_name,
        production_variants=[{
            'InstanceType': 'ml.m5.large',
            'InitialInstanceCount': 1,
            'ModelName': my_model.name,
            'VariantName': 'AllTraffic'
        }],
         tags=custom_tags
        )
endpoint_config.create(session=session)


# Create the Endpoint itself using existing Endpoint Configuration
endpoint_name = "my-endpoint"
from sagemaker.endpoint import Endpoint
endpoint = Endpoint(
    endpoint_name=endpoint_name,
    endpoint_config_name=endpoint_config_name,
    sagemaker_session=session
)

endpoint.create()
```

In this example, I set three custom tags: `ModelVersion`, `DeploymentStage`, and `CreatedBy`. These tags will be associated with the endpoint configuration resource, not the model or endpoint resources. It's crucial to remember these tags are not directly visible when making predictions.

Now, consider the inference code within the model container.  The next code example demonstrates retrieving the endpoint's tags using `boto3`:

```python
import os
import boto3
import json

def get_endpoint_tags():
    endpoint_name = os.environ.get("SAGEMAKER_ENDPOINT_NAME")
    if not endpoint_name:
        return {"error": "SAGEMAKER_ENDPOINT_NAME environment variable not found"}

    try:
        sagemaker = boto3.client("sagemaker")
        response = sagemaker.describe_endpoint(EndpointName=endpoint_name)
        endpoint_config_name = response["EndpointConfigName"]

        config_response = sagemaker.describe_endpoint_config(EndpointConfigName=endpoint_config_name)
        tags = config_response.get("Tags", [])

        tag_dict = {tag['Key']: tag['Value'] for tag in tags}
        return tag_dict
    except Exception as e:
        return {"error": str(e)}

def model_fn(model_dir):
    return None

def input_fn(input_data, content_type):
    return input_data

def predict_fn(data, model):
     tags = get_endpoint_tags()
     # Process data, use tags, and make prediction
     # Example:
     return  { "tags": tags, "prediction": "This is a dummy prediction" }

def output_fn(prediction, content_type):
    return json.dumps(prediction)
```

Here, I retrieve the `SAGEMAKER_ENDPOINT_NAME` environment variable and use it to query the SageMaker API. The response includes a list of tags that were associated with the endpoint's configuration. I then transform this list into a more convenient dictionary for easier access within the `predict_fn` handler. This function demonstrates the retrieval process, including the error handling. Within `predict_fn`, you would then incorporate these retrieved tags into your prediction processing logic. This example utilizes dummy data and a model; the core of accessing tags is within the `get_endpoint_tags()` function.

For more complex scenarios, such as needing to access the model name associated with the deployed endpoint, it's necessary to query further. We must first get the configuration, then find the model name, and finally, find the tags for that model. Here's an example of accessing both endpoint configuration tags and model tags:

```python
import os
import boto3
import json

def get_endpoint_and_model_tags():
    endpoint_name = os.environ.get("SAGEMAKER_ENDPOINT_NAME")
    if not endpoint_name:
        return {"error": "SAGEMAKER_ENDPOINT_NAME environment variable not found"}

    try:
        sagemaker = boto3.client("sagemaker")
        response = sagemaker.describe_endpoint(EndpointName=endpoint_name)
        endpoint_config_name = response["EndpointConfigName"]

        config_response = sagemaker.describe_endpoint_config(EndpointConfigName=endpoint_config_name)
        endpoint_tags = config_response.get("Tags", [])
        endpoint_tag_dict = {tag['Key']: tag['Value'] for tag in endpoint_tags}

        # Get the primary model name
        model_name = config_response["ProductionVariants"][0]["ModelName"]

        model_response = sagemaker.describe_model(ModelName=model_name)
        model_tags = model_response.get("Tags", [])
        model_tag_dict = {tag['Key']: tag['Value'] for tag in model_tags}


        return {"endpoint_tags": endpoint_tag_dict, "model_tags": model_tag_dict}
    except Exception as e:
        return {"error": str(e)}


def model_fn(model_dir):
    return None

def input_fn(input_data, content_type):
    return input_data

def predict_fn(data, model):
    tags = get_endpoint_and_model_tags()
    return {"tags": tags, "prediction":"This is a dummy prediction"}


def output_fn(prediction, content_type):
    return json.dumps(prediction)

```

In this example, I expand the function to also extract the name of the primary model from the endpoint configuration, and then query the model's tags. This demonstrates that the SageMaker API provides metadata access to various resources within a deployment. This function will return both endpoint tags and model tags, should the model also contain custom tags.

I recommend consulting the AWS documentation for both the SageMaker service and the `boto3` library. Specifically, focus on the `describe_endpoint`, `describe_endpoint_config`, and `describe_model` API calls. The documentation provides the most precise description of the request and response structures, and the available parameters. Additionally, familiarity with environment variables within the containerized environment is essential, specifically the `SAGEMAKER_ENDPOINT_NAME` environment variable. Further exploration of the tagging system, specifically its limitations and best practices within AWS, will be useful for production deployments. Finally, the `sagemaker` SDK documentation on endpoint configuration and deployment will aid in understanding the high-level setup process.
