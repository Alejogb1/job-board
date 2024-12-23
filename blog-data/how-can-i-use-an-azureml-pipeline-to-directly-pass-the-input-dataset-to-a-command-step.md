---
title: "How can I use an Azureml pipeline to directly pass the input dataset to a command step?"
date: "2024-12-23"
id: "how-can-i-use-an-azureml-pipeline-to-directly-pass-the-input-dataset-to-a-command-step"
---

,  I've spent quite a bit of time working with Azure Machine Learning pipelines, and passing datasets directly into command steps – without going through intermediate steps that create materialised files—is definitely a common challenge. There are multiple ways to achieve this, some more elegant than others, and it depends quite heavily on the specific requirements of your workflow.

Firstly, let's clarify what we’re trying to do. We want to take a registered Azureml dataset and have that data available directly as input to a script that is executed within a command step in an Azureml pipeline. The critical point here is "directly." We want to avoid unnecessary writes of the dataset to persistent storage (like a datastore) before the command step runs. That write operation can introduce significant delays, especially for large datasets. It's also less efficient if the next step expects the data in memory.

In my past projects, I've frequently encountered situations where preprocessing and model training steps were separated, with the preprocessing output typically being a large dataset. Instead of constantly reading and writing this large file on the datastore, the goal was to pass it directly into training. This approach saved considerable time and resources. The core to understanding this lies in understanding how `Input` objects and named inputs work within the pipeline `Command` step.

The command step in Azure ML accepts various input types, including datasets, which can be a registered dataset or a mounted directory with data files. When a dataset is provided as an input, Azure ML handles it for you automatically, making it accessible inside the execution environment of the command step, either as a mounted location or as an environment variable referencing the data. Let me illustrate with some practical scenarios.

**Scenario 1: Direct dataset as input path to script**

The most common case is that your training script expects a path to the input data, not the data itself in memory. This is often the case with traditional machine learning workflows that involve reading files (CSV, Parquet, etc.). You don’t necessarily need to load the data into memory on the driver node in your command step, or within your script itself, to perform operations efficiently.

Here’s a code snippet that demonstrates this setup:

```python
from azure.ai.ml import command, Input, Output
from azure.ai.ml.entities import Data
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml import  dsl, Input, Output

# Replace with your workspace details
subscription_id = "your_subscription_id"
resource_group = "your_resource_group"
workspace_name = "your_workspace_name"

ml_client = MLClient(
    DefaultAzureCredential(), subscription_id, resource_group, workspace_name
)


# Assuming you have a registered dataset 'my_registered_dataset'
# Create a command component
preprocessing_step = command(
    command="python train.py --input_path ${{inputs.training_data}} --output_dir ${{outputs.model_output}}",
    code="./src",
    inputs={"training_data": Input(type="uri_folder")},  # Using uri_folder for a path
    outputs={"model_output": Output(type="uri_folder")}, # output directory
    environment="azureml:basic-env:1",
    compute="cpu-cluster"
)


@dsl.pipeline(compute="cpu-cluster")
def my_pipeline():
     # Retrieve the registered dataset
    registered_dataset = ml_client.data.get(name="my_registered_dataset", version=1)

    train_job = preprocessing_step(training_data=registered_dataset)
    return {"train_job": train_job}


pipeline_job = my_pipeline()

# Submit pipeline job
returned_job = ml_client.jobs.create_or_update(pipeline_job)
print(returned_job)

```

In the above code, I’m assuming that you have a registered dataset called `my_registered_dataset`. The crucial part is the `training_data` input declaration in the `command` step: `Input(type="uri_folder")`. This indicates that the input should be mounted as a folder, and the path to that folder will be passed as an argument (`--input_path`) to your `train.py` script. The script can then use that path to load data using libraries like pandas or dask. This is very useful for dealing with tabular or file-based data.

**Scenario 2: Passing in-memory data to script using an environment variable**

Sometimes, the data we want to use is not in a file but in a dataframe or a python object. In such cases, reading a large dataset from a mounted folder and doing conversion in your script can be inefficient. Azure ML provides a way to pass small to medium data as environment variables, in this case, you need to convert your data to a string representation. I suggest keeping the size of data under a few MBs for passing using an environment variable.

```python
import pandas as pd
import json
from azure.ai.ml import command, Input, Output
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml import  dsl, Input, Output

# Replace with your workspace details
subscription_id = "your_subscription_id"
resource_group = "your_resource_group"
workspace_name = "your_workspace_name"

ml_client = MLClient(
    DefaultAzureCredential(), subscription_id, resource_group, workspace_name
)

def dataset_to_string(dataset):
    # Assuming dataset is a pandas dataframe
    return dataset.to_json(orient='records')

# Assuming you have a registered dataset 'my_registered_dataset'
# Create a command component
preprocessing_step = command(
    command="python train.py --data_str $DATA_STRING --output_dir ${{outputs.model_output}}",
    code="./src",
    environment="azureml:basic-env:1",
    outputs={"model_output": Output(type="uri_folder")}, # output directory
    compute="cpu-cluster"
)

@dsl.pipeline(compute="cpu-cluster")
def my_pipeline():
    registered_dataset = ml_client.data.get(name="my_registered_dataset", version=1)
    #convert to pandas dataframe
    pandas_dataset = registered_dataset.to_pandas_dataframe()
    # convert to string
    data_string = dataset_to_string(pandas_dataset)

    train_job = preprocessing_step(environment_variables={"DATA_STRING": data_string})
    return {"train_job": train_job}

pipeline_job = my_pipeline()
returned_job = ml_client.jobs.create_or_update(pipeline_job)
print(returned_job)

```

In this example, we retrieve the pandas dataset, convert it to a json string, and pass it as an environment variable `DATA_STRING`. Inside `train.py`, we can load the json string into memory and process it. If using larger than a few MBs data sizes, avoid this method.

**Scenario 3: Using Dataset as Input with specific loading logic**

The final approach involves using an `Input` object to directly pass your registered dataset. This often requires that you write custom data loading logic inside your training script, or if your script already has such logic, this provides great efficiency.

```python
from azure.ai.ml import command, Input, Output
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml import  dsl, Input, Output

# Replace with your workspace details
subscription_id = "your_subscription_id"
resource_group = "your_resource_group"
workspace_name = "your_workspace_name"

ml_client = MLClient(
    DefaultAzureCredential(), subscription_id, resource_group, workspace_name
)


# Assuming you have a registered dataset 'my_registered_dataset'
# Create a command component
preprocessing_step = command(
    command="python train.py --input_path ${{inputs.training_data}} --output_dir ${{outputs.model_output}}",
    code="./src",
    inputs={"training_data": Input(type="uri_folder")}, # uri_folder for a path
    outputs={"model_output": Output(type="uri_folder")}, # output directory
    environment="azureml:basic-env:1",
    compute="cpu-cluster"
)

@dsl.pipeline(compute="cpu-cluster")
def my_pipeline():
     # Retrieve the registered dataset
    registered_dataset = ml_client.data.get(name="my_registered_dataset", version=1)

    train_job = preprocessing_step(training_data=registered_dataset)
    return {"train_job": train_job}


pipeline_job = my_pipeline()

# Submit pipeline job
returned_job = ml_client.jobs.create_or_update(pipeline_job)
print(returned_job)
```

This is almost the same as the first example, with the key distinction being that in your `train.py` script, you would have to handle reading the `training_data` path in the way that best fits the format of your data. Libraries like `pandas` or `dask` could be used.

**Important Considerations and Resources**

*   **Data Format:** Ensure that your `train.py` script is compatible with the format of your dataset.
*   **Data Size:** For very large datasets, consider using data parallelism with Azure ML's distributed training capabilities. You may need to modify your `train.py` script for distributed reads of a dataset.
*   **Environment:** Carefully manage your environment to make sure that the necessary libraries are present in the docker environment.
*   **Data Versioning:** Keep your data versioned. Azure ML registered datasets are versioned. This ensures traceability and reproducibility in your pipeline.

For detailed information, I recommend reviewing the official Microsoft Azure Machine Learning documentation. Specifically, look at the documentation on:

1.  **Azure ML Pipelines:** There are excellent sections on defining pipelines, data inputs, and using the `Command` step.
2.  **Input Objects:** Thoroughly understand the different types of inputs you can provide to a `Command` step, including `uri_file`, `uri_folder` and data objects.
3.  **Datasets in Azure ML:** Read the material on registering and managing datasets in Azure Machine Learning.

By leveraging the appropriate input types, you can significantly optimize your Azure ML pipelines to avoid unnecessary data transfers and enhance overall performance. Understanding these mechanics has been invaluable in my projects, leading to more efficient and scalable machine learning workflows. Remember, the key is to understand how Azure ML handles datasets and to tailor your approach to the specific needs of your pipeline.
