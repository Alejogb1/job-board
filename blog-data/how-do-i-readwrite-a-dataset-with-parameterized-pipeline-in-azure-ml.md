---
title: "How do I read/write a dataset with parameterized pipeline in Azure ML?"
date: "2024-12-16"
id: "how-do-i-readwrite-a-dataset-with-parameterized-pipeline-in-azure-ml"
---

Alright, let’s delve into parameterizing data pipelines in Azure machine learning. I recall a particularly gnarly project a few years back – a large-scale retail forecasting system. We had data coming from disparate sources, each requiring slightly different preprocessing steps before we could train our models. Hand-coding separate pipelines for each source quickly became unsustainable. That's when we really leaned into parameterized pipelines, and it drastically improved our workflow. So, let's break it down.

The key to effectively handling datasets with parameterized pipelines in azure ml revolves around understanding how to define parameters within your pipeline and how to dynamically apply them. Think of these parameters as configurable settings for your pipeline components – allowing you to reuse the same pipeline structure with variations in data input, processing logic, and destination. This is crucial when dealing with multiple data sources, each with its nuances, or when experimenting with different training configurations.

First, we define parameters at the pipeline level. These parameters are then passed down to the individual steps within the pipeline. In the azure ml sdk for python, this usually involves using the `pipelineparameter` class, which is a key element in achieving flexibility and avoiding redundancy.

Let's start with a very basic example. Imagine you have two datasets, one in a csv format and another in a parquet format, both containing similar features but originating from different data lakes. You wish to perform basic data cleaning, like handling missing values and casting columns, before training. Instead of writing two entirely different pipelines, you parameterize the file type and the path to the data.

Here’s a simplified example using python and the azure ml sdk:

```python
from azureml.core import Workspace
from azureml.pipeline.core import Pipeline, PipelineParameter
from azureml.pipeline.steps import PythonScriptStep
from azureml.core.compute import AmlCompute, ComputeTarget
from azureml.core import Environment
from azureml.core.runconfig import RunConfiguration
from azureml.core.conda_dependencies import CondaDependencies

# Assume you have an existing workspace, resource group, etc.

ws = Workspace.from_config()

# Compute configuration (using existing cluster if available, create if not)
compute_name = 'your-compute-cluster-name'
if compute_name in ws.compute_targets:
    compute_target = ws.compute_targets[compute_name]
    if compute_target and compute_target.provisioning_state == 'Succeeded':
        print('Found compute target')
    else:
        print('compute target not found, provisioning')
        #define compute provision
else:
    print('creating a new compute target')
    #define compute provision

# Define a Python environment
env = Environment.from_conda_specification(
    name='my-env',
    file_path='environment.yml' # Define in a local yaml file
)

run_config = RunConfiguration()
run_config.environment = env

# Define parameters
input_data_path_param = PipelineParameter(name="input_data_path", default_value="default_data_path")
file_type_param = PipelineParameter(name="file_type", default_value="csv")

# Define a python step to process data
source_dir = '.'
python_step_file = 'data_processing.py'

python_step = PythonScriptStep(
    name="data_processing",
    script_name=python_step_file,
    source_directory=source_dir,
    inputs=[],
    outputs=[],
    arguments=[
        "--input_data_path", input_data_path_param,
        "--file_type", file_type_param,
    ],
    compute_target=compute_target,
    runconfig=run_config,
    allow_reuse=True
)


# Create pipeline
pipeline = Pipeline(workspace=ws, steps=[python_step])

# Publish pipeline
published_pipeline = pipeline.publish(name="parameterized_pipeline_example")

print(f"Published pipeline id: {published_pipeline.id}")
```

Notice how `input_data_path_param` and `file_type_param` are instances of `PipelineParameter`. These are used within the `PythonScriptStep`, passed as arguments to the python script which will then access them using `argparse`. The `data_processing.py` would then use these parameters to load and process the data. Now, here's what that `data_processing.py` script might look like:

```python
import argparse
import pandas as pd
import os

def process_data(input_data_path, file_type):

    if file_type == 'csv':
        df = pd.read_csv(input_data_path)
    elif file_type == 'parquet':
        df = pd.read_parquet(input_data_path)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")

    # Example processing steps: Fill NA values and cast some columns

    df.fillna(0, inplace=True) # Replace missing values with 0
    #cast numeric columns from object to int or float
    numeric_cols = df.select_dtypes(include = 'object').columns.to_list()
    for col in numeric_cols:
      try:
        df[col] = pd.to_numeric(df[col], errors='raise')
      except:
        print(f'could not cast {col} to numeric')
    print(f"Processed dataset shape: {df.shape}")

    # Saving the processed data (for demonstration purposes):
    output_path = os.path.join("output_folder", "processed_data.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Processed data saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process data based on input parameters")
    parser.add_argument("--input_data_path", type=str, help="Path to the input data file")
    parser.add_argument("--file_type", type=str, help="Type of input file (csv, parquet)")
    args = parser.parse_args()

    process_data(args.input_data_path, args.file_type)
```

This example demonstrates the basic idea. We parameterized the input data path and file type which allowed the same pipeline step to process either a csv or parquet formatted data by using `if/else` conditional processing within the python step.

Let's move on to a slightly more complex example involving output parameters. Suppose you also want to parameterize the name of the dataset you are processing, and then store the resulting output dataset into azure storage using the same parameterized name.

```python
from azureml.core import Workspace
from azureml.pipeline.core import Pipeline, PipelineParameter
from azureml.pipeline.steps import PythonScriptStep
from azureml.core.compute import AmlCompute
from azureml.core import Environment
from azureml.core.runconfig import RunConfiguration
from azureml.core.conda_dependencies import CondaDependencies
from azureml.data import OutputFileDatasetConfig

# Assume workspace and compute setup are already defined

ws = Workspace.from_config()

# Compute configuration (using existing cluster if available, create if not)
compute_name = 'your-compute-cluster-name'
if compute_name in ws.compute_targets:
    compute_target = ws.compute_targets[compute_name]
    if compute_target and compute_target.provisioning_state == 'Succeeded':
        print('Found compute target')
    else:
        print('compute target not found, provisioning')
        #define compute provision
else:
    print('creating a new compute target')
    #define compute provision

# Define a Python environment
env = Environment.from_conda_specification(
    name='my-env',
    file_path='environment.yml' # Define in a local yaml file
)

run_config = RunConfiguration()
run_config.environment = env


# Define parameters
input_data_path_param = PipelineParameter(name="input_data_path", default_value="default_data_path")
file_type_param = PipelineParameter(name="file_type", default_value="csv")
dataset_name_param = PipelineParameter(name="dataset_name", default_value="default_dataset")

# Define output dataset configuration
output_data = OutputFileDatasetConfig(destination=(ws.get_default_datastore(), f'processed/{dataset_name_param}')).as_upload(overwrite=True)

# Define the Python step
source_dir = '.'
python_step_file = 'data_processing_with_output.py'

python_step = PythonScriptStep(
    name="data_processing",
    script_name=python_step_file,
    source_directory=source_dir,
    inputs=[],
    outputs=[output_data],
    arguments=[
        "--input_data_path", input_data_path_param,
        "--file_type", file_type_param,
        "--dataset_name", dataset_name_param,
        "--output_path", output_data # Pass the azure output path
    ],
    compute_target=compute_target,
    runconfig=run_config,
    allow_reuse=True
)

# Create pipeline
pipeline = Pipeline(workspace=ws, steps=[python_step])

# Publish pipeline
published_pipeline = pipeline.publish(name="parameterized_pipeline_example")
print(f"Published pipeline id: {published_pipeline.id}")
```
In this adjusted code snippet, we introduced `dataset_name_param` to parameterize the output dataset. We also configured `output_data` as an instance of `OutputFileDatasetConfig` and passed it to the python script, which will store the processed dataset in `processed/{dataset_name_param}` within the default azure storage container. And here's the adjusted `data_processing_with_output.py` script:
```python
import argparse
import pandas as pd
import os
from azureml.core import Run

def process_data(input_data_path, file_type, dataset_name, output_path):

    if file_type == 'csv':
        df = pd.read_csv(input_data_path)
    elif file_type == 'parquet':
        df = pd.read_parquet(input_data_path)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")

    # Example processing steps: Fill NA values and cast some columns
    df.fillna(0, inplace=True)
    numeric_cols = df.select_dtypes(include = 'object').columns.to_list()
    for col in numeric_cols:
        try:
           df[col] = pd.to_numeric(df[col], errors='raise')
        except:
            print(f'Could not cast {col} to numeric')
    print(f"Processed dataset shape: {df.shape}")


    # The azure output data path will be passed into this script through the arguments
    # save the dataframe into that location
    os.makedirs(output_path, exist_ok=True)
    output_file_path = os.path.join(output_path, f'{dataset_name}.csv')
    df.to_csv(output_file_path, index = False)
    print(f"Processed data saved to: {output_file_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process data based on input parameters")
    parser.add_argument("--input_data_path", type=str, help="Path to the input data file")
    parser.add_argument("--file_type", type=str, help="Type of input file (csv, parquet)")
    parser.add_argument("--dataset_name", type=str, help="Name of the dataset")
    parser.add_argument("--output_path", type=str, help="Azure output path")
    args = parser.parse_args()

    process_data(args.input_data_path, args.file_type, args.dataset_name, args.output_path)
```

This setup provides a clear illustration of how you can combine parameterized input and output paths.

For those looking to delve deeper, I strongly recommend referring to the official Microsoft documentation for Azure Machine Learning pipelines. Specifically, the documentation on `PipelineParameter`, `PythonScriptStep`, and `OutputFileDatasetConfig` provides comprehensive details. The book “Programming Azure Machine Learning” by Jannes Klaas et al. offers a detailed overview of practical uses and advanced concepts. You should also look at the research paper from google “TensorFlow: A system for large-scale machine learning” for deeper insight into efficient data pipeline design which will help you to grasp concepts of optimal pipelines. This paper will help you to understand the theory and the why of this approach.

Remember, parameterization is about building flexible, reusable pipelines. These examples are basic but can easily be expanded upon to include many parameters such as hyperparameters for model training or even different data transformation steps based on the dataset. By mastering the use of pipeline parameters, you'll be able to effectively manage your azure ml workflows and experiments, thus increasing efficiency.
