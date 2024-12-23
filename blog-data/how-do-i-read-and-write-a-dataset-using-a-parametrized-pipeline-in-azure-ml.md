---
title: "How do I read and write a dataset using a parametrized Pipeline in Azure ML?"
date: "2024-12-23"
id: "how-do-i-read-and-write-a-dataset-using-a-parametrized-pipeline-in-azure-ml"
---

Let's tackle that thorny issue of parameterizing your Azure ML pipelines for dataset interaction. It's a common scenario, and one I've bumped into several times over the years, especially when moving models from research to production. Getting it solid requires a careful blend of Azure ML's SDK features and an understanding of how pipelines actually handle data. The key, as you've probably guessed, isn’t hardcoding paths, but rather leveraging parameters to keep things flexible.

I recall one project in particular, involving time-series analysis for industrial equipment. We had different data streams coming in from various sensors and we needed a robust way to train, test and score using different subsets of this data. Hardcoding locations became a maintenance nightmare very quickly. So, let’s break down how to do this effectively.

Firstly, we need to understand that Azure ML pipeline parameters don’t directly handle large datasets themselves. Instead, they deal with metadata, like file paths, which then tell the pipeline where to look for the data. This is why defining dataset inputs as parameters using `PipelineParameter` is pivotal. We will then pass these parameters to our pipeline steps, where we can read from or write to the data storage.

Let's walk through some code examples using the Azure Machine Learning SDK for Python, as that's the typical tooling for this. I'll aim for practicality, assuming you're already familiar with the basics of setting up workspaces and compute targets in Azure ML.

**Example 1: Reading a Tabular Dataset Using a Parameter**

Here's how to define a pipeline that takes a dataset path as a parameter and loads a tabular dataset. We are working with `TabularDataset` type here.

```python
from azureml.core import Workspace, Dataset
from azureml.pipeline.core import Pipeline, PipelineParameter
from azureml.pipeline.steps import PythonScriptStep
from azureml.core.runconfig import RunConfiguration
from azureml.core.conda_dependencies import CondaDependencies

# Load Workspace (replace with your workspace config)
ws = Workspace.from_config()

# Define a pipeline parameter for the dataset path
dataset_path_param = PipelineParameter(name="input_data_path", default_value="path/to/your/data.csv")

# Define the Python script to load and process the dataset
source_dir = "scripts" # directory to hold our python scripts

load_data_script = """
import pandas as pd
import argparse
from azureml.core import Run
from azureml.data import Dataset

parser = argparse.ArgumentParser()
parser.add_argument('--input_data_path', type=str, help='Path to the input data')
args = parser.parse_args()

run = Run.get_context()
workspace = run.experiment.workspace
dataset = Dataset.get_by_path(workspace=workspace, path=args.input_data_path)

# dataset is now a dataset object which you can convert to pandas dataframe
data_df = dataset.to_pandas_dataframe()

print(f"Loaded dataset with {data_df.shape[0]} rows and {data_df.shape[1]} columns.")
"""

with open(f"{source_dir}/load_data.py", "w") as file:
  file.write(load_data_script)


# Define the run configuration
run_config = RunConfiguration()
run_config.conda_dependencies = CondaDependencies.create(conda_packages=['pandas', 'scikit-learn'])

# Create the Python script step, passing the parameter
load_data_step = PythonScriptStep(
    name="Load Tabular Data",
    source_dir=source_dir,
    script_name="load_data.py",
    arguments=["--input_data_path", dataset_path_param],
    compute_target="your-compute-target",  # Replace with your compute target
    runconfig=run_config,
    allow_reuse=True
)

# Create the pipeline
pipeline = Pipeline(workspace=ws, steps=[load_data_step])

# Validate and publish the pipeline
pipeline.validate()
published_pipeline = pipeline.publish(name="Dataset_Param_Pipeline", description="Pipeline that takes dataset path as parameter")
print(f"Published pipeline id: {published_pipeline.id}")


# Example to trigger the pipeline with your data location
pipeline_run = ws.experiments.get("your-experiment-name").submit(published_pipeline,
     pipeline_parameters={"input_data_path": "your/data/relative/location/data.csv"}
)

```

In this code, `input_data_path` acts as the parameter. The `PythonScriptStep` then receives this parameter. Inside `load_data.py` script, the dataset path parameter is received as an argument using `argparse` and we retrieve the dataset using `Dataset.get_by_path()` function. The key aspect here is that `Dataset.get_by_path` needs the full relative path in reference to the storage account linked with the Azure ML workspace.

**Example 2: Writing a Dataset Using a Parameter**

Now, let's consider writing data to a specific location, using a parameter. This is equally important for saving intermediate results from your processing steps.

```python
from azureml.core import Workspace, Datastore
from azureml.pipeline.core import Pipeline, PipelineParameter
from azureml.pipeline.steps import PythonScriptStep
from azureml.core.runconfig import RunConfiguration
from azureml.core.conda_dependencies import CondaDependencies
from azureml.data.datapath import DataPath
import os

# Load Workspace (replace with your workspace config)
ws = Workspace.from_config()

# Fetch default datastore
default_datastore = ws.get_default_datastore()

# Define a pipeline parameter for the output path
output_data_path_param = PipelineParameter(name="output_data_path", default_value="output_data")

# Define the Python script to create some dummy data and store it
source_dir = "scripts"

write_data_script = """
import pandas as pd
import argparse
from azureml.core import Run
from azureml.data.datapath import DataPath
import os

parser = argparse.ArgumentParser()
parser.add_argument('--output_data_path', type=str, help='Path to the output data')
args = parser.parse_args()

run = Run.get_context()
workspace = run.experiment.workspace
default_datastore = workspace.get_default_datastore()
output_path = DataPath(datastore=default_datastore, path_on_datastore=args.output_data_path)


# Create dummy data
data = {'col1': [1, 2, 3], 'col2': [4, 5, 6]}
df = pd.DataFrame(data)

# Write the data to the specified path
output_path.as_mount()
os.makedirs(output_path,exist_ok=True)
output_file_path = os.path.join(output_path, 'output.csv')

df.to_csv(output_file_path, index=False)

print(f"Data written to {output_file_path}")
"""
with open(f"{source_dir}/write_data.py", "w") as file:
    file.write(write_data_script)

# Define the run configuration
run_config = RunConfiguration()
run_config.conda_dependencies = CondaDependencies.create(conda_packages=['pandas'])

# Create the Python script step, passing the parameter
write_data_step = PythonScriptStep(
    name="Write Data",
    source_dir=source_dir,
    script_name="write_data.py",
    arguments=["--output_data_path", output_data_path_param],
    compute_target="your-compute-target", # Replace with your compute target
    runconfig=run_config,
    allow_reuse=True
)

# Create the pipeline
pipeline = Pipeline(workspace=ws, steps=[write_data_step])

# Validate and publish the pipeline
pipeline.validate()
published_pipeline = pipeline.publish(name="Dataset_Param_Pipeline_Write", description="Pipeline that takes dataset output path as parameter")
print(f"Published pipeline id: {published_pipeline.id}")


# Example to trigger the pipeline with your output location
pipeline_run = ws.experiments.get("your-experiment-name").submit(published_pipeline,
     pipeline_parameters={"output_data_path": "your/output/location/folder"}
)
```

Here, the crucial element is the use of `DataPath` class. This function is used to point towards a location in the default datastore associated with the workspace. The `output_data_path_param` parameter is passed to the step, and the `write_data.py` script creates a dummy dataset and saves it to the parametrized location. Note that `DataPath.as_mount()` is used to ensure that the script has access to that path.

**Example 3: Using a Dataset Object as Parameter**

Finally, a more advanced use-case would involve passing an already defined dataset (instead of a path) as a pipeline parameter. This would be beneficial when we need to perform an operation on a particular version of a registered dataset.

```python
from azureml.core import Workspace, Dataset
from azureml.pipeline.core import Pipeline, PipelineParameter
from azureml.pipeline.steps import PythonScriptStep
from azureml.core.runconfig import RunConfiguration
from azureml.core.conda_dependencies import CondaDependencies
from azureml.data import OutputFileDatasetConfig

# Load Workspace (replace with your workspace config)
ws = Workspace.from_config()

# Fetch an existing registered dataset (replace with yours)
existing_dataset = Dataset.get_by_name(ws, name='your-existing-dataset-name')

# Define a pipeline parameter for the dataset
dataset_param = PipelineParameter(name='input_dataset', default_value=existing_dataset)

# Define the Python script to read and process the dataset
source_dir = 'scripts'

process_data_script = """
import pandas as pd
import argparse
from azureml.core import Run
from azureml.data import Dataset
from azureml.core.run import Run
import os

parser = argparse.ArgumentParser()
parser.add_argument('--input_dataset', type=str, help='Input dataset')
args = parser.parse_args()

run = Run.get_context()
workspace = run.experiment.workspace

# Retrieve the dataset using its ID
dataset = Dataset.get_by_id(workspace=workspace, id=args.input_dataset)

# Convert to pandas dataframe and process
data_df = dataset.to_pandas_dataframe()

print(f'Loaded dataset with {data_df.shape[0]} rows and {data_df.shape[1]} columns.')
"""

with open(f"{source_dir}/process_data.py", "w") as file:
    file.write(process_data_script)


# Define the run configuration
run_config = RunConfiguration()
run_config.conda_dependencies = CondaDependencies.create(conda_packages=['pandas', 'scikit-learn'])

# Create the Python script step, passing the parameter
process_data_step = PythonScriptStep(
    name="Process Dataset",
    source_dir=source_dir,
    script_name="process_data.py",
    arguments=["--input_dataset", dataset_param],
    compute_target="your-compute-target",  # Replace with your compute target
    runconfig=run_config,
    allow_reuse=True
)


# Create the pipeline
pipeline = Pipeline(workspace=ws, steps=[process_data_step])

# Validate and publish the pipeline
pipeline.validate()
published_pipeline = pipeline.publish(name="Dataset_Object_Pipeline", description="Pipeline that takes dataset object as parameter")
print(f"Published pipeline id: {published_pipeline.id}")


# Example to trigger the pipeline, no need to pass parameter here since the default value is used
pipeline_run = ws.experiments.get("your-experiment-name").submit(published_pipeline)


```

In this final example, we're not using a path but an actual `Dataset` object. The `PipelineParameter`'s `default_value` is assigned an existing dataset object using `Dataset.get_by_name()`, and the parameter `input_dataset` in our python script takes on the dataset id. We retrieve the dataset using `Dataset.get_by_id()`. This technique allows your pipeline to use a specific version of a dataset without having to track file paths.

To deepen your understanding, I highly recommend delving into the following resources:

*   The official Azure Machine Learning documentation, especially the sections on pipelines, datasets, and the Python SDK: This is your first point of call for clear instructions and examples from the source.
*  “Programming Azure Machine Learning” by Julien Simon:  This book provides detailed practical guides on using Azure ML, with great examples that can enhance your grasp of parameterization and complex pipeline scenarios.

Remember that the keys are using `PipelineParameter` for your metadata, retrieving the dataset from its path or id in your script using either `Dataset.get_by_path` or `Dataset.get_by_id` and utilizing `DataPath` for writing. With these techniques you should have more reliable and flexible Azure ML pipelines.
