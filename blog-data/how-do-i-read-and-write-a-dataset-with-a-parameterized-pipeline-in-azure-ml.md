---
title: "How do I read and write a dataset with a parameterized pipeline in Azure ML?"
date: "2024-12-23"
id: "how-do-i-read-and-write-a-dataset-with-a-parameterized-pipeline-in-azure-ml"
---

Okay, let's tackle this one. I recall a particularly challenging project a few years back, working with a client in the geospatial analysis sector. They were ingesting massive amounts of satellite imagery and processing it through a very specific sequence of algorithms. The core issue was, as you might guess, managing that data flow effectively using parameterized pipelines in Azure Machine Learning. So, I’ve definitely been there. It’s not just a matter of point-and-click; you need a structured approach, especially when dealing with parameters that change based on the dataset or processing stage.

The fundamental concept is to create a pipeline in Azure ML that isn't hardcoded for a single dataset, but instead accepts parameters at runtime, allowing it to operate on various inputs and produce corresponding outputs. This involves defining your pipeline steps, making sure they can consume data passed as inputs, and structuring your code to accommodate these parameters. It’s a modular, scalable approach that's pretty standard practice in mature ML engineering workflows.

Firstly, let's talk about *reading* the dataset. Azure ML provides several ways to ingest data. Often, you’ll find yourself dealing with datasets registered within your workspace. These are essentially pointers to storage locations, making it easier to reference them across your pipeline. When reading, you're not actually moving the data itself into the compute target, you're configuring the compute environment to access the data where it is stored.

Here’s a simplified code snippet illustrating how to pass a data input parameter to an Azure ML pipeline step using the SDK:

```python
from azureml.core import Workspace, Dataset
from azureml.pipeline.steps import PythonScriptStep
from azureml.pipeline.core import Pipeline, PipelineParameter

# Assume workspace and compute target are already defined

# Retrieve the named registered dataset
ws = Workspace.from_config()
dataset = Dataset.get_by_name(ws, name="my_registered_dataset")

# Define a pipeline parameter for the data input
data_input_param = PipelineParameter(name="data_input", default_value=dataset)

# Define a script step
script_step = PythonScriptStep(
    name="data_reader_step",
    script_name="data_reader.py",
    arguments=["--data-input", data_input_param],
    inputs=[data_input_param],
    compute_target=compute_target,
    source_directory='.',
)

# Create pipeline
pipeline = Pipeline(workspace=ws, steps=[script_step])

# Submit pipeline (code to submit not included for brevity)
```

In this snippet, `data_input_param` is a pipeline parameter that defaults to a dataset already registered in Azure ML. The `PythonScriptStep` will then receive this parameter, allowing our script ( `data_reader.py` ) to load and process this dataset. Importantly, the dataset is passed as an *input*, not as an argument alone. This signals to the pipeline that the step depends on that data being available.

Now, on the `data_reader.py` script's side, you'd have code like this to access it:

```python
import argparse
from azureml.core import Dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-input", type=str, help="Data input")
    args = parser.parse_args()

    # Get the Azure ML dataset object from the input
    dataset = Dataset.get_by_id(workspace=None, id=args.data_input) #workspace will get picked up from environment

    # Dataset is now usable with .to_pandas_dataframe(), .to_spark_dataframe(), etc.
    df = dataset.to_pandas_dataframe()
    print(f"Shape of DataFrame read: {df.shape}")
    # Further processing can now occur

if __name__ == "__main__":
    main()
```
The script uses `argparse` to receive the dataset input, retrieve the dataset object, and do something with it (in this case, load it into a pandas DataFrame and print the shape). The key is to recognize that the input parameter arrives as a string representing the dataset's unique ID. We use `Dataset.get_by_id` to obtain the actual `Dataset` object.

Moving on to *writing* data, it’s very similar. Generally, your pipeline step will generate some output data that you need to register or write to storage. You can output your data to a folder, then register a new dataset pointing to it, or you can write data directly to a datastore. Using outputs within the pipeline helps manage dependencies between steps.

Here’s an example demonstrating how to parameterize an output path and write data to it:

```python
from azureml.core import Workspace, Datastore
from azureml.pipeline.steps import PythonScriptStep
from azureml.pipeline.core import Pipeline, PipelineParameter
from azureml.data import OutputFileDatasetConfig
from azureml.core.runconfig import RunConfiguration
from azureml.core.environment import Environment

#Assume workspace and compute target are already defined

ws = Workspace.from_config()
datastore = Datastore.get(ws, datastore_name="my_datastore")

# Define a pipeline parameter for the output folder
output_folder_param = PipelineParameter(name="output_folder", default_value="my_output_data")
output_dataset = OutputFileDatasetConfig(destination=(datastore, output_folder_param)).as_upload(name='processed_data')


# Define a script step
environment = Environment.from_conda_specification(name = "my_environment", file_path = "my_env.yml")
run_config = RunConfiguration()
run_config.environment = environment


script_step = PythonScriptStep(
    name="data_writer_step",
    script_name="data_writer.py",
    arguments=["--output-path", output_dataset],
    outputs=[output_dataset],
    compute_target=compute_target,
    runconfig=run_config,
    source_directory='.',
)

# Create pipeline
pipeline = Pipeline(workspace=ws, steps=[script_step])

# Submit pipeline (code to submit not included for brevity)
```

Here, we define an output parameter, `output_folder_param`, and then use it to create an `OutputFileDatasetConfig`. This ensures the script writes data to the correct location within the datastore. The output is designated as an *output* of the step. Note that the `as_upload()` call specifies we want the output to be uploaded to the datastore from the compute instance after script execution.

Correspondingly, here’s the `data_writer.py` script:

```python
import argparse
import pandas as pd
from azureml.core import Run
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-path", type=str, help="Output path")
    args = parser.parse_args()

    run = Run.get_context()
    output_path = args.output_path
    # Generate sample data
    data = {'col1': [1, 2], 'col2': [3, 4]}
    df = pd.DataFrame(data)
    
    # Ensure the directory exists, can be removed if not needed
    os.makedirs(output_path, exist_ok=True)

    # Construct the output file path
    file_path = os.path.join(output_path, "output_data.csv")

    # Write to the file. This writes to the blob store location referenced by the OutputDataset.
    df.to_csv(file_path, index=False)

    print(f"Data written to: {file_path}")
    # Add a tag to let azure ML know how it did
    run.tag('data-written','true')


if __name__ == "__main__":
    main()
```
This script receives the output path via `argparse` again. Importantly, unlike the input, `OutputFileDatasetConfig` will provide the actual path within the datastore on which we are meant to write the data. The data is created, and we write a simple CSV file. The key to remember is that the path we receive from `args.output_path` is meant to have data *written to* it. The output definition ensures that the content of that directory gets uploaded after the script finishes executing.

For further reading, I’d recommend the official Azure Machine Learning documentation, of course. Specifically, explore the sections related to datasets, pipeline steps, and pipeline parameters. Also, look into 'Machine Learning Engineering' by Andriy Burkov; it provides a broader context on building robust ML pipelines. The Microsoft Learn modules for Azure ML are excellent for hands-on practice as well. Finally, the 'Designing Data-Intensive Applications' by Martin Kleppmann can provide additional background knowledge on data management in large-scale systems.

The parameterization of these pipelines allows for adaptability, reusability, and better control of data flows within complex ML systems. It requires a shift from coding for specific datasets to creating flexible pipelines that can adapt to various inputs and outputs. This, in my experience, is a hallmark of mature ML engineering.
