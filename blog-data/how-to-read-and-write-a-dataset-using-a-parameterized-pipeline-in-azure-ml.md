---
title: "How to read and write a dataset using a parameterized pipeline in Azure ML?"
date: "2024-12-23"
id: "how-to-read-and-write-a-dataset-using-a-parameterized-pipeline-in-azure-ml"
---

Alright, let's unpack this. I've spent a considerable amount of time working with Azure Machine Learning pipelines, and the question of parameterized dataset handling is something I’ve often encountered. It's crucial for repeatability, adaptability, and efficient model development. Getting it *just* right saves a ton of headaches down the road. We’re talking about the ability to not just execute the same pipeline multiple times, but to execute it *with variations* in input data, while ensuring everything runs smoothly.

The core idea revolves around treating your datasets like configurable entities. Instead of hardcoding paths or dataset references into your pipeline steps, you define them as parameters. This allows you to specify at runtime, which dataset you want each step to operate on. This approach lends itself nicely to experimentation, allows for easy deployment to different environments, and makes it simpler to manage large volumes of diverse data. I remember a project where we initially hardcoded data paths, and the refactoring process to parameterize was...let's just say a learning experience. We learned the importance of thinking about data as a dynamic resource early on.

To achieve this in Azure ML, we leverage pipeline parameters, specifically referencing data through `InputPortBinding`, and the crucial use of `Dataset` objects. The `Dataset` object in Azure ML provides an abstraction over your data, regardless of where it's stored (Azure Blob Storage, Data Lake, local file system, etc.). This abstraction is key.

Let me break this down further with some practical examples, showing how you can both read and write datasets in a parameterized way.

**Example 1: Reading a Parameterized Dataset into a Python Script**

Let's start with the simplest scenario: a Python script within an Azure ML pipeline that needs to access a dataset. We’ll define a pipeline parameter of type `Dataset` that represents our input.

```python
from azureml.core import Dataset, Run
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dataset", type=str, help="The path to the input dataset")
    args = parser.parse_args()

    run = Run.get_context()

    # The input_dataset arg contains the actual dataset id as passed by the pipeline definition.
    # We need to obtain the corresponding Dataset object here
    input_ds = run.input_datasets["input_dataset"] # This will have the Dataset object

    print(f"Dataset name: {input_ds.name}")
    print(f"Dataset type: {input_ds.dataset_type}")
    print(f"Dataset description: {input_ds.description}")

    # This is where your actual processing of the dataset would begin.
    # Let's assume you're accessing a tabular dataset, for example:
    df = input_ds.to_pandas_dataframe()

    print(f"Shape of loaded dataset: {df.shape}")

    # Save some sample info for logging.
    first_few_rows = df.head().to_string()
    run.log("Sample data", first_few_rows)


if __name__ == "__main__":
    main()

```

*   **`argparse`**: The `argparse` module allows us to receive parameters passed in through the command line, but note that Azure ML automatically passes the input dataset as an argument under the label we've named it.
*   **`Run.get_context()`**: This function provides runtime information about the current execution of the script in Azure ML. Crucially, it allows access to input datasets, which are passed to the `Run` context, rather than simply as command line arguments.
*   **`run.input_datasets["input_dataset"]`**: Accesses the `Dataset` object based on the name assigned to the `InputPortBinding` when defining the pipeline.
*   **`input_ds.to_pandas_dataframe()`**: This method loads a tabular dataset into a Pandas DataFrame, facilitating data processing within the script.  If your dataset was different (e.g., a file dataset), you would adapt this step to your specific data type.
*    This example loads a dataset and logs some simple information. In practice, you would be doing much more complex data transformation or analysis here, but the key is how you gain access to the dataset.

**Example 2: Writing a Parameterized Dataset from a Python Script**

Now, let's consider writing a resulting dataset from our pipeline step. This works in a similar vein but requires a bit more setup on the pipeline definition side.

```python
from azureml.core import Dataset, Run
import argparse
import os
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dataset", type=str, help="The path to the output dataset")
    args = parser.parse_args()

    run = Run.get_context()

    # Create a small sample DataFrame
    data = {'col1': [1, 2, 3], 'col2': [4, 5, 6]}
    df = pd.DataFrame(data)

    # Ensure the output directory exists
    os.makedirs(args.output_dataset, exist_ok=True)

    # Save the DataFrame as a csv
    output_file_path = os.path.join(args.output_dataset, "output.csv")
    df.to_csv(output_file_path, index=False)


    # Register a dataset in the workspace. This step is optional, you can skip it if you don't
    # need to version control your output data or use it in further steps.
    # output_ds = Dataset.File.from_files(path=args.output_dataset)
    # output_ds.register(workspace=run.experiment.workspace, name="my_output_dataset", create_new_version=True)

if __name__ == "__main__":
    main()
```

*   **`parser.add_argument("--output_dataset", ...)`**: Similar to the input case, this argument receives the output directory as a string value. Note that we’re not directly receiving the output dataset object here, but rather a location where we can store the output.
*   **`os.makedirs(args.output_dataset, exist_ok=True)`**: This line creates the directory if it does not exist. AzureML ensures that the location represented by the output dataset is accessible to the computation.
*   **`df.to_csv(output_file_path, index=False)`**: Saves our data into a CSV file in the location specified by the argument.
*   **Registration (Optional)**: I have included commented-out code that shows how to register the output folder as a new dataset within your Azure ML workspace. This part is optional but helpful if you need to access this output dataset in further pipeline steps or other experiments. You can easily version the datasets if you register the data at each pipeline run.

**Example 3: Defining the Pipeline with Parameterized Datasets**

Now, the crucial piece is defining how we parameterize and use those datasets in the actual Azure ML pipeline definition. The Python code below provides an example using `azureml.pipeline.steps` which sets up the framework to pass the correct arguments to the script.

```python
from azureml.core import Workspace, Environment, Dataset
from azureml.pipeline.steps import PythonScriptStep
from azureml.pipeline.core import Pipeline, PipelineParameter, PipelineData
from azureml.core.runconfig import RunConfiguration

# Load workspace
ws = Workspace.from_config()

# Get an environment - substitute with your environment
env = Environment.get(workspace=ws, name="AzureML-Minimal")

# Define dataset parameters. You would need to provide the actual registered dataset IDs or paths here.
input_dataset_param = PipelineParameter(name="input_dataset_param",
                                       default_value = Dataset.get_by_name(ws, name='your_input_dataset_name'))
output_dataset_param = PipelineParameter(name="output_dataset_param",
                                       default_value = PipelineData(name='output_dataset', datastore=ws.get_default_datastore()).as_directory())


# Create a RunConfiguration and add the dataset as a environment variable
run_config = RunConfiguration()
run_config.environment = env


# Define the input step
input_step = PythonScriptStep(
    name="input_step",
    source_directory=".",
    script_name="input_script.py", # Example 1 script above.
    arguments=["--input_dataset", input_dataset_param],
    inputs = [input_dataset_param],
    runconfig=run_config,
    compute_target="your_compute_target",
    allow_reuse=True
)

# Define the output step
output_step = PythonScriptStep(
    name="output_step",
    source_directory=".",
    script_name="output_script.py", # Example 2 script above.
    arguments=["--output_dataset", output_dataset_param],
    outputs = [output_dataset_param],
    compute_target="your_compute_target",
    runconfig=run_config,
    allow_reuse=True
)


# Create pipeline
pipeline = Pipeline(workspace=ws, steps=[input_step, output_step])

# Submit the pipeline.
pipeline_run = pipeline.submit(experiment_name = "param_pipeline_demo")

pipeline_run.wait_for_completion(show_output=True)

```

*   **`PipelineParameter`**: This class is fundamental for creating pipeline parameters. In this example, we have one for the input dataset, and one to manage the output dataset. Notice we set `default_value` in our parameters here, which can make testing easier, but you can also provide values when submitting the pipeline.
*   **`PipelineData`**: This is how we represent an intermediate data output between steps. In the code, the output data location is made accessible to the computation environment, so we can write our output there.
*  **`PythonScriptStep`**: These steps encapsulate the Python scripts we defined earlier, linking our parameters into the arguments and inputs/outputs of each step respectively. We use `arguments` for passing the locations to the script and `inputs`/`outputs` to ensure the data dependencies between steps are properly represented in the pipeline.

**Key Takeaways**

The core strategy is to use `Dataset` objects and `PipelineParameter` to avoid hardcoding specific data locations. By using the `Run` context within each of the python scripts in the pipeline, you can access the dataset object, even with the passed in parameter. This is paramount for building reproducible and flexible workflows, and it allows you to switch out datasets dynamically without altering the core logic of your scripts.

For a deeper dive into the specifics, I’d recommend looking into the Azure Machine Learning SDK documentation, particularly the sections on `azureml.core.Dataset`, `azureml.pipeline.core.PipelineParameter`, and `azureml.pipeline.steps.PythonScriptStep`. A good starting point would be the official Microsoft documentation on Azure Machine Learning Pipelines. A well-structured tutorial on the Azure ML platform itself, can provide a step by step walkthrough of creating and utilizing pipelines. Lastly, consider diving into *Designing Data-Intensive Applications* by Martin Kleppmann, not directly tied to Azure, but it is an excellent resource to understand the broader context of data handling in complex systems.

Remember, good practice involves carefully considering data lineage and versioning, especially when dealing with multiple data sources. Parameterized datasets within Azure ML pipelines provide the foundation for achieving just that. It’s an investment that pays off handsomely as your projects become more intricate.
