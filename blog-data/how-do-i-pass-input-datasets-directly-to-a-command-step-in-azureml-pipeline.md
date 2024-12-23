---
title: "How do I pass input datasets directly to a command step in AzureML pipeline?"
date: "2024-12-23"
id: "how-do-i-pass-input-datasets-directly-to-a-command-step-in-azureml-pipeline"
---

Okay, let's talk about getting datasets directly into command steps within Azure Machine Learning pipelines. I've encountered this specific challenge more times than I can count, often with data preprocessing pipelines that needed to scale efficiently. It’s a frequent hurdle for anyone moving beyond basic notebook experimentation and into more production-ready scenarios.

The core issue is ensuring that your pipeline steps have the necessary data context – in other words, knowing where to find and how to consume the datasets you intend them to process. Azure ML offers a few different mechanisms, and the one most pertinent here is using input bindings for your command steps. Think of these as pointers, carefully constructed to allow your step to access data securely and without explicit file paths within your scripts.

Before we dive into the specifics, it's worth noting that the best way to manage data within AzureML is through registered datasets. These act as version-controlled, centrally manageable data resources, which simplifies your workflows and increases reproducibility. If you’re not using registered datasets, I highly recommend you consider transitioning to them. They’ll ultimately save you a lot of headaches as your project grows.

Now, regarding command steps and their input data consumption. There are a few ways to pass these datasets.

*   **Using `input` Property in `Command` or `CommandComponent`**: This is arguably the most direct and recommended approach. You specify input bindings as part of the command definition, and Azure ML handles the data routing and accessibility. The command step itself receives these inputs as environmental variables, typically named after the input id. The input path isn't hardcoded into the script, so you can switch data sources easily in the future.

*   **Direct Data Loading Within the Script:** While technically possible, it is less elegant and not as scalable as the first option. You could construct a script that loads data directly by reading it from some mounted storage. This option removes the version control element and leads to brittle pipelines.

I will primarily focus on the first approach, as I have found it leads to more maintainable and robust pipelines. Let me illustrate with a simple example. Let’s say we have a dataset named `raw_training_data` that we wish to pass to a python script to filter the data. First, we register our dataset, and then we define the command step.

**Example 1: Basic Input Binding**

Let’s say our python script (filter_data.py) looks like this.

```python
import pandas as pd
import os
import argparse

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--input_data', type=str, help='Path to the input data file')
  parser.add_argument('--output_data', type=str, help='Path to the output data directory')
  args = parser.parse_args()
  
  input_path = os.path.join(args.input_data, 'data.csv')
  output_path = os.path.join(args.output_data, 'filtered_data.csv')
  
  df = pd.read_csv(input_path)
  filtered_df = df[df['some_column'] > 5]  # Some filter logic
  filtered_df.to_csv(output_path, index=False)
  print(f'Filtered data saved to {output_path}')
```

Here's how you might define your command step in an AzureML pipeline script:

```python
from azure.ai.ml import command
from azure.ai.ml import Input
from azure.ai.ml import Output

# Assuming 'raw_training_data' is a registered dataset already
# We use an AzureML dataset object
raw_training_data_input = Input(type='uri_folder', path='azureml:raw_training_data:1') # replace version with your data's version number

data_filtering_step = command(
    name="data_filtering",
    inputs={"input_data": raw_training_data_input},
    outputs={"output_data": Output(type="uri_folder")},
    command="python filter_data.py --input_data ${{inputs.input_data}} --output_data ${{outputs.output_data}}",
    code="./src/",  # Assuming filter_data.py is in the 'src' folder
    environment="azureml:AzureML-Minimal:1",  # A lightweight environment
)
```

Key points in the above example:

1.  We are creating `Input` object pointing to our registered dataset. We define the type as `uri_folder` because Azure ML exposes it as a mounted directory.
2.  The `command` argument describes how the python script is executed.
3.  The `input` key maps our input object to the `--input_data` parameter used in our python script.
4.  We use `${{inputs.input_data}}` to reference the mounted location of our data within the command. Azure ML manages the resolution of this reference for us.
5. The output parameter is configured using `Output`. Note that our script will write the filtered data within the folder assigned by the service at runtime.

**Example 2: Handling Multiple Inputs**

Often, you will encounter situations requiring more than one input dataset. The mechanism remains similar – you simply define multiple inputs in your command step. Consider you have another dataset called `reference_data` you need in the same step. Let’s extend our python script.

```python
import pandas as pd
import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_data', type=str, help='Path to the main data file')
    parser.add_argument('--reference_data', type=str, help='Path to the reference data file')
    parser.add_argument('--output_data', type=str, help='Path to the output data directory')
    args = parser.parse_args()

    input_path = os.path.join(args.input_data, 'data.csv')
    reference_path = os.path.join(args.reference_data, 'reference.csv')
    output_path = os.path.join(args.output_data, 'processed_data.csv')

    df_main = pd.read_csv(input_path)
    df_reference = pd.read_csv(reference_path)

    # Some processing using both dataframes
    merged_df = pd.merge(df_main, df_reference, on='common_column')
    merged_df.to_csv(output_path, index=False)
    print(f'Processed data saved to {output_path}')
```

Now, you'd modify your Azure ML pipeline step to include two inputs:

```python
from azure.ai.ml import command
from azure.ai.ml import Input
from azure.ai.ml import Output


# Assuming 'reference_data' is another registered dataset with a version number
reference_data_input = Input(type='uri_folder', path='azureml:reference_data:2') # replace with version number

data_processing_step = command(
    name="data_processing",
    inputs={"input_data": raw_training_data_input, "reference_data": reference_data_input},
    outputs={"output_data": Output(type="uri_folder")},
    command="python process_data.py --input_data ${{inputs.input_data}} --reference_data ${{inputs.reference_data}} --output_data ${{outputs.output_data}}",
    code="./src/",
    environment="azureml:AzureML-Minimal:1",
)
```

Notice how we are now mapping two `Input` objects to our python script. The important part to take away from this, is that the script itself does not contain any specific information about where the data files are stored, and neither does the pipeline step. The path to these data files are exposed as environment variables during runtime.

**Example 3: Using Download mode**

By default, the `input` parameter is configured to mount the data at runtime. There are cases where you do not want the data to be mounted, and would rather copy it to local storage prior to execution. This can be configured using the `mode` property. Let's modify the previous example to show this:

```python
from azure.ai.ml import command
from azure.ai.ml import Input
from azure.ai.ml import Output

# Assuming 'raw_training_data' is a registered dataset already
raw_training_data_input = Input(type='uri_folder', path='azureml:raw_training_data:1', mode='download')

data_filtering_step = command(
    name="data_filtering",
    inputs={"input_data": raw_training_data_input},
    outputs={"output_data": Output(type="uri_folder")},
    command="python filter_data.py --input_data ${{inputs.input_data}} --output_data ${{outputs.output_data}}",
    code="./src/",  # Assuming filter_data.py is in the 'src' folder
    environment="azureml:AzureML-Minimal:1",  # A lightweight environment
)
```

In this example, the data will be copied to a local storage prior to the execution of the python script. In the `filter_data.py` file, the path used in `input_path = os.path.join(args.input_data, 'data.csv')` would reflect the local path.

**Further Learning:**

For a deeper dive, I highly recommend exploring the following:

*   **Azure Machine Learning SDK documentation:** The official documentation provides the most comprehensive explanation, particularly the sections on pipelines, command components, and data handling. Start with the high-level pipeline guides and then dive into the details of the SDK.
*  **"Designing Data-Intensive Applications" by Martin Kleppmann:** While not specific to Azure ML, this book provides foundational knowledge of data management, which is essential when building robust data pipelines. It includes best practices for versioning and data accessibility.
*   **"Data Pipelines Pocket Reference" by James Densmore:** This book offers a very practical approach to designing and implementing data pipelines. It provides guidance applicable to various platforms, including but not limited to Azure ML.

In summary, using the `input` property within your `command` or `commandcomponent` definitions is the standard, efficient, and maintainable way to handle data input for your command steps in Azure ML pipelines. It decouples your scripts from hardcoded file paths and gives you the flexibility to swap datasets easily while making good use of Azure ML’s data management features. Remember to utilize registered datasets and choose between download or mount based on your specific needs. If you follow these guidelines, you’ll build considerably more robust and reproducible data pipelines.
