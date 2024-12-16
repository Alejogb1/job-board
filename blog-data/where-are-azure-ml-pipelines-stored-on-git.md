---
title: "Where are Azure ML pipelines stored on Git?"
date: "2024-12-16"
id: "where-are-azure-ml-pipelines-stored-on-git"
---

Alright, let's tackle this one. It's a question that, honestly, I've seen trip up quite a few newcomers, and even some experienced folks who haven't fully grokked Azure Machine Learning's underlying architecture. The short answer is: Azure ML pipelines, in their *definitive form*, aren't directly stored as code files in a git repository like your standard Python modules. It’s more nuanced than that. What we're really storing, tracking, and versioning in Git are the *definitions* of those pipelines, often in the form of Python scripts or YAML files, along with any source code required to execute the pipeline's components. I’ve certainly been through the confusion myself, back when I was architecting a large-scale sentiment analysis pipeline and initially expected a single monolithic file to track everything.

Let’s break it down a bit further, and I’ll relate it to how I've handled this in past projects, along with illustrating working code examples. Think of an Azure ML pipeline as a directed acyclic graph, describing the sequence of steps needed to process data, train models, or deploy them. This graph itself resides within Azure Machine Learning's workspace environment. Your local git repository, on the other hand, is mainly where you keep the instructions for *constructing* this graph.

First, consider the most common scenario: defining your pipeline using the Azure ML SDK for Python. You're essentially crafting a Python script that creates your pipeline using the SDK's classes and functions. This script, along with any custom modules or data processing scripts required by your pipeline steps, is what you store in git. For example, you might have a file structure that looks something like this:

```
project_repo/
├── src/
│   ├── data_prep/
│   │   ├── data_cleaning.py
│   │   ├── feature_engineering.py
│   ├── training/
│   │   ├── train.py
│   │   ├── model_evaluation.py
│   ├── scoring/
│       ├── score.py
│   ├── pipeline_definition.py
│   ├── utils.py
├── config/
│   ├── azureml_workspace.json
│   ├── compute_target.json
└── .git/
```

The `pipeline_definition.py` file, for instance, would contain the code to assemble your Azure ML pipeline using classes like `azureml.pipeline.steps.PythonScriptStep` and `azureml.pipeline.core.Pipeline`. This script is committed to git, allowing you to version the *design* of your pipeline.

Here’s a highly simplified code snippet to illustrate:

```python
# pipeline_definition.py

from azureml.core import Workspace, Environment, Experiment
from azureml.core.compute import ComputeTarget
from azureml.pipeline.steps import PythonScriptStep
from azureml.pipeline.core import Pipeline
from azureml.core.runconfig import RunConfiguration
import os
import json

def create_pipeline(workspace_config, compute_target_name, experiment_name):
    ws = Workspace.from_config(path=".", auth=None, _file_name=workspace_config)
    compute_target = ComputeTarget(workspace=ws, name=compute_target_name)
    
    env = Environment.get(workspace=ws, name="AzureML-Minimal")
    run_config = RunConfiguration()
    run_config.environment = env

    # Step 1: Data Preparation
    data_prep_step = PythonScriptStep(
        name="Data Prep",
        source_directory="./src/data_prep",
        script_name="data_cleaning.py",
        compute_target=compute_target,
        runconfig=run_config,
        allow_reuse=True
    )
    # Step 2: Training
    training_step = PythonScriptStep(
        name="Model Training",
        source_directory="./src/training",
        script_name="train.py",
        compute_target=compute_target,
        runconfig=run_config,
        inputs=[data_prep_step],
        allow_reuse=True
    )
    
    pipeline = Pipeline(workspace=ws, steps=[data_prep_step, training_step])
    experiment = Experiment(workspace=ws, name=experiment_name)
    return pipeline, experiment

if __name__ == '__main__':
    with open('./config/azureml_workspace.json', 'r') as f:
        workspace_config = json.load(f)
    with open('./config/compute_target.json','r') as f:
        compute_target_config = json.load(f)

    pipeline, experiment = create_pipeline(workspace_config['config_file'], compute_target_config['name'], 'my-experiment')
    pipeline_run = experiment.submit(pipeline)
    pipeline_run.wait_for_completion()
```

Notice how the Python code defines the steps and their dependencies. This is the part that’s stored in Git, not a runtime representation of the pipeline. When this code runs (typically via a CI/CD pipeline triggered by a git commit), it *constructs* the pipeline within the specified Azure ML workspace.

Secondly, you might define pipelines using YAML definitions. The YAML files specify the pipeline steps, compute resources, and other configuration, similar to the Python SDK, just in a declarative format. These YAML files and accompanying scripts/code are then version controlled through git as well. Here’s a basic example in YAML:

```yaml
# pipeline.yml

name: "my-pipeline"
description: "Example ML Pipeline"
display_name: "my-display-pipeline"
jobs:
- job:
    type: command
    name: "Data Preparation"
    component:
        file: src/data_prep/data_cleaning.yaml
        code: "./src/data_prep"
    compute: "my-compute-target"
- job:
    type: command
    name: "Model Training"
    component:
        file: src/training/train.yaml
        code: "./src/training"
    compute: "my-compute-target"
    inputs:
      data_prep_step: ${{jobs.Data Preparation.outputs.output_data}}
```

This `pipeline.yml` file, and the `data_cleaning.yaml` and `train.yaml` files, are stored in git. They aren’t the pipeline *itself*, but rather the blueprint. When deployed, Azure ML reads these files and builds a corresponding pipeline. It is important to be aware that sometimes, users may be storing the *results* of pipeline runs in external git-compatible storages, such as DVC and Git-LFS, but the pipelines definitions are what's being tracked in this case.
Finally, consider the pipeline definition through the Azure ML Designer, a low-code drag-and-drop UI experience. Even here, while you build the pipeline visually, the underlying representation is a JSON structure and related source code that describes the pipeline’s configuration. This configuration *can* be exported and checked into git. While not as transparent as direct Python or YAML files, it’s still a set of instructions that define the pipeline’s structure. Here’s a very simplified example of the kind of JSON you might get when you export a designer pipeline:

```json
{
  "version": "1.0",
  "nodes": [
    {
      "id": "node1",
      "type": "PythonScript",
       "source_dir": "./src/data_prep",
       "script_name": "data_cleaning.py",
      "outputs": [
        {
          "name": "cleaned_data",
           "mode": "WriteToOutputPort"
        }
      ]
    },
    {
      "id": "node2",
       "type": "PythonScript",
       "source_dir": "./src/training",
       "script_name": "train.py",
      "inputs": [
        {
          "source": "node1",
          "output": "cleaned_data"
        }
      ]
    }
  ],
  "connections": [
    {
      "source": "node1",
      "target": "node2",
      "source_port": "cleaned_data"
    }
  ]
}
```

This JSON, or a more complex version of it, can be committed to a git repository and used to recreate the pipeline. It's the declaration of how to build the pipeline, not the pipeline in its executable state.

Essentially, the pipeline's 'code' exists as instructions in git - whether python, yaml or json - while the actual executable pipeline with run history and specific runtime settings exist within the Azure Machine Learning workspace.

To delve deeper, I'd recommend reviewing the Azure Machine Learning SDK documentation, especially the sections on pipelines. The official Azure documentation is pretty comprehensive, and if you're serious about this, I suggest looking at "Designing Machine Learning Systems" by Chip Huyen, which provides an excellent architectural view on deploying machine learning models. Also, for specific guidance on pipeline design patterns, the paper "Machine Learning Pipelines: Challenges and Pitfalls" is insightful. You won’t find the definitive answer as a file in Git, but rather the ability to *recreate* and version pipelines from what *is* stored in Git. It’s a conceptual difference, but one that's crucial for effectively managing Azure ML projects at scale.
