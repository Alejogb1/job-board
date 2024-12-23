---
title: "How do I save Azure ML pipelines in GIT or backup somewhere?"
date: "2024-12-23"
id: "how-do-i-save-azure-ml-pipelines-in-git-or-backup-somewhere"
---

Alright, let's tackle this. Having wrestled (oops, sorry, slipped there for a second) with pipeline versioning and backup strategies in Azure Machine Learning more times than I care to count, I can certainly offer some insight. It’s a crucial aspect often overlooked in the initial excitement of model development, and neglecting it can lead to considerable headaches down the line. Think of it like this: your pipelines are more than just ephemeral scripts; they're the core recipes for your machine learning models, and keeping them safe and versioned is non-negotiable.

The core challenge stems from the fact that Azure Machine Learning pipelines, as you define them through the SDK or portal, are metadata descriptions of compute workflows rather than actual code files. This distinction is critical. What you're building isn't inherently a static piece of text to be directly stored in git. Instead, you are constructing a sequence of operations, defining dependencies, and specifying compute resources within Azure's infrastructure. So, "saving" them isn't about grabbing a single file; it’s about capturing that entire configuration.

My experience comes from a project involving time-series forecasting, where pipeline integrity was paramount for compliance reasons. We needed to reconstruct exact model training procedures months later, including data processing, feature engineering, and model selection. What we initially attempted – manually logging parameters and script locations – rapidly became a maintenance nightmare. We quickly learned the value of robust, automated methods.

The primary approach for capturing these pipelines involves two distinct but complementary techniques: storing the pipeline *definition* and preserving the *supporting code*.

The pipeline *definition* is basically the configuration that constructs the pipeline in Azure ML, which can be captured through the Azure Machine Learning SDK for Python. I've found the SDK to be exceptionally reliable here. You're essentially serializing the pipeline object into something you can store and reconstitute. The most effective method involves capturing a JSON representation of the pipeline. Here's a simplified example of how you might accomplish this:

```python
import json
from azureml.core import Workspace
from azureml.pipeline.core import Pipeline

# Assuming you have a Workspace object named 'ws' and a Pipeline object named 'pipeline'
def save_pipeline_definition(pipeline, filename="pipeline_definition.json"):
    pipeline_dict = pipeline.serialize()
    with open(filename, "w") as outfile:
        json.dump(pipeline_dict, outfile, indent=4)

def load_pipeline_definition(filename="pipeline_definition.json"):
    with open(filename, "r") as infile:
        pipeline_dict = json.load(infile)
    return Pipeline.deserialize(pipeline_dict)

# Example usage
# save_pipeline_definition(pipeline)
# restored_pipeline = load_pipeline_definition()
```

This code snippet shows how you can serialize and deserialize a pipeline. The `save_pipeline_definition` function takes a pipeline object and serializes it to a JSON file. The `load_pipeline_definition` takes this JSON and reconstructs the pipeline. We stored this JSON file along with the Python scripts required for individual pipeline steps in the git repository. This is key – the definition itself doesn’t include the code executed in each step, just references to those scripts.

Moving on, the second key element is managing your *supporting code*. This is where git really shines. All Python scripts, configuration files, and any other necessary code used in the pipeline should be version controlled in git. Moreover, you *must* ensure that the pipeline definition references specific commits of these code repositories. The pipeline definition should not be pointing to, say, the "latest" version of your code but to a particular commit hash. This offers reproducibility. Without this, reconstructing a past pipeline with a precise version is impossible.

Here’s how to integrate code version control using Azure Machine Learning's `ScriptStep`. This example assumes you're running a Python script:

```python
from azureml.core import Workspace
from azureml.core.compute import ComputeTarget
from azureml.core.environment import Environment
from azureml.pipeline.steps import PythonScriptStep
from azureml.pipeline.core import Pipeline, PipelineData
from azureml.core.runconfig import RunConfiguration
from azureml.core.runconfig import CondaDependencies
from azureml.core.datastore import Datastore

# assuming workspace object 'ws', compute target 'compute_target' and default datastore 'default_ds'

def create_script_step(compute_target, source_directory, script_name, entry_point, params, pip_packages=None, conda_packages=None):
    env = Environment(name="pipeline_env")
    if pip_packages is not None:
       pip_dependencies = pip_packages
       env.python.conda_dependencies = CondaDependencies.create(
           pip_packages=pip_dependencies
       )
    if conda_packages is not None:
        conda_dependencies = conda_packages
        env.python.conda_dependencies = CondaDependencies.create(
            conda_packages = conda_dependencies
       )

    run_config = RunConfiguration()
    run_config.environment = env

    step = PythonScriptStep(
        source_directory=source_directory,
        script_name=script_name,
        arguments=[f"--{key}", value for key, value in params.items()],
        compute_target=compute_target,
        runconfig=run_config,
        name=entry_point,
        inputs=[],
        outputs=[]
       )
    return step


#Example usage
my_params = {'input_data':'path_to_data',
              'output_data':'output_dir',
              'other_param':10}

my_step = create_script_step(compute_target="my-compute-target",
                             source_directory="src_dir/",
                             script_name="train_model.py",
                             entry_point="model_training",
                             params=my_params,
                             pip_packages=["scikit-learn", "pandas", "numpy"])

# pipeline = Pipeline(workspace=ws, steps=[my_step])
```

This snippet creates a Python script step, with configurable parameters for different script versions. Your `train_model.py` script and related files are located in the `src_dir` folder which should be under git control and committed before the pipeline definition is generated. The `source_directory` parameter of the `PythonScriptStep` is critical; this is the path to the folder with your scripts. You would version this `src_dir` alongside the JSON pipeline definition.

Thirdly, and this is a crucial element for reproducibility, make sure your pipeline scripts explicitly record all versions of packages or dependencies. Avoid relying on global or default environments. Ensure that each step within your pipeline operates within an isolated and well-defined environment. By including the conda or pip package lists within the environment definition (as shown in `create_script_step`) you make sure the same set of libraries are used every single time. You must also save these environments to git as well if necessary. This way, we eliminate ambiguity about why the pipeline behaved one way today and another way tomorrow.

```python
# example saving and loading environment

def save_environment(env, file_path="env.json"):
    env_dict = env.serialize_to_dictionary()
    with open(file_path, "w") as outfile:
        json.dump(env_dict, outfile, indent=4)

def load_environment(file_path="env.json"):
    with open(file_path, 'r') as f:
       env_dict = json.load(f)
    return Environment.deserialize(env_dict)

# Example Usage
# env = create_environment(...)
# save_environment(env)
# restored_env = load_environment()
```

This final snippet shows how to serialize and deserialize the environment associated with the script step above. All of these files (the `pipeline_definition.json`, the code in `src_dir/` and the environment `env.json`), should all be committed together into git for proper versioning. If a version of the pipeline needs to be restored, make sure to checkout the same commit of git for all three, including code, environment and pipeline defintion.

It might feel like overkill at first, but you'll thank yourself later. For further exploration, I strongly suggest looking into the documentation on `azureml.pipeline.core` and `azureml.pipeline.steps` within the Azure Machine Learning SDK. The official Azure Machine Learning documentation is quite good, but for a deeper theoretical understanding of reproducible research, I recommend researching principles of version control and build automation techniques discussed in "The Pragmatic Programmer" by Andrew Hunt and David Thomas. Additionally, "Continuous Delivery" by Jez Humble and David Farley will provide valuable insight on how to build and deploy data science pipelines.

In summary, while Azure ML pipelines aren't directly git-compatible as single files, a combination of serializing the pipeline definition, strict version control of all associated code, and diligent environment specification provide a reliable, auditable, and reproducible way to preserve your work. This approach requires a little upfront planning, but it prevents countless hours of troubleshooting and offers peace of mind.
