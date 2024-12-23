---
title: "Where do I save Azure ML pipelines on GIT?"
date: "2024-12-23"
id: "where-do-i-save-azure-ml-pipelines-on-git"
---

, so you're facing the classic version control quandary with Azure Machine Learning pipelines. I've definitely been down that road, and let me tell you, there’s a bit more to it than just “put it in git”. My experience, especially back in the days of dealing with multi-team projects heavily leveraging Azure ML, has hammered home the importance of a structured approach. The quick answer is, yes, use Git, but where and how you organize your repo really impacts maintainability, collaboration, and ultimately, your deployment success.

The problem isn't really *where* you save them - Git handles any kind of file - but rather *how* you structure and manage the different aspects that make up an Azure ML pipeline. A typical pipeline doesn’t just consist of python code; it encompasses configuration files, script dependencies, environment definitions, and potentially even packaged data artifacts. Putting it all in one giant directory is a recipe for chaos.

The core issue centers on separation of concerns. We want to treat our pipeline definitions, which are typically json or yaml, and our source code as separate entities that are related but should be versioned and managed independently. Think about it: your core ML algorithms might undergo frequent changes, while the fundamental structure of your pipeline might remain relatively consistent. Mixing these can muddy your version history and make debugging a nightmare.

The recommended approach, and one I've seen work successfully time and again, involves structuring your Git repository with a dedicated directory for the Azure ML pipelines and then sub-structuring within that directory to further organize pipeline definitions, code, and environments. Let’s break it down.

First, I recommend that within your git repo you establish at least these three top-level directories: `/src`, `/pipelines`, and `/envs`.

The `/src` directory houses your actual code; things like your training script, preprocessing scripts, evaluation scripts, and any helper functions. This isn't exclusive to Azure ML; any general utility or machine learning code should reside here, structured as you see fit within best practices for your language. This can be python packages, organized folders, whatever makes sense for your specific project.

The `/pipelines` directory is specifically for your Azure ML pipeline definitions. This will contain files that define the individual steps within a pipeline. This includes .json or .yaml representations of your Azure ML pipeline, sometimes along with any data configuration files that accompany them. You should structure the /pipelines directory with a folder structure that reflects the logical organization of your pipelines.

Finally, the `/envs` directory is where your environment definitions live. This will hold any .yml files defining your conda environments, any dockerfile needed to build containers, or pointers to Azure ML environments. This ensures your pipelines are always run within repeatable environments, and ensures you can reproduce your runs.

Now, for some code examples. Let's imagine we’re using Python and the Azure ML SDK, and are trying to setup a simple pipeline that trains a model.

**Example 1: Basic Directory Structure**

Here is the basic folder structure we are talking about.

```
my-ml-project/
├── .git/
├── src/
│   ├── train/
│   │   └── train.py
│   └── preprocess/
│        └── data_prep.py
├── pipelines/
│   ├── training_pipeline/
│   │   └── pipeline.yaml
├── envs/
│    └── training_environment.yml
├── README.md
```

This shows a clear separation: core logic resides within `/src`, pipeline structure is in `/pipelines`, and environment definitions are in `/envs`.

**Example 2: A Simplified Pipeline YAML**

Now, let's take a look at a simplistic `/pipelines/training_pipeline/pipeline.yaml`:

```yaml
$schema: https://azuremlschemas.azureedge.net/latest/pipelineJob.schema.json
type: pipeline
display_name: TrainingPipeline
description: A simple training pipeline
jobs:
  train_job:
    type: command
    code: ../../src/train # relative reference to your code
    command: "python train/train.py --data_path ${{inputs.training_data}}"
    environment: azureml:training_environment@latest # reference to a versioned env
    inputs:
      training_data:
          type: uri_folder
          path: <your data path or datastore reference>
    outputs:
      model_output:
          type: uri_folder
    compute: azureml:<your compute name>
  data_prep_job:
    type: command
    code: ../../src/preprocess
    command: "python data_prep.py --data_path ${{inputs.raw_data}} --output_path ${{outputs.preprocessed_data}}"
    environment: azureml:training_environment@latest
    inputs:
        raw_data:
            type: uri_folder
            path: <your data path or datastore reference>
    outputs:
        preprocessed_data:
            type: uri_folder
```

Notice that the paths are relative, the `code` property points to a relative location of your code in `/src`. We refer to a registered environment in azure, though we can define that in our /envs directory. You would create such an environment with the Azure CLI, or by uploading an environment file and then referencing it in your pipeline using azureml:<environment name>@<environment version>. Having a specific version of the environment helps with reproducibility as well.

**Example 3: Environment Definition (training_environment.yml)**

Lastly, lets consider an example of an environment definition, that would be stored in `/envs/training_environment.yml`

```yaml
name: training_environment
dependencies:
  python:
    - scikit-learn
    - pandas
    - numpy
    - azureml-mlflow
  pip:
    - azureml-sdk
channels:
    - conda-forge
```

This file, along with a dockerfile if you need that level of customizability, defines all of the libraries that are required for the python program you would run from within the `/src` folder. You can then use this yml file to create an environment in Azure ML and reference it in your pipeline.

This structured approach gives us multiple advantages. First, it is easier to reason about our system. We have a distinct place for our code, our pipeline, and our environments. Second, this structure allows you to modify your code without modifying your pipelines as long as the interface is still the same. Likewise, you can update your pipeline and your environment without modifying your code. This reduces code churn, especially if you are running many experiments. Finally, it is also more git-friendly, allowing you to more easily manage versions and changes to components.

In terms of best practices, it's critical that you version control *everything* you require to reproduce your pipeline. This means not only the pipeline definitions and code but also the environment definitions and potentially even your packaged data artifacts if you are using versioned data (this is beyond the scope of this response but is crucial for true reproducibility). Remember that data is the most crucial part of most machine learning processes, so be sure to include that in your considerations for version control and experiment tracking.

For further study, I strongly recommend digging into "Designing Data-Intensive Applications" by Martin Kleppmann for foundational concepts on data and infrastructure management and “Software Engineering at Google” by Titus Winters, Tom Manshreck, and Hyrum Wright for practical software engineering advice at scale, much of which is relevant to ML projects. Within the ML domain, the paper “Hidden Technical Debt in Machine Learning Systems” by Sculley, et al. (Google) is a critical read that will give you a good overview of the real-world difficulties of managing large-scale ML systems and how to avoid them.

In essence, the key takeaway is to treat your Azure ML pipeline as a structured software project, not just a collection of scripts. By adopting a clear and organized approach in your git repository, you'll drastically improve the maintainability, scalability, and reproducibility of your machine learning deployments. It has worked for me in multiple contexts, and I'm confident it will serve you well.
