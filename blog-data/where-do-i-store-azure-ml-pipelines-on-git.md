---
title: "Where do I store Azure ML pipelines on GIT?"
date: "2024-12-16"
id: "where-do-i-store-azure-ml-pipelines-on-git"
---

, let’s talk about where to store your Azure ml pipelines on git. It’s a topic that often comes up, and honestly, it’s not always as straightforward as a quick search might suggest. I've had my share of figuring this out in production environments, and believe me, there's a significant difference between a hello-world example and maintaining a complex mlops pipeline across a team. So let’s unpack this systematically.

The core issue isn't just *where* to store the files, but *how* to organize them for a manageable and scalable mlops workflow. The answer isn't a single directory but rather a structured repository with clear conventions. I learned this the hard way when I first joined a team migrating from individual data scientist notebooks to an automated system. Chaos ensued, mostly because of differing storage methods and versioning protocols.

First, understand that your git repository for Azure ML pipelines should ideally encompass *more* than just the pipeline definition itself. It needs to handle scripts for data preprocessing, model training, model evaluation, and deployment. Let’s break this down. You need a consistent structure to help with versioning, collaboration and code maintainability.

I’d recommend organizing your git repo into these main sections:

1. **`src/` or `code/`:** This directory houses all of your Python code related to data manipulation, model training, evaluation, and custom components used within your azure ml pipeline. We can further break down this folder like: `src/data/`, `src/models/` and `src/training/`, depending on the size and complexity of your project. Each module can be a sub-directory, providing a clearer organization.
2.  **`pipelines/`:** This is where the actual Azure ML pipeline definitions are stored. These are typically YAML files or Python scripts that define the sequence of steps, their dependencies, and compute configurations. Each unique pipeline should have its own file or set of files.
3.  **`config/` or `configs/`:** This directory houses configuration files needed for the pipeline, including environment variable settings, compute targets, datasets connections, and parameters. These configuration parameters should be versioned and tracked.
4.  **`tests/`:** A crucial, often-overlooked section. This houses unit and integration tests for your data processing scripts, custom components, and possibly the overall pipeline (e.g., testing that the pipeline executes).
5.  **`docs/`:** Important for team collaboration, especially with the long term in mind. This houses documentation related to the pipeline, any design decisions, dependencies, and how to operate it.

Now, let's talk about the *how* with code examples. Suppose we have a simple pipeline involving data preprocessing and model training.

**Example 1: Data processing script located within `src/data/preprocess.py`:**

```python
# src/data/preprocess.py
import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(filepath):
    """Loads data from a csv file."""
    df = pd.read_csv(filepath)
    return df

def preprocess(df):
   """Preprocesses the data:
      - Handles missing values
      - Splits data into features and target
   """
   df = df.dropna()
   X = df.drop('target_column', axis=1)
   y = df['target_column']
   return X,y

def split_data(X,y, test_size=0.2, random_state=42):
   """Splits data into training and testing sets."""
   X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=test_size, random_state=random_state)
   return X_train, X_test, y_train, y_test

if __name__ == '__main__':
  # Example of use.
  df = load_data('data.csv')
  X,y = preprocess(df)
  X_train, X_test, y_train, y_test = split_data(X,y)
  print(f"Shape of X_train: {X_train.shape}")
  print(f"Shape of X_test: {X_test.shape}")
  print("Preprocessing completed.")
```
This is a Python script residing in the `src/data/` directory, that can be used as a preprocessing step in the Azure ML pipeline.

**Example 2: The pipeline definition within `pipelines/train_pipeline.yaml`:**

```yaml
# pipelines/train_pipeline.yaml
$schema: https://azuremlschemas.azureedge.net/latest/pipeline.schema.json
type: pipeline
display_name: training_pipeline
jobs:
  preprocess_job:
    type: command
    component: azureml:/component/preprocess_component:1
    inputs:
      data_path:
        type: uri_folder
        path: azureml:my_input_data:1
    outputs:
      preprocessed_data:
        type: mltable
  train_job:
    type: command
    component: azureml:/component/train_component:1
    inputs:
      preprocessed_data: ${{parent.jobs.preprocess_job.outputs.preprocessed_data}}
    outputs:
      trained_model:
         type: uri_folder
```

This `pipelines/train_pipeline.yaml` file defines the entire pipeline composed of two jobs `preprocess_job` and `train_job`. Notice it references custom components (e.g., `preprocess_component`), which, by best practice, should be also tracked and versioned in a proper `components/` directory. These components might contain references to the `preprocess.py` or other python script, thus having a clear path. Note that `azureml:/component/preprocess_component:1` refers to the named reference to a registered azure ml component.

**Example 3: Azure ml run configuration within `config/training.json`:**

```json
{
    "compute_target": "my-compute-cluster",
    "environment": "my_conda_environment:1",
    "input_dataset_name":"my_input_data",
    "input_dataset_version": "1",
    "train_parameters":{
      "model_type": "LogisticRegression",
      "learning_rate": 0.01,
      "epochs": 100
    }
}
```
This is a json file that keeps configuration variables that are shared among all pipeline executions. For example, this file keeps the name and version of the registered environment, and the input datasets. You can access this configuration during runtime using environment variables.

Now, why this approach? It provides several advantages. Firstly, *reproducibility*. By having all the code, pipeline definitions, and configurations version-controlled, we can easily reproduce experiments or rollback changes. Secondly, *collaboration*. Different team members can easily work on separate parts of the pipeline, knowing where to store their changes and how these are related to others. Thirdly, *scalability*. As your pipelines become more complex, a well-defined structure ensures your codebase is easy to maintain and understand.

For further reading and authoritative sources that can assist you in creating a complete MLOps pipeline, I would recommend these resources:

1. **"Designing Data-Intensive Applications" by Martin Kleppmann:** While not directly ML-focused, it provides indispensable insights on building reliable, scalable systems – crucial for productionized ML pipelines.
2.  **"Continuous Delivery for Machine Learning" by Martin Fowler and team:** This book delves into the practical applications of CI/CD for machine learning, showcasing methodologies for continuous training and deployment.
3.  **Azure Machine Learning official documentation:** The official microsoft documentation should always be a go to point for all aspects of Azure ml. They are continually updated, providing the most up-to-date information.

Finally, I would emphasize that there is no single *perfect* layout; it’s essential to find the one that suits your team and project. However, the general structure I’ve outlined here is a strong, production-ready foundation for managing your Azure ML pipelines with git. Remember to adapt as your specific use cases evolve, and always focus on creating a system that is transparent, reproducible, and robust. That is the most important thing.
