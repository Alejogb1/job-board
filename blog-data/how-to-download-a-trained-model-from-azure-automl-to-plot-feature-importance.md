---
title: "How to download a trained model from Azure AutoML to plot feature importance?"
date: "2024-12-16"
id: "how-to-download-a-trained-model-from-azure-automl-to-plot-feature-importance"
---

Okay, let’s tackle this. I've personally dealt with this scenario a few times, extracting models from Azure AutoML to analyze feature importance, and it's not always as straightforward as one might hope. The process involves a few crucial steps, and understanding the underlying components is key to doing it properly. The challenge primarily revolves around the way AutoML packages trained models and the diverse model types it can generate. Let’s break this down.

First, remember that when you train a model with Azure AutoML, it’s not immediately a standalone pickle file waiting for you. Instead, AutoML constructs a complex pipeline including preprocessing steps, feature engineering, and the final model itself. To extract the feature importance, you typically need to access the model instance *after* applying the preprocessing.

So, the initial hurdle is locating that fully assembled model. Typically, the way to get at this is by using the `mlflow` integration that Azure AutoML uses. When you initiate an AutoML run, the model and its associated artifacts get logged in the experiment’s run history, accessible through the `mlflow` tracking URI. Now, the exact code will depend slightly on if you're using the SDKv1 or SDKv2 for Azure Machine Learning. For this example, let's consider we’re using SDKv2 as it's the more current and recommended approach.

Here's a general overview of the workflow:

1.  **Retrieve the best model's run ID:** You need the specific run ID that corresponds to your optimal model selected by AutoML.
2.  **Access the MLflow run:** Use the run ID to connect to the MLflow tracking service.
3.  **Load the model:** Download the serialized model from MLflow.
4.  **Extract the fitted model:** Because AutoML models are often wrapped in a preprocessing pipeline, we need to extract the final estimator object.
5.  **Access feature importances:** Finally, use the estimator's API to access the feature importances, and prepare it for plotting.

Let’s put this into actual code. Here's a Python code snippet demonstrating how to achieve this in a controlled environment, using the Azure Machine Learning SDKv2:

```python
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from mlflow.tracking import MlflowClient
import mlflow
import pandas as pd
import matplotlib.pyplot as plt

def get_automl_model_feature_importance(experiment_name: str, run_id: str):
    """
    Downloads an AutoML model from AzureML and plots feature importance.
    """
    # Initialize the MLClient
    credential = DefaultAzureCredential()
    ml_client = MLClient.from_config(credential=credential)

    # Get mlflow tracking uri
    mlflow_tracking_uri = ml_client.workspaces.get(ml_client.workspace_name).mlflow_tracking_uri

    # Connect to mlflow
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow_client = MlflowClient()

    # Get the mlflow run associated with the best automl model
    run_info = mlflow_client.get_run(run_id)

    # Get the artifact URI
    model_artifact_uri = run_info.info.artifact_uri

    # Load the fitted model from the artifact URI
    model = mlflow.pyfunc.load_model(model_artifact_uri + "/model")

    # Extract the estimator from the pipeline
    estimator = model.steps[-1][1]

    # Now, let's extract feature importance
    if hasattr(estimator, 'feature_importances_'):
         importances = estimator.feature_importances_
    elif hasattr(estimator, 'coef_'):
         importances = estimator.coef_
    else:
         raise ValueError("The model type does not support feature importance extraction.")


    # Get feature names
    feature_names = model.steps[0][1].get_feature_names_out()  # Get feature names from the first step

    # Ensure feature names is a list, if not convert
    if isinstance(feature_names, list):
        pass
    elif hasattr(feature_names, 'tolist'):
        feature_names = feature_names.tolist()
    else:
        raise ValueError("Feature names could not be extracted correctly")

    # create feature importance dataframe
    feat_importance = pd.DataFrame(
        {'feature': feature_names, 'importance': importances}
    ).sort_values('importance', ascending=False)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.barh(feat_importance['feature'], feat_importance['importance'])
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importance Plot')
    plt.gca().invert_yaxis()  # To show most important on top
    plt.show()

if __name__ == '__main__':
    # Replace with your actual experiment name and run ID
    experiment_name = "your-experiment-name"
    run_id = "your-run-id"
    get_automl_model_feature_importance(experiment_name, run_id)
```

This snippet does the following: First, it establishes a connection to Azure Machine Learning using the `MLClient`. Then it retrieves the MLflow tracking URI, and using that, connects to the MLflow tracking service. After that, it loads the model using `mlflow.pyfunc.load_model()`, which includes both the preprocessing pipeline and the model. It then extracts the model estimator from the fitted model. Following that, the code checks for either `feature_importances_` or `coef_` attributes, a commonality across several scikit-learn style model classes that AutoML can employ. Finally, it plots those importances as a horizontal bar chart.

However, it's important to note that the `model.steps[-1][1]` part presumes that the final step of your pipeline contains the estimator. This might not always be the case if AutoML has done some more elaborate preprocessing or wrapping of the model. Therefore, you may need to inspect the `model.steps` object to pinpoint which step holds the final model to correctly extract it.

Now, let’s consider a scenario when a more complex model is trained by AutoML, like a LightGBM, XGBoost or similar ensemble-based tree methods. The approach is conceptually the same but requires additional steps to handle the output from the model, particularly where the model is wrapped by a model pre-processing stage.

```python
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from mlflow.tracking import MlflowClient
import mlflow
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def get_complex_automl_model_feature_importance(experiment_name: str, run_id: str):
    """
    Downloads an AutoML model (e.g., LightGBM) from AzureML, extracts
    feature importance and plots it. This variant handles
    more complex wrapped model outputs.
    """

    # Initialize the MLClient
    credential = DefaultAzureCredential()
    ml_client = MLClient.from_config(credential=credential)

    # Get mlflow tracking uri
    mlflow_tracking_uri = ml_client.workspaces.get(ml_client.workspace_name).mlflow_tracking_uri

    # Connect to mlflow
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow_client = MlflowClient()

    # Get the mlflow run associated with the best automl model
    run_info = mlflow_client.get_run(run_id)

    # Get the artifact URI
    model_artifact_uri = run_info.info.artifact_uri

    # Load the fitted model from the artifact URI
    model = mlflow.pyfunc.load_model(model_artifact_uri + "/model")

    # Inspect model steps to locate the actual estimator (may vary!)
    estimator = None
    for step in reversed(model.steps):
         if hasattr(step[1], 'feature_importances_'):
              estimator = step[1]
              break
    if not estimator:
         raise ValueError("Could not locate an estimator with feature importance.")


    # Extract feature importance (might vary based on model type)
    if hasattr(estimator, 'feature_importances_'):
        importances = estimator.feature_importances_
    elif hasattr(estimator, 'coef_'):
        importances = estimator.coef_
    else:
       raise ValueError("The model type does not support feature importance extraction.")

    # Get feature names (again, must adapt to preprocessor type)
    feature_names = model.steps[0][1].get_feature_names_out()

    # Ensure feature names is a list, if not convert
    if isinstance(feature_names, list):
        pass
    elif hasattr(feature_names, 'tolist'):
        feature_names = feature_names.tolist()
    else:
        raise ValueError("Feature names could not be extracted correctly")


    # create feature importance dataframe
    feat_importance = pd.DataFrame(
        {'feature': feature_names, 'importance': importances}
    ).sort_values('importance', ascending=False)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.barh(feat_importance['feature'], feat_importance['importance'])
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importance Plot')
    plt.gca().invert_yaxis()  # To show most important on top
    plt.show()


if __name__ == '__main__':
    # Replace with your actual experiment name and run ID
    experiment_name = "your-complex-experiment-name"
    run_id = "your-complex-run-id"
    get_complex_automl_model_feature_importance(experiment_name, run_id)
```

Here, instead of directly assuming the final pipeline step houses the model, we iterate in reverse to locate the step with the `feature_importances_` attribute. This is a more robust approach, that better adapts to the nuances of AutoML's pipeline architecture.

Lastly, if you are using a model which has a `coef_` instead of `feature_importances_`, the approach remains the same, but you would need to change your if statement to check for the appropriate attribute.

```python
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from mlflow.tracking import MlflowClient
import mlflow
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def get_linear_model_feature_importance(experiment_name: str, run_id: str):
    """
    Downloads a linear AutoML model (e.g., LogisticRegression) from AzureML, extracts
    feature importance (coefficients) and plots it.
    """

    # Initialize the MLClient
    credential = DefaultAzureCredential()
    ml_client = MLClient.from_config(credential=credential)

    # Get mlflow tracking uri
    mlflow_tracking_uri = ml_client.workspaces.get(ml_client.workspace_name).mlflow_tracking_uri

    # Connect to mlflow
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow_client = MlflowClient()

    # Get the mlflow run associated with the best automl model
    run_info = mlflow_client.get_run(run_id)

    # Get the artifact URI
    model_artifact_uri = run_info.info.artifact_uri

    # Load the fitted model from the artifact URI
    model = mlflow.pyfunc.load_model(model_artifact_uri + "/model")

    # Inspect model steps to locate the actual estimator (may vary!)
    estimator = None
    for step in reversed(model.steps):
         if hasattr(step[1], 'coef_'):
              estimator = step[1]
              break
    if not estimator:
         raise ValueError("Could not locate an estimator with feature coefficients.")


    # Extract feature coefficients
    if hasattr(estimator, 'coef_'):
      importances = estimator.coef_

      # Check if this is a multi-class classification
      if len(importances.shape) == 2:
        # For multi-class, take the absolute value of each class's coefficients
        importances = np.mean(np.abs(importances), axis=0)
      else:
        importances = np.abs(importances)

    else:
       raise ValueError("The model type does not support feature coefficient extraction.")

    # Get feature names (again, must adapt to preprocessor type)
    feature_names = model.steps[0][1].get_feature_names_out()

    # Ensure feature names is a list, if not convert
    if isinstance(feature_names, list):
        pass
    elif hasattr(feature_names, 'tolist'):
        feature_names = feature_names.tolist()
    else:
        raise ValueError("Feature names could not be extracted correctly")

    # create feature importance dataframe
    feat_importance = pd.DataFrame(
        {'feature': feature_names, 'importance': importances}
    ).sort_values('importance', ascending=False)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.barh(feat_importance['feature'], feat_importance['importance'])
    plt.xlabel('Feature Importance (Absolute Coefficient Value)')
    plt.ylabel('Feature')
    plt.title('Feature Importance Plot')
    plt.gca().invert_yaxis()  # To show most important on top
    plt.show()


if __name__ == '__main__':
    # Replace with your actual experiment name and run ID
    experiment_name = "your-linear-experiment-name"
    run_id = "your-linear-run-id"
    get_linear_model_feature_importance(experiment_name, run_id)
```

This script extends the example further to handle models that use coefficients. The main adjustment is to locate the relevant object, check if `coef_` is available, and use that to access the importance values.

For deeper understanding of MLflow, I’d recommend consulting the official documentation at [mlflow.org](https://mlflow.org/). Additionally, “Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow” by Aurélien Géron provides very sound insights into the practicalities of various machine learning models, particularly understanding feature importances. Finally, for a more focused exploration of model interpretation methods, look to “Interpretable Machine Learning” by Christoph Molnar.

Remember, model extraction from cloud-based AutoML services requires an understanding of the tooling that connects you to your models. Understanding the workflow and knowing how to adapt the pipeline extraction logic is crucial. I hope these practical examples and explanations prove helpful.
