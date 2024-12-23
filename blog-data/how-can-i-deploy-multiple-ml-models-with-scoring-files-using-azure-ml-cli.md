---
title: "How can I deploy multiple ML models with scoring files using Azure ML CLI?"
date: "2024-12-23"
id: "how-can-i-deploy-multiple-ml-models-with-scoring-files-using-azure-ml-cli"
---

Let's unpack this. I've certainly been in the trenches deploying multiple machine learning models, especially using Azure ml's command-line interface (cli). It’s not uncommon to find yourself juggling numerous models, each potentially having distinct scoring scripts, and needing a reliable, reproducible deployment strategy. The key, as I learned early on in a project involving personalized recommendation engines, is structured management and leveraging the cli's capabilities for automation.

The challenge you're facing fundamentally boils down to orchestrating several moving parts: the trained models themselves, their respective scoring scripts, environment dependencies, and the actual deployment process onto Azure compute resources. Each of these elements requires careful consideration and precise configuration to ensure seamless operation. Let's break it down, focusing on a typical pattern I’ve seen evolve and refine over various projects.

First, we establish a clear directory structure. For example, consider the following setup:

```
ml_project/
├── models/
│   ├── model_a/
│   │   ├── model.pkl
│   │   ├── scoring_script.py
│   │   └── environment.yml
│   ├── model_b/
│   │   ├── model.joblib
│   │   ├── scoring_script.py
│   │   └── environment.yml
│   └── model_c/
│       ├── model.h5
│       ├── scoring_script.py
│       └── environment.yml
└── config/
    └── deployment_config.yaml
```

In this structure, each model resides in its own subdirectory, containing the trained model file (`.pkl`, `.joblib`, `.h5` – formats vary), its corresponding scoring script, and a `environment.yml` file specifying the necessary python dependencies. The `config` directory houses overall deployment configurations.

The `environment.yml` file for each model is paramount because it defines the exact dependencies required for that model's scoring script. This is where you prevent frustrating 'works on my machine' scenarios. For example, the `model_a/environment.yml` might look like this:

```yaml
name: model_a_env
channels:
  - conda-forge
dependencies:
  - python=3.8
  - scikit-learn=0.24.2
  - pandas=1.2.0
```

This ensures that when the container for model `a` is built in azure, it has precisely the python environment it needs. Similarly, `model_b` and `model_c` can have different environments. This granular control over environment management is critical, especially when you start to use different versions of packages or have conflicts.

Now, let's dive into the core deployment logic. We'll use the azure ml cli to register each model, its environment, and then deploy. I'm using conceptual commands that would be similar to the actual cli calls. I would recommend referring to the official Azure documentation as well as resources like the "Machine Learning Engineering in Action" book by Ben Trevett to understand the full nuances of the Azure ml ecosystem.

Here's how I would register and deploy model `model_a`:

```bash
# register the environment
az ml environment create --file models/model_a/environment.yml -n model_a_env

# register the model
az ml model create --name model_a \
    --path models/model_a/ \
    --type custom \
    --scoring-script models/model_a/scoring_script.py \
    --environment model_a_env
    --description "Machine learning model a."

# create the online endpoint
az ml online-endpoint create --name my-endpoint \
    --auth-mode key

# create the online deployment
az ml online-deployment create --name model-a-deployment \
    --endpoint-name my-endpoint \
    --model model_a \
    --instance-type Standard_DS3_v2 \
    --instance-count 1
```

This process registers the environment, registers the model along with the path and the scoring script, establishes an online endpoint to receive requests, and then creates a deployment for this specific model to serve. Notice that the cli provides the ability to directly use the model and the environment that was just registered.

Let's look at an example scoring script to illustrate what that looks like. Suppose the `model_a/scoring_script.py` looks like this:

```python
import json
import joblib
import numpy as np
import pandas as pd
from azureml.core.model import Model

def init():
    global model
    # Load the model from the model path provided
    model_path = Model.get_model_path(model_name='model_a')
    model = joblib.load(model_path + '/model.pkl')

def run(raw_data):
    try:
        data = json.loads(raw_data)["data"]
        data = pd.DataFrame(data)
        prediction = model.predict(data)
        return json.dumps({"prediction": prediction.tolist()})
    except Exception as e:
        return json.dumps({"error": str(e)})
```
This `scoring_script.py` does the following: uses `azureml.core.model.Model` to load the previously registered model file, parses json input, converts it to dataframe, and uses the model to predict, then returns the result as json.

We would repeat a similar process for models `b` and `c`, adjusting the parameters accordingly. This is where the power of scripting truly shines. Rather than manually running each command, I typically create a python script or a shell script to loop through the different models and perform the registration and deployment.

For example, we can abstract the registration and deployment process into a python script as a wrapper. This is a simplified illustration and could be enhanced with error handling and logging.
```python
import os
import subprocess
import yaml

def deploy_model(model_name, model_path, environment_path, scoring_script_path):
    env_name = f"{model_name}_env"

    # Create environment
    subprocess.run(
        ["az", "ml", "environment", "create", "--file", environment_path, "-n", env_name],
        check=True,
    )

    # Register model
    subprocess.run(
        [
            "az", "ml", "model", "create",
            "--name", model_name,
            "--path", model_path,
            "--type", "custom",
            "--scoring-script", scoring_script_path,
            "--environment", env_name,
            "--description", f"Machine learning model {model_name}."
        ],
        check=True,
    )

    # Deploy model to online endpoint
    subprocess.run(
        [
            "az", "ml", "online-deployment", "create",
            "--name", f"{model_name}-deployment",
            "--endpoint-name", "my-endpoint",
            "--model", model_name,
            "--instance-type", "Standard_DS3_v2",
            "--instance-count", "1"
        ],
        check=True,
    )

if __name__ == "__main__":
    model_dir = 'models'
    for model_name in os.listdir(model_dir):
        if not os.path.isdir(os.path.join(model_dir,model_name)):
            continue
        model_path = os.path.join(model_dir, model_name)
        environment_path = os.path.join(model_path, 'environment.yml')
        scoring_script_path = os.path.join(model_path, 'scoring_script.py')

        deploy_model(model_name,model_path, environment_path, scoring_script_path)
    print("All models deployed!")
```
This simplified script iterates over the model directories and executes the az cli commands for each model. This ensures consistency of the deployment procedure.

A key aspect I’ve found helpful, and which this python script touches upon, is to utilize configuration files for managing deployment parameters. In the initial directory structure I described, there was a config folder containing a `deployment_config.yaml`. This file can house the online endpoint names, the instance types, the instance counts, or other relevant parameters. This would allow modification of the deployment without hardcoding parameters in the script.

This structured approach ensures that every time you deploy your models, you are doing so consistently and predictably. This reduces the risks associated with human error and makes the entire process easily repeatable. It’s about automation and configuration management, not just running individual commands.

Finally, always test the deployed models thoroughly. Azure provides ways to test endpoints through the ui or with the cli. Thorough testing ensures the models are performing as expected in the deployed environment.

For further learning on Azure ml concepts and best practices, I would suggest checking out the official Microsoft documentation on Azure machine learning. Also, “Programming Machine Learning: From Coding to Deploying” by Paolo Perrotta offers a solid foundation on the general workflow of developing and deploying machine learning models, which is valuable context. By combining the theoretical knowledge and the practical experience of deploying various models via Azure ML cli, you will have a strong toolset for deploying and maintaining complex deployments.
