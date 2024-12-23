---
title: "How to deploy multiple ML models using the Azure ML CLI?"
date: "2024-12-16"
id: "how-to-deploy-multiple-ml-models-using-the-azure-ml-cli"
---

Alright, let's talk about deploying multiple machine learning models using the Azure ML cli, something I've had to navigate quite a bit, particularly during a project involving personalized recommendation systems a few years back. We had multiple variations of the same core model, each fine-tuned for different user segments. Scaling that up and managing it required a robust deployment strategy, and the Azure ML cli was instrumental. The critical element here isn't just getting *a* model deployed; it's the systematic, maintainable deployment of *multiple* models, often concurrently and possibly with different configurations.

When we delve into this, we're not simply pushing code to a server. We need a systematic methodology. First, let's understand the key concepts. In the azure ml ecosystem, a model is often represented as a registered asset within your workspace. Deployment, then, involves creating endpoints or containers that serve that model for inference. The cli interacts with your azure ml workspace to handle all of this process.

The most straightforward way, and frankly the one I often lean on for quick prototyping, is to deploy models individually. Let's start with a basic example:

```python
# example 1: deploying a single model with the Azure ML CLI
# assuming you have a registered model named 'my-model-v1' and an inference configuration file

az ml online-endpoint create --name my-endpoint \
    --resource-group <your-resource-group> \
    --location <your-location> \
    --auth-mode Key

az ml online-deployment create --name blue \
    --endpoint-name my-endpoint \
    --model azureml:<your-model-name>:<your-model-version>  \
    --instance-type Standard_DS3_v2 \
    --instance-count 1 \
    --code-configuration ./src \
    --scoring-script  ./src/score.py  \
    --environment azureml:<your-environment-name>:<your-environment-version>
```

In this initial snippet, we are initiating two commands. The first creates a named online endpoint which acts as the gateway through which requests reach our models. The second command deploys the specified model version under the alias 'blue'. This deployment includes specifying the virtual machine type, number of instances, path to your inference scripts (`./src`), and an environment to ensure consistency.

Now, this method is  for one or two models, but quickly becomes impractical at scale. Therefore, using deployment definitions through yaml files comes next, and it's a necessity when deploying multiple versions or variations of a model. This brings us to a more maintainable approach where you define everything in a configuration file, allowing for easier reproduction and modification.

Here's a sample configuration, let's name it `deployment_config.yaml`:

```yaml
# deployment_config.yaml
$schema: https://azuremlschemas.azureedge.net/latest/onlineDeployment.schema.json
name: blue
endpoint_name: my-endpoint
model: azureml:<your-model-name>:<your-model-version>
instance_type: Standard_DS3_v2
instance_count: 1
code_configuration:
  code: ./src
  scoring_script: score.py
environment: azureml:<your-environment-name>:<your-environment-version>
```

And you deploy using this configuration like so:

```python
# example 2: deploying using a configuration file
az ml online-deployment create --file deployment_config.yaml
```

This is a much cleaner way to manage things, as you can track changes to configurations using version control. Furthermore, you can use environment variables in your yaml config, which is useful in creating a generic template for multiple models. For example, use `${MY_MODEL_NAME}` and `${MY_MODEL_VERSION}` and then populate those variables before executing the CLI command.

Now, scaling up to multiple models, the most effective approach I've used is leveraging traffic splitting and A/B testing. You're deploying several versions, and we need to manage how traffic is routed to each. For this, you use a concept called "deployment traffic." This directs a certain percentage of incoming requests to one deployment versus another. It's invaluable for canary releases and gradual rollouts.

Let’s see an illustration. Imagine you have a new version of your model ('my-model-v2') that you want to test in a production setting. You would: first, create a new deployment, then set up traffic distribution.

```python
# example 3: Deploying and handling traffic splitting
# create a new deployment, this time named 'green'
az ml online-deployment create --name green \
    --endpoint-name my-endpoint \
    --model azureml:<your-model-name-v2>:<your-model-version-v2> \
    --instance-type Standard_DS3_v2 \
    --instance-count 1 \
    --code-configuration ./src \
     --scoring-script  ./src/score.py  \
    --environment azureml:<your-environment-name>:<your-environment-version>

# update the traffic allocation
az ml online-endpoint update --name my-endpoint \
    --traffic  '{"blue": 80, "green": 20}'
```

Here, we create the 'green' deployment with the newer version of the model. Then, we update the endpoint configuration to route 80% of requests to the existing 'blue' deployment and 20% to the new 'green' deployment. This allows you to monitor performance of the new model in real conditions, and to slowly increase the traffic to the new model.

This workflow using the cli provides incredible flexibility. It facilitates complex deployment scenarios like shadow deployments, where you evaluate models with production data without routing traffic, or blue/green deployments to minimize service interruptions during updates. The key takeaway is that by using deployment configurations, variables and traffic management options, you establish a scalable and reproducible process. This moves beyond a simple single deployment into an efficient, professional, multi-model pipeline.

For anyone wanting to dive deeper, I'd highly recommend getting hands-on with the official Azure documentation for the Machine Learning cli. There is a good whitepaper from Microsoft research, “large scale Machine learning on Azure”, that delves further into managing ML pipelines with a focus on deployment methodologies. Additionally, "Designing Data-Intensive Applications" by Martin Kleppmann provides excellent general concepts that are relevant to building robust systems around deployed models, especially when scaling up to multiple deployments. Keep in mind that while the cli is powerful, the real effort is in creating robust deployment strategies that fit your team's needs.
