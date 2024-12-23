---
title: "How can I create an Azure container registry and deploy an image to it within the same Azure DevOps build pipeline using Bicep?"
date: "2024-12-23"
id: "how-can-i-create-an-azure-container-registry-and-deploy-an-image-to-it-within-the-same-azure-devops-build-pipeline-using-bicep"
---

,  I've been through this scenario quite a few times, usually when setting up new microservices or refactoring legacy deployments to containerized environments. It’s a common requirement to automate both the infrastructure provisioning and image deployment process within a single pipeline. Doing it via Bicep and Azure DevOps is definitely a solid approach, offering repeatability and consistency.

The fundamental idea here is to use Bicep to define your Azure Container Registry (ACR) resource, and then within the same pipeline, build and push your Docker image, before finally deploying it. This eliminates the need for manual creation of the registry or separate deployment steps. I’m going to walk you through this, keeping things fairly practical.

Initially, you need to define your ACR resource using Bicep. This is your infrastructure-as-code layer. Here’s how that might look:

```bicep
//acr.bicep
param location string = resourceGroup().location
param acrName string
param skuName string = 'Basic'
param adminUserEnabled bool = false

resource acr 'Microsoft.ContainerRegistry/registries@2023-07-01' = {
  name: acrName
  location: location
  sku: {
    name: skuName
  }
  properties: {
    adminUserEnabled: adminUserEnabled
  }
}

output acrLoginServer string = acr.properties.loginServer
```

This snippet defines a basic ACR resource. Notice that we're using parameters for things like location, name, sku, and admin user enable/disable. This allows you to reuse this file across different environments or deployments with varying configurations. The `skuName` defaults to ‘Basic’ but can be configured as needed for your use case. Crucially, I've included an output for `acrLoginServer` which is required later in the pipeline to push your image to the registry.

Next, you’ll incorporate this Bicep deployment into your Azure DevOps pipeline. The first task of this stage is to deploy the bicep template. Within your azure-pipelines.yml file, this step might look something like this:

```yaml
# azure-pipelines.yml
stages:
  - stage: DeployInfra
    jobs:
      - job: DeployACR
        pool:
          vmImage: 'ubuntu-latest'
        steps:
          - task: AzureCLI@2
            displayName: 'Deploy ACR with Bicep'
            inputs:
              azureSubscription: 'your-service-connection' # Replace with your service connection
              scriptType: 'ps'
              scriptLocation: 'inlineScript'
              inlineScript: |
                az deployment group create \
                --resource-group your-resource-group \ # Replace with your target resource group
                --template-file ./acr.bicep \
                --parameters acrName=myacregistry location=eastus
                --query properties.outputs.acrLoginServer.value -o tsv
              outputVariable: ACR_LOGIN_SERVER
```

Here, we use the `AzureCLI@2` task to deploy the `acr.bicep` template. Key points to consider are: *`your-service-connection`* should be replaced with the actual name of your Azure service connection configured in Azure DevOps. Also, ensure that *`your-resource-group`* is replaced with the name of the resource group where the ACR should be deployed. In the `script` section, `az deployment group create` is used to execute the bicep deployment. The output of that command has the login server name, which is then saved in a variable called `ACR_LOGIN_SERVER`. This is done with `--query` and `-o tsv` parameters to get a clean result. Note the `outputVariable`, this is critical for later steps to reference the login server.

The next logical step is building and pushing the docker image to the ACR. Continuing from the previous pipeline file, this would look something like this:

```yaml
 - stage: BuildAndPush
   dependsOn: DeployInfra
   jobs:
     - job: BuildPushImage
       pool:
         vmImage: 'ubuntu-latest'
       steps:
         - task: Docker@2
           displayName: 'Build and Push Docker Image'
           inputs:
             command: 'buildAndPush'
             repository: 'myimage' # Replace with your image name
             dockerfile: '**/Dockerfile' # Replace with the path to your dockerfile
             containerRegistry: 'myacregistry.azurecr.io' # Replace with your acr name
             tags: '$(Build.BuildId)' # Adding build id as tag, but you can replace with your tagging strategy
             buildContext: '.'
             push: true
             addBaseImageNameAsTag: false
         - task: Docker@2
           displayName: 'Login to ACR'
           inputs:
             command: 'login'
             containerRegistry: '$(ACR_LOGIN_SERVER)'
             connectToACR: 'false'
         - task: Docker@2
           displayName: 'Push Docker Image'
           inputs:
             command: 'push'
             containerRegistry: '$(ACR_LOGIN_SERVER)'
             repository: 'myimage' # Replace with your image name
             tags: '$(Build.BuildId)' # Adding build id as tag
```

This stage depends on the successful completion of `DeployInfra` stage. Within the `BuildAndPush` job, we have multiple Docker tasks. Initially, the first task is to build and push the image. Replace `myimage` with your desired image name, and ensure the `dockerfile` path is correct. Notice I have replaced `myacregistry.azurecr.io` in `containerRegistry` attribute with your ACR name in format of `myacrname.azurecr.io`. The `tags` are set to the build id as a simple example, but you can customize this with your versioning strategy. Afterwards, it’s essential that the second Docker task logs into the ACR using the previously stored variable `ACR_LOGIN_SERVER`. Finally the last task is where the image is pushed, referencing the login server and tagging strategy defined previously.

There are some points I want to emphasize based on my experiences:

Firstly, the `AzureCLI@2` task’s output variable parameter, `outputVariable`, is critical to pass the login server name between jobs. Incorrectly implementing or omitting this will result in failures later in the pipeline.

Secondly, the Docker task is sensitive to the `containerRegistry` setting. The build step requires `myacrname.azurecr.io` format while pushing image require `myacrname.azurecr.io` as well as a separate login action requires `$(ACR_LOGIN_SERVER)`. These details may seem small, but can create significant headaches in debugging if overlooked.

Thirdly, consider incorporating build arguments into your Docker build process when necessary, using the Docker task’s `buildArgs` parameter. This will make the process more robust, allowing for greater flexibility.

For deeper understanding and troubleshooting, I strongly recommend checking the official Microsoft Azure documentation for Bicep and Azure DevOps. Also, "Cloud Native Patterns" by Cornelia Davis offers valuable insights into cloud-native deployments. If you're new to containerization, "Docker Deep Dive" by Nigel Poulton provides a thorough introduction to Docker fundamentals.

I hope this detailed explanation and the practical examples helps you to successfully configure your Azure DevOps pipelines for ACR deployment using Bicep. Don’t hesitate to refer to the mentioned materials for further understanding and clarification.
