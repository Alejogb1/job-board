---
title: "How can I use a private Git repo in Azure Functions and Container Apps?"
date: "2024-12-16"
id: "how-can-i-use-a-private-git-repo-in-azure-functions-and-container-apps"
---

, let’s tackle this. I’ve definitely been in this situation more than once, needing to pull code from a private git repository into Azure Functions and Container Apps, and it’s always a little more nuanced than just pointing and clicking. The fundamental challenge boils down to authentication: how do we securely provide these services access to your private repository without exposing your credentials? It's a common problem, and fortunately, there are several robust ways to address it. Let's break down the best practices, including detailed examples, based on my experiences over the years.

My earliest encounters involved trying to simply embed credentials directly into deployment configurations, which, as you can imagine, is a terrible idea, security-wise. Thankfully, much more sophisticated options have emerged. The core issue revolves around providing a secure, short-lived authentication token or key that grants access to your private repository specifically for the deployment process, without storing static secrets.

The preferred method, and the one I’ll detail first, is using **deployment keys or personal access tokens (PATs)**. These are essentially scoped access credentials. For Azure Functions, where deployment is more often handled via zip deploy or similar methods, you'll typically need to use the deployment key/PAT during the deployment stage. You won't be directly cloning within the running function’s environment. Think of it as prepping the ingredients *before* cooking rather than fetching them on demand *while* cooking. With Container Apps, the authentication methods are applied in a manner very similar to Azure Kubernetes Service (AKS), because the platform underlying Container Apps is built on AKS. Container Apps can pull private images, but in the context of your question, we're focusing on private *Git* repositories. Here's a breakdown:

**1. Using Deployment Keys/PATs in Azure Functions**

Let’s say you're deploying a python function app using zip deploy via Azure CLI, and you need to fetch an auxiliary python package from a private git repository. The process might resemble this. Note that you typically do this during a CI/CD process, not manually from your machine in a production context. We'll assume you already have a deployment key or PAT set up with *read-only* access to your repository (a critical security practice).

```python
# Example: Function App Deployment using Azure CLI with Deployment Key/PAT
# Assuming you have a python requirements.txt that lists the package from the private git repo
# example: git+https://<username>:<deployment_key_or_PAT>@github.com/<owner>/<repo>.git@<branch>#egg=<package_name>
# First, package up your function app files
import os
import shutil
import zipfile

def create_zip_package(folder_path, zip_path):
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(folder_path):
           for file in files:
                abs_path = os.path.join(root,file)
                zf.write(abs_path, os.path.relpath(abs_path, folder_path))
    return zip_path

function_app_folder = './my_function_app'  # your function app folder
package_path=create_zip_package(function_app_folder,'./package.zip')


# Then deploy using the Azure CLI, setting the git repo details in the requirements.txt

# The az functionapp deployment command will pick this up via the pip install step
# Ensure the requirements.txt has the private git entry in this form:
# git+https://<username>:<deployment_key_or_PAT>@github.com/<owner>/<repo>.git@<branch>#egg=<package_name>

# the PAT should be generated specifically for the deployment stage with read only access
# and stored safely in your CI/CD system
# This can also work for other package managers.
# Note: the username in the URL is sometimes unnecessary but left for explicitness. 

# Now, deploy the function app using the Azure cli
import subprocess

def azure_cli_deploy(package_path):
    result=subprocess.run(['az', 'functionapp', 'deployment', 'source', 'config-zip','-g', 'your_resource_group','--src', package_path, '-n', 'your_function_app_name'], capture_output=True, text=True)
    if result.returncode == 0:
        print("Function App deployed successfully.")
    else:
        print("Error deploying Function App:\n", result.stderr)

azure_cli_deploy(package_path)

# You should replace the values of `your_resource_group`,`your_function_app_name` and the repository details above
```

In this example, the crucial part lies within the `requirements.txt` file, where the private git repository is referenced using the PAT (or deployment key). Azure Functions' deployment process automatically handles this via the `pip install` step, assuming you've included the appropriate package specifier ( `git+https://<username>:<deployment_key_or_PAT>@...` ) .

**2. Using Deployment Keys/PATs in Container Apps**

For Azure Container Apps, you typically build a Docker image that already contains your application and its dependencies. The deployment key or PAT is again used *during* the build process, not at runtime. You wouldn't be using this technique to pull code dynamically at runtime. Here's an example docker file scenario:

```dockerfile
# Dockerfile Example: Using Deployment Key/PAT during the build
FROM python:3.9-slim-buster

# Replace with your requirements.txt path
COPY requirements.txt /app/requirements.txt

# Create a .netrc file with your PAT
# Important: this is an example and you need to use environment variables or secrets for production
# This means the PAT should not be hardcoded here!
# Typically this will be passed as a build arg, not hard coded
# A good approach is to use a secure key vault and retrieve this at build time
ARG GIT_DEPLOYMENT_KEY
RUN echo "machine github.com\n\tlogin <username>\n\tpassword $GIT_DEPLOYMENT_KEY" > /root/.netrc

# Install python dependencies, including any private git packages
# using the same method as in the function app example
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app
WORKDIR /app

CMD ["python","app.py"]
```

In this case, we're using the `.netrc` file which is a well-understood method to supply login credentials for git. *Crucially*, the `GIT_DEPLOYMENT_KEY` should be provided as a build argument, not hardcoded into the Dockerfile. This value should also be carefully stored using a secure secret management system and passed to the docker build command as an ARG via `--build-arg GIT_DEPLOYMENT_KEY=your_key`. Your CI/CD system should handle this securely.

**3. Managed Identities and Azure DevOps Pipelines/GitHub Actions**

Another, often better, approach involves using Managed Identities in conjunction with Azure DevOps pipelines or GitHub Actions. Instead of directly handling PATs, you grant the pipeline/action a managed identity with permissions to access Azure resources, including your container registry and potentially even a key vault to securely retrieve credentials if absolutely needed for certain situations. This greatly simplifies secret management. Here's a conceptual example with a generic CI/CD configuration.

```yaml
# CI/CD Pipeline Example using Managed Identity

# Assume you have a pipeline that triggers on commits to the main branch

trigger:
- main

pool:
  vmImage: 'ubuntu-latest'

steps:
- task: AzureCLI@2 # this applies to azure devops; github action equivalent available
  displayName: 'Login to Azure'
  inputs:
    azureSubscription: 'your_azure_subscription' # You can store this
    scriptType: 'pscore'
    scriptLocation: 'inlineScript'
    inlineScript: |
      az login --identity

- task: Docker@2
  displayName: 'Build and Push Docker Image'
  inputs:
    command: 'buildAndPush'
    dockerfile: 'Dockerfile'
    containerRegistry: 'your_container_registry' # Azure Container Registry
    repository: 'your_image_repo'
    tags: '$(Build.BuildId)'

# Then you can use that image with your Container App

- task: AzureCLI@2
  displayName: 'Deploy Container App'
  inputs:
    azureSubscription: 'your_azure_subscription'
    scriptType: 'pscore'
    scriptLocation: 'inlineScript'
    inlineScript: |
     az containerapp update -g your_resource_group -n your_container_app --image your_container_registry/your_image_repo:$(Build.BuildId)

```

In this scenario, the managed identity attached to the pipeline is authenticated with Azure via `az login --identity`. The pipeline, not a hard coded PAT, is able to build and push the image. And the Container App can then pull this image, assuming its own managed identity also has access permissions to the container registry.

**Recommendations:**

For deeper understanding of these concepts, I strongly recommend exploring the following:

*   **_Continuous Delivery with Azure DevOps_ by Jeffrey Palermo:** This book provides a comprehensive guide on implementing CI/CD pipelines with a strong emphasis on security best practices. It goes into detail on managing secrets within pipelines, which is essential for the scenarios we’ve discussed.
*   **The official Azure documentation on Managed Identities:** Microsoft provides great documentation on how to leverage these identities for different resources, including Container Apps, Functions, and Azure Pipelines/GitHub Actions.
*   **The documentation for your specific git hosting provider (GitHub, GitLab, Bitbucket):** Be sure to review how they handle deployment keys and PATs. Also, examine their recommendations on using secure pipelines for deployments. Understanding the specific security models of these systems is crucial.
*  **_Docker Deep Dive_ by Nigel Poulton:** For those looking to deepen their understanding of container security (e.g. not hard coding build args in docker files), this is an excellent resource.

In conclusion, pulling code from private Git repositories into Azure Functions and Container Apps requires careful attention to security. Deployment keys/PATs and managed identities, when used correctly, provide robust solutions to this challenge. Always aim for the least privilege principle and ensure that your secret management processes are implemented securely. The examples I've provided, combined with the recommended reading materials, should provide a solid foundation for achieving secure and reliable deployments.
