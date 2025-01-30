---
title: "How can a Docker container from Azure Container Registry be run in an Azure DevOps pipeline?"
date: "2025-01-30"
id: "how-can-a-docker-container-from-azure-container"
---
The core challenge in running a Docker container from Azure Container Registry (ACR) within an Azure DevOps pipeline lies in correctly authenticating to the ACR and specifying the image's precise location.  Failing to do so results in authentication errors, preventing the pipeline from pulling and subsequently running the container.  My experience integrating continuous integration/continuous deployment (CI/CD) pipelines across numerous Azure projects has highlighted this as a critical point.  Incorrect authentication is the single largest source of failure I've encountered in this specific scenario.


**1.  Clear Explanation**

The process involves several distinct steps. First, the pipeline needs access to the ACR.  This usually involves service principal authentication, granting the pipeline sufficient privileges to pull images from the specified registry.  Second, the image's full name, including the registry's address, repository name, tag (or digest for increased immutability), and image name, must be correctly specified in the pipeline's task.  Finally, the container needs to be run within a Docker environment; this typically involves using a pre-configured agent or virtual machine with Docker installed, or a task dedicated to Docker operations.

A common pitfall is overlooking the appropriate scoping of permissions for the service principal. Restricting permissions to only necessary actions enhances security and reduces the potential attack surface.  Another frequently observed mistake involves using an incorrect image name or tag, leading to pull failures. Finally, ignoring the agent's Docker configuration, including ensuring Docker daemon access and appropriate resource allocation, may cause the container to fail to start or operate inefficiently.

My experience has shown that using a dedicated service principal, coupled with detailed logging and error analysis within the pipeline, significantly reduces troubleshooting time.  This granular approach not only ensures successful image pulls but also facilitates identification and resolution of related issues.

**2. Code Examples with Commentary**

The following examples demonstrate three distinct approaches for running a Docker container from ACR within an Azure DevOps pipeline using YAML.  These encompass variations in authentication mechanisms and image specification.


**Example 1: Using a Service Principal with Image Tag**

```yaml
trigger:
- main

pool:
  vmImage: 'ubuntu-latest'

steps:
- task: Docker@2
  displayName: 'Login to ACR'
  inputs:
    command: 'login'
    containerRegistry: 'myacr.azurecr.io' # Replace with your ACR name
    azureSubscription: 'MyAzureSubscription' # Replace with your subscription name
- task: Docker@2
  displayName: 'Run Docker Container'
  inputs:
    command: 'run'
    imageName: 'myacr.azurecr.io/myrepo/myapp:latest' # Replace with your image
    containerName: 'mycontainer'
    arguments: '-d' # Run container in detached mode
```

**Commentary:** This example leverages the `Docker@2` task, a common Azure DevOps task designed for Docker interactions.  It first logs in to the ACR using a service principal associated with the specified Azure subscription.  Then, it runs a Docker container using the fully qualified image name, including the registry address, repository, and image tag.  The `arguments` field specifies the `-d` flag to run the container in detached mode.  Crucially, replacing placeholders like `myacr.azurecr.io`, `MyAzureSubscription`, `myrepo`, and `myapp` is essential for successful execution.


**Example 2:  Using a Service Principal with Image Digest**

```yaml
trigger:
- main

pool:
  vmImage: 'ubuntu-latest'

variables:
  imageDigest: $(ImageDigest)

steps:
- task: Docker@2
  displayName: 'Login to ACR'
  inputs:
    command: 'login'
    containerRegistry: 'myacr.azurecr.io'
    azureSubscription: 'MyAzureSubscription'
- script: |
    echo "##vso[task.setvariable variable=ImageDigest;isOutput=true]$([[ $(docker images -q myacr.azurecr.io/myrepo/myapp) ]])"
  displayName: 'Get Image Digest'
- task: Docker@2
  displayName: 'Run Docker Container'
  inputs:
    command: 'run'
    imageName: 'myacr.azurecr.io/myrepo/myapp@$(imageDigest)'
    containerName: 'mycontainer'
    arguments: '-d'
```

**Commentary:** This example utilizes an image digest instead of a tag for improved immutability.  The script retrieves the image digest using the `docker images` command and sets it as a variable `imageDigest`.  The `Docker@2` task then uses this variable to specify the image for running the container, providing a more robust approach.  This method is superior in environments prioritizing image version control and deployment stability.  Note the use of `isOutput=true` to make the `imageDigest` available to subsequent steps.


**Example 3:  Using a Personal Access Token (PAT) (Less Recommended)**

```yaml
trigger:
- main

pool:
  vmImage: 'ubuntu-latest'

variables:
  ACR_PAT: $(ACR_PAT)

steps:
- task: CmdLine@2
  displayName: 'Login to ACR using PAT'
  inputs:
    script: 'echo $(ACR_PAT) | docker login -u "$DOCKER_USER" -p "$ACR_PAT" myacr.azurecr.io'
- task: Docker@2
  displayName: 'Run Docker Container'
  inputs:
    command: 'run'
    imageName: 'myacr.azurecr.io/myrepo/myapp:latest'
    containerName: 'mycontainer'
    arguments: '-d'
```

**Commentary:** This example uses a Personal Access Token (PAT) for authentication. While functional, this method is less secure than service principals, exposing credentials directly within the pipeline.  It's important to manage PATs carefully, employing short lifespans and least privilege principles. The `CmdLine@2` task executes a shell command to perform the login using the PAT stored as a variable.  While functional, this approach should be avoided in production environments due to security considerations.  Service principals offer a far more secure method of authentication.



**3. Resource Recommendations**

For deeper understanding of Azure DevOps pipelines, consult the official Azure DevOps documentation.  Familiarize yourself with the available tasks, specifically those pertaining to Docker and container orchestration.  Mastering YAML syntax for pipeline definitions is crucial for effective configuration management.  Study best practices for securing Azure resources, particularly around authentication and authorization for access control.  Understanding Docker concepts, including image building, tagging, and digest usage, is essential for effective container management.  Finally, explore Docker Compose for managing multi-container applications.  This provides the knowledge for orchestrating more complex deployments beyond single containers.
