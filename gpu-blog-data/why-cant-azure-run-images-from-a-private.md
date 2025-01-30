---
title: "Why can't Azure run images from a private Azure repository?"
date: "2025-01-30"
id: "why-cant-azure-run-images-from-a-private"
---
Azure, by design, does not directly execute container images stored within a private Azure Container Registry (ACR) in the same way it might deploy a web app directly from a git repository. This distinction is paramount to understanding the deployment workflow within Azure's ecosystem. Instead of direct image execution, Azure resources, such as Azure Container Instances (ACI), Azure Kubernetes Service (AKS), or Azure App Service (in the context of Docker containers), **pull** the images from the ACR. The security model mandates this pull operation; it does not allow a "push to execute" workflow.

The inability to directly execute from a private ACR stems from Azure's foundational architecture separating resource management from image storage. ACR is fundamentally a registry, designed to reliably and securely store container images, analogous to a GitHub repository for code. Azure resources needing those images operate as clients, authenticating against the ACR to access and then deploy the containerized applications. This separation offers considerable advantages in terms of scalability, security, and resource lifecycle management.

Specifically, the crucial step revolves around authentication and authorization. When a deployment service, for instance ACI, needs to create a container based on an image within a private ACR, it does not have inherent access. The deployment service requires explicit credentials, typically an identity or service principal, and authorization granted through Role-Based Access Control (RBAC) to authenticate with the ACR. This authentication step ensures that only authorized services can access the private images, adding a critical layer of security and preventing unauthorized access. Moreover, the actual pulling of the image involves transferring the image layers from ACR to the compute node where the container will execute. This transfer is a pull, not a direct execution from the registry.

Consider an analogy. A software development company keeps its code in a private repository (like ACR). Developers (like Azure services) don't directly run code from the repository. Instead, they clone or download the necessary files, compile them if required, and then execute the compiled application. Similarly, Azure services must pull the image from the ACR to instantiate the container. This pull process includes downloading all the necessary layers and then performing the necessary operations to launch the container in its intended environment.

Let's examine this with a few conceptual code examples, focusing on demonstrating the authentication and pull mechanisms, and the necessary resource configuration.

**Example 1: Azure Container Instance (ACI) deployment using Service Principal**

```yaml
# This is a simplified YAML representation of an ACI deployment template
apiVersion: '2019-12-01'
type: 'Microsoft.ContainerInstance/containerGroups'
location: 'westus2'
name: 'my-aci-instance'
properties:
  osType: 'Linux'
  containers:
  - name: 'my-container'
    properties:
      image: 'myacr.azurecr.io/my-private-image:v1'
      resources:
        requests:
          cpu: 1
          memoryInGB: 1.5
  imageRegistryCredentials:
  - server: 'myacr.azurecr.io'
    username: '<SERVICE_PRINCIPAL_CLIENT_ID>'
    password: '<SERVICE_PRINCIPAL_CLIENT_SECRET>'
  restartPolicy: 'Never'
```
This YAML configuration illustrates how the ACI resource specifies the ACR server name, along with the service principal credentials. The `imageRegistryCredentials` section is critical. It provides the identity needed for ACI to authenticate against the ACR and then pull the `my-private-image:v1`. The `image` property points to the specific image in the registry. Note that the image path is a pull target, not a direct execution path. If these credentials are not configured correctly, ACI cannot access the image and the deployment will fail.  The service principal used here requires the `ACRPull` role at minimum for the ACR. This demonstrates the principle of explicit credential management for private registries.

**Example 2: Azure Kubernetes Service (AKS) Deployment with Image Pull Secret**

```yaml
# This is a simplified Kubernetes deployment manifest
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-deployment
spec:
  replicas: 3
  selector:
    matchLabels:
      app: my-app
  template:
    metadata:
      labels:
        app: my-app
    spec:
      containers:
      - name: my-container
        image: myacr.azurecr.io/my-private-image:v1
      imagePullSecrets:
      - name: my-acr-secret
---
apiVersion: v1
kind: Secret
metadata:
  name: my-acr-secret
  namespace: default
type: kubernetes.io/dockerconfigjson
data:
  .dockerconfigjson: <BASE64_ENCODED_DOCKER_CONFIG>
```

In this Kubernetes manifest, the image to be deployed is specified within the container definition. However, the key component here is the `imagePullSecrets` which reference a Kubernetes Secret (`my-acr-secret`). This secret, of type `kubernetes.io/dockerconfigjson`, holds the base64 encoded Docker configuration file containing the ACR credentials (typically, the username and password derived from service principal credentials or ACR admin user). When the Kubernetes deployment is initiated, it leverages this secret to authenticate with the ACR and pull the image. This secret is critical and similar to the ACI example, it’s not an automatic direct execution path from the ACR. If missing or incorrect, Kubernetes will fail to pull and start the Pods using this image. This demonstrates an alternative credential handling approach within Kubernetes.

**Example 3: App Service with Docker Container**

```
// This demonstrates how the App Service is configured for a Docker image
// This representation is not executable code, but conceptual
App Service Configuration:
    Image: 'myacr.azurecr.io/my-private-image:v1'
    Registry Type: Azure Container Registry
    Registry URL: 'myacr.azurecr.io'
    Authentication Type: Managed Identity
    Managed Identity Client ID: '<MANAGED_IDENTITY_CLIENT_ID>'
```

The simplified configuration shows that an Azure App Service can deploy a Docker container. Instead of storing secrets directly, this uses Managed Identity. When configured, the App Service will use the associated managed identity to authenticate against ACR and pull the container image from there. This demonstrates yet another mechanism for authentication. App Service, like ACI and AKS, always *pulls* the image as part of its deployment process, illustrating the separation of container storage from execution within Azure.  The managed identity will also need at least the ACRPull role on the target registry for authorization.

In all three examples, the common thread is the requirement to establish a secure, authenticated link between the service wanting to run a container and the ACR storing its image. This demonstrates that direct execution from ACR is not possible. The service needs explicit credentials, authorization, and a pull mechanism to retrieve the image before execution.

For understanding the broader scope and fine-grained details, it's crucial to consult the following resource categories, all directly published by Microsoft:

*   **Azure Container Registry Documentation:** Provides comprehensive explanations on image storage, management, authentication, and security considerations.  It covers topics ranging from service principal creation and ACR roles to best practices for registry maintenance.
*   **Azure Container Instances Documentation:** Details how to create and deploy container instances with proper authentication against various registries, including ACR. It goes in depth on the different credential options.
*   **Azure Kubernetes Service Documentation:** Describes the configuration of container images within a Kubernetes cluster, including the concepts of image pull secrets and integration with ACR. This documentation is necessary for secure and scalable container orchestration.
*   **App Service Documentation for Containerized Applications:** Provides detailed instructions on how to deploy custom container images to App Service, including authentication methods with a private ACR. It includes best practices for managing your app images.
*   **Azure Active Directory Documentation:** Understanding Azure AD is critical to understanding how managed identities and service principals work and how to use them to access private resources including ACR. This covers the permissions and policies needed to grant access between services.

These resources provide a comprehensive view of the mechanisms required to use container images securely within the Azure ecosystem. They reinforce the fundamental principle that Azure services always *pull* images from ACR; they do not execute them directly within the registry's context.
