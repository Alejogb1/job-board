---
title: "How can I use a private Git repo for Azure Functions and Container Apps?"
date: "2024-12-23"
id: "how-can-i-use-a-private-git-repo-for-azure-functions-and-container-apps"
---

Alright, let’s talk about managing private git repositories with Azure Functions and Container Apps. This is something I’ve dealt with quite a bit in past projects, and it’s definitely a common hurdle when moving beyond basic deployments. It’s more than just plopping code; it's about security, workflow, and maintaining control over your source.

Often, when you begin with cloud deployments, the natural inclination might be to just directly integrate with public git repositories, especially for learning or small prototypes. However, once you start handling sensitive code or work within team environments, a move to private git is non-negotiable. I distinctly recall a scenario where we inadvertently exposed a crucial internal api’s base url through a public repo – a painful, but ultimately instructive, lesson.

So, how do we achieve this securely? There are multiple approaches, and the 'best' one often depends on the specific constraints of your environment. I’ll outline three methods I've implemented that generally cover a wide array of situations: leveraging personal access tokens, managed identities, and utilizing deployment keys, especially for container apps.

First, let's start with **personal access tokens (PATs)**. This approach is arguably the simplest to grasp and implement, especially for Azure Functions. The concept is straightforward. You generate a PAT within your git provider (e.g., GitHub, Azure DevOps, GitLab) with the appropriate read access to the private repository. This token becomes part of your connection string within Azure.

Now, here is a conceptual python snippet for azure functions to fetch code from a private git using a PAT. Please note, this is for illustration purposes and will vary depending on the hosting method you opt for within Azure Functions:

```python
import os

# Environment variables are the most common way to manage sensitive information.
repo_url = os.environ.get("GIT_REPO_URL", "your-repo-url")
pat = os.environ.get("GIT_PERSONAL_ACCESS_TOKEN", "your-personal-access-token")

if not repo_url or not pat:
    print("Missing either repository url or personal access token")
else:
    # construct git clone URL with PAT embedded
    git_clone_url = repo_url.replace(
        "https://", f"https://{pat}@"
    )

    # perform git clone into some directory
    command = f"git clone {git_clone_url} /tmp/my_repo"

    # execute system command to clone
    os.system(command)

    print("Code cloned from private git successfully")
```

Important considerations: always use environment variables, never hardcode your PAT in code, just like the example above. Store the token in a secure configuration such as Azure Key Vault and retrieve it as an environment variable, this prevents accidentally checking sensitive data into code. Also ensure the PAT has the *least privilege* required to perform the clone. This avoids the potential for abuse if compromised.

While PATs work, they have lifecycle management complexities. They can expire, and someone needs to remember to rotate them. This is where **managed identities** for Azure resources shines. Managed identities provide an automatically managed identity in Azure Active Directory, and your Azure Function or Container App can use this to authenticate to other azure resources, including accessing a private git repo.

Here's how you would achieve this with an Azure Function, specifically using an Azure DevOps private git repo for illustrative purposes. I will show a conceptual python script as example:

```python
import os
from azure.identity import DefaultAzureCredential
from azure.devops.connection import Connection
from msrest.authentication import BasicAuthentication

# Environment variables are the most common way to manage sensitive information.
organization_url = os.environ.get("AZURE_DEVOPS_ORG_URL", "your-devops-org-url")
project_name = os.environ.get("AZURE_DEVOPS_PROJECT_NAME", "your-project-name")
repo_id = os.environ.get("AZURE_DEVOPS_REPO_ID", "your-repo-id")

if not organization_url or not project_name or not repo_id:
    print("Missing organization, project or repository id")
else:
    # use default azure credential to login
    credential = DefaultAzureCredential()

    # Create a connection to Azure DevOps
    credentials = BasicAuthentication('', credential)
    connection = Connection(base_url=organization_url, creds=credentials)

    # Get the Git client
    git_client = connection.get_client('git')

    # Get repository info for clone url purposes
    repo = git_client.get_repository(repo_id, project=project_name)

    # build clone url
    git_clone_url = repo.remote_url

    # Perform git clone
    command = f"git clone {git_clone_url} /tmp/my_repo"

    # execute command
    os.system(command)

    print("Code cloned from Azure DevOps repo using managed identity.")

```

In this example, you’re not dealing with explicit tokens, the *DefaultAzureCredential* automatically handles authentication by obtaining a token for your managed identity. The function itself needs the ‘reader’ role on the repository. This is managed entirely within Azure’s IAM. From a management standpoint, this is easier as the identity and access is centrally controlled and token rotations are done for you, so there is no need to do it yourself. It is crucial to understand the minimal permissions your managed identity needs.

Finally, when it comes to container apps, the approach differs slightly. While you could potentially use a PAT or a managed identity, **deployment keys** offer a more fine-grained and secure approach. Deployment keys, in essence, are ssh keys that provide read-only access to your repository. They don't require user accounts. You create an ssh key pair, add the public key as a deployment key to your git repository, and then configure your container app deployment to use the private key. Here is a conceptual docker file entry command example using a deployment key:

```dockerfile
FROM python:3.9-slim-buster

WORKDIR /app

# Install git
RUN apt-get update && apt-get install -y git openssh-client

# Set up the deployment key
COPY id_rsa /root/.ssh/id_rsa
RUN chmod 600 /root/.ssh/id_rsa && \
    ssh-keyscan github.com >> /root/.ssh/known_hosts

# Define git repo url and the private key
ARG GIT_REPO_URL
ENV GIT_REPO_URL=$GIT_REPO_URL
ARG GIT_PRIVATE_KEY
ENV GIT_PRIVATE_KEY=$GIT_PRIVATE_KEY

# set up git config user (required for commit purposes)
RUN git config --global user.name "containerapp" && \
    git config --global user.email "containerapp@notreal.com"

# clone the repo
RUN git clone $GIT_REPO_URL /app

# Install dependencies (as needed)
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application code
COPY src .

# Run command
CMD ["python", "main.py"]

```

Here, the dockerfile copies the deployment key during the image build process. This is not ideal for sensitive data, and this is why tools such as Azure Container Registry, or build agents should be utilized to ensure secure handling of keys. The critical part is the usage of `ssh-keyscan` to avoid man in the middle attack when cloning. Using environment variables makes this docker file much more generic, so you can clone different git repos using the same dockerfile. You'd then provide the private key via Azure's configuration and the repository via a build argument or environment variable. This approach is favored as it avoids user-level authentication and enables a finer control over access.

When deploying from these repositories, you will need to also consider things like CI/CD pipelines as this is not an automated process. You can leverage the azure CLI, GitHub actions, Azure DevOps pipelines or other third party CI/CD tools to deploy the code. It will be essential to manage secrets using those services as well, and avoid hard coding secrets in code.

For a deeper dive into this topic, I’d recommend exploring "Git Internals" by Scott Chacon and Ben Straub for a thorough understanding of how git works, which will help significantly when debugging deployment issues. For more on Azure resource management, "Microsoft Azure Resource Manager Cookbook" by Jason Lee and James Broadhead provides insights into managing resource identities and permissions. Finally, the official Azure documentation, specifically on Azure Functions and Azure Container Apps, provides a wealth of information and is constantly updated with the latest guidance.

In short, managing private git repositories with Azure functions and container apps involves several viable approaches. Choose the one that best suits your needs, security posture and operational capabilities. My experience has taught me that focusing on security, automation, and employing the least privileged approach are keys to deploying and maintaining these services successfully. Remember, start with a clear understanding of each method's pros and cons, and consistently evaluate and adapt your setup as your requirements evolve.
