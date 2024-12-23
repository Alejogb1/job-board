---
title: "How do I use a python package stored in private Git Repo for Azure Functions and Container Apps?"
date: "2024-12-23"
id: "how-do-i-use-a-python-package-stored-in-private-git-repo-for-azure-functions-and-container-apps"
---

, let's tackle this one. I've actually spent a decent chunk of my career navigating this exact scenario – the joys of private dependencies and serverless deployments. Specifically, I recall a project a few years back where we had some very sensitive data processing logic that absolutely had to stay out of public repos, and getting that to play nice with our Azure Functions and Container Apps was... an experience. It's not inherently complex, but it does require a specific, methodical approach.

The core challenge here is that, by default, Azure Functions and Container Apps don't have direct access to private Git repositories. You’re essentially dealing with an environment that’s designed for public consumption, and you're trying to introduce something that isn’t. There are a few established routes to get around this. We'll focus on the most common and generally recommended methods, breaking down the technical specifics and providing illustrative examples along the way.

**The Primary Approaches: Credentials and Package Management**

The fundamental concept revolves around providing Azure services with the credentials needed to access your private repository. This boils down to two main techniques which are often used together: using environment variables or managed identities to provide credentials and employing either pip or a similar package manager to resolve dependencies during deployment.

**Approach 1: Using Environment Variables with Pip**

This method involves setting up environment variables within the Azure environment (Function App or Container App) to provide the necessary authentication when pip resolves your private package. It's a relatively straightforward method. You essentially pass your git credentials when pip installs your packages. For Git-based repos, this typically means embedding your authentication into the git url, or in the case of Azure DevOps, use a PAT token.

Here’s how it generally looks.

First, you will want to construct the pip install statement that includes authentication. You will have to encode the git credentials inside the url itself. For example, if your private repo url is `https://myorg.visualstudio.com/MyProject/_git/MyPrivateRepo` you will want something like this.

```python
# Sample requirements.txt with git package auth

git+https://{username}:{pat}@myorg.visualstudio.com/MyProject/_git/MyPrivateRepo@main#egg=myprivatepackage
```
In this example replace `{username}` with a valid username, and `{pat}` with a personal access token that has the correct permissions to read the repository.

Then you want to configure the deployment script for your Azure function to install the packages from this requirements.txt, while using the authenticated pip install. The following python code would be run in the build script of your Azure function or container app.

```python
# example script to install dependencies with authentication

import subprocess

def install_dependencies():
    try:
        # Install packages from requirements.txt, adding verbose for debugging purposes
        subprocess.check_call(['pip', 'install', '-r', 'requirements.txt', '--verbose'])
        print("Packages installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error installing packages: {e}")

if __name__ == "__main__":
    install_dependencies()
```

This code segment reads the requirements.txt and runs the pip install command. You would typically incorporate this into your Azure deployment pipeline.

**Important Notes:**

*   **Security:** Directly embedding credentials in your `requirements.txt` is generally *not* recommended for production, and this is why i used personal access tokens and not passwords. We will see a more secure approach with managed identities, but this is an easy way to quickly test your process.
*   **Git Authentication**: Git authentication can vary significantly, particularly with Azure DevOps, which benefits from using PAT tokens. For other providers like GitHub or Bitbucket, your authentication strategy may slightly change, such as using username/password. Ensure you’re using a PAT if available.
*   **Dependency Management:** Pinning package versions in your `requirements.txt` file is highly recommended for predictable builds. This helps avoid unexpected breaking changes as dependencies evolve.

**Approach 2: Managed Identities**

This is a far more secure and robust approach, leveraging Azure’s Managed Identity feature. Managed identities provide your Azure resources with an identity in Azure AD, eliminating the need for hardcoding credentials.

Here's a general breakdown of the process:

1.  **Enable Managed Identity:** First, enable a system-assigned or user-assigned managed identity on your Azure Function App or Container App.
2.  **Grant Repository Access:** Grant this managed identity the necessary permissions to access your private Git repository. For Azure DevOps this would typically involve giving the identity 'read' access on the repository.
3.  **Configure Pip Install:** Modify the pip install command to authenticate using the managed identity. For Azure DevOps, we use `azure-devops-cli-extension` to establish this authentication. We typically use this tool to get a PAT token which we then use for authentication. This token is temporary and does not expose any persistent keys.

This is somewhat more complex to configure, and it requires using the azure cli, but is the recommended method. The python code within your build script would resemble the following:

```python
import subprocess
import os

def install_dependencies():
    try:
        # Login to Azure and get temporary PAT for Azure DevOps

        # Check if we are running in an Azure environment
        if "WEBSITE_SITE_NAME" not in os.environ:
           print("Not running in an azure environment.")
           return

        # Get PAT from Azure CLI. This requires az cli extension
        # This also assumes AZURE_DEVOPS_ORG is an environment variable
        # which you will need to set in the app configuration.
        print("Getting PAT from Azure CLI")
        org_name = os.environ.get("AZURE_DEVOPS_ORG")
        pat_process = subprocess.run(["az", "devops", "configure", "--list", "--output", "json"], capture_output=True, text=True, check=True)
        pat_json = json.loads(pat_process.stdout)

        # Loop to find PAT for this particular org
        pat = None
        for config in pat_json:
          if config['organization'] == org_name:
           pat = config['token']

        if not pat:
           print("Could not find Azure devops PAT")
           return

        # Construct the git url, and pip install command with PAT
        print("Installing packages with PAT")
        repo_url = "https://{}:{}@myorg.visualstudio.com/MyProject/_git/MyPrivateRepo".format("", pat)
        install_command = ['pip', 'install', f'git+{repo_url}@main#egg=myprivatepackage', '--verbose']
        subprocess.check_call(install_command)

        print("Packages installed successfully.")

    except subprocess.CalledProcessError as e:
        print(f"Error installing packages: {e}")
    except Exception as e:
        print(f"An unexpected error occured: {e}")


if __name__ == "__main__":
    install_dependencies()
```

In this code, we use the azure cli to retrieve a token that is then used to authenticate against Azure DevOps. As before, this should be a part of your build and deploy pipeline.

**Important Notes:**

*   **Principle of Least Privilege:** Always grant the managed identity only the necessary permissions to access the repository, following the principle of least privilege. This prevents unauthorized access to other resources.
*   **Complexity:** Configuring managed identities can involve more steps than using environment variables, but the improved security is worth it for production environments.
*   **Monitoring:** Ensure proper monitoring of failed deployments when employing these methods to quickly identify potential authentication problems. Check your deployment logs for debugging information.
*   **Azure CLI:** The az cli is necessary for managed identities with Azure DevOps, but equivalent options exist for github or bitbucket.

**Recommendations and Resources**

*   **_Effective Python_ by Brett Slatkin:** This is essential reading for best practices around python, especially when considering deployment and maintainability. This is a comprehensive guide covering many aspects including virtual environments and package management which is critical for serverless deployments.
*   **Microsoft’s official Azure documentation:** The Azure documentation is the canonical resource for working with Azure Functions and Container Apps. Specifically pay attention to sections related to identity management and deployments.
*   **_Continuous Delivery_ by Jez Humble and David Farley:** Understanding the principles of continuous delivery is critical for building reliable deployment processes. This book provides many key concepts that should be considered when designing your pipeline.

**A Few Closing Thoughts**

While seemingly a minor detail, managing private dependencies is a foundational aspect of production-ready serverless deployments. Choose the approach that best suits your security needs and project complexity. I’ve personally seen the move towards managed identities pay off significantly in the long run. Remember that proper dependency management and versioning is just as important as secure authentication, and always favor best practices in python for all your projects. By following these steps and leveraging the resources available, you can create robust and secure deployment processes for your private python packages in Azure environments.
