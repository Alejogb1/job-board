---
title: "How can I use a Python package stored in a private Git Repo for Azure Functions?"
date: "2024-12-23"
id: "how-can-i-use-a-python-package-stored-in-a-private-git-repo-for-azure-functions"
---

, let's unpack this. I've encountered this specific challenge many times in my career, particularly when deploying internal tools and microservices within Azure environments. The need to utilize private python packages in azure functions isn't uncommon, and the solutions require understanding the interplay between git authentication, azure function deployment, and python package management. It’s more nuanced than a simple public PyPI package install, that’s for certain.

My experience stems from a project involving several data processing functions which relied on a proprietary analytics library. This library, deemed sensitive, resided in a private git repository. Getting it to work seamlessly with our Azure Functions required a bit of planning, and we hit some interesting hiccups along the way that taught us valuable lessons. The core issue is that Azure Function deployment pipelines aren't directly aware of your private git credentials or, by default, how to properly fetch packages from them. We need to bridge that gap.

First, understand the underlying mechanism. When an Azure Function gets deployed, it essentially creates a virtual environment, and it's within this virtual environment that your function code and its dependencies execute. We need to ensure that our private package is correctly installed into *that* virtual environment during the deployment process. There are multiple approaches, but I've found that consistently the best solution involves leveraging either the azure deployment pipeline or a custom deployment script that can handle git authentication and pip package installation. Here’s how I’ve generally handled it in real-world scenarios, and I’ll provide a few different methods with accompanying code examples.

**Method 1: Using a Deployment Pipeline with a Service Principal**

This approach is often the cleanest and most secure for CI/CD workflows. Here, we'll use Azure DevOps pipelines (or similar) to manage the deployment. The key is to grant a service principal access to our private git repository.

*   **Service Principal Setup:** First, you need a service principal in Azure Active Directory. This can be created in the portal or via Azure CLI. Grant this service principal 'read' access to your private git repository. How you manage this access depends on your git platform (e.g., bitbucket, github, azure repos). Consult their documentation for specific instructions, but, generally, this involves adding the service principal as a collaborator to your private repository and ensuring it has 'pull' permissions.
*   **Pipeline Configuration:** In your Azure DevOps pipeline (or similar), include a step to perform the pip installation. We will use a slightly modified `requirements.txt` which specifies the private git repo location. This requires using the `-e` flag for editable install, which means pip will clone and install directly from the source repo, and the repository’s url is prepended by `git+`:

```yaml
# Example Azure DevOps Pipeline snippet
steps:
- task: AzureFunctionApp@1
  displayName: 'Deploy Azure Function'
  inputs:
    azureSubscription: 'YourSubscriptionName'
    appName: 'YourFunctionName'
    package: '$(System.DefaultWorkingDirectory)/app_package' # path to function app package
    deploymentMethod: 'zipDeploy'
- task: Bash@3 # use a bash task to install deps
  displayName: 'Install Private Dependencies'
  inputs:
    targetType: 'inline'
    script: |
      cd $(System.DefaultWorkingDirectory)/app_package
      pip install -r requirements.txt # install all the required packages from the requirements.txt file.
```

And here's how your requirements.txt might look:

```
# requirements.txt
git+https://<service_principal>:<PAT>@github.com/your-org/your-private-repo.git@main#egg=my_private_package
# regular PyPI dependencies here as usual
requests
pandas
```

The critical part is using `git+https://...` and including the specific branch, tag, or commit using `@<branch>`, in this case main branch and `#egg=my_private_package` defines the name to use for the install which is usually your repository name with `-` swapped for `_`.

The important parts are the service principal’s id and its client secret, which we use as a personal access token (PAT). We will need to create it from the service principal and pass them to the pipeline. The url above is an example using a service principal and a github repo, but it can be easily modified to different platforms and authentication methods. This will allow the pipeline to pull from your private repo, and install using pip. Note, make sure to scope and store service principals securely. Do not embed credentials directly into code.

**Method 2: Custom Deployment Script with SSH Keys**

This method is more hands-on but gives you precise control over the deployment process, which can be beneficial if your setup is very unique or you need very fine-grained control. This method involves using ssh keys.

1.  **Generate and Configure SSH Keys:** Create an ssh key pair. Add the *public* key to the authorized keys of your private git repository. Treat your private key as you would any other sensitive credential.
2.  **Function App Settings:** In your Azure Function App configuration, add an app setting to store the *private* key (base64 encoded). It is recommended to use the Azure Key Vault for such sensitive information to enhance security, however, for demonstration, this app setting is sufficient.
3.  **Deployment Script:** Create a custom deployment script (e.g. `deploy.sh`) in the function app directory:

```bash
#!/bin/bash
# deploy.sh
set -e

# Decode the base64 encoded key from the app setting.
echo "$DEPLOYMENT_SSH_KEY" | base64 --decode > ~/.ssh/id_rsa
chmod 600 ~/.ssh/id_rsa
ssh-keyscan github.com >> ~/.ssh/known_hosts # Add the git host to known_hosts
git clone git@github.com:your-org/your-private-repo.git ./my_private_package
pip install -e ./my_private_package # Install your package
pip install -r requirements.txt # install the rest of the requirements.

```

4. **Modify the .funcignore file:** make sure `my_private_package` is ignored during deployments if it is in your root folder or it can cause issues during subsequent deployments. Add the folder to the ignore file by:
```
# .funcignore
my_private_package
```

5.  **Deployment:** Configure your deployment to execute this script. There are multiple ways to achieve this. One of the easiest ways is to add it to the `deployment` section of the `host.json` config file in your project:
```
{
    "version": "2.0",
    "logging": {
    },
    "extensions": {
    "http": {
        "routePrefix": "api"
      }
    },
  "customHandler": {
    "description": {
      "defaultExecutablePath": "python",
      "workingDirectory": ".",
    }
    },
    "deployment": {
      "runFromPackage": false,
      "scriptFile": "deploy.sh"
    }
}
```

Make sure to set the app setting `DEPLOYMENT_SSH_KEY` in your function app.

This approach gives more flexibility, but remember that security is critical. The private key must be secured and handled correctly. The above snippet is a demonstration for clarity; you’d want to store the key in a secure vault and retrieve it through an appropriate mechanism if you’re going to use this in production.

**Method 3: Using a Private Package Registry**

Another approach to consider is hosting your private packages in a private package registry (like Azure Artifacts or GitHub Packages). This requires a separate setup but can significantly simplify dependency management and make the process a lot more straightforward.

*   **Private Registry Setup:** Configure a private registry and upload your custom package to it. This involves a bit more upfront setup, such as creating an access token for authentication.
*   **Deployment:** Your deployment pipeline would need to configure the python environment to authenticate with this private registry, using the access token and the correct registry address. This usually is done by setting the pip config correctly during deployment.

The pip config would look something like this in your `.pip/pip.conf` file or alternatively the `PIP_INDEX_URL` environment variable.

```
[global]
index-url = https://<your-registry>/pypi/<your-org>/simple
extra-index-url = https://pypi.org/simple

[pypi]
username = <your-username>
password = <your-token>
```

Once you’ve set up the registry correctly, you can reference it in the `requirements.txt` file of your Azure function deployment as usual, and it would install packages from your registry without needing any extra git cloning or SSH key management. This is often preferred for more complex deployments, as it gives more control and separation over the different stages.

**Considerations:**

*   **Security:** Always handle your keys and tokens securely. Avoid embedding them directly in your code or scripts. Azure Key Vault is a recommended practice for secure storage of secrets.
*   **Authentication:** The way you handle git authentication might vary based on the repository platform. Make sure your chosen method works reliably.
*   **Performance:** Cloning large git repos can slow down deployment. Consider using a tag or a specific commit rather than the entire repo or cloning the `egg` of the python package using `-e`.
*   **Maintenance:** Regularly review and update your deployment processes, especially as your infrastructure evolves.

**Recommended Resources:**

*   **"Python Packaging User Guide"**: Provides in-depth documentation on python packaging including how to create and install editable installs with git using `-e` flag, which is a critical point here. It is available online at [packaging.python.org](https://packaging.python.org/).
*   **"Azure DevOps Documentation":** Azure DevOps documentation has excellent documentation on setting up CI/CD pipelines for Azure functions and also how to use deployment scripts and custom steps.
*   **"Pro Git" by Scott Chacon and Ben Straub:** This book is a comprehensive guide to git. It provides in-depth knowledge about how to handle git authentication and other relevant git concepts.

In summary, while using private Git repositories for Azure Functions can be initially tricky, a good understanding of the deployment mechanisms and authentication methods allows you to seamlessly incorporate your private Python packages into your Azure Function deployments. Choose the method that best suits your needs while keeping security best practices in mind. I have personally found that using a service principal with an Azure DevOps pipeline (or other similar tools) provides the most robust solution overall.
