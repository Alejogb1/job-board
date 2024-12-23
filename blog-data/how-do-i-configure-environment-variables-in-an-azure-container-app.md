---
title: "How do I configure environment variables in an Azure Container App?"
date: "2024-12-23"
id: "how-do-i-configure-environment-variables-in-an-azure-container-app"
---

Okay, let's tackle this. Environment variables in Azure Container Apps, that’s a topic I’ve encountered numerous times, particularly when migrating existing applications from more traditional platforms. I've seen first-hand how crucial they are for managing configurations, especially when dealing with microservices and ensuring consistent behavior across different deployment stages. Let me walk you through the process and share some insights I've picked up.

At its core, Azure Container Apps relies on Kubernetes under the hood. However, you don't directly interact with Kubernetes configuration files. Instead, Microsoft abstracts this complexity, offering a streamlined interface for managing your containerized applications, including how you handle environment variables. There are several ways to configure these, and the method you choose often depends on the complexity and specific needs of your deployment.

The simplest and most common method is through the Azure Portal. Navigating to your Container App resource, you'll find a 'Container' section, and within that, an 'Environment variables' area. This offers a UI to add or modify variables using key-value pairs. While convenient, it's less suited for infrastructure-as-code approaches and complex scenarios involving many variables. For development and quick tests, it's perfectly adequate. I've used this method frequently for debugging and rapid prototyping.

For production environments or more complex setups, using Azure CLI or ARM/Bicep templates is far more robust and reproducible. I strongly advocate for infrastructure-as-code principles, which significantly reduce human errors and streamline the deployment process. When using the CLI, the `az containerapp update` command is your friend. You can modify the environment variables using a JSON array that includes the keys `name` and `value`.

Here’s an example of how I've structured a typical command, assuming I need to set a database connection string and an API key:

```bash
az containerapp update \
  --name my-container-app \
  --resource-group my-resource-group \
  --set properties.configuration.secrets=[] \
  --set properties.template.containers[0].env="[{\"name\":\"DATABASE_URL\", \"value\":\"mydbserver.database.windows.net\"}, {\"name\":\"API_KEY\", \"value\":\"some-secure-api-key\"}]"
```

Notice that I first clear existing secrets with `--set properties.configuration.secrets=[]` which is a necessary safety precaution to remove any accidental old settings, then the `--set properties.template.containers[0].env` adds the variables. This snippet directly modifies the container app configuration, which results in a new revision being deployed. Remember, each change in container app configuration leads to a new revision being created. This process helps with managing different releases of your app.

Now, let's talk about secrets. You wouldn't want to hardcode sensitive information like API keys or database passwords directly in the configuration. That's where secrets management comes in. Azure Container Apps allows you to define secret values, and these can be referenced as environment variables. You can define the secret either directly through the Azure portal by specifying the name and value or through the `az containerapp update` command using the `--set properties.configuration.secrets` option. In my experience, I prefer using Azure Key Vault to manage secrets in production. This enhances security and offers more robust control over secret lifecycles. The process involves first storing the secrets in Azure Key Vault and then referencing them as environment variables in your container app.

Let's imagine I have the same database connection string, but I want to store it securely in Key Vault. Here’s the command structure, assuming that we've created a secret in Azure Key Vault named `database-secret`:

```bash
az containerapp update \
    --name my-container-app \
    --resource-group my-resource-group \
    --set properties.configuration.secrets="[{\"name\":\"database-secret\", \"value\":\"/subscriptions/your_subscription_id/resourcegroups/your_resource_group/providers/microsoft.keyvault/vaults/your_key_vault_name/secrets/database-secret\"}]" \
    --set properties.template.containers[0].env="[{\"name\":\"DATABASE_URL\", \"secretRef\":\"database-secret\"}, {\"name\":\"API_KEY\", \"value\":\"some-secure-api-key\"}]"
```

Key differences are in that here we create the secret with a reference to the keyvault secret, and then use `secretRef` to map to the environment variable. The container app will now fetch the value during runtime. The key point here is that you need to have appropriate access policies set on the key vault so your container app can actually retrieve the secret. This setup enhances security, separating the configuration from the actual values.

Finally, consider that when building pipelines for your container apps, you'll find that ARM/Bicep templates offer a repeatable and declarative approach. Using such templates, you would define `env` and `secrets` under `properties.template.containers` and `properties.configuration`, respectively. Here's a condensed snippet of how this would appear in a Bicep template:

```bicep
resource containerApp 'Microsoft.App/containerApps@2023-05-01' = {
  name: 'my-container-app'
  location: location
  properties: {
    configuration: {
      secrets: [
        {
          name: 'database-secret'
          value: '/subscriptions/your_subscription_id/resourcegroups/your_resource_group/providers/microsoft.keyvault/vaults/your_key_vault_name/secrets/database-secret'
        }
      ]
    }
    template: {
        containers: [
           {
              name: 'my-container'
              image: 'myregistry.azurecr.io/my-image:v1'
              env: [
               {
                 name: 'DATABASE_URL'
                 secretRef: 'database-secret'
               }
               {
                   name: 'API_KEY'
                   value: 'some-secure-api-key'
               }
             ]
           }
        ]
      }
  }
}
```

This snippet illustrates how you’d define environment variables and secrets within a Bicep template, which would then be deployed by Azure Resource Manager. This approach helps in versioning your infrastructure configurations, and keeps consistent deployments.

To delve deeper into these topics I'd recommend a few resources: specifically the official Microsoft Azure documentation on Container Apps, including articles focusing on configuration and secrets management. Also, I find 'Kubernetes in Action' by Marko Luksa to be particularly helpful for understanding the underlying concepts even though you're not directly working with Kubernetes manifests. Understanding these deeper principles can be beneficial, even with the abstractions that container apps provide. Another valuable resource is the 'Cloud Native Patterns' book by Cornelia Davis, it offers excellent insights into designing and deploying cloud applications including configuration techniques.

In summary, while the Azure Portal is convenient for simple tasks, adopting Azure CLI and ARM/Bicep templates facilitates a more manageable and secure approach for configuring environment variables in Azure Container Apps, especially for production. It allows for the application of infrastructure-as-code principles, making the deployment process repeatable, consistent, and less prone to errors. Remember to separate sensitive information into Azure Key Vault and then reference them appropriately in your deployments to enhance overall security.
