---
title: "How can I build a Docker Compose stack using Azure Container Instances?"
date: "2024-12-23"
id: "how-can-i-build-a-docker-compose-stack-using-azure-container-instances"
---

Let's tackle this, shall we? I've spent a fair bit of time navigating the intersection of docker compose and cloud container services, and, specifically, Azure container instances (aci). It's not a direct mapping, as you'll discover, but there are workarounds and strategies that get you surprisingly close to that familiar docker compose workflow. I recall a particularly challenging project a couple of years back, where we had to rapidly deploy a multi-container application for a proof of concept, and we absolutely needed a lean, serverless option—that's when we really dove deep into this area.

The critical thing to understand is that docker compose is, fundamentally, designed for local development or orchestrated environments like docker swarm or kubernetes. Aci, conversely, is a container-as-a-service offering with different constraints and operational models. Directly deploying a `docker-compose.yml` isn't natively supported. Instead, we must translate the concepts embedded within that yaml file into individual aci deployments, often orchestrated by other azure tools.

The core challenge is the translation of inter-container communication and dependency management that docker compose handles elegantly. With aci, you're dealing with individual container instances, so you're responsible for providing connectivity yourself. Let’s explore a few ways we can accomplish this, starting with the most straightforward and then moving to more sophisticated options.

Firstly, consider the simple scenario: you have a web application, and a database declared in your `docker-compose.yml`. Typically, the compose file sets up the network connections. In aci, you'd have to define these manually, either by using azure virtual networks or, in a basic scenario, relying on exposed ports and external connectivity. Let's represent that in a working example.

Assume a standard `docker-compose.yml`:

```yaml
version: "3.8"
services:
  web:
    image: nginx:latest
    ports:
      - "80:80"
  db:
    image: postgres:13
    environment:
      POSTGRES_USER: user
      POSTGRES_PASSWORD: password
      POSTGRES_DB: my_database
    ports:
       - "5432:5432"
```

Now, if we were to translate that into aci-friendly commands, using the azure cli, it would look like this:

```bash
# deploy postgres instance
az container create \
  --resource-group myresourcegroup \
  --name mydatabase \
  --image postgres:13 \
  --environment-variables POSTGRES_USER=user POSTGRES_PASSWORD=password POSTGRES_DB=my_database \
  --ports 5432 \
  --cpu 1 --memory 1 \
  --no-wait

# deploy the web instance
az container create \
  --resource-group myresourcegroup \
  --name mywebserver \
  --image nginx:latest \
  --ports 80 \
  --cpu 1 --memory 1 \
  --no-wait
```

This is a simplistic example and it's important to emphasize that each container is exposed as a public ip in this case, and there’s no direct internal networking. It should be understood that exposing the database publicly, as in this demonstration, is a poor security practice. In reality, you'd ideally place these into a virtual network, where the web container could then reach the database using a private ip address. We’re simplifying to keep the illustration manageable. This first approach does not offer the same internal network communication of docker compose. You would need to adjust your web service's configuration to use the database's public ip or, if in a vnet, it’s private ip. This approach also lacks any sort of orchestration or dependency management between the web container and database containers; they are independent deployments.

A slightly more nuanced approach would be to utilize Azure Resource Manager (arm) templates or bicep templates. This offers repeatability and enables us to express our deployment declaratively which is more closer to the spirit of a `docker-compose.yml`.

Here’s a simplified bicep example of that same setup:

```bicep
resource webcontainer 'Microsoft.ContainerInstance/containerGroups@2023-05-01' = {
  name: 'mywebcontainer'
  location: resourceGroup().location
  properties: {
    osType: 'Linux'
    containers: [
      {
        name: 'nginx'
        properties: {
          image: 'nginx:latest'
          resources: {
            requests: {
              cpu: 1
              memoryInGB: 1
            }
          }
          ports: [
            {
              port: 80
            }
          ]
        }
      }
    ]
    ipAddress: {
      ports: [
        {
          port: 80
          protocol: 'TCP'
        }
      ]
       type: 'Public'
    }
    restartPolicy: 'Never'
  }
}


resource dbcontainer 'Microsoft.ContainerInstance/containerGroups@2023-05-01' = {
    name: 'mydatabasecontainer'
    location: resourceGroup().location
    properties: {
        osType: 'Linux'
        containers:[
            {
                name: 'postgres'
                properties: {
                    image: 'postgres:13'
                    environmentVariables: [
                        {
                           name: 'POSTGRES_USER'
                           value: 'user'
                        }
                       {
                           name: 'POSTGRES_PASSWORD'
                           value: 'password'
                       }
                       {
                           name: 'POSTGRES_DB'
                           value: 'my_database'
                        }
                    ]
                  resources:{
                      requests:{
                           cpu: 1
                           memoryInGB: 1
                      }
                    }
                     ports: [
                       {
                           port: 5432
                       }
                     ]
                  }
               }
          ]
      ipAddress: {
        ports: [
        {
          port: 5432
          protocol: 'TCP'
        }
      ]
         type: 'Public'
    }
         restartPolicy: 'Never'
    }
}
```

This bicep template provides a more structured way to deploy your containers. You could then deploy it with the azure cli by using `az deployment group create`. Though this is a bit more verbose, it is considerably more maintainable over the cli commands as you scale. It also allows you to incorporate parameters and reuse the template for other projects. It still, however, exposes containers publicly and does not implement the networking and dependency management of docker compose.

A more robust approach involves using Azure Container Apps (aca), which are built on top of kubernetes and offer features that can simplify this. With aca, you can group container instances and leverage its environment features to setup internal networking similar to docker compose networks and implement ingress options. aca would be a closer solution to a `docker compose up` implementation. While not using aci directly, it handles inter-container communication more elegantly. To illustrate, I’ll demonstrate an equivalent implementation using the azure cli that creates a simple container app with two containers within.

```bash
# Create a container apps environment
az containerapp env create \
  --name myenv \
  --resource-group myresourcegroup \
  --location eastus

# Create a web container app
az containerapp create \
  --name mywebapp \
  --resource-group myresourcegroup \
  --environment myenv \
  --image nginx:latest \
  --ingress 'external' \
  --target-port 80

# Create a database container app (no public ingress)
az containerapp create \
  --name mydbapp \
  --resource-group myresourcegroup \
  --environment myenv \
  --image postgres:13 \
  --environment-variables POSTGRES_USER=user POSTGRES_PASSWORD=password POSTGRES_DB=my_database \
  --target-port 5432 \
  --ingress 'none'
```

In this example, the web application is exposed publicly, and can access the db container on a internal dns name. The networking is handled by azure container apps and is significantly more robust. The inter-container networking is abstracted, and you do not need to use any external ips which were previously necessary in our simpler examples.

For further reading and deep dives into this, I recommend the official Azure documentation for both Azure Container Instances and Azure Container Apps. The "Programming Azure" book by Michael Collier and Robin Shahan is also an invaluable resource if you need to delve deeper into ARM templates and Bicep. Lastly, "Kubernetes in Action" by Marko Luksa, is helpful for understanding the underlying concepts of Azure Container Apps.

In conclusion, while you cannot directly use `docker compose` with aci, you can emulate its functionality through a mixture of cli commands, arm/bicep templates, and utilizing services like azure container apps, each with their own nuances. Understanding their limitations and strengths is essential for a successful deployment. Hopefully, these examples and references are helpful, and best of luck applying these approaches!
