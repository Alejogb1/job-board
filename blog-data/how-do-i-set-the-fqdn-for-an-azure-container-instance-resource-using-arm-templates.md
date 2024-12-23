---
title: "How do I set the FQDN for an Azure Container Instance resource using ARM templates?"
date: "2024-12-23"
id: "how-do-i-set-the-fqdn-for-an-azure-container-instance-resource-using-arm-templates"
---

Okay, let's tackle this. Over the years, I've wrestled—*ahem*, I mean, *dealt with*—my fair share of Azure deployments, and setting up container instances with the correct FQDN using ARM templates is a common scenario. It's not immediately obvious, and it often requires a little understanding of how Azure manages DNS. Let me walk you through it.

The challenge isn’t in the core container instance configuration itself, but in how we expose the application running within that instance to the outside world, and how we map a user-friendly domain name to it. The core problem is this: an Azure Container Instance (ACI) by itself doesn't automatically get an FQDN. It gets an IP, or if configured as a private instance, a private IP within a vnet, and we need a way to associate a name with that IP. We achieve this using the combination of a public IP address, a DNS label (which creates a subdomain within `azurecontainer.io`), and optionally, by bringing your own custom domain.

Let’s first discuss the DNS label which creates the `azurecontainer.io` subdomain. Within the ACI resource definition, we utilize the `ipAddress` property, specifically focusing on the `dnsNameLabel` field within the `ipAddress` property to define our FQDN, structured as `[dnsNameLabel].[region].azurecontainer.io`. The region comes from where the resource is deployed. This is good for getting up and running rapidly, but I wouldn't recommend using this in a production setting. If you choose to go that route, you'd include something like this in the `properties` section of your ACI resource definition in your ARM template:

```json
"properties": {
    "ipAddress": {
        "type": "Public",
        "ports": [
          {
            "protocol": "TCP",
            "port": 80
          }
        ],
        "dnsNameLabel": "my-demo-aci"
    },
    "containers": [
      {
        "name": "mycontainer",
        "properties": {
            "image": "nginx",
            "resources": {
                "requests": {
                    "cpu": 1,
                    "memoryInGB": 1
                }
            },
            "ports": [
              {
                  "port": 80
              }
            ]
        }
      }
    ],
    "osType": "Linux",
    "restartPolicy": "Never"
  }
```

In this snippet, "my-demo-aci" becomes part of the FQDN assigned to your ACI. Your actual FQDN will look something like `my-demo-aci.eastus.azurecontainer.io` (depending on your deployment region, of course). The 'eastus' portion will be specific to the region where you’re deploying the ACI.

However, the more robust and professional way is using a custom domain. This involves first creating an Azure Public IP resource. Then we'll attach that to your ACI during configuration and then finally, use that Public IP with a DNS service, either with Azure DNS or an external provider. Here’s an example of the Public IP resource definition in an ARM template. This is the IP address we’ll associate with our container instance:

```json
{
  "apiVersion": "2023-04-01",
  "type": "Microsoft.Network/publicIPAddresses",
  "name": "[parameters('publicIpName')]",
  "location": "[parameters('location')]",
  "sku": {
      "name": "Basic"
  },
  "properties": {
      "publicIPAllocationMethod": "Dynamic",
      "dnsSettings": {
          "domainNameLabel": "[parameters('dnsNameLabel')]"
      }
    }
}
```

Notice here we have a `domainNameLabel`. This is similar to the DNS name label we used earlier in the ACI properties, but it’s tied to the Public IP.

With the public IP defined, we can then modify the ACI definition. We will no longer use the `dnsNameLabel` of ACI instead reference the public ip by it's id.

```json
"properties": {
    "ipAddress": {
        "type": "Public",
        "ports": [
          {
            "protocol": "TCP",
            "port": 80
          }
        ],
        "ipAddress": "[resourceId('Microsoft.Network/publicIPAddresses', parameters('publicIpName'))]"
    },
    "containers": [
      {
        "name": "mycontainer",
        "properties": {
            "image": "nginx",
            "resources": {
                "requests": {
                    "cpu": 1,
                    "memoryInGB": 1
                }
            },
            "ports": [
              {
                  "port": 80
              }
            ]
        }
      }
    ],
    "osType": "Linux",
    "restartPolicy": "Never"
  }
```

We are now using the public ip resource by referencing it using `resourceId`. This is a far better way to manage and expose your application.

Finally, this is not a complete FQDN solution. The public IP we just created, while having a dns label, is still tied to an Azure managed domain. To get the truly custom FQDN that’s needed for production, you would then go to the DNS provider (Azure DNS or any third-party DNS provider) and create a `CNAME` record. This CNAME record would point your custom domain (e.g., `www.mycompany.com`) to the automatically generated domain from the public ip (`<dnsNameLabel>.<region>.cloudapp.azure.com`). So for this scenario, you will be creating a `CNAME` record that will point `www.mycompany.com` to something like `my-custom-dnslabel.eastus.cloudapp.azure.com`.

For a deeper understanding of networking in Azure, I highly recommend reading "Microsoft Azure Networking Cookbook" by Thomas W. Shinder and Michael G. Crump. It provides an excellent overview of Azure networking concepts, which is invaluable when deploying and configuring container instances in a production environment. Also, the official Azure documentation on Public IP addresses and DNS management is essential. Furthermore, the official Azure documentation on Azure Resource Manager templates is a must-have. These resources will explain the nuances of Azure networking much better than any brief snippet I can offer here.

Remember, when dealing with anything DNS related, patience is key. Propagation can take some time. Don't start changing everything without verifying the basics first. These examples should give you a solid foundation for setting up your FQDN for your Azure Container Instances with both simple and complex scenarios. I’ve found that understanding the underlying principles of how Azure manages DNS is essential for a successful implementation and debugging.
