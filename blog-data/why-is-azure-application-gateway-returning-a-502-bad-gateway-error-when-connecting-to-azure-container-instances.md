---
title: "Why is Azure Application Gateway returning a 502 Bad Gateway error when connecting to Azure Container Instances?"
date: "2024-12-23"
id: "why-is-azure-application-gateway-returning-a-502-bad-gateway-error-when-connecting-to-azure-container-instances"
---

Okay, let's delve into this 502 error situation with Azure Application Gateway and Azure Container Instances (ACI). It’s a fairly common headache, and I’ve seen it firsthand multiple times, often during the pressure cooker of deployment phases. The root cause, as is typical, isn't usually a single issue, but rather a confluence of potential misconfigurations or limitations. I'm not going to pretend it’s always straightforward, but I can break down the core reasons based on my experiences and provide some troubleshooting steps with code examples.

The core problem we are observing is a 502 'Bad Gateway' error originating from the Azure Application Gateway. This error essentially signifies that the gateway, which acts as a reverse proxy in front of our ACI instances, cannot establish a successful connection to the backend service – in this case, your containerized application running within ACI. Now, let’s pinpoint the common culprits:

**1. Network Configuration and Reachability**

Perhaps the most frequent issue lies in network connectivity. Application Gateway needs to be able to resolve and communicate with the backend IPs of the ACI instances. This seems basic, but various roadblocks can appear. I’ve often found cases where:

*   **ACI Instances are not in a subnet accessible by the Application Gateway:** If your ACI instances are deployed in a virtual network (vnet) that’s not accessible or is blocked by a network security group (NSG) rule to the subnet where the Application Gateway resides, the gateway won't establish a connection. They might be in different vnets entirely, or even in the same vnet but behind firewalls or routing configurations that the app gateway is not configured to traverse.
*   **Incorrect Backend Pool Settings:** The Application Gateway has a "backend pool" section where you define the endpoints (IP addresses or FQDNs) of your backend services. If the IP addresses listed here do not match the actual private IP addresses of your ACI instances, the gateway will fail. Dynamic IP assignments for your ACIs, if not handled correctly, can cause frequent mismatches. You might think you are pointing to an ACI instance, but the IP has changed, or the configuration points to a different backend altogether.
*   **DNS Resolution Issues:** If you're using FQDNs rather than private IPs for your ACI backend pool endpoints, ensure that the Application Gateway can actually resolve those FQDNs, and they correctly point to the assigned private IPs of the ACI instances. This is a particular issue in environments with customized DNS resolution and non-standard setups. I've encountered misconfigurations here where custom DNS servers were not being accessed by the Application Gateway.
*  **Private Endpoints Misconfiguration** If the ACIs are only accessible via private endpoints, ensuring the app gateway is configured to communicate with these endpoints is critical. Any misconfiguration in the private endpoint integration will lead to connectivity issues.

**2. Application Health Probes**

Application Gateway relies heavily on health probes to verify backend instance health. These probes send HTTP or HTTPS requests to your ACI instances to see if they’re responding correctly. Issues here can include:

*   **Incorrect Probe Configuration:** I’ve noticed many scenarios where the specified path on the health probe is incorrect or does not exist, or the port configured is not correct for the application. This leads to the probe continuously failing, which results in Application Gateway marking the ACI instance as unavailable, even if it's running correctly. The probe could also be failing because it is looking for `HTTP 200 OK`, and the server is only returning another status code, like `HTTP 202`.
*   **Network Issues Impacting Probes:** Even if the health probe path is correctly specified, network issues (such as routing misconfiguration) can prevent probes from reaching the ACI container's endpoint, leading to the 502 error. This might be intermittent, which is even harder to troubleshoot.
*   **Application Unresponsive:** If the application itself is unresponsive or failing to start properly within the ACI, it will naturally fail health probes. So it's always good to take a look at the logs in your container first, to make sure it's not failing to start correctly.

**3. Application Gateway Limits and Resources**

While less common, resource limits and incorrect settings on Application Gateway itself could cause these errors:

*   **Request Timeouts:** The Application Gateway has timeout settings which, if set too low, can lead to failures if the ACI container takes too long to respond. This can happen if an operation within the container takes longer than the set gateway timeout.
*   **Insufficient Application Gateway Resources:** If your ACI application handles a high load, the Application Gateway might struggle if it does not have the appropriate size/sku, and you might experience performance issues like gateway timeouts and 502s. This has happened to me before when we underestimated how much load the application was going to handle, and that impacted the throughput of the app gateway.
* **Connection Limits** Application Gateway has limits on the number of simultaneous connections that it will allow to each backend. If there are too many requests coming through or long-lived connections, you can hit these limits and see a 502.

Let’s look at some code snippets to illustrate some of the common issues I've described. Please remember that these are examples and may need adjustment based on your specifics:

**Example 1: Configuring Application Gateway Backend Pool using Azure CLI**

This example assumes you have an existing App Gateway and an ACI instance.

```bash
# Assuming Resource Group: myResourceGroup, App Gateway: myAppGateway, and ACI instance has a Private IP $aci_private_ip

aci_private_ip=$(az container show --resource-group myResourceGroup --name myAciInstance --query ipAddress.ip --output tsv)

az network application-gateway address-pool create \
  --gateway-name myAppGateway \
  --resource-group myResourceGroup \
  --name aciBackendPool \
  --servers $aci_private_ip
```

This snippet gets the private IP address of your ACI instance and adds it as a backend server to your Application Gateway's address pool. If you are experiencing connection problems, verify that this is the correct IP address in your Application Gateway configuration. I’ve had scenarios where the IP had changed after the ACI instance was restarted.

**Example 2: Configuring Health Probe with Azure CLI**

Here, we’ll create a custom health probe that checks the `/health` endpoint:

```bash
# Assuming Resource Group: myResourceGroup, App Gateway: myAppGateway, and you want a health check on the /health endpoint. Using HTTP on port 80

az network application-gateway probe create \
  --gateway-name myAppGateway \
  --resource-group myResourceGroup \
  --name aciHealthProbe \
  --protocol Http \
  --host "127.0.0.1" \ # For HTTP probes, the actual backend IP is used.
  --path /health \
  --port 80
```
This example creates a health probe, targeting port 80 and the `/health` endpoint. When troubleshooting, it's vital to confirm if this path is correct and reachable. Sometimes, the application might expose a health endpoint at a different location, like `/healthz`. Furthermore, always ensure that the application responds to the health check with a `200 OK` status code.

**Example 3: Checking the ACI Logs**

To get the logs from ACI, you can use this command:

```bash
az container logs --resource-group myResourceGroup --name myAciInstance
```

I've often found that the 502 was only caused because the container was simply not starting up correctly and was thus not responding to probes. This command will surface any start-up errors the container has. Ensure that your container starts without errors and is listening on the correct port specified in the application gateway and its health probe configuration.

**Recommended resources for deeper learning:**

For a more comprehensive understanding, I'd suggest exploring these specific resources:

*   **"Azure Application Gateway documentation"**: Look at Microsoft's official documentation for the best explanation of all settings and how to best troubleshoot gateway related issues.
*   **"Container Networking in Azure"** You will find much deeper dive into ACI networking and concepts, so you can understand how it interacts with other services and components, such as app gateways.
*   **"Troubleshooting Application Gateway"** I recommend finding specific articles on how to troubleshoot Azure Application Gateways, as this will be specifically aligned with the kinds of problems I've outlined in this response.

In summary, diagnosing 502 errors with Application Gateway and ACI requires a methodical approach. Check your network configuration, backend pool settings, health probes, and resource limitations. Always examine the ACI container logs as your first port of call and ensure your application is actually working before trying to troubleshoot connectivity. This error is almost always one of the above issues, and I’ve personally found that systematic review and checking all of them will help you get to the root cause.
