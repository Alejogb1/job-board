---
title: "How do I connect to Kafka running in Azure Container Instance from outside?"
date: "2024-12-23"
id: "how-do-i-connect-to-kafka-running-in-azure-container-instance-from-outside"
---

Alright, let's talk about connecting to Kafka running in Azure Container Instances (ACI) from external networks. This isn't exactly a plug-and-play scenario, and I've definitely seen my share of head-scratching moments tackling this in past projects. Remember that time at *SynapseTech*? We had a similar setup, but our initial approach was... let's just say, less than optimal. We were battling network config issues for days. The short version is, you're dealing with the inherent isolation of ACI and need a strategic approach to bridge that gap.

The core problem boils down to ACI's network isolation. By default, an ACI instance is not directly accessible from outside its virtual network. To establish external connectivity to Kafka, we need to expose the Kafka brokers, and that typically means routing network traffic through another Azure service designed to handle this kind of external exposure. There are a few valid paths, each with its trade-offs, but the main ones I've consistently found success with involve using an Azure Load Balancer or an Azure Application Gateway. I'll lean into the Load Balancer approach initially because it’s generally simpler and more cost-effective for straightforward TCP-based communication.

Let's start by assuming your Kafka cluster is up and running within your ACI group and using a well-defined internal port (say, `9092`). Now, to expose this service, we’ll deploy an Azure Load Balancer in front of the ACI instances. Here’s how that generally works: we create a load balancer with a public IP address, configure backend pools pointing to your ACI instance(s), and set up health probes to ensure the load balancer only directs traffic to healthy instances. Lastly, we configure load balancing rules to forward traffic on the specified external port to the internal port of the target container. Crucially, for Kafka to function correctly you need to properly handle port mapping, and ensure both the internal and external port information is correctly configured in Kafka's configuration files. Otherwise, you may end up with the client connecting to the load balancer, but Kafka will advertise itself with the internal IP address and port which the client cannot access.

For a basic configuration, the Azure load balancer will expose a single IP address, and clients outside can connect to that public IP address using the defined port. The load balancer will then take care of routing it internally. The key configuration components typically involve: front end IP address configuration (public IP address), backend pool (that contains your ACI instance), health probes and load balancing rules.

Now, let’s get practical. Here’s a code snippet using Azure CLI, illustrating how to create such a setup. While this code is simplified for demonstration, it illustrates the core concepts and should be used as a template for your specific needs. We'll assume you've created your ACI instance and have the resource group and other necessary details already established.

```bash
# --- Create a Resource Group (if you haven't already) ---
# az group create --name myResourceGroup --location eastus
# --- Get the Resource Group Name --
RESOURCE_GROUP_NAME="myResourceGroup"
# --- Create a Public IP Address ---
PUBLIC_IP_NAME="myPublicIP"
az network public-ip create \
  --resource-group $RESOURCE_GROUP_NAME \
  --name $PUBLIC_IP_NAME \
  --sku Standard
# --- Retrieve Public IP Address
PUBLIC_IP=$(az network public-ip show --resource-group $RESOURCE_GROUP_NAME --name $PUBLIC_IP_NAME --query ipAddress --output tsv)
echo "Public IP address is: $PUBLIC_IP"

# --- Create a Load Balancer ---
LOAD_BALANCER_NAME="myLoadBalancer"
az network lb create \
    --resource-group $RESOURCE_GROUP_NAME \
    --name $LOAD_BALANCER_NAME \
    --sku Standard \
    --frontend-ip-configurations "{\"name\": \"$PUBLIC_IP_NAME\", \"publicIpAddress\": \"$PUBLIC_IP\"}"
# --- Add Load Balancer Backend Pool, point this to the ACI ---
ACI_IP="<Your ACI private IP address>"
az network lb address-pool create \
   --resource-group $RESOURCE_GROUP_NAME \
   --lb-name $LOAD_BALANCER_NAME \
   --name "backendPool"
az network lb address-pool address add \
   --resource-group $RESOURCE_GROUP_NAME \
    --lb-name $LOAD_BALANCER_NAME \
   --pool-name "backendPool" \
   --backend-ip-address $ACI_IP

# --- Add Health Probe ---
az network lb probe create \
  --resource-group $RESOURCE_GROUP_NAME \
  --lb-name $LOAD_BALANCER_NAME \
  --name "tcpProbe" \
  --protocol Tcp \
  --port 9092

# --- Configure Load Balancing Rule (assuming external port 9092) ---
az network lb rule create \
    --resource-group $RESOURCE_GROUP_NAME \
    --lb-name $LOAD_BALANCER_NAME \
    --name "kafkaRule" \
    --protocol Tcp \
    --frontend-port 9092 \
    --backend-port 9092 \
    --frontend-ip-name $PUBLIC_IP_NAME \
    --backend-pool-name "backendPool" \
    --probe-name "tcpProbe"
```

In the script above, replace `<Your ACI private IP address>` with the actual private IP address of your ACI instance running Kafka. Make sure the TCP probe points to the correct port on the ACI instance.

You'll likely need to adjust Kafka's `server.properties` configuration to listen on the correct interface and, importantly, advertise the correct broker address. For example, the following configurations are key to setting up a listener. Replace `<Your ACI private IP address>` with the ACI instance private ip.

```properties
listeners=PLAINTEXT://0.0.0.0:9092
advertised.listeners=PLAINTEXT://<Your ACI private IP address>:9092
```

It's critical that `advertised.listeners` uses either the public IP of the load balancer or an FQDN that resolves to the public IP, and it’s something that we learned by almost running into a wall on that *SynapseTech* project. The internal address will not work as the client cannot resolve it, and this will prevent the client from connecting and consuming.

Here's another code snippet, this time showing a slightly different approach by employing the Azure application gateway which is more robust and provides more powerful capabilities:

```bash
# --- Create a Public IP Address (if needed) ---
# (already created in previous example, if you're continuing, you can skip)
PUBLIC_IP_NAME="myPublicIP"
az network public-ip show --resource-group $RESOURCE_GROUP_NAME --name $PUBLIC_IP_NAME --query ipAddress --output tsv

# --- Create a Virtual Network and Subnet ---
VNET_NAME="myVnet"
SUBNET_NAME="mySubnet"
az network vnet create \
    --resource-group $RESOURCE_GROUP_NAME \
    --name $VNET_NAME \
    --address-prefixes 10.0.0.0/16 \
    --subnet-name $SUBNET_NAME \
    --subnet-prefixes 10.0.1.0/24

VNET_ID=$(az network vnet show --resource-group $RESOURCE_GROUP_NAME --name $VNET_NAME --query id --output tsv)
SUBNET_ID=$(az network vnet subnet show --resource-group $RESOURCE_GROUP_NAME --vnet-name $VNET_NAME --name $SUBNET_NAME --query id --output tsv)

# --- Create an Application Gateway ---
APP_GW_NAME="myAppGateway"
az network application-gateway create \
    --resource-group $RESOURCE_GROUP_NAME \
    --name $APP_GW_NAME \
    --sku Standard_v2 \
    --public-ip-address $PUBLIC_IP_NAME \
    --vnet-name $VNET_NAME \
    --subnet $SUBNET_ID \
    --http-settings-protocol Http \
    --port 8080 \
    --capacity 2
# --- Create the Backend Pool pointing to ACI instance (adjust as needed)
# here the backend is assumed to be http since we're not terminating TLS at the LB.
az network application-gateway address-pool create \
    --resource-group $RESOURCE_GROUP_NAME \
    --gateway-name $APP_GW_NAME \
    --name "backendPool"
# add the ip of the container instance where Kafka resides.
az network application-gateway address-pool address add \
    --resource-group $RESOURCE_GROUP_NAME \
    --gateway-name $APP_GW_NAME \
    --pool-name "backendPool" \
    --backend-ip-address $ACI_IP
# --- Create an HTTP Listener
az network application-gateway http-listener create \
    --resource-group $RESOURCE_GROUP_NAME \
    --gateway-name $APP_GW_NAME \
    --name "httpListener" \
    --frontend-ip-name "appGatewayFrontendIP" \
    --frontend-port 8080
# --- Create Routing Rule for kafka port using http as protocol
az network application-gateway rule create \
    --resource-group $RESOURCE_GROUP_NAME \
    --gateway-name $APP_GW_NAME \
    --name "kafkaRule" \
    --rule-type Basic \
    --http-listener "httpListener" \
    --backend-address-pool "backendPool" \
    --backend-http-settings "appGatewayHttpSettings"
```
While this looks similar, the application gateway approach is more involved, especially as it requires you to configure HTTP settings, as it was built to handle http workloads. In this case, we are bypassing the HTTP layer and using plain TCP so we don’t introduce additional complexity. But Application Gateway provides additional features and options, such as TLS termination, web application firewall (WAF), and so on. For many production cases these are requirements, which is why this was included here.

Finally, another approach that can be used involves setting up an SSH tunnel which acts as a secure proxy. This method however, involves another setup in an instance, to handle the tunneling, so this is not ideal for production and more suitable for local testing purposes only. In this example, I am setting the ssh connection from the outside to a dedicated virtual machine (bastion) in Azure which is in the same vnet as the ACI instance, and from there tunnelling the traffic to the container instance.

```bash
# Get the ACI private ip address (from above)
ACI_IP="<Your ACI private IP address>"
# Replace <username> with the username for your bastion instance
ssh -i <your-ssh-private-key> -L 9092:$ACI_IP:9092 <username>@<bastion-public-ip>
```
This command sets up an ssh tunnel to the bastion machine and forwards traffic from your local port 9092 to the container port 9092. You'll need to replace the place holders to match your environment. This method is quite useful for debugging and initial testing.

For more in-depth knowledge, I'd suggest going through the official Azure documentation for Load Balancer and Application Gateway. Also, the "Kafka: The Definitive Guide" by Neha Narkhede, Gwen Shapira, and Todd Palino is indispensable for understanding Kafka's internal workings. Specifically pay attention to the listener configurations section, as it's frequently a pain point for those starting out with Kafka. "Understanding Networking in Azure" (available on Microsoft Learn) is also a good foundational resource to strengthen your understanding of the underlying concepts.

Keep in mind, security is paramount. Ensure that your load balancer or application gateway has proper access controls, and that your Kafka instances are not exposed to the open internet without adequate protection (e.g., TLS). Properly configure firewalls, both at the Azure level and within the ACI environment.

In short, the key is understanding the networking concepts in Azure and carefully configuring your load balancer or application gateway to properly bridge the external clients with your internal Kafka cluster running in the ACI. It’s definitely not a one-size-fits-all solution, but these approaches should give you a solid foundation. Good luck!
