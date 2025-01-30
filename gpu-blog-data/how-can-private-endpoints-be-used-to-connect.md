---
title: "How can private endpoints be used to connect services within different resource groups?"
date: "2025-01-30"
id: "how-can-private-endpoints-be-used-to-connect"
---
Private endpoints offer a secure method for connecting virtual networks (VNets) residing within distinct Azure resource groups, circumventing the public internet. My experience implementing this in large-scale enterprise deployments has highlighted the critical role of careful VNet peering and service principal configuration.  The key fact to grasp is that private endpoints don't inherently bridge resource groups; rather, they leverage existing networking constructs to achieve secure, private connectivity between services, irrespective of their resource group affiliation.  Therefore, understanding VNet peering and appropriate access control is paramount.


**1. Clear Explanation**

The core functionality involves creating a private endpoint within a target VNet, pointing it at a specific service (e.g., a storage account, a SQL database, or a Kubernetes cluster) hosted in a different resource group.  This private endpoint acts as a private DNS entry, resolving to a private IP address within the target VNet.  Traffic destined for that service from within the source VNet is routed exclusively through the private endpoint, effectively bypassing the public internet.

Crucially, the source and target VNets must be able to communicate. This is typically achieved through VNet peering.  VNet peering establishes a transitive connection between the VNets, allowing traffic to flow between them as if they were a single network.  Once peering is established, the private endpoint’s private IP address within the target VNet becomes reachable from the source VNet.

Security is maintained by controlling access to the service through network security groups (NSGs) applied to the subnet containing the private endpoint and through Azure role-based access control (RBAC).  This ensures that only authorized virtual machines (VMs) or services within the source VNet can access the service via the private endpoint.  Furthermore, the service itself needs to be configured to allow access from the private endpoint’s private IP address.

Failing to correctly configure VNet peering, NSGs, and RBAC will result in connectivity issues.  During a previous engagement involving a multi-tenant SaaS solution, neglecting proper RBAC led to security vulnerabilities, requiring a significant remediation effort.  Hence, meticulous planning and rigorous testing are indispensable.



**2. Code Examples with Commentary**

The following examples use PowerShell.  Remember to replace placeholders like `<resourceGroupName>`, `<vnetName>`, `<subnetName>`, and `<serviceName>` with your actual values.

**Example 1: Creating a VNet Peering Connection**

```powershell
# Establish peering connection between source and target VNets
$sourceVNet = Get-AzVirtualNetwork -ResourceGroupName <sourceResourceGroupName> -Name <sourceVNetName>
$targetVNet = Get-AzVirtualNetwork -ResourceGroupName <targetResourceGroupName> -Name <targetVNetName>

$peeringConfig = New-AzVirtualNetworkPeeringConfig -Name "PeeringToTarget" -AllowVirtualNetworkAccess $true -AllowForwardedTraffic $false -AllowGatewayTransit $false -UseRemoteGateways $false
$peering = New-AzVirtualNetworkPeering -VirtualNetwork $sourceVNet -PeeringProperties $peeringConfig

Set-AzVirtualNetworkPeering -VirtualNetwork $sourceVnet -Name $peering.Name -PeeringProperties $peeringConfig

#Repeat the process, reversing source and target, to make the connection bidirectional (if needed)
```

This code snippet demonstrates the creation of a VNet peering connection.  Note the parameters `AllowVirtualNetworkAccess`, controlling whether VMs in one VNet can access VMs in the other. Setting  `AllowForwardedTraffic` and `AllowGatewayTransit` to `$false` is often a good practice for security.  Bidirectional peering might be necessary depending on the architecture.


**Example 2: Creating a Private Endpoint for a Storage Account**

```powershell
# Create a private endpoint for a storage account
$privateEndpointConfig = New-AzPrivateEndpointConfig -SubnetId "/subscriptions/<subscriptionId>/resourceGroups/<resourceGroupName>/providers/Microsoft.Network/virtualNetworks/<vnetName>/subnets/<subnetName>" -PrivateIpAddressVersion IPv4 -Name "privateEndpointStorage"
$privateEndpoint = New-AzPrivateEndpoint -ResourceGroupName <resourceGroupName> -Name "privateEndpointStorage" -Location "<location>" -PrivateEndpointConfig $privateEndpointConfig -VirtualNetworkId "/subscriptions/<subscriptionId>/resourceGroups/<targetResourceGroupName>/providers/Microsoft.Network/virtualNetworks/<targetVNetName>"

$privateLinkServiceConnection = New-AzPrivateLinkServiceConnection -Name "storageConnection" -GroupId "storage" -PrivateLinkServiceId "/subscriptions/<subscriptionId>/resourceGroups/<targetResourceGroupName>/providers/Microsoft.Storage/storageAccounts/<storageAccountName>"
Add-AzPrivateEndpoint -PrivateEndpoint $privateEndpoint -PrivateLinkServiceConnection $privateLinkServiceConnection
```

This example creates a private endpoint within the `<resourceGroupName>`  pointing to a storage account residing in `<targetResourceGroupName>`.  The crucial elements are specifying the correct subnet ID, private IP version, and the `PrivateLinkServiceId` which uniquely identifies the service.


**Example 3: Configuring NSGs**

```powershell
# Configure NSGs to allow traffic to the private endpoint
$nsg = Get-AzNetworkSecurityGroup -ResourceGroupName <resourceGroupName> -Name <nsgName>
$rule = New-AzNetworkSecurityRuleConfig -Name "AllowToPrivateEndpoint" -Access Allow -Protocol Tcp -Direction Inbound -Priority 100 -SourceAddressPrefix "*" -SourcePortRange "*" -DestinationAddressPrefix "<privateEndpointIP>" -DestinationPortRange "443"

Add-AzNetworkSecurityRule -NetworkSecurityGroup $nsg -NetworkSecurityRuleConfig $rule

```

This example shows how to add an inbound rule to an NSG to explicitly allow traffic to the private endpoint’s private IP address.  Remember to adjust the protocol, port range, and source address prefix to match your service's requirements.  Consider using service tags for more granular control instead of wildcard IP addresses whenever possible.  Implementing NSGs on the subnet of the private endpoint is a security best practice.


**3. Resource Recommendations**

For a deep dive into the intricacies of private endpoints, I suggest consulting the official Azure documentation.  Furthermore,  Microsoft Learn offers various modules and tutorials specifically designed for networking and security in Azure.  Finally, reviewing Azure's networking best practices document provides valuable guidance for designing robust and secure cloud deployments.  Supplementing this with practical experience through hands-on labs will consolidate your understanding and build your confidence in this crucial technology.
