---
title: "Why is Server 2019 failover cluster not responding correctly when forwarding public traffic through the firewall?"
date: "2024-12-23"
id: "why-is-server-2019-failover-cluster-not-responding-correctly-when-forwarding-public-traffic-through-the-firewall"
---

Let's get down to it then, shall we? I recall a rather... *memorable* situation a few years back involving a windows server 2019 failover cluster that exhibited precisely the behavior you've described. We were migrating a fairly critical application, and during the testing phase, the cluster would just... *stop responding* to external requests when those requests transited our firewall. Internally, things were fine; the nodes communicated flawlessly, application instances migrated successfully – the whole shebang behaved perfectly. But, as soon as public traffic flowed through the firewall, it would all go pear-shaped. The core of the issue, as it often is, resided in a layered problem. It wasn't just a single, easily isolated configuration error.

The first layer was, surprisingly, not the firewall itself, but a fundamental misunderstanding of how windows failover clustering handles network address translation (NAT) and port forwarding, particularly when dealing with public IP addresses. We'd initially set up the cluster using private ip addresses for node-to-node communication and a separate virtual ip for the cluster service itself. We then configured the firewall to forward incoming traffic (ports 80 and 443) to that cluster ip. What we failed to fully grasp initially was how the cluster was responding to the source ip of the client connecting from the public internet – an IP address that didn't directly correspond to any addresses within the cluster subnet. This was crucial.

The failover cluster relies on windows-based network name resolution and registration which, by default, assumes clients connecting to the cluster are doing so from the *same* subnet as the cluster nodes. The incoming traffic, having been translated through the firewall, had a source IP address different than the private IPs used by the cluster. The cluster, when attempting to respond, would often try to direct responses using that source IP, which then became an invalid target. This essentially resulted in dropped packets and the feeling that the cluster wasn’t responding. The firewall wasn't technically blocking, but rather the cluster was routing responses to a place that made little sense given the NAT between the cluster and internet. It's not about blocked traffic; it’s about misdirected responses.

The second layer revolved around how windows failover clustering handles its resource groups and network adapters. Each resource group, for example an application service, gets its own virtual ip address. This virtual ip moves to the node holding the active role of that resource group. When the firewall forwards traffic to the cluster’s ip, it isn't forwarding directly to a node’s ip but to one of these resource group vips. If the resource group was actively on a given node, the client would successfully connect through the firewall's port forwarding. However, the source address mismatch, as mentioned before, would still lead to routing issues, often causing intermittent connections.

The third component was related to the inherent complexities of network load balancing across failover clusters and firewall rules. The cluster's virtual ip address was used as the destination of the forwarded traffic, but we had not created specific rules on the nodes or the cluster itself to handle traffic sourced from an IP address outside of its internal network range. Windows failover cluster, by default, doesn't automatically compensate for NAT traversal; it needs to be guided.

Okay, so how did we get it all working, finally? There isn't a single magic bullet; it's a combination of tweaks. First, on the cluster nodes, we implemented what's called a "loopback adapter." This acts like an internal virtual interface that we bound to a subnet matching the public facing network (as seen by the NAT gateway). This was critical to create a reference point within the cluster for the public IP source addresses of clients. Second, we used PowerShell to create firewall rules on each cluster node that specifically allowed inbound traffic on the forwarded ports, not just from private subnets, but from the *entire internet*, but only for the *cluster’s virtual ip address*. We also adjusted the cluster configuration to use the newly created loopback ip for the cluster’s response and outbound connections. Lastly, we created a dedicated routing rule within the cluster to handle outbound responses destined for the firewall and to use the external network’s IP address as the source.

Here are three code snippets demonstrating the main elements of these configurations:

**Snippet 1: Creating a Loopback Adapter (powershell, run as admin)**

```powershell
# Install a loopback adapter with a specific name and an internal ip
New-NetAdapter -Name "Loopback Adapter Public" -InterfaceDescription "Loopback adapter for public traffic"
$loopbackAdapter = Get-NetAdapter -Name "Loopback Adapter Public"
$loopbackAdapter | New-NetIPAddress -IPAddress 192.168.1.250 -PrefixLength 24
$loopbackAdapter | Get-NetAdapterAdvancedProperty -RegistryKeyword *offload* | set-netadapteradvancedproperty -displayvalue "disabled" # Disable all offloads to prevent interference
```

*   This code creates a new loopback adapter named "loopback adapter public", assigns it an internal ip, and disables offloading to prevent hardware-based conflict. The IP address (192.168.1.250) should reside on a subnet that matches the firewall's external interface subnet, but should not conflict with other real-world addresses.

**Snippet 2: Adding Firewall Rules on Each Node (powershell, run as admin)**

```powershell
# assuming the cluster virtual ip is 10.0.0.100
$clusterVIP = "10.0.0.100"
# Add a firewall rule to allow HTTP (port 80) from any IP to our cluster virtual IP
New-NetFirewallRule -DisplayName "Allow HTTP Cluster Traffic" -Direction Inbound -Action Allow -Protocol TCP -LocalPort 80 -LocalAddress $clusterVIP -RemoteAddress Any
# Add a firewall rule to allow HTTPS (port 443) from any IP to our cluster virtual IP
New-NetFirewallRule -DisplayName "Allow HTTPS Cluster Traffic" -Direction Inbound -Action Allow -Protocol TCP -LocalPort 443 -LocalAddress $clusterVIP -RemoteAddress Any

# Add a firewall rule for cluster communication
New-NetFirewallRule -DisplayName "Allow Cluster Traffic" -Direction Inbound -Action Allow -Protocol TCP -LocalPort 3343 -RemoteAddress 192.168.1.0/24,10.0.0.0/24 # Internal Networks Only, example
```

*   This snippet generates specific firewall rules on each node, allowing inbound traffic on tcp ports 80 and 443, destined for the cluster's virtual ip address. We can use “any” for the remote address. I’ve included an example firewall rule for internal cluster communication for comparison. Note: this assumes cluster ips in both 192.168.1.0/24 and 10.0.0.0/24.

**Snippet 3: Adding a Route to the loopback interface (powershell, run as admin)**

```powershell
#Assuming the loopback adapter public ip is 192.168.1.250 and our external gateway is 192.168.1.1
New-NetRoute -InterfaceAlias "Loopback Adapter Public" -DestinationPrefix "0.0.0.0/0" -NextHop 192.168.1.1 #default route
```

* This code establishes a default route directing all outbound traffic from the loopback adapter towards the gateway used for external communication on the external network (firewall’s external interface). We also need to ensure that the routing table is adjusted so all responses are directed through this route using the loopback adapter’s IP address as a source.

By addressing these three components—the source ip confusion, the resource group awareness, and the lack of external routing guidance—we were able to successfully operate the cluster in a public-facing environment, using port forwarding on the firewall. It’s a prime example of how seemingly simple network setups can have complex interactions when you're dealing with a cluster environment that needs to communicate across firewall boundaries.

To dive deeper, i would recommend the classic *TCP/IP Illustrated* series by W. Richard Stevens, specifically *Volume 1: The Protocols* for a comprehensive understanding of networking protocols. For a deep dive into Windows Server Failover Clustering, the official Microsoft documentation on failover clustering is your best resource, which includes articles on networking considerations and advanced cluster configuration. And, if you want more insights into windows networking specifically, *Windows Internals* by Mark Russinovich et al. will give you a comprehensive view of how the operating system behaves internally.

In summary, what often appears as a cluster not responding through the firewall is, more precisely, the cluster incorrectly routing its responses. By implementing a few targeted network adjustments, we can make that cluster communicate effectively with the world. This situation certainly tested our knowledge, but taught us a great deal about the nuances of how windows failover clusters operate in routed environments.
