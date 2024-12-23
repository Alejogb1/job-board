---
title: "Why can't Tailscale connect to the exit node's LAN?"
date: "2024-12-23"
id: "why-cant-tailscale-connect-to-the-exit-nodes-lan"
---

Let's dissect this Tailscale exit node connectivity issue, shall we? It's a common stumbling block, and over the years, I've seen it crop up in various forms, each time requiring a slightly different approach. The core problem, as you’re experiencing, is that while your Tailscale clients connect to the exit node itself, they can't seem to reach the devices on the exit node's local area network (LAN). This usually isn’t a problem with Tailscale itself, but rather how routing and network address translation (NAT) are configured within that local network.

My first encounter with this was back when I was setting up a home lab and wanted remote access to my NAS. The Tailscale connection worked flawlessly to the box running the exit node, but I couldn't ping, access, or otherwise communicate with the NAS on the same LAN. This initially led me down a few rabbit holes, but after some focused investigation, the pattern became clear.

The typical scenario is this: your Tailscale client, let's say, `100.x.y.z`, connects to your exit node, perhaps `100.a.b.c`. The exit node acts as a gateway to the internet for your tailnet clients. However, devices on your LAN, often on a range like `192.168.1.0/24`, have no awareness of the `100.x.y.z` subnet, nor do they know that the exit node should act as a router for it. Furthermore, the exit node, by default, doesn’t forward the traffic destined for the LAN. This is a core networking principle – routing packets needs to be explicitly configured.

There are three common solutions, and your mileage may vary depending on your network setup.

**1. IP Forwarding and NAT Masquerading:**

This method is probably the most straightforward and is often the first one to try. It involves enabling IP forwarding on the exit node itself (allowing it to act as a router) and then setting up NAT masquerading (making the traffic appear to originate from the exit node's LAN IP address).

Here's how this looks on a Linux-based system. I've had success with similar commands on systems ranging from raspberry pis to more robust servers:

```bash
# Enable IP forwarding
sudo sysctl net.ipv4.ip_forward=1

# Masquerade traffic going to your LAN
sudo iptables -t nat -A POSTROUTING -o <your_lan_interface> -j MASQUERADE

# To make this persistent across reboots, consider
# installing a utility like 'iptables-persistent'
# and saving your rules. For example:
# sudo netfilter-persistent save
```
Replace `<your_lan_interface>` with the actual network interface name connected to your LAN (e.g., `eth0`, `wlan0`). `sysctl` temporarily enables forwarding; this requires a more permanent method like editing `/etc/sysctl.conf` if you need it to survive a reboot. The second `iptables` line is what really allows the traffic from tailscale network to be seen as local and handled on the LAN side.

**2. Subnet Routes via Tailscale:**

Tailscale also provides functionality to advertise subnet routes. This essentially tells your tailnet clients that traffic for a specific subnet (your LAN) should be routed via the exit node. This method is cleaner than NAT masquerading because the source IP address is preserved, making it easier to identify the originating device. This requires configuring your Tailscale settings and potentially adding some commands on the exit node.

Assuming your local LAN is on `192.168.1.0/24`, the tailscale configuration on the exit node should include this:

```bash
# On the exit node
tailscale up --advertise-routes=192.168.1.0/24 --accept-routes
```
`--advertise-routes` tells Tailscale to announce the `192.168.1.0/24` route to the Tailnet. The `--accept-routes` command is necessary to enable any routes advertised by other nodes on your tailnet.

After this change, the traffic from your tailscale clients will be routed through the exit node to the LAN, with client IPs preserved in routing on the LAN side. No NAT required in this scenario. This assumes your LAN router will correctly respond to packets from IPs outside of your `192.168.1.0/24` subnet but on the same interface. Some residential routers might drop packets from outside the configured subnet.

**3. Combining IP Forwarding with Subnet Routing:**

There might be cases where just one of the above isn't sufficient, possibly due to network complexities like intermediary routers or multiple network interfaces. In such situations, it becomes necessary to use a combination of IP forwarding, NAT and subnet routing. I ran into a particularly tricky case at a former job where the exit node was connected to two different networks with overlapping IP ranges, but with distinct use cases. The solution was to add IP forwarding rules, masquerading rules specifically for each interface, and advertise subnet routes for both networks.

Here is a slightly more involved example:

```bash
# On a Linux based system acting as exit node with interfaces eth0 for LAN 1 (192.168.1.0/24) and eth1 for LAN 2 (192.168.2.0/24)

# Enable IP forwarding
sudo sysctl net.ipv4.ip_forward=1

# Masquerade traffic going out eth0 (LAN 1)
sudo iptables -t nat -A POSTROUTING -o eth0 -j MASQUERADE

# Masquerade traffic going out eth1 (LAN 2)
sudo iptables -t nat -A POSTROUTING -o eth1 -j MASQUERADE

# Tailscale command advertising both subnets and accepting routes
tailscale up --advertise-routes=192.168.1.0/24,192.168.2.0/24 --accept-routes
```
In this setup, we enable forwarding and then specify that traffic to either of our LAN subnets going out through their specific interfaces will be treated with NAT, plus advertise routes for both on the tailscale network.

It’s important to note that these approaches work best in controlled environments, such as home labs or small business networks. In larger, more complex environments, things get a bit more interesting. You may require more specific routes, firewall rules on your LAN, or even changes on your router if it's dropping packets from unexpected source addresses, so a thorough understanding of the network is essential. For these more intricate cases, consulting networking documentation and gaining experience working with the particular hardware you are using is vital.

**Further Exploration:**

For a deeper understanding of network routing and forwarding, I recommend diving into "TCP/IP Illustrated, Volume 1: The Protocols" by W. Richard Stevens. It’s a classic for a reason. Also, understanding iptables (if you’re using linux) is crucial, and the official iptables documentation, as tedious as it is, is the best resource. For Tailscale-specific knowledge, their official documentation and any github discussions are valuable, since they are constantly making updates.

In essence, the inability to reach the LAN behind your Tailscale exit node usually stems from a gap in routing and NAT rules. By applying one of these methods carefully and with a solid understanding of the network under consideration, you can unlock the full potential of your Tailscale network. It requires some hands-on experience to truly grasp the nuances, but these steps should give you a robust starting point for troubleshooting and resolution.
