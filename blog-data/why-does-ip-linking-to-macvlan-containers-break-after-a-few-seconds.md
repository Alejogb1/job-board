---
title: "Why does IP linking to macvlan containers break after a few seconds?"
date: "2024-12-16"
id: "why-does-ip-linking-to-macvlan-containers-break-after-a-few-seconds"
---

Let's unpack this peculiar issue of macvlan container network instability. I've seen this particular problem surface in a few diverse environments over the years, ranging from small, experimental setups to larger, more complex container orchestration platforms. The core issue, as you've observed, is that connectivity to a macvlan container often works initially, only to fail after a brief period – sometimes just a few seconds. It's a frustrating situation because it often seems like everything *should* be working according to the configuration. The heart of the matter usually involves a confluence of factors, predominantly related to how networking protocols handle ARP (Address Resolution Protocol) and the quirks of virtualized network interfaces.

My past experiences with this usually point towards the interaction between the host network and the macvlan interface. When a container initially starts, it sends out an ARP request to discover the MAC address corresponding to a target IP address, allowing for communication outside of the container. The switch on the physical network where the host is connected learns this MAC address and associates it with the port the host is using. The problem is, a macvlan interface uses the *same* physical port as the host, but presents itself with a *different* MAC address. This can lead to confusion at the layer 2 level, especially when dealing with switches that implement port security features, like MAC address learning or port flapping detection.

The initial communication succeeds because the switch quickly learns the new MAC address of the macvlan container. However, when the host also sends out ARP requests (which it does constantly as part of its regular IP stack operation), it's sending packets with *its* MAC address out the same port. This can result in the switch potentially thinking that the MAC address it learned for the container has suddenly disappeared, or "flapped," and thus it may invalidate the learned mapping, breaking communication for the container until it performs another ARP request. It isn't *always* the switch that's causing the problem, but it’s the most common scenario I’ve seen, especially on networks with robust security measures.

Let's illustrate the point with some simplified scenarios and code. First, consider a basic Dockerfile for our demo container:

```dockerfile
FROM alpine:latest
RUN apk add --no-cache iputils
CMD ["/bin/sh","-c","while true; do ping -c 1 8.8.8.8; sleep 2; done"]
```

This will simply ping an external server every 2 seconds. We can then build and run this image using a macvlan network:

```bash
docker build -t my-ping-container .
docker network create -d macvlan \
    -o parent=eth0 \
    --subnet=192.168.1.0/24 \
    --gateway=192.168.1.1 macnet
docker run --rm --net macnet --ip 192.168.1.100 my-ping-container
```

This setup, on many standard networks, may exhibit the exact issue: initially, the ping works, then fails after a few moments. This happens because the switch (or even your router acting as a switch) is confused by the presence of multiple mac addresses on the same port.

To observe what’s happening, consider this python script, running on the host, while the container is running, that uses `scapy` to capture packets:

```python
from scapy.all import *

def packet_callback(packet):
    if ARP in packet and packet[ARP].op == 1:  # 1 is for ARP request
        print(f"ARP Request: Source MAC={packet[ARP].hwsrc}, Target IP={packet[ARP].pdst}")

if __name__ == '__main__':
    sniff(prn=packet_callback, filter="arp", iface="eth0")
```

Running this will show the ARP requests being sent from the host and from the container. This reveals the MAC addresses and target IPs, which can be extremely useful to debug this kind of problem. In particular, you might notice host ARP requests interfering with the container's ARP responses learned at the switch.

A more subtle variation I've encountered is the incorrect setup of the parent interface on the docker network creation. For instance:

```bash
docker network create -d macvlan \
    -o parent=eth1 \  #Incorrect!
    --subnet=192.168.2.0/24 \
    --gateway=192.168.2.1 wrong_macnet
```

In this instance, if the host interface `eth1` isn't the interface connected to the intended network, containers created using `wrong_macnet` will experience similar connectivity problems because the physical link isn't correctly matched.

So, how do we go about addressing this issue reliably? Well, in my experience, there are a few pragmatic solutions:

First, check the network configuration closely. The `parent` interface specified during the `docker network create` process *must* be the actual physical interface connected to the network. In multi-homed environments, this is an easy place to make a mistake. The script using scapy, along with standard system tools like `ip link` can be invaluable here to understand how interfaces are configured.

Second, consider using a different virtual networking solution entirely. Alternatives to macvlan such as overlay networks, like those provided by Docker Swarm or Kubernetes, offer better isolation and do not face the same issues with L2 address confusion. They often use solutions like VXLAN or Geneve to encapsulate network traffic at L3, which avoids the issues of MAC address conflicts on the physical switch port. While a more complex topic, these are often necessary for real-world applications.

Third, where using macvlan is essential for reasons of network separation or direct access to a physical network, you may need to configure the network switch to relax its port security settings. Disabling port security or enabling a sticky mac-address learning mode can often alleviate these problems, although this must be done carefully because it could introduce a security weakness. A conversation with the network administrator may be required to make such changes.

For further reading, I suggest delving into the documentation on network namespaces, in particular the paper "Network Namespaces: A Linux Foundation Guide" by the Linux Foundation Networking Working Group; it provides an excellent breakdown of how these concepts work at a low level. Also, the book "Linux Networking Internals" by Klaus Wehrle, Christian Vogt, and Hartmut Ritter offers a comprehensive look at the linux networking stack which can help develop a better understanding of how network interfaces function under the hood. Specifically relevant in that book are chapters on address resolution, link layer functionality, and the interaction of physical and virtual interfaces. Additionally, the official Docker documentation on macvlan networks is a valuable resource to help understand Docker's specific implementation of these concepts. Lastly, consulting the specific documentation of your network switch or router will help you to understand how it handles MAC address learning and port security. By combining this information, and by paying attention to the low-level details, you can resolve these often-frustrating macvlan networking issues.
