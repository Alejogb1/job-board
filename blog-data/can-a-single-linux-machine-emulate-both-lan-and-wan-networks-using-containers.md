---
title: "Can a single Linux machine emulate both LAN and WAN networks using containers?"
date: "2024-12-23"
id: "can-a-single-linux-machine-emulate-both-lan-and-wan-networks-using-containers"
---

Alright, let's get into this. I've had my share of head-scratching moments trying to wrangle network topologies with limited hardware, and emulating both lan and wan behaviors on a single linux box using containers certainly fits that bill. It's not just theoretically possible, it's actually quite practical and even indispensable for certain testing and development scenarios. Let's break down why and how we can accomplish this, starting with some fundamental concepts.

The core idea is leveraging linux's built-in networking capabilities in conjunction with containerization tools like docker (or podman, whichever you prefer) to create isolated network environments. We're essentially crafting multiple virtual networks within the single physical host. Think of it like having multiple switches and routers existing entirely in software, all within your machine. The linux kernel, with its advanced network namespace support, forms the bedrock of this setup. These namespaces provide the necessary segregation to allow multiple network stacks, each with their own interfaces, routing tables, and firewall rules, to coexist peacefully.

Specifically, achieving both lan and wan emulation requires understanding the key differences between these network types. A lan (local area network) is generally characterized by high bandwidth, low latency, and primarily operates within a single administrative domain, often employing private ip address ranges. A wan (wide area network), on the other hand, connects multiple lans over long distances, typically involves lower bandwidth and higher latency, and uses public ip address ranges for internet routing. Crucially, for our emulation, we need to simulate these characteristics within our containerized setup.

Here’s how we can approach this technically. First, we'll create our "lan" within a docker network. This is pretty straightforward; docker makes it relatively easy. The containers on this network will be able to communicate with each other directly, mimicking a local network. We can even control the ip addressing using subnet masks and gateways to further refine the lan characteristics.

For our "wan" emulation, we’ll take a slightly different tack. Instead of relying solely on a direct docker network, we’ll simulate wan properties, such as packet loss and latency, using tools like `tc` (traffic control). This involves using linux’s qdiscs and filters to introduce realistic wan behavior on the egress interface connected to the "wan" network. This level of granular control makes all the difference when you're testing how your application behaves on less-than-ideal network conditions. The beauty of using containers is that our "wan" side can be another set of containers, effectively creating a container-based "internet".

Now, let's move to code. The following examples will demonstrate how to set this up using docker. Remember, podman would be equally valid; I’m sticking with docker here due to wider familiarity.

**Example 1: Creating a Local Area Network (LAN)**

```bash
# Create a docker network named "lan-net" with a specific subnet
docker network create --subnet=172.18.0.0/16 lan-net

# Run two containers connected to the "lan-net" network
docker run -d --name lan-container-1 --net lan-net --ip 172.18.0.10 busybox sh -c "while true; do sleep 1; done"
docker run -d --name lan-container-2 --net lan-net --ip 172.18.0.20 busybox sh -c "while true; do sleep 1; done"

# Test connectivity between containers
docker exec -it lan-container-1 ping 172.18.0.20
```

In this first example, we've created a private network, `lan-net`, with a subnet of `172.18.0.0/16`. We then launch two containers, `lan-container-1` and `lan-container-2`, assigning static ip addresses within that subnet. The ping command in the last line verifies that these containers can communicate directly with each other, just like in a real lan environment. You can add more containers to simulate more clients/servers on this network.

**Example 2: Simulating a Wide Area Network (WAN) Using `tc`**

```bash
# Create a docker network to represent the "wan" side
docker network create wan-net

# Run a router container that connects our lan and wan
docker run -d --name wan-router \
    --net lan-net --ip 172.18.0.1 \
    --net wan-net \
    --sysctl "net.ipv4.ip_forward=1" \
    busybox sh -c "while true; do sleep 1; done"

# Apply tc to the wan facing interface of the router (eth1 will need to be found using 'docker exec wan-router ip a')
docker exec wan-router tc qdisc add dev eth1 root handle 1: htb default 10
docker exec wan-router tc class add dev eth1 parent 1: classid 1:10 htb rate 10mbit
docker exec wan-router tc qdisc add dev eth1 parent 1:10 handle 10: netem delay 50ms 20ms distribution normal loss 1%

# Run a "server" container on the wan
docker run -d --name wan-server --net wan-net busybox sh -c "while true; do sleep 1; done"

# Test connectivity from LAN to "WAN" (requires ip forwarding set in wan-router container)
# docker exec -it lan-container-1 ping <wan-server-ip> (need to find the wan-server ip with 'docker exec wan-server ip a' first)
```

This example is more complex. We introduce a "router" container that sits between the `lan-net` and `wan-net`. This container has ip forwarding enabled. The critical part is the `tc` commands applied within the router. These introduce artificial latency (50ms with 20ms jitter) and a small amount of packet loss (1%) on the router's "wan" interface. The actual interface (eth1 in the example) needs to be confirmed inside the router container as it could be named differently, so double-check the output of `docker exec wan-router ip a`. This `tc` command is crucial in emulating a real-world wan connection with limited bandwidth, latency, and packet loss. To fully test this example you would need to run the last line, after discovering the `wan-server`'s ip address, and set up routing using the `wan-router`.

**Example 3: Network Address Translation (NAT) for 'Internet' Access**

```bash
# Add a nat rule to the router container (requires iptables which isn't in busybox but is in most images)
docker exec wan-router iptables -t nat -A POSTROUTING -s 172.18.0.0/16 -j MASQUERADE

# Then we need an external facing container
docker run -d --name internet-server --net host busybox sh -c "while true; do sleep 1; done"

# And route any traffic from wan destined for external host to internet-server
# (assuming the external server on the host has ip 192.168.1.50 - replace with your correct host's ip if different)
docker exec wan-router iptables -t nat -A PREROUTING -d 192.168.1.50 -j DNAT --to-destination <your local machines host ip>
docker exec wan-router route add default via <your local machines host ip>

# Test connectivity from LAN to internet-server
# docker exec -it lan-container-1 ping 192.168.1.50
```

This example builds upon the previous one by adding NAT. This enables containers on the LAN to access an 'external' server, which is in this case running on the host, but we could use another network if required. Note that we use the host network for `internet-server` because it is the simplest way to provide access to a resource outside of the container network. We need to find out our local machines host ip address and enter that in `<your local machines host ip>` to enable this routing. Then our container on the `lan-net` will be able to reach the external 'internet server' via the `wan-router`.

These examples provide a solid foundation. The core takeaway is this: with linux's network namespaces and tools like docker and `tc`, you can indeed create incredibly realistic network emulations on a single machine. This is especially useful in scenarios where you need to simulate diverse network conditions without deploying a large and expensive physical lab.

For more in-depth knowledge, I strongly recommend delving into *Linux Networking Cookbook* by Carla Schroder, which is a practical resource covering advanced techniques, and *TCP/IP Illustrated, Volume 1* by W. Richard Stevens for a deep understanding of network protocols. Additionally, the official documentation for `tc` and docker are invaluable resources. Exploring research papers on network emulation techniques used in academic settings will also provide you with further background knowledge.

The key is to experiment and adapt these setups to your specific needs. Over time, you’ll develop a practical intuition for how these virtualized networks behave, which is a great asset to any tech professional who frequently deals with networking challenges. It’s certainly been invaluable in my career.
