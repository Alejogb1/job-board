---
title: "Why doesn't Tailscale reconnect after a WAN failover?"
date: "2024-12-23"
id: "why-doesnt-tailscale-reconnect-after-a-wan-failover"
---

Let's unpack the complexities of why Tailscale might struggle to reconnect after a wan failover, something I’ve certainly encountered in my time. It's not always as straightforward as a simple network hiccup; there are a number of interlocking components that need consideration. My experience, particularly with setting up multi-site vpn configurations, has highlighted these challenges, and while tailscale simplifies a lot, it doesn't eliminate all the underlying issues.

The core of the problem often lies in how tailscale, and indeed most vpn solutions, establish and maintain connections. Think of it less like a direct wire and more like a carefully orchestrated dance between multiple systems, each dependent on the others. When your primary wan connection drops, it isn’t just the loss of internet access; it’s a seismic shift in the network landscape as far as your devices and tailscale’s control servers are concerned.

One critical aspect is the reliance on stable identifiers. Each device using tailscale gets an internal identifier which is then associated with your account. Furthermore, they often rely on the publicly routable ip address of your router (or the machine running tailscale directly) for initial connection handshakes. If the wan failover results in a change of public ip address, the initial connection details cached by both your device and the tailscale control servers may no longer be valid. This isn't always an immediate issue because there are mechanisms for NAT traversal and keep-alive packets, but with a sufficient period of disruption, these connections will naturally time out.

Another consideration is the statefulness of the connection. Tailscale maintains an encrypted tunnel. When a wan connection is lost, this tunnel is abruptly terminated. It’s not a graceful closure where each side can announce its intentions to disconnect; it's more akin to yanking a plug out of the wall. The endpoints might not immediately realize that they have been severed. They will eventually recognize this through the absence of keep-alive packets, but it takes time to initiate a new handshake. The tailscale client must now rediscover the appropriate tailscale control servers, usually via dns resolution, and initiate a new secure tunnel based on the changed networking circumstances.

And finally, there’s the network configuration itself. A wan failover often involves more than just switching from one public ip to another. It might also include changes in dns settings, gateway information, and mtu (maximum transmission unit) values. These network details play a crucial role in establishing any kind of reliable connection. Even if tailscale successfully rediscovers its control servers and attempts a handshake, underlying network issues can prevent that from occurring smoothly or promptly.

Let’s illustrate this with code examples, keeping in mind these aren't complete, functioning snippets but rather conceptual representations to demonstrate the points discussed.

First, consider the hypothetical scenario of a device maintaining a cache of known tailscale peers (simplified):

```python
class TailscalePeer:
    def __init__(self, peer_id, public_ip, last_seen):
        self.peer_id = peer_id
        self.public_ip = public_ip
        self.last_seen = last_seen

known_peers = {
    "peer123": TailscalePeer("peer123", "203.0.113.1", "2024-07-26T12:00:00Z"),
    "peer456": TailscalePeer("peer456", "198.51.100.2", "2024-07-26T12:05:00Z")
}

def update_peer_ip(peer_id, new_ip):
    if peer_id in known_peers:
        known_peers[peer_id].public_ip = new_ip
        known_peers[peer_id].last_seen = datetime.now().isoformat()
    else:
       print (f"peer {peer_id} not found, not able to update")

# After a wan failover, the public_ip might change:
update_peer_ip("peer123", "172.217.160.142")

```

This code shows how the cached public ip is updated, but it doesn't demonstrate how that initial discovery occurs or how the system might react if that initial data is significantly stale due to a wan disruption.

Next, let’s look at a simplified representation of a keep-alive process:

```python
import time
import socket

def send_keepalive(ip, port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    while True:
       try:
           sock.sendto(b"keepalive", (ip, port))
           print (f"keepalive sent to {ip}")
       except Exception as e:
           print(f"Error sending keepalive: {e}")
           break
       time.sleep(30)

# Simulating the loss of connection
# after this send, a send_keepalive would fail
#send_keepalive("203.0.113.1", 41641)

```

This highlights the simple mechanism of periodic data transmission to maintain the connection. When the network fails, these transmissions cease, and the connection eventually expires. The actual tailscale implementation is more nuanced, involving encryption and sequence numbers, but this captures the fundamental concept.

Finally, the need for dns resolution is highlighted here, as an example to demonstrate, not actual code execution:

```
import dns.resolver

def resolve_tailscale_server(hostname):
    try:
       answers = dns.resolver.resolve(hostname, "A")
       for rdata in answers:
           print (f"resolved ip: {rdata.address}")
    except dns.resolver.NXDOMAIN:
       print ("host not found")
       return None
    except Exception as e:
        print (f"error during resolution: {e}")
        return None


#Example: Resolving a tailscale control server address.
#This would be initiated after wan restoration.
resolve_tailscale_server("controlplane.tailscale.com")

```

This code shows the necessity of dns resolution. If the dns server settings change during a wan failover and the tailscale client is not informed of this change, it won't be able to resolve the control server address, stalling the connection.

To address these challenges in real-world deployments, you need robust networking configurations. Ensure that the failover process itself is as seamless as possible, minimize the time taken for the failover, and verify that your dns settings remain consistent and reachable after the failover. Furthermore, it can be helpful to use dynamic dns services if your isp is not able to guarantee a static ip, to keep consistent name records. In these more complex setups, there are often layers of networking between your internal devices and the tailscale servers and the stability of this middle networking layer, is also critically important to a consistent connection

For a deeper dive, I strongly recommend exploring the "tcp/ip illustrated" series by richard stevens, particularly volume 1, which provides an in-depth understanding of network protocols. Additionally, “computer networking: a top-down approach" by kurose and ross covers these topics in detail and is a valuable resource. Specifically, delving into the sections on network address translation, connection establishment, and reliable data transfer will greatly aid in understanding these challenges. These texts are essential for truly grasping the nuances behind these often subtle issues. While tailscale strives to simplify network configuration, a solid comprehension of the underlying network principles remains crucial for troubleshooting these kind of issues.
