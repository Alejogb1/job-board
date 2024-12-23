---
title: "Why does Tailscale not reconnect after a WAN failover on an upstream router?"
date: "2024-12-23"
id: "why-does-tailscale-not-reconnect-after-a-wan-failover-on-an-upstream-router"
---

,  I've certainly seen this exact scenario play out more than once in complex network deployments, and it's never quite as straightforward as we'd hope. When a wan failover occurs on an upstream router while using tailscale, the lack of immediate reconnection often boils down to a combination of factors centered around how tailscale establishes and maintains its connections, and how routers handle network address translation (nat) and route propagation. It's less about tailscale itself being "broken," and more about the interplay between its assumptions and the underlying network changes.

To begin, we need to understand how tailscale initially establishes a connection. It uses a method called nat traversal, which attempts to establish peer-to-peer connections directly between devices whenever possible. This minimizes reliance on tailscale servers and reduces latency, which is generally good for performance. However, this also means that the established connection is often tied to the specific public ip address of the network the device is on. If the wan ip address changes on the router during a failover, it can invalidate the previously established connection in a few ways.

First, consider the state of nat mappings on the router. Most routers employ nat, which translates the private ip addresses used on a local network to the router’s single public ip address when traffic is sent out to the internet. When tailscale establishes a connection, the router creates nat mappings that associate specific ports and private ip addresses with the public ip address. A failover event that results in the router obtaining a new public ip often wipes out those existing nat mappings. Since the established tailscale connection is based on the old nat mappings and public ip, packets directed to the old connection will now fail to reach their intended destination.

Second, there's the issue of connection tracking. Routers often keep track of active network connections and their associated states for security and performance reasons. When a failover occurs, the router’s connection tracking table is likely to be flushed. This means that even if the old private ip/port mapping could be somehow kept associated, the actual state information for that connection will be missing, making further data transfer difficult.

Third, tailscale relies on the local network’s dns server to resolve tailscale network device names and associated ip addresses. When a wan failover happens, sometimes the local network’s dns server might not be available or might need to refresh its cached mappings. This disruption can prevent tailscale from resolving its peer’s ip address, further hindering reconnection efforts. Although tailscale has its own discovery mechanism, the initial dns discovery can significantly impact initial reconnection.

Now, let’s look at this from the perspective of the involved machines running tailscale. Tailscale itself uses a combination of techniques to maintain connections, including keep-alive packets. These packets are normally sent periodically to check that the network path is still active. However, the problem here isn't usually that tailscale isn't sending keep-alives; it's that the packets aren't being routed to the correct endpoint because of the routing changes introduced by the failover, or the packets are being dropped because of the nat mapping changes, or the endpoint has changed its public ip address, even if the routing appears to be working, but the keep-alives use outdated connection info.

To illustrate, let's assume a simple scenario where we have device 'a' and device 'b' connected via tailscale, using nat-based routing and a failing router.

**Example 1: The Nat Problem**

Let’s examine the scenario through a simple python script that attempts to simulate a connection with a specific endpoint, and that highlights the need for the correct public ip and ports.

```python
import socket
import time

def establish_connection(host, port):
    try:
      sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
      sock.connect((host, port))
      print(f"Connection successful to {host}:{port}")
      return sock
    except Exception as e:
      print(f"Connection failed: {e}")
      return None

def send_keepalive(sock):
  if sock:
      try:
          sock.sendall("ping".encode())
          data = sock.recv(1024)
          if data:
              print(f"Received data: {data.decode()}")
          else:
            print("No data received")
      except Exception as e:
          print(f"Error sending keepalive: {e}")

if __name__ == "__main__":
    # Assume these are values resolved before the failover
    old_public_ip = "203.0.113.10" # old router public ip
    remote_port = 12345   # pre-established port mapping
    
    sock = establish_connection(old_public_ip, remote_port)
    if sock:
       send_keepalive(sock)
       time.sleep(5) # Simulate some time passed between checks
       send_keepalive(sock) # keep-alive will likely fail now if there is a wan failover.
       sock.close()
       
    # Then after the failover the ip address is changed on the router. The code has not been able to update itself with the new values
    new_public_ip = "198.51.100.20" # new router public ip, which the previous connection is not aware of.
    sock2 = establish_connection(new_public_ip, remote_port)
    if sock2:
        send_keepalive(sock2)
        sock2.close()
    else:
        print("Second connection failed, as expected, since the old connection was broken.")
```

This script demonstrates how a direct connection using an outdated public ip will not function after a failover. Tailscale will eventually discover the new ip address if the dns is functional, but this can cause a noticeable delay during the reconnection process.

**Example 2: The Router's Connection Tracking**

This example will be a simplified simulation of a stateful firewall tracking a specific connection, showing that a failover wipes the connection state and requires a new handshake.

```python
import time

class Firewall:
    def __init__(self):
      self.connections = {}

    def create_connection(self, source_ip, source_port, dest_ip, dest_port):
      key = (source_ip, source_port, dest_ip, dest_port)
      if key not in self.connections:
        self.connections[key] = { "status": "active", "timestamp": time.time() }
        print(f"Connection created {key}")
        return True
      else:
         print(f"Connection exists {key}")
         return False

    def is_connection_active(self, source_ip, source_port, dest_ip, dest_port):
        key = (source_ip, source_port, dest_ip, dest_port)
        if key in self.connections and self.connections[key]["status"] == "active":
           print(f"Connection is active: {key}")
           return True
        else:
           print(f"Connection is not active {key}")
           return False

    def close_connection(self, source_ip, source_port, dest_ip, dest_port):
      key = (source_ip, source_port, dest_ip, dest_port)
      if key in self.connections:
        self.connections[key]["status"] = "closed"
        print(f"Connection closed {key}")
    
    def clear_all_connections(self):
        self.connections = {}
        print("All connections have been cleared")


if __name__ == "__main__":
    fw = Firewall()

    src_ip = "192.168.1.100"
    src_port = 54321
    dst_ip = "203.0.113.10" # Old public ip
    dst_port = 12345

    fw.create_connection(src_ip, src_port, dst_ip, dst_port)
    fw.is_connection_active(src_ip, src_port, dst_ip, dst_port)

    fw.clear_all_connections() # Simulate Router failover.

    fw.is_connection_active(src_ip, src_port, dst_ip, dst_port) # connection will now be inactive, and tailscale will need to establish a new one.
    
    new_dst_ip = "198.51.100.20" # new public ip, and a new connection will need to be established.

    fw.create_connection(src_ip, src_port, new_dst_ip, dst_port)
    fw.is_connection_active(src_ip, src_port, new_dst_ip, dst_port) # a new connection has been correctly established.

```

This shows how the firewall/router's connection tracking is disrupted by the failover, and it needs to start from scratch. Tailscale has to re-establish and re-authenticate all of its connections.

**Example 3: A simplified DNS issue**

```python
import time

class DnsResolver:
    def __init__(self):
        self.records = {
            "device_b.tailscale": "203.0.113.10" # old ip of device b before failover
        }

    def resolve(self, hostname):
        print(f"Attempting to resolve {hostname}")
        if hostname in self.records:
           return self.records[hostname]
        else:
            print(f"{hostname} could not be resolved")
            return None

    def refresh_records(self, new_record):
        print(f"Refreshing dns records with {new_record}")
        self.records = new_record


if __name__ == "__main__":
    dns = DnsResolver()

    device_b_ip = dns.resolve("device_b.tailscale")
    print(f"Resolved ip for device_b: {device_b_ip}")
    
    time.sleep(2)

    new_records = {"device_b.tailscale" : "198.51.100.20"} # the ip of device b after the router failover
    dns.refresh_records(new_records)

    device_b_new_ip = dns.resolve("device_b.tailscale")
    print(f"Resolved new ip for device_b: {device_b_new_ip}")

```

This demonstrates how a disruption in dns will prevent tailscale from quickly reconnecting because its endpoint has changed and it needs to be resolved with the new public ip address.

To mitigate these issues, a few strategies can be effective. First, reducing the reliance on the local router’s dns by having each tailscale device have a static ip from the tailscale ip range rather than from the local ip range can help. Secondly, configuring shorter connection keep-alive times, within reasonable limits, can prompt tailscale to detect changes faster, although this can increase network overhead. Finally, ensure that the tailscale client is configured to use the fastest and most reliable dns servers possible.

For further learning, the following resources are helpful:
*   “TCP/IP Illustrated, Vol 1: The Protocols” by W. Richard Stevens: It provides a detailed explanation of tcp/ip communication, and includes discussion about nat, which is critical for the understanding of this issue.
*   “Internetworking with TCP/IP Vol 1 Principles, Protocols, and Architecture” by Douglas Comer: Focuses on the core principles and protocols of the internet. This includes details of how routing works and it is a solid background for understanding network failovers.
*    “Linux Firewalls: Attack Detection and Response” by Michael Rash: Explains how stateful firewalls operate and their connection tracking mechanisms, which are critical for understanding why connections fail after a failover.

While tailscale is good at handling many network disruptions, changes at the router level, especially those involving nat and ip address changes, can present unique challenges. Understanding the interplay between these systems, as illustrated in these code snippets, is key to diagnosing and mitigating such issues.
