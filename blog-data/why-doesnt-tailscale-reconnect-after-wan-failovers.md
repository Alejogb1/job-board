---
title: "Why doesn't Tailscale reconnect after WAN failovers?"
date: "2024-12-16"
id: "why-doesnt-tailscale-reconnect-after-wan-failovers"
---

Okay, let's talk about Tailscale and those pesky disconnects after a WAN hiccup. From my own experience, troubleshooting network issues—especially those involving overlay networks like Tailscale—often feels like navigating a maze with shifting walls. You'd think a reconnect would be automatic, but there are a few nuances at play here that can cause headaches. I've spent quite a few late nights staring at tcpdump outputs to get to the bottom of similar situations, and I can tell you, there's usually a logical reason for the lack of automatic recovery.

First off, it’s essential to understand that Tailscale, while presenting a seamless network abstraction, operates on top of your existing network infrastructure. This means it relies heavily on things like network address translation (NAT), the stability of your underlying connection, and its own internal state management. When a wide area network (WAN) connection fails, a cascade of events occurs that can disrupt Tailscale’s ability to smoothly recover.

The core issue often stems from a combination of two things: Tailscale’s peer discovery mechanisms and the ephemeral nature of network sessions. Tailscale uses a technique called *hole punching*, or more accurately, NAT traversal techniques, to allow your devices to communicate directly despite being behind NAT. This process involves establishing sessions through various publicly reachable servers, referred to as DERP servers. These connections are essentially temporary stateful connections. When your WAN connection drops, these sessions break down, and Tailscale needs to re-establish them.

The problem is, re-establishing these sessions isn’t always instantaneous. Tailscale clients typically rely on a combination of techniques for network probing. One technique is periodic network checks to determine connectivity and peer reachability. Another approach involves relying on the server's "magic" DNS records and network information. However, these probes might not be aggressive enough or occur frequently enough during a WAN failover. In addition, the IP addresses on the router's WAN interface might change as it re-establishes a connection with the ISP. This change requires Tailscale to update its network state and re-negotiate the direct peer-to-peer connections, a process that can lag or even completely fail.

Another aspect to consider is that, during a WAN failover, your router might go through a brief period where it isn’t fully operational. During this interval, your local network might appear to be up and functional, but the outside world is unreachable. Therefore, the Tailscale client might think that it has connectivity while it’s technically on the fringes of the network. The client assumes that a disconnect isn’t due to a complete WAN disruption, so it doesn't aggressively attempt a total network re-evaluation.

Now, let’s break this down with some practical code snippets. While I cannot give you literal Tailscale internals, I can show the logical steps a client might need to take when it loses its WAN connection and then needs to reconnect. Keep in mind these examples are conceptual and designed to illustrate the process.

**Example 1: Basic Network Connectivity Check**

Here's a simplistic Python snippet demonstrating a basic check that a client might perform.

```python
import socket
import time

def check_connectivity(host="8.8.8.8", port=53, timeout=3):
    try:
        socket.setdefaulttimeout(timeout)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((host, port))
        sock.close()
        return True
    except socket.error as ex:
        return False

def main():
    while True:
        if check_connectivity():
            print("Network is up.")
            time.sleep(10) # Wait before checking again
        else:
            print("Network down, attempting recovery...")
            # Placeholder: add recovery logic here
            # For example, trigger a tailscale restart
            # os.system("tailscale restart")
            time.sleep(5)

if __name__ == "__main__":
    main()
```

This code snippet is a simplified version of how a client might check for network connectivity. It establishes a connection with a public DNS server and, based on the success or failure, it logs whether the network is up. This shows a basic mechanism by which a client might recognize that a reconnection process should occur. Note that `os.system("tailscale restart")` is a simplification of more comprehensive network state recovery logic that Tailscale would use, and in practice, restarting tailscale can have unintended side effects in more complex scenarios.

**Example 2: Detecting IP Changes**

Here's how a client might detect a change in its external IP, which is critical for reconnecting using updated connection parameters:

```python
import requests
import time

def get_external_ip():
    try:
       response = requests.get('https://api.ipify.org?format=json', timeout=3)
       response.raise_for_status()
       data = response.json()
       return data['ip']
    except requests.exceptions.RequestException as e:
      return None

def main():
    previous_ip = None
    while True:
        current_ip = get_external_ip()
        if current_ip and previous_ip != current_ip:
          if previous_ip:
              print(f"External IP changed from {previous_ip} to {current_ip}. Initiating Tailscale Reconnection")
              # Placeholder: Trigger tailscale reconnection logic here
              # os.system("tailscale up --reset")
          else:
              print(f"Initial IP Address Detected: {current_ip}")
          previous_ip = current_ip
        elif not current_ip:
          print("Could not get external IP address. Network issue detected.")
          # Placeholder: Implement failure handling
        time.sleep(10)

if __name__ == "__main__":
    main()
```

This example fetches the external IP address of the device using the `ipify.org` API (it's recommended to use a more robust method for production). When the IP changes (likely due to the WAN failover), it flags that a reconnect process needs to happen. This is crucial because Tailscale's peer-to-peer connections are often keyed off this external address. Similar to the previous example, the command `os.system("tailscale up --reset")` is a basic and potentially disruptive way of forcing Tailscale to renew its network connections and shouldn't be used without an understanding of its implications.

**Example 3:  Simple DERP server Connection Check**

Here's a conceptual check for DERP server connectivity:

```python
import socket
import time

def check_derp_connectivity(derp_server="derp.tailscale.com", port=443, timeout=3):
  try:
      socket.setdefaulttimeout(timeout)
      sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
      sock.connect((derp_server, port))
      sock.close()
      return True
  except socket.error as ex:
      return False

def main():
  while True:
    if check_derp_connectivity():
        print("DERP server reachable.")
        time.sleep(10)
    else:
      print("DERP server unreachable, potential network issue. ")
        # Placeholder: implement a tailored recovery strategy that targets DERP connectivity issues
        # In a real system this would involve several steps like reconnecting using a different DERP server or waiting longer between connection attempts.
      time.sleep(5)

if __name__ == "__main__":
   main()
```

This script illustrates a basic method to ensure that the DERP server is accessible, as this is a crucial prerequisite for re-establishing communication within the Tailscale network. A failure to connect to the DERP server might mean that the client will not be able to reconnect until the underlying issue is resolved.

In practice, Tailscale uses significantly more complex mechanisms for handling this. These examples illustrate the basic building blocks of Tailscale's reconnect process.

To dig deeper into these topics, I would highly recommend a couple of resources. First, thoroughly reading the official Tailscale documentation is essential, especially regarding their NAT traversal methods. Second, consider reading “TCP/IP Illustrated, Volume 1: The Protocols” by W. Richard Stevens, as it provides an incredibly thorough explanation of TCP/IP fundamentals, which are vital for understanding overlay networks. Finally, while it doesn't focus directly on Tailscale, “Computer Networks” by Andrew S. Tanenbaum is another authoritative source for foundational networking concepts. These resources will give you a strong background in the underpinnings of Tailscale's operation and why issues like these can occur.

The apparent lack of automatic reconnects after a WAN failure isn’t necessarily a flaw in Tailscale; it’s often a result of the inherent complexity of network reconnections. The client needs to ensure it has a working network path and a consistent network state. The code examples I have provided should help shed light on the challenges inherent in these sorts of network recoveries. Troubleshooting often requires a meticulous analysis, including, perhaps, a capture using tcpdump, to pinpoint the exact nature of the issues. Hopefully, this explanation and examples provide a better grasp of the situation.
