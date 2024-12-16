---
title: "Why doesn't Tailscale reconnect after WAN failover?"
date: "2024-12-16"
id: "why-doesnt-tailscale-reconnect-after-wan-failover"
---

Alright, let's tackle this one. I’ve seen this scenario play out a good number of times, usually when dealing with network setups that require high availability – a critical need where a simple wan failover can unfortunately cascade into other connection issues, like with Tailscale. You'd expect a clean switchover from one internet connection to another, but Tailscale, like other vpn solutions, can sometimes hiccup. Let me break down why this happens and what you can do about it, drawing from my experience with a particularly troublesome setup at a small datacenter years ago where we had redundant wan links.

The core issue isn’t that Tailscale is inherently flawed, it’s more about how network environments and their protocols behave during a failover event. Here's the gist: Tailscale relies on establishing secure connections through something known as 'tunnels'. These tunnels are built upon persistent, usually long-lived, connections, and during a wan failover, the underlying internet protocol address (ip address) changes on the router and this disrupts the established tunnels. Think of it like a phone call: the moment your phone's network changes – maybe you move from wifi to cellular or your isp's system switches over – the call needs to be re-established. That’s a crude analogy, but it captures the essence.

When a wan failover occurs, your router, by design, is relinquishing one ip address and receiving a new one from the backup isp. Tailscale, or any similar system, is aware of and uses this original ip address to communicate with your other nodes. When that address is no longer active, the established tunnels to other machines suddenly have no valid endpoint to send data. These tunnels don't auto-magically detect the new ip address and update their configuration. The connections are essentially severed, leading to your 'offline' node status. Tailscale’s magic happens through a combination of coordinating via their coordination servers, performing nat traversal techniques to make sure nodes connect despite the barriers of different networks, and persistent connections; a change in an endpoint means those persisted connections need re-establishment.

To make things more complex, there's often a timing factor involved. It’s not just the ip address change; the dns resolution might not instantly reflect the new routing, and the client software might take some time to recognize the new configuration details. Therefore, sometimes the failover itself is clean, but the software needs a bit of time to understand and implement those changes. This can lead to timeouts, connection failures, and a temporarily unavailable service.

Here's a practical piece of experience that might offer some insight. During a project, we were operating a bunch of remote machines in a facility with flaky internet. We used Tailscale for management. When the primary isp failed, machines would become unreachable. The core reason? The change in external ip caused the tailscale tunnels to break, even when the connection was restored using the secondary internet connection. The solution wasn't to blame Tailscale; it was to think how to make the system more robust.

Let's illustrate this with some pseudo-code snippets. These aren’t full working applications, but rather illustrative examples to give you a better understanding. First, let's show how a basic connection setup might look:

```python
# Simplified pseudo-code representation of initial connection

class TailscaleConnection:
    def __init__(self, server_ip, client_ip):
        self.server_ip = server_ip
        self.client_ip = client_ip
        self.is_connected = False

    def connect(self):
        print(f"Connecting to {self.server_ip} from {self.client_ip}...")
        # Simulate a successful connection establishment
        self.is_connected = True
        print("Connection established.")

    def send_data(self, data):
        if self.is_connected:
            print(f"Sending data: {data} to {self.server_ip}")
        else:
            print("Connection not established. Cannot send data.")

    def close(self):
        print("Closing connection.")
        self.is_connected = False

# Initial connection
connection1 = TailscaleConnection("192.0.2.10", "192.168.1.50")  # Hypothetical IP
connection1.connect()
connection1.send_data("Test message")
```

This shows a simple connection establishment where we assume a successful initial connection. Now let's see how a wan failover would disrupt this.

```python
# Pseudo-code representation of failover event

def wan_failover(connection, new_server_ip):
    print("WAN failover detected.")
    connection.close()  # Existing tunnel is no longer valid
    connection.server_ip = new_server_ip # IP change
    print(f"New server IP address: {connection.server_ip}")
    print("Reconnection is needed to the new address.")

# Simulate wan failover event
new_server_ip = "203.0.113.20" # New Hypothetical IP
wan_failover(connection1, new_server_ip)

# Try to use old connection - Fails
connection1.send_data("Message after Failover")

# Establish new connection:
connection1.connect()
connection1.send_data("Message after reconnecting")
```

Here, you can see that upon a simulated failover, the existing connection is closed, the server's ip address is updated, and subsequent attempts to send data using the old tunnel will fail. We would need to create a new connection.

Finally, a very simplified representation of what a reconnection function might involve:

```python
# Pseudo-code illustrating reconnection attempt after failover
import time

def attempt_reconnect(connection, max_retries=3, retry_interval=5):
    retries = 0
    while retries < max_retries:
        print(f"Attempting reconnection ({retries + 1}/{max_retries})...")
        try:
            connection.connect()
            if connection.is_connected:
                print("Reconnection successful.")
                return True
            else:
                print("Connection failed during retry.")
        except Exception as e:
            print(f"Error during reconnection: {e}")
            print("Trying again.")
        retries += 1
        time.sleep(retry_interval)
    print("Reconnection attempts failed.")
    return False

# Attempt reconnection after failover
reconnected = attempt_reconnect(connection1)
if reconnected:
    connection1.send_data("Test Message After Reconnect")
```

This final example shows a basic retry mechanism to handle reconnection after a network change. It tries connecting a few times before finally giving up.

These examples highlight the core issue: the established tunnels become invalid during a wan failover, and we need a mechanism to re-establish them.

Now, what practical steps can you take to mitigate this? While there isn't a single magical solution, here are a few things you can implement:

1.  **Implement Robust Retry Logic:** Tailscale has its own retry logic, but it sometimes benefits from adding an additional retry mechanism at the system level in scripts or custom software where feasible. Consider a systemd service that watches for network changes and triggers a restart of the tailscale service or a specific script that force a re-establishment of connections, perhaps after a specific period.

2.  **Persistent Configuration and DNS:** Ensure your configuration remains persistent across reboots and network changes. This includes using fixed Tailscale hostnames, not ip addresses, where possible and making sure all your devices have stable dns settings and access to tailscale's coordination servers.

3. **Utilize Tailscale's DERP Servers:** Tailscale's DERP servers act as relay servers and can assist in connecting behind firewalls and complex network configurations, this is configured by default, but if not consider enabling and configuring them to increase the chances of reconnecting. These also are helpful for devices behind nat.

For deep dives, I recommend you look into the following resources:

*   **“TCP/IP Illustrated, Volume 1: The Protocols” by W. Richard Stevens:** This classic gives a comprehensive understanding of tcp/ip, and is crucial for diagnosing and solving network related issues. You’ll find invaluable info regarding how connections are established and maintained.

*   **"Understanding Linux Network Internals" by Christian Benvenuti:** While not specifically about Tailscale, understanding the linux networking stack is vital for those administering devices using linux. This book dives into the architecture of networking within linux and can help with debugging connection related issues in linux.

*   **The Tailscale Documentation:** Tailscale provides comprehensive documentation on its website that outlines its configuration, troubleshooting, and general networking principles. This is the first stop for anyone working with the service, it goes into fine details about how the protocol works.

Ultimately, ensuring a smooth reconnection after a wan failover comes down to understanding the underlying networking mechanics and tailoring your system with redundant network strategies. No system is foolproof, but with the knowledge of the basic building blocks, your setups become more robust and predictable. It's a journey of learning and adapting, and there’s no reason why these kinds of issues can't be handled effectively with the right approach.
