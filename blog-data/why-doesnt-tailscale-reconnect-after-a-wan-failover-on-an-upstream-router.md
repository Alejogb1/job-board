---
title: "Why doesn't Tailscale reconnect after a WAN failover on an upstream router?"
date: "2024-12-23"
id: "why-doesnt-tailscale-reconnect-after-a-wan-failover-on-an-upstream-router"
---

Okay, let's talk about Tailscale and those frustrating disconnects following a WAN failover. I've spent more late nights than I care to remember chasing similar issues in various network environments, so this feels familiar. In your scenario, the root of the problem likely isn't Tailscale itself, but rather the transient changes in network addressing and routing that occur during the failover on your upstream router. Tailscale, at its core, relies on establishing a stable connection to its control plane (the coordination server) and then to other peers, using mechanisms like NAT traversal and DERP relays when direct connections aren't possible.

The crux of the issue lies in the fact that a WAN failover often results in a new public IP address being assigned to your router. This change, while seemingly innocuous, significantly impacts the underlying networking that Tailscale depends upon. Here's a breakdown of why:

*   **IP Address Change**: Tailscale initially establishes connections based on the original public IP address. When your upstream router switches to a backup WAN link, it likely receives a *new* public IP address from your internet service provider. This renders the previously established connection information stale. While Tailscale does have mechanisms for detecting network changes, these aren't always instantaneous, and can depend on the specifics of your network configuration and the timing of various events.

*   **NAT Mappings**: Network Address Translation (NAT) on your router plays a crucial role. When a device within your local network initiates a connection, your router creates a NAT mapping, associating its local IP and port to a specific public IP and port. This mapping is crucial for return traffic to reach the correct device. When the failover happens and the public IP changes, these mappings become invalid. Tailscale needs to re-establish these mappings to maintain a connection.

*   **DERP Relays**: If a direct connection isn't feasible, Tailscale uses DERP relays – servers that facilitate communication between peers. Even if the DERP relay connection is persistent after a change, it may still struggle if one peer has a stale NAT or IP address associated with its connection.

*   **Keep-Alive Timers**: Although Tailscale utilizes keep-alive mechanisms, these timers can be insufficient to deal with the complete network reset implied by a WAN failover. The default interval might not be aggressive enough to trigger the full re-negotiation process needed. The system might still be operating under the presumption of a persistent, albeit temporarily interrupted, network path.

So, how do we go about addressing this? The solution typically involves strategies that facilitate quicker detection and adaptation to network changes. I’ll detail some concrete steps and illustrate with code examples.

**Strategies and Code Examples:**

1.  **More Aggressive Keep-Alive Intervals (Within Reason)**: You can influence how frequently Tailscale checks for network changes and re-establishes connections. While modifying core processes is not recommended, understanding the configurable parameters is important. The configuration you would alter here isn’t within Tailscale directly, but instead within network configurations that influence the behavior of connection-tracking, firewalls, and NAT. However, if the issue relates to the machine running the client, you should look at network interface configuration.

    ```python
    # This example isn't Tailscale configuration, but rather an illustration
    # of how you might set up more frequent keepalives at the system level.
    # Actual syntax and methods would vary based on your operating system.

    #On Linux using iproute2
    #get existing timer
    #ip link show dev eth0 | grep "link/ether"
    #the example below will vary greatly on the os. Consult relevant manual pages.

    # This example would vary significantly on macos and windows.
    import subprocess

    def set_keepalive(interface, timeout):
        try:
            subprocess.run(['sudo', 'sysctl', f'net.ipv4.tcp_keepalive_time={timeout}'], check=True)
            subprocess.run(['sudo', 'sysctl', f'net.ipv4.tcp_keepalive_intvl={timeout}'], check=True)
            subprocess.run(['sudo', 'sysctl', 'net.ipv4.tcp_keepalive_probes=5'], check=True)
            print(f"Keepalive timers set for {interface} with timeout {timeout}")
        except subprocess.CalledProcessError as e:
            print(f"Error setting keepalive timers: {e}")


    if __name__ == "__main__":
      interface_name = "eth0" #replace with appropriate name
      keepalive_timeout = 30  # Seconds, adjust as required, don't set too low
      set_keepalive(interface_name, keepalive_timeout)

    ```
    *Note*: This python example changes network connection properties at the system level and does not directly modify tailscale. It is critical that you use the appropriate commands for your operating system. Consult your system documentation for more specific details. Lowering keep-alive intervals too much can create additional network traffic and stress, so start conservatively and monitor for stability issues. This helps the local system maintain established connections at a lower level, and therefore aids in tailscale working with a more stable network environment.

2.  **Using a Dynamic DNS (DDNS) Service:** Tailscale can be configured to use a hostname rather than a specific IP address for each device. By using a DDNS, your host will still be reachable even after the public ip changes on your router, provided your router supports updating the assigned ip with your DDNS provider. Your tailscale clients will still need to reconnect, but the process will be much more reliable as the address of your peers will remain constant, even as your public facing ip address changes.

    ```python
    # Example using a hypothetical DDNS API client, this is purely illustrative.

    import requests

    class DDNSClient:
        def __init__(self, api_url, api_key, hostname):
            self.api_url = api_url
            self.api_key = api_key
            self.hostname = hostname

        def update_ip(self, new_ip):
            payload = {'hostname': self.hostname, 'ip': new_ip}
            headers = {'Authorization': f'Bearer {self.api_key}'}
            try:
                response = requests.post(self.api_url, json=payload, headers=headers)
                response.raise_for_status() # Raise an exception for bad status codes
                print(f"DDNS updated successfully to {new_ip} for {self.hostname}")
            except requests.exceptions.RequestException as e:
                print(f"DDNS update failed: {e}")

    #This example needs to be run on the router directly or on a network host with access to the router api.
    def get_external_ip():
        try:
            response = requests.get('https://api.ipify.org?format=json')
            response.raise_for_status()
            return response.json()['ip']
        except requests.exceptions.RequestException as e:
            print(f"Unable to retrieve external IP: {e}")
            return None


    if __name__ == "__main__":
        ddns_api_url = "https://myddnsservice.com/api/update"
        ddns_api_key = "your_ddns_api_key" #replace with actual api key
        ddns_hostname = "yourhostname.myddns.net" #replace with actual hostname

        ddns_client = DDNSClient(ddns_api_url, ddns_api_key, ddns_hostname)

        new_public_ip = get_external_ip()

        if new_public_ip:
            ddns_client.update_ip(new_public_ip)


    ```

    *Note:* This example illustrates a DDNS API interaction, which would require a service compatible with your specific needs and a script designed to run on either your router or within your network. The actual API calls will vary widely. You should consult the documentation of your chosen provider to tailor the script. If the router allows it, use its built-in DDNS functionality over running any external scripts.

3.  **Explicitly Resetting Tailscale Client (As a Last Resort)**: While not ideal as a regular solution, sometimes, a complete restart of the Tailscale service on the affected device can be necessary. This forces the client to re-establish connections with updated network information.

    ```bash
    #!/bin/bash
    # Example for a linux system.
    # This bash script is system specific. Consult your operating system's documentation

    # Check if tailscaled is running
    if pgrep tailscaled > /dev/null; then
       echo "Tailscaled is running. Restarting..."
       sudo systemctl restart tailscaled
       echo "Tailscale service restarted."
    else
       echo "Tailscaled is not running. Starting service."
       sudo systemctl start tailscaled
       echo "Tailscale service started."
    fi

    # for macos you may use: launchctl stop com.tailscale.tailscaled ; launchctl start com.tailscale.tailscaled
    # for windows you may use: Stop-Service -Name "Tailscale" ; Start-Service -Name "Tailscale"

    ```
    *Note:* This is a bash example and would need to be adapted based on the operating system of your device, and run as an administrator or root user. Such a script could be set to run automatically upon detecting network changes, although doing so requires some system-specific knowledge for detection events, and should be treated with caution. Forcing restarts is not a preferred method, and should be done only in situations where it is necessary and is the most efficient solution.

**Recommendations for further reading:**

To deepen your understanding, I recommend looking into these resources:

*   **"TCP/IP Illustrated, Volume 1: The Protocols" by W. Richard Stevens:** This book is a classic for understanding the underlying network protocols, including TCP, IP, and NAT, which are essential for understanding why network failovers affect Tailscale.

*   **RFC 3489: STUN - Simple Traversal of User Datagram Protocol (UDP) Through Network Address Translators (NATs):** Reading the RFC behind STUN (Session Traversal Utilities for NAT), the technique Tailscale often uses to punch holes in NAT, will help you understand limitations and why connection re-establishment is sometimes necessary.

*   **"Computer Networking: A Top-Down Approach" by James Kurose and Keith Ross:** This provides a comprehensive understanding of computer networks and will greatly benefit your broader understanding of these topics.

In summary, while Tailscale is designed to be robust, the fundamental nature of network address changes during a WAN failover requires some degree of mitigation. Implementing more aggressive keep-alive intervals and a good DDNS setup are usually sufficient. However, understanding these deeper underlying network mechanisms will enable you to tackle more complex situations.
