---
title: "Why is Pi-Hole in Portainer inaccessible at the specified IP address, but accessible at the public IP?"
date: "2024-12-23"
id: "why-is-pi-hole-in-portainer-inaccessible-at-the-specified-ip-address-but-accessible-at-the-public-ip"
---

, let's tackle this Pi-hole in Portainer access puzzle. I've seen this exact scenario pop up a few times over the years, particularly when dealing with containerized deployments and network configurations. It's almost always a variation on network isolation or incorrect port mapping within the docker environment rather than a failure of Pi-hole itself.

The core issue, as you're likely experiencing, is that while the Pi-hole instance within Portainer is reachable through your server's public IP, it becomes inaccessible using the internal IP assigned to that server or, more specifically, the container's assigned ip within the docker network. This divergence typically points to a problem with how network traffic is routed and managed between your host machine, the docker network, and the outside world. Here’s a breakdown of the common pitfalls and how to debug them.

First, let’s consider how Docker networks function. When you deploy a container within Portainer (or using the docker cli), docker by default places these containers within a bridge network. This means that containers within this network, typically named `bridge` by default, can communicate with each other using their internal IPs (e.g., 172.17.0.2) but are isolated from your host's main network unless specifically configured. So, when you attempt to access Pi-hole using your server's LAN IP address, you're essentially trying to reach a service that is, for all practical purposes, on a separate network segment. The traffic isn’t reaching the container.

The fact that it *is* accessible using the public IP indicates a couple of things. One, your docker container is correctly set up to listen on ports exposed to the host; and two, your host's network interface is configured such that requests to the public IP are being correctly routed and then sent to the associated container through port mapping configuration. The problem is not within the container as such, but that the intended internal route to the container is not correctly configured.

Second, let's look at the different types of port mapping. Docker utilizes either direct port mapping (e.g. `-p 53:53/udp -p 80:80`) or published port mapping via a docker bridge network. In the former direct mapping case, you explicitly tell docker to forward traffic from a specific port on the host machine to a port in the container. This is usually how you expose the web UI or DNS services of pi-hole. When you access the public IP address, you're directly using the mapped host port which then gets sent to the container. However, when you attempt to access with the local server IP and it does not work, this is indicative that either the docker bridge interface has not been configured to be bound to the host network interface or firewall issues are blocking connections. If you are using the bridge network, the docker daemon must also handle routing and forwarding, and often there can be a mismatch between what is set and the host.

Thirdly, consider the docker container's network configuration itself. It's possible, though less likely with the default pi-hole image, that you have explicitly bound the container to listen only on a specific interface, the public interface that the docker network has configured for the outside route. If the container is bound to a single interface, the local interface address will be unable to reach the running container. To verify this, you would need to examine the container configuration itself either using docker commands or through the Portainer interface.

Now, let's solidify this with some illustrative examples in code snippets. Assume we're using docker-compose as a common configuration method since Portainer is very docker-centric.

**Example 1: Standard Bridge Network with Direct Port Mapping**

```yaml
version: "3.7"

services:
  pihole:
    image: pihole/pihole:latest
    container_name: pihole
    ports:
      - "53:53/tcp"
      - "53:53/udp"
      - "67:67/udp"
      - "80:80/tcp"
    environment:
      TZ: 'America/New_York'
      WEBPASSWORD: "changeme"
    volumes:
      - './etc-pihole:/etc/pihole'
      - './etc-dnsmasq.d:/etc/dnsmasq.d'
    restart: unless-stopped
```

In this example, the ports are mapped directly. In this configuration, you’d likely experience the issue you described. The Pi-hole service is exposed on ports 53 (DNS) and 80 (web UI) on the *host* network interface. Requests via the server’s public IP address would be directed to these ports, and the docker daemon would route them into the pi-hole container. However, since this docker container is in its own isolated bridge network, requests directly via the server's internal IP will fail because they are not being routed via the host's network stack and bridge network to the correct port within the docker container.

**Example 2: Host Network Mode**

```yaml
version: "3.7"

services:
  pihole:
    image: pihole/pihole:latest
    container_name: pihole
    network_mode: host
    environment:
      TZ: 'America/New_York'
      WEBPASSWORD: "changeme"
    volumes:
      - './etc-pihole:/etc/pihole'
      - './etc-dnsmasq.d:/etc/dnsmasq.d'
    restart: unless-stopped
```

By using `network_mode: host`, we force the container to use the host's network stack *directly*. The pi-hole container now shares the server's network namespace. This configuration often resolves access problems with the internal network as now the ports (53 and 80) are exposed and listening directly on your server's IP interfaces. This is generally only recommended in specific scenarios as it bypasses docker's network isolation capabilities. This would also, however, mean that the server's internal firewall now controls all access. This configuration should allow access through the internal network interface address as well as the public address.

**Example 3: Docker Bridge with Explicit Bridge Interface Binding**

```yaml
version: "3.7"

services:
  pihole:
    image: pihole/pihole:latest
    container_name: pihole
    ports:
      - "0.0.0.0:53:53/tcp"
      - "0.0.0.0:53:53/udp"
      - "0.0.0.0:67:67/udp"
      - "0.0.0.0:80:80/tcp"
    environment:
      TZ: 'America/New_York'
      WEBPASSWORD: "changeme"
    volumes:
      - './etc-pihole:/etc/pihole'
      - './etc-dnsmasq.d:/etc/dnsmasq.d'
    restart: unless-stopped
```

In this setup, we explicitly bind the exposed ports on the container to all network interfaces by utilizing the wildcard `0.0.0.0` prefix in the docker-compose ports section. This directs the docker bridge to forward packets from all interfaces on the host to the container's ports. This resolves situations where the bridge interface is only listening on an external interface. This should then resolve the issue of inaccessibility on the local server address. You should be able to reach the container with the external IP as well as the internal IP address for the server.

Now for recommended reading, I would recommend diving into the Docker documentation directly at [docs.docker.com](https://docs.docker.com). Specifically, look at the section on networking. Also, a deep dive into network namespaces, perhaps via something like *Understanding Linux Network Internals* by Christian Benvenuti, would also provide useful context. *Linux Kernel Networking* by Rami Rosen is another useful resource.

In my experience, the key to troubleshooting this specific problem is methodically ruling out possibilities. Start by verifying the container is running and healthy using `docker ps` or the Portainer interface. Then, check the ports configuration in the docker-compose file (or docker run parameters). Next, look at your server’s firewall (iptables or firewalld) to ensure it’s not blocking traffic on the necessary ports or to the relevant bridge interface. Finally, if you’re still stuck, use `docker inspect` on the container to see its IP and port configuration and verify that those port maps are available on the host. Often a problem arises from an overlooked, very minor typo in the port mapping.

In conclusion, the inaccessibility of your Pi-hole via the local IP address is almost certainly related to network isolation or routing issues within the docker network configuration. By carefully analyzing your configuration, understanding how Docker networks function, and using the suggested debugging strategies, you can resolve the issue and have your Pi-hole working correctly on all interfaces.
