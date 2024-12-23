---
title: "Why aren't DNS queries in Docker caching?"
date: "2024-12-23"
id: "why-arent-dns-queries-in-docker-caching"
---

, let's unpack this. From my experience, I’ve seen the frustration firsthand when containers seem to ignore DNS caching, or rather, behave inconsistently with it. It often surfaces as slow or intermittent connections to external services, leading to a lot of head-scratching and time spent trying to diagnose what *should* be working flawlessly. The core issue often isn't that DNS queries *aren't* being cached, but rather *where* they’re being cached, and how that caching interacts with Docker's network stack. It’s less about Docker magically ignoring the fundamentals and more about understanding its layering and some default behaviors.

Let's start with a bit of background. In any typical operating system, DNS resolution isn’t a single event; it’s a series of lookups managed at multiple layers. Your application makes a request, which usually goes through a local resolver, typically a lightweight service like `systemd-resolved` or `dnsmasq` on Linux. This resolver, in turn, queries configured DNS servers, caching the results locally to speed up subsequent queries. The idea is to reduce the latency and load on the upstream DNS servers.

Docker, however, introduces a layer of abstraction via its container network. By default, Docker containers operate inside their own isolated network namespaces. This means they essentially get their own virtual network interface, their own routing table, and crucially, their own `resolv.conf` file, which defines their DNS settings. The ‘magic’ that makes this all work is Docker's own embedded DNS server that sits inside the bridge network or a custom defined network. This internal DNS is what the containers query.

The problem arises because this internal DNS server inside Docker, by default, doesn’t actively cache upstream DNS responses in the way a traditional resolver does on your host operating system. This is because, historically, it is more of a forwarding resolver than a full caching resolver, designed to route requests to your host’s DNS and just relay results. While this is a valid design to provide easy access to host services, it limits how much DNS caching occurs within the container ecosystem.

Now, before diving deeper, I'd suggest anyone interested in fully grasping network behaviour in Linux delve into W. Richard Stevens' "TCP/IP Illustrated, Vol. 1: The Protocols". It’s an invaluable resource for foundational understanding, especially the chapters on DNS and network layers. For a more specific deep dive into Docker networking itself, “Docker Deep Dive” by Nigel Poulton is a great practical guide that addresses these quirks in detail.

Let's get to some practical examples. Imagine you have a simple Python application inside a Docker container that needs to connect to an external service, `api.example.com`.

Here’s a basic Python application:

```python
import socket
import time

def resolve_domain(domain):
    start_time = time.time()
    try:
        socket.gethostbyname(domain)
        end_time = time.time()
        print(f"DNS lookup for {domain} took: {end_time - start_time:.4f} seconds")
    except socket.gaierror as e:
        print(f"DNS lookup failed for {domain}: {e}")

if __name__ == "__main__":
    domain_name = "api.example.com"
    for _ in range(3):
        resolve_domain(domain_name)
        time.sleep(1)
```

If you run this application inside a standard Docker container, you would typically see similar times for all three DNS lookups, suggesting a lack of caching. The problem, as mentioned, isn't that caching is completely absent, but where it resides, or rather, the lack of a caching DNS server at the Docker-container level.

Here is an example of a basic Dockerfile that can be used to test this:

```dockerfile
FROM python:3.9-slim-buster

WORKDIR /app

COPY ./app.py /app

CMD ["python", "app.py"]

```

This Dockerfile creates a base environment to run the python script. You can build it using the command `docker build -t dns-test .` and then run it `docker run --rm dns-test`. When you execute this application without custom caching mechanisms at the docker level, you will notice that the resolution times are generally the same for each lookup.

Now, to demonstrate how we can introduce proper DNS caching, we can modify the Dockerfile by installing `dnsmasq` inside the container.

Here’s an updated Dockerfile:

```dockerfile
FROM python:3.9-slim-buster

RUN apt-get update && apt-get install -y dnsmasq

WORKDIR /app

COPY ./app.py /app

# Configure dnsmasq
RUN echo 'server=8.8.8.8' > /etc/dnsmasq.conf \
    && echo 'port=53' >> /etc/dnsmasq.conf \
    && echo 'cache-size=1000' >> /etc/dnsmasq.conf
# Ensure that the dnsmasq service is launched in the foreground within the container
CMD ["/usr/sbin/dnsmasq", "--no-daemon"]
```

In this version, we've installed `dnsmasq`, configured it to use Google's public DNS servers, configured a listening port of 53, and set a cache size. The `--no-daemon` argument ensures that the container doesn’t immediately exit. Note that for proper configuration, you might need to set `dns-opt` with the container options to configure its `/etc/resolv.conf` to use localhost as the nameserver. Here's how you might start this container with the correct settings:

```bash
docker run -d --rm --dns=127.0.0.1 --dns-opt="ndots:0" -p 53:53/udp dns-test
```

Now you would need to modify the Python program to resolve against the local dnsmasq.

```python
import socket
import time

def resolve_domain(domain):
    start_time = time.time()
    try:
        # Here, we're explicitly specifying 127.0.0.1 as our DNS server
        socket.gethostbyname_ex(domain, family=socket.AF_INET,  hosts = [('127.0.0.1', 53)])
        end_time = time.time()
        print(f"DNS lookup for {domain} took: {end_time - start_time:.4f} seconds")
    except socket.gaierror as e:
        print(f"DNS lookup failed for {domain}: {e}")

if __name__ == "__main__":
    domain_name = "api.example.com"
    for _ in range(3):
        resolve_domain(domain_name)
        time.sleep(1)
```

After modifying and running the Python application inside the modified container, the subsequent lookups will now be significantly faster due to `dnsmasq` caching the initial result. There are some caveats to note here, such as a requirement to bind the dnsmasq service to a specific IP. For the context of this example, it's running on the local interface.

The main takeaway here is that by embedding a caching DNS resolver within the container, we directly addressed the lack of caching at the container level. In a more complex environment, such a setup isn’t always the ideal solution. It often requires managing these caching servers in each container image, which isn’t scalable for large deployments.

Instead of embedding caching resolvers directly inside each container, an alternative approach is to utilize Docker's `--dns` flag. By specifying an external caching DNS server at container runtime, all containers share and leverage it. For example, you could configure the docker daemon to use a dedicated caching resolver running on your host system or a different machine. This can be configured by modifying the `/etc/docker/daemon.json` file to include the following:

```json
{
  "dns": ["192.168.1.10"]
}
```

Where `192.168.1.10` is your caching DNS resolver address. You will then need to restart the docker service for this configuration to take effect. You will not have to modify your dockerfile for this approach.

In summary, the lack of DNS caching inside Docker containers by default isn’t a bug, but a result of the layered network architecture and Docker’s default use of a simple, non-caching resolver. By understanding these layers and leveraging configurations or by deploying in-container caching DNS resolvers, or external caching servers, we can effectively improve DNS resolution times. The choice depends on your specific deployment context and scale. Each method introduces its own operational overhead and complexity, but the core issue ultimately revolves around where caching occurs in the network stack. For further insight, I’d recommend looking into the documentation surrounding docker networking, specifically the documentation detailing how to manipulate DNS settings and the various options available for container network configuration.
