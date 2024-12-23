---
title: "How do I configure the volume in a Pi-hole docker Portainer Stack?"
date: "2024-12-23"
id: "how-do-i-configure-the-volume-in-a-pi-hole-docker-portainer-stack"
---

Okay, let's unpack this. I've definitely been down this particular rabbit hole before, wrestling with the nuances of docker volumes, specifically within the confines of a Portainer stack while trying to tame a Pi-hole instance. It's a fairly common scenario, and thankfully there's a pretty established path to get this right. You're essentially dealing with the persistence of Pi-hole data across container restarts, which is crucial if you want to maintain your blocklists, settings, and query logs. The core issue stems from the ephemeral nature of docker containers; without proper volume configuration, any data created or modified within the container will vanish when the container is stopped or removed.

The primary method for data persistence in docker is, of course, through the utilization of volumes. These volumes can be either host-mounted (binding a directory on your host machine to a directory within the container) or docker-managed volumes (where docker handles the storage location). For Pi-hole within a Portainer stack, I usually gravitate towards the former – host-mounted volumes. It offers direct visibility and control over where the data is stored on the host, which, in my experience, simplifies troubleshooting and backups.

Now, let's address *how* this translates into the configuration of your docker compose file, which Portainer will then consume when creating or updating your stack. The crucial piece is the `volumes` section within your `pihole` service definition. Let’s break down a sample compose file segment showcasing this, and then I'll follow with more concrete examples.

Here's the core concept:

```yaml
services:
  pihole:
    image: pihole/pihole:latest
    container_name: pihole
    ports:
      - "53:53/tcp"
      - "53:53/udp"
      - "80:80/tcp"
    volumes:
      - ./pihole/etc-pihole:/etc/pihole
      - ./pihole/etc-dnsmasq.d:/etc/dnsmasq.d
    environment:
      - TZ=Your/Timezone
```

In this snippet, ` ./pihole/etc-pihole:/etc/pihole` maps the directory `/etc/pihole` *inside* the container to a directory named `etc-pihole` located inside a directory named `pihole` that will be created in the same location where the `docker-compose.yml` file lives *on your host*. The same logic applies to ` ./pihole/etc-dnsmasq.d:/etc/dnsmasq.d`.  Any change to files in `/etc/pihole` within the container will also appear in `./pihole/etc-pihole` on your host, ensuring your Pi-hole settings and configuration persist even when the container is removed.

This leads me to the first, slightly more developed example, where we incorporate more of the common environment variables you might need:

```yaml
version: "3.7"

services:
  pihole:
    image: pihole/pihole:latest
    container_name: pihole
    ports:
      - "53:53/tcp"
      - "53:53/udp"
      - "80:80/tcp"
    volumes:
      - ./pihole/etc-pihole:/etc/pihole
      - ./pihole/etc-dnsmasq.d:/etc/dnsmasq.d
    environment:
      - TZ=America/New_York # Set your timezone appropriately
      - WEBPASSWORD=YourSecureWebPassword
      - DNS1=1.1.1.1
      - DNS2=1.0.0.1
```
Here, in addition to the volume mappings, I have also included environment variables to specify my timezone, set the web interface password, and configure the upstream dns servers. The key here is understanding the relationship between the host directories on the left of the colon and the container directories on the right. These volumes will ensure that critical Pi-hole settings are stored persistently on the host machine.

Now, let's address a more complex scenario, one that I experienced a couple of years back involving using a specific host interface on which to listen. This required tweaking the network settings, and consequently, the compose file volume setup. For simplicity, I will again bind to a single host interface and not use host networking to demonstrate the volume configuration. This configuration is more tailored for scenarios where you have multiple interfaces and you do not want the docker container to use all interfaces. This will include an explicit `interface` variable.

```yaml
version: "3.7"
services:
  pihole:
    image: pihole/pihole:latest
    container_name: pihole
    ports:
      - "53:53/tcp"
      - "53:53/udp"
      - "80:80/tcp"
    volumes:
      - ./pihole/etc-pihole:/etc/pihole
      - ./pihole/etc-dnsmasq.d:/etc/dnsmasq.d
    environment:
      - TZ=Europe/London # Set your timezone appropriately
      - WEBPASSWORD=AnotherSecurePassword
      - DNS1=8.8.8.8
      - DNS2=8.8.4.4
      - INTERFACE=eth0 # Or your specific interface name
```

In this example, you’ll notice that we have added the variable `INTERFACE=eth0`. The crucial part, from our topic's perspective, is that even with this additional networking complexity, our volume configuration remains identical. We are still mapping `/etc/pihole` and `/etc/dnsmasq.d` to a local directory on our host system, maintaining data persistence irrespective of network configuration changes or specific interface usage.

Finally, let's examine a practical example where, like me, you might want additional local customisations. I encountered one such scenario where I wanted to place custom DNS entries in local.list. Therefore, I decided to map the file within the container to a file on my host machine. You'll notice the shift from a directory volume mapping to a single file volume mapping.

```yaml
version: "3.7"
services:
  pihole:
    image: pihole/pihole:latest
    container_name: pihole
    ports:
      - "53:53/tcp"
      - "53:53/udp"
      - "80:80/tcp"
    volumes:
      - ./pihole/etc-pihole:/etc/pihole
      - ./pihole/etc-dnsmasq.d:/etc/dnsmasq.d
      - ./pihole/etc-pihole/local.list:/etc/pihole/local.list
    environment:
      - TZ=Australia/Sydney
      - WEBPASSWORD=YetAnotherPassword
      - DNS1=9.9.9.9
      - DNS2=149.112.112.112

```

Notice that in addition to the familiar directory mappings, I have added ` ./pihole/etc-pihole/local.list:/etc/pihole/local.list`. This will ensure that the `local.list` file within the Pi-hole container is mapped directly to the `local.list` on the host, allowing you to maintain and modify the custom dns entries on your host, and the changes will be reflected in the Pi-hole container.

When working with docker volumes, I recommend consulting *Docker in Action* by Jeff Nickoloff and *The Docker Book* by James Turnbull. These are, in my opinion, excellent resources for developing a deeper understanding of docker concepts, especially regarding volumes. For more specifics on Pi-hole configurations within a docker context, the official Pi-hole documentation is a must-read.

In summary, configuring volumes for Pi-hole in a Portainer stack is essential for persistence. The core idea is to map the relevant directories within the Pi-hole container to directories on your host, ensuring your settings, logs, and customisations are preserved across restarts. The actual method of doing this involves carefully crafting the `volumes` section in your docker-compose.yml file. This approach provides a reliable and transparent method for managing your Pi-hole data within the docker ecosystem. Remember to tailor the file paths and configurations to your specific needs. If you encounter further complexities, exploring docker volume drivers or named volumes is the next step.
