---
title: "Why doesn't the nspawned container resolve certain DNS names?"
date: "2024-12-23"
id: "why-doesnt-the-nspawned-container-resolve-certain-dns-names"
---

Okay, let’s tackle this. It's a question I've seen crop up more times than I'd care to count, and usually, the solution boils down to a few common culprits when it comes to `systemd-nspawn` containers and DNS resolution issues. I've personally spent a good chunk of time debugging similar problems, particularly back when we were migrating our microservices over to a container-based architecture, and I remember the frustration vividly. So let's break it down.

The core of the problem often revolves around how `nspawn` containers handle network namespaces and their interactions with the host system's DNS configuration. By default, `nspawn` uses a virtual network interface within the container. This interface, while providing network connectivity, doesn't automatically inherit all of the host's networking setup, particularly DNS resolution. When you encounter a scenario where a container resolves some names but not others, the discrepancy usually points to a misconfigured or incomplete DNS setup within the container itself.

Let's consider the most frequent scenarios. Firstly, the container might be relying solely on the default resolver configuration which, upon its creation, might not fully replicate the host's resolver settings. Secondly, a failure might stem from a missing or misconfigured `resolv.conf` within the container. Thirdly, and perhaps less obvious, are the potential complexities of split DNS configurations where different domains resolve to different DNS servers. Finally, the host system may be using systemd-resolved, which further complicates the resolution as the container may need specific configurations to communicate with systemd-resolved on the host.

Here’s how I’ve typically approached these problems, coupled with code examples for clarity:

**Scenario 1: The Default Resolver is Insufficient**

Often, a newly created `nspawn` container starts with a very basic, often loopback-only, DNS configuration. If the host system uses specific DNS servers configured via `/etc/resolv.conf` or systemd-resolved, the container won't automatically utilize them unless explicitly told to.

*Example Code Snippet (Using `nspawn`'s `--bind` option)*

```bash
# Inside the host system
# This will bind the host's /etc/resolv.conf into the container
sudo systemd-nspawn -b -n --bind=/etc/resolv.conf:/etc/resolv.conf -D /var/lib/machines/mycontainer
```

This solution is straightforward: by using `--bind`, we mount the host’s `resolv.conf` inside the container. This means the container's DNS settings mirror the host’s, addressing the issue of a deficient default configuration. While often sufficient, it’s not ideal for all scenarios, particularly if the container requires its own specific configurations or if `systemd-resolved` is involved.

**Scenario 2: Explicitly Setting DNS Servers within the Container**

Sometimes, direct control of the container’s DNS servers is preferred. This scenario is particularly useful when the container needs access to private DNS servers or needs a different configuration altogether. This can be accomplished by editing the `/etc/resolv.conf` file within the container.

*Example Code Snippet (Directly manipulating `/etc/resolv.conf` in the container)*

```bash
# Within the running container (after you've entered it, e.g., using systemd-nspawn -j /var/lib/machines/mycontainer)
echo "nameserver 8.8.8.8" | sudo tee /etc/resolv.conf
echo "nameserver 8.8.4.4" | sudo tee -a /etc/resolv.conf
```

This snippet assumes you are within the container environment. The `tee` command, with `sudo`, allows writing to the `/etc/resolv.conf` file, setting the nameservers to Google's public DNS servers (8.8.8.8 and 8.8.4.4). This approach bypasses the host’s configuration entirely, giving you complete control within the container. Ensure that the chosen nameservers are reachable from within the container network.

**Scenario 3: Using `systemd-resolved` and `nspawn`'s `--network-veth`**

When the host system uses `systemd-resolved`, things become a bit more involved. `systemd-resolved` listens on a local socket rather than using the traditional `/etc/resolv.conf` approach. A simple bind-mount of `resolv.conf` will not suffice. In this case, we can leverage `nspawn`'s `--network-veth` option to bridge the container into the host's networking environment and configure it to properly communicate with `systemd-resolved`.

*Example Code Snippet (Using `nspawn` with a virtual ethernet interface and link to the resolved socket)*

```bash
# On the host system
# Set up a veth interface for the container and link to resolved's socket
sudo systemd-nspawn -b -n \
    --network-veth \
    --bind=/run/systemd/resolve/stub-resolv.conf:/etc/resolv.conf \
    -D /var/lib/machines/mycontainer
```

Here, we use `--network-veth` which creates a virtual ethernet interface connected to the host's network. We then *also* use the `--bind` option to mount the `stub-resolv.conf` file from the host. This `stub-resolv.conf` file is specifically designed to interact with `systemd-resolved` and allows the container to resolve DNS names using the host’s systemd-resolved instance. You also need to make sure that `systemd-resolved` is configured to listen on the loopback address so that the container can reach it. This is often done by configuring the `ListenAddress` and `DNS` directives in the `/etc/systemd/resolved.conf` file.

It is important to note that a simple `--bind` of `/etc/resolv.conf` in combination with using the `--network-veth` option may not always work as intended if the container tries to use a traditional resolver when `systemd-resolved` is active on the host. This requires understanding that the container is now within the host’s network namespace, but may not have the corresponding resolver config to use the host's `systemd-resolved`. This is why the `--bind` specifically targets the stub file and allows the container to resolve via `systemd-resolved` which is now in the host's networking namespace.

**Further Recommendations**

For deeper dives, I'd recommend the following resources:

*   **"Understanding the Linux Kernel" by Daniel P. Bovet and Marco Cesati:** This book, while broad, gives you a very solid foundation on networking concepts, including namespaces, which is essential to understanding these issues. Specifically focus on the chapters dealing with networking and containers.
*   **The `systemd-nspawn` man page:** The official documentation is indispensable. Read it thoroughly, especially the sections on networking options. Understanding how the various options interact is paramount to troubleshooting these situations.
*   **The `systemd-resolved` man page:** This provides detailed information on how `systemd-resolved` works, its configuration, and how to integrate with other systems, like containers. It's crucial if your host system utilizes `systemd-resolved` for name resolution.

In essence, DNS resolution issues with `nspawn` containers typically stem from a mismatch between the host's DNS settings and the container's configuration. Carefully examining your configuration within the container and understanding the mechanisms of `nspawn` networking and how they interact with `systemd-resolved`, if applicable, is generally the key to solving these situations. It's also beneficial to keep in mind the difference between a full network stack within the container, versus simply relying on the host's name resolution functionality.
