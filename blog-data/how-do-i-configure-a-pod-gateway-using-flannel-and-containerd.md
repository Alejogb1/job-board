---
title: "How do I configure a pod gateway using Flannel and containerd?"
date: "2024-12-23"
id: "how-do-i-configure-a-pod-gateway-using-flannel-and-containerd"
---

Alright, let's talk about pod gateways with Flannel and containerd. I’ve tackled this setup a few times in the past, most notably while orchestrating a multi-node cluster for a distributed analytics platform where network performance was absolutely critical. It’s a bit more involved than just slapping together a quick demo, so let’s break it down piece by piece.

At its core, what we’re trying to achieve is a way for pods running on different nodes to communicate with each other, even though they’re in different network namespaces. Flannel, a rather prevalent CNI (Container Network Interface) plugin, handles that by creating a virtual overlay network. Think of it as a separate, abstract network that runs on top of your physical infrastructure, allowing containers to have their own unique IP addresses, irrespective of their host's IPs. Containerd, on the other hand, is our underlying container runtime. It's responsible for actually running those containers and interfacing with the kernel.

The process involves a few moving parts, and while there are alternative ways to do this (such as using Calico or Weave Net instead of Flannel), I'll focus on the Flannel/containerd combination since that's the specific request. The primary challenge is not just installing the tools, it's ensuring the configuration is correct so the pod-to-pod networking actually functions as intended. This means configuring Flannel to create the overlay network correctly and then making sure that containerd is using the CNI plugin correctly.

Let’s start with the containerd configuration. Containerd relies on a configuration file, usually found at `/etc/containerd/config.toml`. The crucial parts related to CNI are within the `[plugins."io.containerd.grpc.v1.cri".containerd.runtimes.runc.options]` section. Specifically, you need to specify the CNI configuration directory and the CNI plugin binary directory. Here's what a relevant section might look like:

```toml
[plugins."io.containerd.grpc.v1.cri".containerd.runtimes.runc.options]
  SystemdCgroup = true
  [plugins."io.containerd.grpc.v1.cri".containerd.runtimes.runc.options.cni]
    bin_dir = "/opt/cni/bin"
    conf_dir = "/etc/cni/net.d"
```

In this snippet:

- `SystemdCgroup = true`: This isn't directly related to CNI, but it's a good practice to have for proper resource management with systemd.
- `bin_dir`: This points to where the actual CNI plugin binaries are located. For Flannel, that would include `flannel` and likely some helper utilities.
- `conf_dir`: This specifies the directory containing the CNI configuration files. Each configuration file in this directory represents a CNI network definition.

Now, you’ll need a configuration file for Flannel in `/etc/cni/net.d`. Let's call it `10-flannel.conflist`. This file defines how Flannel should work. Below is a typical configuration, and I've highlighted key parameters:

```json
{
  "cniVersion": "0.3.1",
  "name": "flannel",
  "plugins": [
    {
      "type": "flannel",
      "delegate": {
        "hairpinMode": true,
        "isDefaultGateway": true,
        "ipMasq": true
      }
    },
    {
        "type": "portmap",
        "capabilities": {"portMappings": true}
    }
  ]
}
```

Let's break this down:

- `"cniVersion"`: Specifies the version of the CNI specification used.
- `"name"`: The name of the network.
- `"plugins"`: This is an array of plugins that are executed in order to configure the network. The first plugin is of type `"flannel"`.
   -  `"hairpinMode": true`: Enables communication between pods on the same node using the loopback interface. This was crucial in some of my past deployments where service discovery required same-node pod interactions.
   - `"isDefaultGateway": true`: This makes Flannel the default gateway for pods, directing traffic appropriately across the overlay network.
   - `"ipMasq": true`: Enables masquerading, which allows the pods to communicate with external networks (like the internet) using the node's IP address.
- The second plugin is of type `"portmap"`, which is a standard CNI plugin used to handle port forwarding.

These files alone don’t make the system work. Flannel relies on a configuration backend, usually etcd, to distribute network configuration across all nodes. You'll need to ensure that each node in the cluster can access your etcd instance. After that, Flannel will write network configuration based on the cluster's node and pod CIDR.

Finally, we come to actually installing the software. You'll need to install the Flannel binary and the `cni-plugins` package on all nodes within your cluster. The specific method varies based on your distribution, but it usually involves extracting tarballs from releases and placing binaries in `/opt/cni/bin`. You can obtain the release archives from the official github repository of `flannel` and `cni-plugins`. You will also need to ensure containerd is correctly configured. Here is example of command that can be used to restart containerd after configuration changes:

```bash
systemctl restart containerd
```
This is important to ensure containerd picks up the configuration changes related to the CNI.

Now, to verify this is all working, you'll need to create some pods. Ensure that the pod configuration includes annotations that point to the correct CNI configuration (in our case, "flannel"). You could do this using kubernetes deployment yaml or other tools such as docker compose. The easiest way to test connectivity will be to start two pods, on different nodes if you have them, and try to `ping` each other using the pod ip addresses. You should see those pings succeed if everything is working as intended. You could also use `kubectl exec` command to debug pod networking with common tools such as `ping`, `traceroute`, and `tcpdump`. I also found that a good tool to use during development is `cnitool`, which can be used to test CNI configurations outside of the context of containerd or kubernetes.

That covers the basics. It's not a simple process, but with these configuration steps in place, Flannel should set up its overlay network and allow your pods to communicate with each other as if they're on the same local network.

For further reading, I’d highly recommend the following resources:

1. **The CNI Specification** (available on the CNCF website): This is essential for understanding the fundamental building blocks of how CNI plugins operate. The official specification will be the most authoritative definition.
2. **The official Flannel documentation:** You can find it on the Flannel Github repository. This provides a detailed overview of Flannel’s features and how to configure it for different environments.
3. **"Programming Kubernetes" by Michael Hausenblas and Stefan Schimanski:** This book provides a deep dive into Kubernetes networking, including how CNI plugins such as Flannel work within the Kubernetes ecosystem. Although we're focusing on containerd directly, understanding the kubernetes model also clarifies many underlying concepts.

By carefully following these steps and consulting the resources, you should be well on your way to creating a functioning pod network using Flannel and containerd. It is a foundational skill when creating distributed systems, and spending the time to understand it now will pay huge dividends in the future.
