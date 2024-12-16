---
title: "How can I skip TLS cert checks for crictl while pulling from private repos?"
date: "2024-12-16"
id: "how-can-i-skip-tls-cert-checks-for-crictl-while-pulling-from-private-repos"
---

Alright, let's tackle this. The situation you’re describing—needing to bypass TLS certificate verification when using `crictl` to pull images from a private registry—is a scenario I’ve certainly bumped into a few times, and it can be quite frustrating if you're not sure how to navigate it. Let me walk you through the options, drawing from some hands-on experience.

It’s important to understand why this problem occurs. Typically, when `crictl` (or any container runtime client) communicates with a registry, it establishes a secure TLS connection. This connection is verified using certificates issued by trusted certificate authorities (CAs). However, private registries often use self-signed certificates or certificates signed by a private CA. Unless the system running `crictl` trusts this private CA, certificate verification will fail. Skipping the verification entirely is definitely *not* a recommended practice in a production setting, as it introduces significant security risks, but during development or in a controlled environment, it might be a necessary evil.

I've seen this happen most frequently when setting up local development environments, or when a testing environment requires pulling from an internal registry that's still using a self-signed certificate. For the sake of this discussion, let's assume you understand the risks, and I'll approach the problem with that perspective.

The most direct, though not universally applicable, way to skip TLS verification with `crictl` is via an environment variable specific to the container runtime in use. Usually, that's containerd. For `containerd`, the relevant variable is `CONTAINERD_INSECURE_REGISTRY`. However, this approach bypasses verification on a per-registry basis, rather than a global skip. Before we get to the code snippets, here are some theoretical ideas. The core problem we're addressing is that by default `containerd` will refuse to connect to the registry without a trusted certificate. The fix will either involve adding your custom certificate to the trusted list, or ignoring certificate checks completely. The later should be done with caution.

Let's explore the practicalities.

**Option 1: Using the `CONTAINERD_INSECURE_REGISTRY` Environment Variable**

This method targets `containerd` directly. It tells the runtime to treat certain registries as "insecure," thereby skipping TLS checks for them.

```bash
# Example 1: Skipping cert checks for a single registry
export CONTAINERD_INSECURE_REGISTRY="my.private.registry:5000"
crictl pull my.private.registry:5000/myimage:latest

# Example 2: Skipping cert checks for multiple registries
export CONTAINERD_INSECURE_REGISTRY="my.private.registry:5000,another.private.registry:5000"
crictl pull my.private.registry:5000/myimage:latest
crictl pull another.private.registry:5000/anotherimage:latest
```

Here's what's happening in the code: We're setting `CONTAINERD_INSECURE_REGISTRY` to contain a comma separated list of registries and specifying the port. When `containerd` subsequently tries to pull images, it'll consult this variable. Any registry included here has its TLS certificate check bypassed. This is convenient for dev environments where you might not want to deal with certificate management, and it gives you a working proof of concept. Bear in mind that this approach isn't portable across different runtimes. If you swap out `containerd` for, say, CRI-O, you'll need a different mechanism. Note also that this *completely* disables certificate verification for that registry. There’s no partial bypass here; either the verification occurs fully, or it's skipped. This method is best for testing or local development, but for a more secure approach, consider the next option.

**Option 2: Adding the Private CA to the System's Trusted Store**

A more robust solution involves importing the private CA's certificate into the system’s trust store. This way, the TLS connection is still secure, using a trusted certificate and the bypass isn't global but per-system. This involves adding your private root CA certificate into the trusted certificate authority list on your host.

Here is how you can import the root CA certificate, on a Linux System using the `update-ca-certificates` command:

```bash
# Ensure you have the root certificate in a file, say `my-ca.crt`

# Copy the certificate to the trusted certificates directory
sudo cp my-ca.crt /usr/local/share/ca-certificates/

# Update the certificate store
sudo update-ca-certificates

# Now try your pull
crictl pull my.private.registry:5000/myimage:latest
```

In this approach, we are not bypassing any verification at all. Instead, we are making the system trust the certificate, just as it would trust any other certificates issued by a trusted certificate authority. This approach is more involved, but it doesn’t compromise the security of the connection and is portable. The system now trusts your custom root CA certificate and will trust any certificates derived from it. This would be my recommendation for most environments. It should be noted that this process varies depending on the operating system. It is similar on modern mac systems, for instance, but differs widely on windows.

**Option 3: Modifying the Containerd Configuration**

The final method I’ll cover involves editing the `containerd` configuration file directly, which can sometimes be a cleaner approach than environment variables if you're dealing with a more complex setup.

Usually, `containerd`'s configuration file lives in `/etc/containerd/config.toml`.

```toml
# /etc/containerd/config.toml
[plugins."io.containerd.grpc.v1.cri".registry]
  [plugins."io.containerd.grpc.v1.cri".registry.mirrors]
    [plugins."io.containerd.grpc.v1.cri".registry.mirrors."my.private.registry:5000"]
      endpoint = ["https://my.private.registry:5000"]
  [plugins."io.containerd.grpc.v1.cri".registry.configs]
    [plugins."io.containerd.grpc.v1.cri".registry.configs."my.private.registry:5000"]
      [plugins."io.containerd.grpc.v1.cri".registry.configs."my.private.registry:5000".tls]
        insecure_skip_verify = true
```
After you modify this configuration file, you'll need to restart the `containerd` service:
```bash
sudo systemctl restart containerd
```

Here’s the breakdown. Inside the `config.toml` file, I am directly specifying an insecure registry by setting `insecure_skip_verify = true` for that specific registry. This is more granular than the environment variable since it targets a specific registry directly through the configuration of the `cri` plugin. Also this approach is persistent across system reboots, unlike the environment variable that would be lost when the session ends. This is useful if you wish to automate the configuration of a development environment, or want to ensure consistent behavior across reboots. As with the other approaches, this method is only applicable to the `containerd` runtime, but it is more resilient and more granular.

**Recommended Reading**

To get a deeper understanding, I would suggest:

*   **"Docker Deep Dive" by Nigel Poulton:** This book provides a solid understanding of container technology and how runtimes work. While it focuses on Docker, it builds a strong foundation for understanding underlying concepts.
*   **"Kubernetes in Action" by Marko Lukša:** This book offers a detailed look into Kubernetes, which often utilizes `containerd` or similar runtimes under the hood. This resource is especially helpful if you are dealing with Kubernetes and container images simultaneously.
*   **The official containerd documentation:** This is a good reference for the configuration options, especially if you decide to modify the config files directly.

These resources will help you understand not just the "how" but also the "why," especially when it comes to security concerns.

To wrap it up, skipping TLS checks directly should be a last resort. It's more prudent to add your private CA to the system's trust store whenever possible. If you absolutely must skip it, be sure you are fully aware of the risks, and the code snippets and suggestions I've provided are in a development or internal environment.
