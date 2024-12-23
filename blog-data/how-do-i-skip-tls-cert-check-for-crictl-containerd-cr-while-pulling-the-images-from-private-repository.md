---
title: "How do I skip TLS cert check for crictl (containerd CR) while pulling the images from private repository?"
date: "2024-12-23"
id: "how-do-i-skip-tls-cert-check-for-crictl-containerd-cr-while-pulling-the-images-from-private-repository"
---

Let’s tackle this. So, you're facing a common hurdle when working with `crictl` and private image registries: the dreaded TLS certificate verification. I’ve been there, more times than I care to recall, and it usually manifests when either using self-signed certificates or when dealing with internal registries behind a company firewall. The goal here isn't to haphazardly disable security; instead, it's about understanding the underlying issue and applying the necessary tweaks in a controlled manner.

Frankly, blindly disabling TLS verification is a risky proposition, even in development environments. It exposes you to potential man-in-the-middle attacks. Ideally, you should always try to use properly signed and validated certificates. However, sometimes pragmatism wins, especially in certain development or test scenarios.

The fundamental problem lies in how `crictl`, which interacts with the containerd runtime, handles TLS connections. By default, it performs strict verification of the server’s certificate against the system's trust store. When using self-signed or internally generated certificates, the verification fails because these certificates aren’t trusted by default.

To skip the certificate check, we need to influence containerd's behavior as `crictl` delegates to it. Critically, this isn't done via a simple command-line flag to `crictl` itself. The configuration needs to happen at the containerd level. The primary method involves tweaking containerd's configuration file, usually `config.toml`, found typically at `/etc/containerd/`.

Here’s a breakdown of how to do this, and I'll provide a few illustrative code snippets that directly show the required configuration modification.

First, let’s look at the structure of the `config.toml`. We are looking for the `registry.mirrors` section and specifically how we define the registry endpoint. If you are using a direct registry access, we need to add a section under `plugins."io.containerd.grpc.v1.cri".registry.mirrors` to define exception for our registry. If you are using a mirror, same rules apply to mirror configuration.

Here's an example of a `config.toml` snippet, modified to skip certificate verification for a fictional registry `my-private-registry.local:5000`:

```toml
[plugins."io.containerd.grpc.v1.cri".registry]
  [plugins."io.containerd.grpc.v1.cri".registry.mirrors]
    [plugins."io.containerd.grpc.v1.cri".registry.mirrors."my-private-registry.local:5000"]
      endpoint = ["https://my-private-registry.local:5000"]
      insecure_skip_verify = true
```
In this example, `endpoint` specifies the address of your registry, and `insecure_skip_verify = true` is the magic line that directs containerd to skip certificate validation.

To give you a complete working example, let’s assume a more complicated scenario where you might have multiple private registries with different access modes:

```toml
[plugins."io.containerd.grpc.v1.cri".registry]
  [plugins."io.containerd.grpc.v1.cri".registry.mirrors]
    [plugins."io.containerd.grpc.v1.cri".registry.mirrors."my-private-registry.local:5000"]
        endpoint = ["https://my-private-registry.local:5000"]
        insecure_skip_verify = true
    [plugins."io.containerd.grpc.v1.cri".registry.mirrors."secure-registry.example.com"]
        endpoint = ["https://secure-registry.example.com"]
        # This one defaults to TLS verification as 'insecure_skip_verify' is not set.
    [plugins."io.containerd.grpc.v1.cri".registry.mirrors."another-insecure-registry.internal:5000"]
        endpoint = ["https://another-insecure-registry.internal:5000"]
        insecure_skip_verify = true
```

Here you can see we are mixing secure registries (where we use valid certificates) and insecure registries (where we skip certificate verification). This configuration allows flexibility based on different registry setup.

After making these changes, it’s crucial to restart the containerd service for the configuration to take effect:
```bash
systemctl restart containerd
```

Always remember to double-check the `config.toml` path on your system. It might vary based on your Linux distribution or installation choices. Also, note the importance of restarting the service for the changes to actually apply.

It’s also worth mentioning the different ways you might configure the registries. If you are using a mirrored setup, the changes are a little bit different, here's an example:
```toml
[plugins."io.containerd.grpc.v1.cri".registry]
    config_path = "/etc/containerd/certs.d"
  [plugins."io.containerd.grpc.v1.cri".registry.mirrors]
      [plugins."io.containerd.grpc.v1.cri".registry.mirrors."docker.io"]
        endpoint = ["https://my-mirror.example.com"]
         insecure_skip_verify = true
```
In this example, the `docker.io` is mirrored to `my-mirror.example.com`, and for that mirror, we are disabling the verification. This configuration is useful when pulling docker hub images via an internal mirror, especially if you use a private certificate for your mirror.

Beyond skipping certificate verification, there are other things to consider regarding registry access. Authentication, for instance, requires proper configuration of credentials, usually stored in a `config.json` file (usually located at `~/.docker/config.json`), although these aren't directly related to skipping certificate checks. If you face an authentication error, you will see that on the logs of the containerd or `crictl`. In a production context, leveraging proper secrets management and not storing credentials directly in the configuration files is paramount.

For a deeper dive into these configurations, I would strongly suggest referring to the official containerd documentation on registry configuration, specifically focusing on the `config.toml` file structure and its `registry.mirrors` section. The documentation provides exhaustive information regarding the various parameters and options available. I also suggest "Programming Kubernetes" by Michael Hausenblas and Stefan Schimanski as a good resource for understanding container runtimes. The section on containerd provides good insights. Additionally, the Kubernetes documentation, although not specifically for `crictl` itself, has useful information related to interacting with container registries and secrets management, as well as containerd in general.

Remember, skipping certificate checks should be approached with caution and used only when necessary, such as in testing environments. It’s usually better to resolve the root cause of certificate validation failure. But sometimes, you need to bypass the check, for example, in a lab or staging environment where you are not dealing with real client data. This method allows you to proceed with testing and development without being blocked by certificate issues. And, most importantly, it's done at the containerd level, which is how it should be.
