---
title: "How to skip TLS cert checks for crictl while pulling images from private repo?"
date: "2024-12-23"
id: "how-to-skip-tls-cert-checks-for-crictl-while-pulling-images-from-private-repo"
---

Alright, let's tackle this. I've definitely been down that particular rabbit hole myself, particularly during a large-scale migration of our internal container infrastructure. Skipping TLS certificate verification for `crictl` when pulling images from a private registry, while seemingly straightforward, requires a bit of nuanced understanding to do it safely and effectively. It's certainly not something you'd want to do in a production environment without a clear strategy, but for development or controlled testing, it can be a necessary evil.

The fundamental challenge arises because `crictl`, a command-line interface for interacting with container runtimes compliant with the container runtime interface (cri), by default, expects valid TLS certificates when communicating with a registry over https. This mechanism is there for security, ensuring you're connecting to the intended registry and that the data transmission is encrypted. When dealing with self-signed or otherwise invalid certificates common in internal setups, `crictl` will correctly refuse the connection, throwing errors related to certificate verification failure. We can’t just bypass this blindly without considering the ramifications.

Now, there isn’t a single, direct flag in `crictl` to disable certificate verification, and that’s probably a good thing. What we need to do instead, is to influence the underlying container runtime (e.g., containerd, cri-o) which `crictl` interacts with. This influence primarily involves modifying the runtime’s configuration to trust our custom registry. This is done differently across various runtimes. My experience is primarily with `containerd`, so that’s the primary focus here, but I will briefly mention cri-o as well.

Let's illustrate how to bypass TLS checks with examples, starting with `containerd`.

**Example 1: Modifying `containerd` Configuration (`containerd.toml`)**

The main configuration file for `containerd` usually resides at `/etc/containerd/config.toml` (the location might vary slightly depending on your distribution). We need to add our registry to the `plugins.cri.registry.mirrors` section. Within this section, you'll configure the specific registry and instruct `containerd` to skip TLS verification when connecting to it. I found this approach to be the most reliable in my team's experience, as it provides granular control.

```toml
[plugins."io.containerd.grpc.v1.cri".registry]
  [plugins."io.containerd.grpc.v1.cri".registry.mirrors]
    [plugins."io.containerd.grpc.v1.cri".registry.mirrors."my.private.registry:5000"] # Use your registry address
      endpoint = ["https://my.private.registry:5000"]
      skip_verify = true
```

In this snippet, replace `my.private.registry:5000` with the actual address of your private registry, including the port number if it's non-standard. The `skip_verify = true` line is what tells `containerd` to bypass TLS checks for connections to this particular registry. Remember, making this change means you are trusting this registry without the security guarantees that certificate verification provides, so it should be done with caution.

After modifying the `config.toml` file, you need to restart the `containerd` service for the changes to take effect:

```bash
systemctl restart containerd
```

With this configuration change, `crictl` should now be able to pull images from your private registry without encountering certificate verification errors. It's crucial to understand that this is a system-wide change, affecting all containers pulled using `containerd`.

**Example 2: Using `crictl` with a Custom Docker Config File**

While modifying `containerd` directly is the most common and robust method, an alternative approach is to leverage a custom docker configuration file. This approach might be useful in scenarios where directly altering the `containerd` configuration is not easily possible, or you need a more container-specific solution. We can accomplish this with the `crictl pull` command using the `--creds` parameter.

First, create a docker config file named `config.json`. Replace placeholders with actual values:

```json
{
  "auths": {
    "my.private.registry:5000": {
      "auth": "<base64 encoded username:password>",
      "insecure_skip_tls_verify": true
    }
  }
}
```

The `<base64 encoded username:password>` should be the base64 encoded version of your username and password, joined with a colon. You can generate this using base64 encoding tools (e.g., `echo -n 'username:password' | base64`). The crucial part here is `insecure_skip_tls_verify`: `true` which tells the CRI to skip TLS verification.

Now use the following `crictl pull` command:

```bash
crictl pull --creds /path/to/config.json my.private.registry:5000/my-image:latest
```

The `--creds` parameter points to the custom docker config file we created and this approach applies only to this pull command.

**Example 3: cri-o Alternative Approach (Brief Mention)**

If you happen to be using `cri-o` instead of `containerd`, the configuration process differs slightly. You'll typically modify `/etc/crio/crio.conf`. Inside, you'll find a `[registries]` section where you can add configurations for specific registries:

```toml
[registries.insecure]
  prefixes = ["my.private.registry:5000"]
```

The crucial part here is adding the prefix of your registry into the `prefixes` array under the `[registries.insecure]` config group. This will enable `cri-o` to bypass TLS for the specific mentioned registry prefix. You will need to restart the `cri-o` service to apply changes after making modifications to the configuration file using:

```bash
systemctl restart crio
```

These configurations, especially enabling `skip_verify = true` or equivalent, need careful consideration. They lower the security bar, making you vulnerable to man-in-the-middle attacks if the network between the node and the registry is compromised. Thus, only implement these in controlled and isolated environments like local development setups, test clusters, or private networks where the risk is considered acceptable.

For authoritative resources on this topic, I would highly recommend consulting the official documentation for `containerd` and `cri-o`. Specifically, look for sections related to registry configuration, mirrors, and authentication. Also, the kubernetes CRI documentation is very helpful to understand how CRI runtimes communicate with kubelet. These resources will provide the most accurate and up-to-date information. For a deeper dive into TLS and certificate management in general, consider checking out "Serious Cryptography" by Jean-Philippe Aumasson, which contains great foundational knowledge about applied cryptography in different contexts. And, for a comprehensive understanding of containers, I recommend reading "Docker Deep Dive" by Nigel Poulton.

Remember, modifying security settings should always be done with full awareness of the potential risks and with a plan to revert or enhance security as needed. Security is not a one-time configuration, but a continuous process of adaptation and vigilance.
