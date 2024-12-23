---
title: "How do I skip TLS certificate checks for crictl when pulling from a private repo?"
date: "2024-12-23"
id: "how-do-i-skip-tls-certificate-checks-for-crictl-when-pulling-from-a-private-repo"
---

,  I recall a particularly frustrating incident back in my early cloud-native days. We had a small, isolated development cluster with a private image registry. The goal was simplicity: quick iterations, minimal overhead. We initially opted for self-signed certificates, and boy, did that introduce complications, especially with `crictl`. Ignoring TLS verification seemed like a quick fix initially, but, as you'll see, it's not always the best approach. There’s a proper way to navigate this, and I’d rather avoid the usual advice that glosses over the underlying mechanics.

The core issue stems from the secure nature of TLS. `crictl`, a command-line interface for container runtimes that follow the CRI (Container Runtime Interface) standard, inherently validates the certificates presented by the remote registry. When using self-signed or certificates issued by an untrusted authority, the validation process fails, preventing image pulls. Just disabling this check globally can introduce security risks. Hence, the need for more nuanced solutions.

Let me break down the common approaches, starting with the least recommended and moving towards more secure practices:

1. **The Forceful Bypass (Use With Caution):** This involves explicitly telling `crictl` to ignore TLS verification. In most cases, you'd achieve this through environment variables. For example, the `CRI_CONTAINERD_INSECURE_REGISTRY` environment variable might tempt us into this path. We might be setting `CRI_CONTAINERD_INSECURE_REGISTRY="my-private-registry.local"`, but this is highly discouraged, particularly in production or any environment exposed to external threats. It circumvents the security that TLS is designed to provide. In my experience, this approach was a stepping stone in the beginning because of time constraints, and it ultimately lead to more work down the line.

   Here's how it might be implemented (avoid it if you can):

    ```bash
    export CRI_CONTAINERD_INSECURE_REGISTRY="my-private-registry.local:5000"
    crictl pull my-private-registry.local:5000/my-image:latest
    ```

   This sets the insecure flag for that specific registry. Though it might initially seem straightforward, understand that it weakens the overall security model by neglecting certificate verification. This approach should ideally be temporary.

2. **Properly Configuring `containerd` with Certificates:** The much better way is to instruct `containerd`, the container runtime often used behind the scenes with `crictl`, to trust your certificate authority. This means placing your private registry's certificate (or the certificate authority's certificate if your registry uses a certificate from a custom ca) where `containerd` can access and validate it. This method is more secure and recommended for any persistent environment. `containerd` uses a specific directory and format for storing these trusted certificates.

   Typically, you would:

   * Obtain the certificate (`.crt` or `.pem`).
   * Place it into `/etc/containerd/certs.d/my-private-registry.local:5000/`. Note, the directory name corresponds to the full registry address.
   * The certificate file must have a `.crt` extension, for instance `ca.crt`.
   * Restart `containerd` or send a signal to reload its config.

   The configuration looks like this:

   ```bash
   # (Assuming you have your certificate in ca.crt)
   sudo mkdir -p /etc/containerd/certs.d/my-private-registry.local:5000
   sudo cp ca.crt /etc/containerd/certs.d/my-private-registry.local:5000/ca.crt
   sudo systemctl restart containerd
   ```

   After doing this, `crictl` should be able to pull images from that registry without issues because `containerd` now trusts the certificates. It's important to use the full registry address, including the port, in the directory structure. This ensures that the correct certificate is used when connecting to the registry.

3. **Configuring the CRI Config for Remote Registries:** Often a more scalable approach, especially if you manage multiple registries, is to manage the registry configuration via the `crictl` configuration. This method allows you to manage the registry configuration using a CRI configuration file, rather than directly manipulating filesystem certificates. The CRI configuration file is specific to the container runtime being used (in this case, `containerd`). This file is commonly located at `/etc/crictl.yaml` and often referenced via the `--config` flag of `crictl` commands. If you examine the `registries` section of this config file, you will find that you can configure registry endpoint entries to include `insecure_skip_verify` or to configure `tls_config` settings to reference `cert_file`, `key_file` or `ca_file`.

   Here’s what a snippet in your `/etc/crictl.yaml` file might look like. Note that this assumes you have the correct `ca.crt` available, but you don’t want to manipulate the `containerd` directories directly.

    ```yaml
    runtime-endpoint: unix:///run/containerd/containerd.sock
    image-endpoint: unix:///run/containerd/containerd.sock
    timeout: 2
    debug: false
    pull-image-on-create: true
    registries:
      'my-private-registry.local:5000':
        insecure_skip_verify: false
        tls_config:
           ca_file: /path/to/your/ca.crt
    ```
  Then, when running `crictl`, you should specify this config:

   ```bash
   crictl --config /etc/crictl.yaml pull my-private-registry.local:5000/my-image:latest
   ```

   This method centralizes the registry configuration and is cleaner to maintain, especially in environments with a high volume of registry interactions. This also allows for more dynamic registry configurations as you can provide different certificates for multiple registries without making direct file system changes.

It's critical to understand that these methods interact differently with the various layers of container infrastructure. Bypassing TLS validation introduces risks and is generally advised against except for very controlled development environments. The preferred strategy involves ensuring that `containerd` (or your container runtime) trusts the certificate of the registry. If you want to go further in depth into registry configurations and the different ways they can be handled in `containerd`, you should be sure to check the `containerd` documentation itself. Additionally, the Container Runtime Interface (CRI) spec documents how configuration is supposed to work. You can also find excellent information on TLS and certificate management in the excellent *Network Security with OpenSSL* by Dr. John Viega, and *Cryptography Engineering* by Niels Ferguson, Bruce Schneier, and Tadayoshi Kohno.

In conclusion, when dealing with TLS certificate issues when pulling images with `crictl`, avoid resorting to insecure workarounds like `CRI_CONTAINERD_INSECURE_REGISTRY`. Opt for either placing your certificate within `containerd`'s certificate directory or configuring your CRI config correctly. These secure methods ensure your deployments remain robust and your images are validated against trusted registries. Each method provides a different trade off, and it is important to select the one best for your specific circumstances and long-term needs.
