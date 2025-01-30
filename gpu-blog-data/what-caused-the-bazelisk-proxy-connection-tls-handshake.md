---
title: "What caused the Bazelisk proxy connection TLS handshake timeout?"
date: "2025-01-30"
id: "what-caused-the-bazelisk-proxy-connection-tls-handshake"
---
The core of Bazelisk proxy connection TLS handshake timeouts typically stems from misconfigurations or resource limitations impacting the secure communication channel between Bazelisk and the remote Bazel registry or server. I've observed this firsthand across several large-scale build environments, each exhibiting slightly different contributing factors.

Let's break down the handshake process and examine potential pitfalls. The TLS handshake, at its most basic, involves the client (Bazelisk, in this case) and the server agreeing upon a secure encryption algorithm and generating session keys. This sequence entails several messages exchanged: client hello, server hello, certificate exchange, key exchange, and finally, the change cipher spec and finished messages. A timeout during any of these steps indicates a failure to complete the handshake in an allotted period.

One common issue is network connectivity. If the client's network path to the registry or server is unreliable, packets related to the handshake might get dropped, delayed, or corrupted. These delays can easily push the exchange past the predefined timeout. Packet loss or significant latency, often due to firewalls, proxy servers, or congested networks, creates the most apparent class of issues. I’ve troubleshot such scenarios where a seemingly healthy network nonetheless had high levels of packet loss on the network segment between the CI server and the corporate registry.

Another frequently encountered root cause lies within the server’s configuration. The server might be under heavy load, lack available resources, or have misconfigured cipher suites. When the server struggles to process incoming handshake requests, it may be delayed or entirely unresponsive, culminating in the timeout on the client-side. On one occasion, the registry was overwhelmed during the end-of-day build pipeline spike, resulting in sporadic handshake failures across the entire CI infrastructure. Similarly, I've seen issues when the registry's configured TLS ciphers were incompatible with the client's advertised capabilities, leading to a communication stall.

Furthermore, the client's configuration, Bazelisk in our specific case, can also play a role. In older versions, there might be inconsistencies or bugs in the TLS negotiation logic. Although less common, incorrect or outdated system configurations on the machine executing Bazelisk may also cause problems. These might involve the local root CA certificates not being up-to-date, causing the client to distrust the server's certificate. Moreover, I’ve worked with setups where a locally configured proxy server (not the Bazel registry proxy) interfered with the TLS handshake, creating similar timeouts.

To illustrate potential solutions, consider the following scenarios and their respective code examples.

**Scenario 1: Network Intermittency**

Suppose network latency is suspected. Using `curl`, you could diagnose network performance directly:

```bash
#!/bin/bash

START_TIME=$(date +%s.%N)
curl -v https://<your_registry_or_server_url>/some-resource > /dev/null 2>&1
END_TIME=$(date +%s.%N)

ELAPSED=$(echo "$END_TIME - $START_TIME" | bc)

echo "Request time: $ELAPSED seconds"

if [[ $(echo "$ELAPSED > 5" | bc) -eq 1 ]]; then
    echo "Possible network latency issue detected"
fi
```

**Commentary:** This script uses `curl` to access a resource from the suspect server while providing verbose output via `-v`. The script measures the request time, and if it exceeds a threshold (here, 5 seconds), it reports a potential latency problem. In production environments, you'd want more sophisticated methods for monitoring, but this serves as a starting point. In the verbose `curl` output, one should look for indications of slow connection times, retries, or TLS handshake errors.

**Scenario 2: Client-Side Proxy Interference**

If a local proxy is interfering, you might bypass it for testing. Here is how you can explicitly tell Bazelisk not to use any proxy:

```bash
#!/bin/bash

NO_PROXY="<your_registry_or_server_domain>" bazelisk build //...

if [[ $? -ne 0 ]]; then
  echo "Bazelisk build failed. Check if NO_PROXY or proxy configuration is incorrect."
fi
```

**Commentary:**  The environment variable `NO_PROXY` will bypass any configured HTTP or HTTPS proxies for specific domains or IP addresses. In this example, if your Bazel registry URL resolves to `registry.company.com`, replace the placeholder with this domain to disable proxying to that registry. Running this and re-testing can often help establish if a proxy is interfering.

**Scenario 3: Incompatible Cipher Suites**

If cipher mismatches are at play, you can attempt to configure the client with a narrower set of ciphers. Bazelisk itself does not directly expose options to control cipher suites, as these are typically handled by underlying libraries such as gRPC. Therefore, one must examine the server setup, but as a stopgap, or if you are unable to influence the server configuration, one can attempt this using `openssl`. I've found it instructive:

```bash
#!/bin/bash

openssl s_client -connect <your_registry_or_server_url>:443 -cipher 'ECDHE-RSA-AES256-GCM-SHA384'

if [[ $? -ne 0 ]]; then
  echo "Openssl connection failed. Check certificate configuration and cipher compatibility."
fi
```

**Commentary:** This script attempts a direct TLS handshake using `openssl` using a specific, commonly compatible cipher suite. If this connection fails, it indicates a mismatch between the client and server cipher suites. This helps to isolate the issue to the TLS protocol.

When troubleshooting these issues, I typically begin by verifying network connectivity and latency using tools like `ping`, `traceroute`, or the aforementioned `curl` tests. I'd also recommend inspecting system logs for error messages related to network or TLS failures, both on the client and the server side, if possible. Detailed log analysis can often provide clues about the exact point of failure. On the client-side, enabling Bazelisk’s verbose output, via the `--verbose_failures` or `-v` flags, will enhance debugging.

Furthermore, I advise keeping all involved components up to date. This includes Bazelisk, Bazel, the operating system, and the TLS libraries. Outdated software may contain bugs that affect the TLS handshake process. I’ve encountered issues with older versions of OpenSSL having default cipher settings which did not match the server defaults.

When examining server configurations, confirm the server's available resources (CPU, memory), verify configured cipher suites, and ensure certificates are valid and correctly installed. If a load balancer is in place, verify it’s healthy and configured correctly. For the client side, examine firewalls, any existing proxy configurations, and any network or routing related settings.

In terms of further resources, I've found that understanding the structure of TLS handshake messages (e.g., the Client Hello, Server Hello, and so forth) is beneficial. Documentation and tutorials detailing TLS/SSL troubleshooting using tools like `openssl` or `wireshark` prove insightful. Finally, the official Bazelisk documentation often has useful pointers, though these may not provide specific help for TLS handshake issues. Look instead for general error messages and patterns common to underlying libraries like gRPC.
