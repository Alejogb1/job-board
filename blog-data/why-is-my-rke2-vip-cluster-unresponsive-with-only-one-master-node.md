---
title: "Why is my RKE2 VIP cluster unresponsive with only one master node?"
date: "2024-12-23"
id: "why-is-my-rke2-vip-cluster-unresponsive-with-only-one-master-node"
---

,  I’ve seen this particular scenario unfold more times than I care to count, and it usually stems from a fundamental misunderstanding of how RKE2’s high-availability components interact, especially with a single master node. When a virtual ip (VIP) for the api server becomes unresponsive in this configuration, it’s less about some deep, hidden bug and more about architecture limitations being exposed. This isn’t about 'blaming' rke2, but about understanding its design. Let me explain.

The core issue lies in the design of the virtual ip mechanism in a high-availability setup. Typically, a vip in a distributed system relies on a mechanism for failover, typically provided by technologies such as keepalived or similar. The virtual ip, ideally, should be moving between the active nodes as needed to ensure requests can still reach the control plane even if a node is offline. However, in a cluster with *only one* master node, this failover mechanism doesn't have anyplace to fail *to*. It's a bit like having a backup generator wired only to the primary circuit – if that primary fails, the backup has nowhere to direct its power.

In rke2, when using a vip, the expectation is that there will be at least two, or preferably three, control plane (master) nodes. These nodes run services, such as the kube-api server and etcd, and participate in an election process to decide which of them should have their copy of these services exposed via the vip. If there’s only one node, that election is never fully resolved because there’s no peer to elect amongst. The vip configuration is still there, but the mechanism that actively transfers the vip to a healthy node doesn't operate correctly. Thus, if the node running the api-server service is unhealthy, the vip effectively points to a black hole.

Let's break it down further by looking at what typically happens with a proper multi-master node rke2 cluster. The virtual ip is often managed by a service like `kube-vip`, which runs on each master node. In an ideal scenario, here’s a simplified version of the process:

1.  **Initialization:** Each master node starts and participates in an election. One of them wins and becomes the 'leader'.
2.  **Virtual IP assignment:** The leader claims the vip, making it accessible via that node.
3.  **Monitoring:** Other nodes monitor the leader. If the leader goes down, another election is triggered, a new leader is selected, and the vip gets moved to the new leader.
4.  **Redundancy:** Should any node fail, others automatically take over services using the virtual ip.

With only one master node, we bypass this whole orchestrated dance. The node still *has* the vip configured on its interface. However, without the other nodes, there’s no way to transfer the vip if the node’s services are down or unhealthy, even if the node itself is online. To compound this, the vip does not represent a service residing on the host's local interface. Instead, this vip is used to connect to the internal cluster components; meaning local node services may still operate while the cluster services, accessed by the vip, are unavailable.

Here are three examples that illustrate common problems and how they appear in code or configuration snippets:

**Example 1: Incorrect rke2 configuration for a single master node**

This is often due to a misunderstanding of the documented requirements, and a misconfiguration can be a major issue, particularly in production setups.

```yaml
# This is an incorrect configuration
server: true
token: "some-token"
tls-san:
  - "my-single-master.mydomain.com"
  - "10.10.10.10" # This is the ip address for the node
cluster-cidr: "10.42.0.0/16"
service-cidr: "10.43.0.0/16"
# This config *does not specify* the need for a vip
# or any external load balancers.
```

In the code above, there is no mention of a load balancer, virtual ip address, or secondary host to failover to. While you might expect rke2 to function without the need for HA, this configuration still activates the HA components of rke2, but without other hosts to support it. The vip is declared, but the election mechanism lacks a quorum or peer to act with. This would cause the vip to be unresponsive if services on that host fail. The correct solution in this case is to not use a virtual IP at all, and point client connections directly to the master node’s IP, since there are no failover nodes to direct to.

**Example 2: kube-vip configuration attempting to manage a non-HA ip address**

This is an example of incorrect or incomplete `kube-vip` configuration:

```yaml
# This is *incorrect* in a single-node setup
apiVersion: kube-vip.io/v1
kind: KubeVipConfig
metadata:
  name: kube-vip
  namespace: kube-system
spec:
  vip: "10.10.10.100"  # This is the virtual IP
  # This configuration *does not provide* another node to take over if the primary goes down
  interface: "eth0" # Interface where the VIP should be bound
  # This is the root cause of the problem with single nodes.
```

Here, the configuration itself is trying to create a highly available virtual ip. The missing piece is that the virtual ip needs to be moved to *another* control plane node if this node stops responding to api requests. Because no such secondary exists, the api server will become unresponsive via the vip if it fails on this host.

**Example 3: Misunderstanding of vip use case**

This involves assuming the VIP will just 'work' on a single node setup with a vip enabled in rke2 config file (this often happens when someone has copied over a cluster config meant for a multi-master setup)

```bash
# rke2 config file - assumes we are using HA
server: true
token: "some-token"
tls-san:
  - "my-cluster.mydomain.com"
  - "10.10.10.100" # This is the virtual IP
cluster-cidr: "10.42.0.0/16"
service-cidr: "10.43.0.0/16"
# no other HA related settings included...
```

In this instance, the configuration specifies a `tls-san` using a vip, which rke2 interprets as an attempt at HA. But without the necessary components in place (other servers), the `kube-vip` instance on the single node is trying to manage failovers to a node which does not exist, and so the vip itself is unreachable.

To fix this, for a single master node setup, you must either *not* use a virtual ip at all, and access the api server directly via the host IP address, or understand that a *working* cluster will require additional master nodes.

For understanding the concepts here in depth, I’d recommend looking into the following resources:

*   **“Kubernetes in Action” by Marko Lukša:** This book provides a good overview of kubernetes architecture, including the control plane and how HA is typically implemented.
*   **“Designing Data-Intensive Applications” by Martin Kleppmann:** The chapters on consensus and distributed systems can help develop a better intuition for how quorum-based systems like rke2 and etcd work, explaining why a single node cannot maintain a working consensus.
*   **The Kubernetes documentation itself:** Specifically, look into the sections on the control plane, highly available configurations, and load balancing. These are key to understanding how a vip operates in a multi-node setup. Also, review the specific RKE2 documentation related to High Availability configurations.

In summary, when you find your rke2 cluster unresponsive with a vip on a single node, the underlying cause will almost always be related to the inherent limitations of a high-availability solution being used in a non-high-availability environment. It's critical to grasp the architecture at play. The fix isn't usually about 'fixing' rke2, but adjusting expectations and configurations based on the available resources, and planning your clusters to match real world needs.
