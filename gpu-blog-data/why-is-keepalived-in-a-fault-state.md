---
title: "Why is Keepalived in a FAULT state?"
date: "2025-01-30"
id: "why-is-keepalived-in-a-fault-state"
---
Keepalived transitioning to a FAULT state indicates a critical issue preventing it from maintaining its assigned role in a high-availability cluster. During my tenure as a systems engineer, I've frequently encountered this state, and debugging it requires a systematic approach, as numerous underlying factors can trigger the behavior. It's rarely a single, isolated problem, but rather a cascade of events typically stemming from connectivity, configuration, or resource constraints.

The core function of Keepalived rests on the VRRP (Virtual Router Redundancy Protocol), which facilitates the election of a primary router (master) amongst a group of redundant servers. A successful election and continuous master role are dependent on consistent communication with the other members. A FAULT state essentially implies the local instance of Keepalived is unable to establish or maintain its position within the group based on VRRP criteria. This can occur in both master and backup roles. The most common causes stem from either loss of network communication with peers, failing health checks, or underlying configuration errors.

Network connectivity issues are the most frequent culprit. Loss of communication with other Keepalived instances in the group results in an inability to receive or send VRRP advertisement packets. Without these packets, Keepalived cannot ascertain its place within the active/standby configuration. This can be due to a faulty network interface, misconfigured firewall rules, or issues within the network infrastructure like switch failures or incorrect VLAN configurations. It’s essential to verify basic network reachability using standard tools like ping and traceroute to isolate the potential problem at the physical, routing or network layers, and rule out these core communication issues before diving deeper.

Failing health checks are another primary source of FAULT states. Keepalived relies on script executions to verify the overall health of the system and associated applications. These scripts, configured within the keepalived configuration file, return an exit code indicating success or failure. A non-zero exit code is interpreted by keepalived as a failure, which can lead to a state transition. For instance, a health check that verifies the availability of a backend application might fail if the application process crashes or the server it runs on experiences resource limitations. Thoroughly testing and debugging these health check scripts is vital to confirm their accuracy. Furthermore, analyzing log files related to keepalived and health checks provides crucial insight into any error conditions.

Configuration issues can also drive the faulty state. Improper configuration, be it VRRP parameters, authentication settings or incorrect health check scripts will cause keepalived to operate outside the desired parameters, which will ultimately lead to a failover or prevent it from taking over when it needs to. This may mean incorrectly configured `vrrp_instance` parameters, like the priority values for each node, or a misplaced character within the configuration file.

Below are three practical examples, with their commentary and how they relate to a FAULT state.

**Example 1: Network Connectivity Issue**

This scenario involves the scenario where a network firewall is blocking VRRP packets between two keepalived nodes.

```bash
# /etc/keepalived/keepalived.conf (Node 1)
vrrp_instance VI_1 {
    state MASTER
    interface eth0
    virtual_router_id 51
    priority 150
    advert_int 1
    authentication {
        auth_type PASS
        auth_pass mysecretpassword
    }
    virtual_ipaddress {
        192.168.1.100/24
    }
}
```

```bash
# /etc/keepalived/keepalived.conf (Node 2)
vrrp_instance VI_1 {
    state BACKUP
    interface eth0
    virtual_router_id 51
    priority 100
    advert_int 1
    authentication {
        auth_type PASS
        auth_pass mysecretpassword
    }
    virtual_ipaddress {
        192.168.1.100/24
    }
}
```

**Commentary:** In this basic setup, two keepalived nodes are configured. Normally, Node 1 would assume the MASTER role because of its higher priority. However, if a firewall rule blocks the VRRP multicast packets (typically 224.0.0.18 on IP protocol 112) between Node 1 and Node 2, both nodes will not acknowledge the others’ presence. Each node will transition to a `MASTER` role, and log `VRRP_Instance(VI_1) Changing to MASTER STATE` and similar entries related to the loss of an existing master. This will result in address conflicts, and effectively put them in a FAULT state, since VRRP requires a master. This demonstrates the importance of verifying network traffic flow, and in this specific case the firewall rules. If the rule were later removed, both nodes would need a short time to reestablish their roles, based on their priority.

**Example 2: Failing Health Check**

This example demonstrates a failing health check leading to a state transition.

```bash
# /etc/keepalived/keepalived.conf
vrrp_instance VI_1 {
    state MASTER
    interface eth0
    virtual_router_id 51
    priority 150
    advert_int 1
    authentication {
        auth_type PASS
        auth_pass mysecretpassword
    }
    virtual_ipaddress {
        192.168.1.100/24
    }
    track_script {
      chk_app
    }
}

vrrp_script chk_app {
    script "/usr/local/bin/check_app.sh"
    interval 2
    weight -20
}

```

```bash
# /usr/local/bin/check_app.sh
#!/bin/bash
if ! systemctl is-active myapp; then
    exit 1
fi
exit 0

```

**Commentary:** This example introduces a health check script `/usr/local/bin/check_app.sh` that verifies the status of the 'myapp' service. If ‘myapp’ is not active, the script returns a non-zero exit code, which then causes keepalived to reduce the priority of the instance using the `weight -20` parameter. If the weight is low enough, the machine relinquishes the master role. Additionally, if both nodes were in the backup state, and one of the nodes experienced this health check failure it could cause all nodes to fall into a FAULT state, as none would be able to achieve Master status. The logs will contain entries related to script execution and priority adjustments. This highlights the importance of scrutinizing the health check scripts, and the associated application.

**Example 3: Configuration Misconfiguration**

This scenario involves a subtle configuration mismatch.

```bash
# /etc/keepalived/keepalived.conf (Node 1)
vrrp_instance VI_1 {
    state MASTER
    interface eth0
    virtual_router_id 51
    priority 150
    advert_int 1
    authentication {
        auth_type PASS
        auth_pass mysecretpassword
    }
    virtual_ipaddress {
        192.168.1.100/24
    }
}
```

```bash
# /etc/keepalived/keepalived.conf (Node 2)
vrrp_instance VI_2 {
    state BACKUP
    interface eth0
    virtual_router_id 51
    priority 100
    advert_int 1
    authentication {
        auth_type PASS
        auth_pass mysecretpassword
    }
    virtual_ipaddress {
        192.168.1.100/24
    }
}
```

**Commentary:** Here, the issue isn’t as immediately apparent. The key difference is in the `vrrp_instance` name. In Node 1 it's set to `VI_1`, and in Node 2 it’s set to `VI_2`. This configuration error will cause both instances to enter a FAULT state. Because Keepalived associates the instance ID and name as part of the VRRP protocol and its identifier, they must be identical across all the keepalived members that are in a high availability configuration. Mismatched instance names prevent correct VRRP operation, and both will revert to the FAULT state, as neither is aware of a second member. This highlights the importance of exact parameter matching within the configuration file.

In summary, troubleshooting a Keepalived FAULT state requires a methodical approach. Focus on network connectivity checks, meticulous review of health check scripts, and careful evaluation of the keepalived configuration. Log analysis of both keepalived and application logs provides essential clues and context to the events that transpire.

For further learning, consider the official keepalived documentation, available from the project's source repository or website, and any books detailing high availability system design principles. These documents explain underlying concepts and how they are applied. Additionally, articles focused on networking and VRRP can improve your understanding of the protocol's mechanics.
