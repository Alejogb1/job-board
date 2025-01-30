---
title: "Why is ACI not receiving UDP packets?"
date: "2025-01-30"
id: "why-is-aci-not-receiving-udp-packets"
---
Application Convergence Infrastructure (ACI) fabrics, while offering significant advantages in network virtualization and automation, can present challenges when dealing with UDP traffic.  My experience troubleshooting similar issues in large-scale deployments points to a core reason:  the default behavior of ACI's policy-based architecture often filters or implicitly drops UDP packets unless explicitly permitted. This isn't necessarily a bug; it's a security feature designed to prevent unwanted or malicious UDP floods.


**1. Explanation: Policy-Based Filtering in ACI**

ACI's strength lies in its policy-driven approach.  Every aspect of network communication, including which endpoints can communicate and the types of traffic allowed, is defined through policies.  This policy-based approach, while incredibly powerful for managing large, complex networks, implicitly means that any traffic not explicitly permitted is blocked.  UDP, being a connectionless protocol, doesn't inherently benefit from the connection tracking and stateful inspection mechanisms often used to filter TCP traffic.  Consequently, if no specific policy allows UDP traffic between the source and destination endpoints, or if the relevant port isn't explicitly defined, the packets will be dropped at various points within the ACI fabric.  These points include:

* **Ingress Access Policy:** This policy dictates what traffic is allowed into the ACI fabric from external networks.  If the UDP traffic originates from an external network, it must be explicitly allowed through the appropriate contract and access list configurations.

* **Endpoint Group and Contract Associations:**  Endpoints (virtual machines, servers, etc.) are grouped, and contracts define which groups can communicate.  If the source and destination endpoint groups aren't associated with a contract allowing UDP on the specified port, the traffic will be dropped.

* **Bridge Domains and VLANs:** Although less common, misconfigurations related to VLANs and bridge domain assignments can also prevent UDP traffic from reaching its destination. Incorrectly configured VLAN tags on the packets might lead to filtering within the fabric.

* **Security Policies:**  ACI offers various security policies (such as firewall policies) that can explicitly block UDP traffic based on various criteria (source/destination IP, port numbers, etc.). These policies are usually configured to enhance security but can inadvertently block legitimate UDP traffic if not carefully managed.


Failure to explicitly define policies for UDP traffic results in its implicit rejection.  This is where many administrators encounter unexpected behavior.


**2. Code Examples with Commentary**

These examples utilize the APIC (Application Policy Infrastructure Controller) REST API, reflecting my experience managing ACI environments programmatically.  However, the underlying principles apply regardless of the management tool used.


**Example 1:  Creating an Access List to Permit UDP Traffic**

```json
{
  "fvTenant": {
    "attributes": {
      "dn": "uni/tn-common",
      "name": "common"
    }
  },
  "fvCtx": {
    "attributes": {
      "dn": "uni/tn-common/ctx-external",
      "name": "external"
    }
  },
  "fvAEPg": {
    "attributes": {
      "dn": "uni/tn-common/ctx-external/ap-external-epg",
      "name": "external-epg"
    }
  },
  "fvRsCons": {
    "attributes": {
      "tnFvCtxName": "external",
      "tnVzBrCPName": "external-brc",
      "status": "created"
    }
  },
  "vzBrCP": {
    "attributes": {
      "dn": "uni/tn-common/brc-external-brc",
      "name": "external-brc"
    }
  },
  "vzFilter": {
    "attributes": {
      "dn": "uni/tn-common/brc-external-brc/flt-allow-udp",
      "name": "allow-udp",
      "descr": "Allows UDP traffic on port 53"
    },
    "children": [
      {
        "vzEntry": {
          "attributes": {
            "name": "udp-entry",
            "arpEtherType": "0x0800",
            "dFromPort": "53",
            "dToPort": "53",
            "prot": "udp",
            "sFromPort": "53",
            "sToPort": "53",
            "action": "permit"
          }
        }
      }
    ]
  }
}
```

This JSON snippet demonstrates the creation of an access list (`vzFilter`) that explicitly permits UDP traffic on port 53.  This is crucial for allowing DNS traffic, a common service relying on UDP.  The access list is then associated with the appropriate bridge domain (`vzBrCP`) and endpoint group (`fvAEPg`).  Note the explicit specification of `prot: "udp"` and the port numbers.


**Example 2:  Defining a Contract to Allow Inter-EPG Communication**

```json
{
  "fvTenant": {
    "attributes": {
      "dn": "uni/tn-common",
      "name": "common"
    }
  },
  "fvAp": {
    "attributes": {
      "dn": "uni/tn-common/ap-app-tier",
      "name": "app-tier"
    }
  },
  "fvEpg": {
    "attributes": {
      "dn": "uni/tn-common/ap-app-tier/epg-web-servers",
      "name": "web-servers"
    }
  },
  "fvRsCons": {
    "attributes": {
      "tnVzBrCPName": "brc-default",
      "status": "created"
    }
  },
  "vzBrCP": {
    "attributes": {
      "dn": "uni/tn-common/brc-default",
      "name": "brc-default"
    }
  },
  "vzSubj": {
    "attributes": {
      "dn": "uni/tn-common/ap-app-tier/epg-web-servers/subj-web-servers-subj",
      "name": "web-servers-subj"
    }
  },
  "vzFilter": {
    "attributes": {
      "dn": "uni/tn-common/ap-app-tier/epg-web-servers/subj-web-servers-subj/flt-allow-udp-communication",
      "name": "allow-udp-communication",
      "descr": "Allows UDP communication between EPGs"
    },
    "children": [
      {
        "vzEntry": {
          "attributes": {
            "name": "udp-communication-entry",
            "prot": "udp",
            "action": "permit"
          }
        }
      }
    ]
  }
}
```

This example shows the creation of a contract (`vzFilter`) that allows UDP communication between Endpoint Groups (EPGs).  This contract is then associated with the relevant EPGs.  While not specifying specific ports, this ensures that any UDP traffic is allowed – which might be necessary for applications that dynamically use various UDP ports.  Caution is advised with this approach, favoring more granular port-level control when possible.


**Example 3:  Verifying Policy Application**

This example doesn't involve creating policies, but rather verifying their application.

```bash
apic-cli -c "show fvAEPg -filter "name eq 'web-servers'" -format json"
```

This command-line interface (CLI) command, using the APIC CLI, retrieves information about a specific Endpoint Group. Examining the JSON output allows verification that the associated contracts and filters are correctly applied and allow UDP traffic as intended.  Similar CLI commands can be used to verify the application of access lists and other policies.



**3. Resource Recommendations**

For deeper understanding of ACI policy configuration, I strongly recommend the official Cisco ACI documentation.  Pay close attention to the sections dealing with access control lists, contracts, and endpoint group management.  Additionally, Cisco's official training courses on ACI are invaluable for gaining practical skills and addressing complex troubleshooting scenarios. Finally, reviewing deployment guides and best practice documents for your specific ACI hardware and software version is vital for avoiding common pitfalls.  These resources provide detailed guidance on proper policy construction and troubleshooting methodologies.
