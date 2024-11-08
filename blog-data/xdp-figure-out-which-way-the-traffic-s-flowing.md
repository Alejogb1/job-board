---
title: "XDP: Figure Out Which Way the Traffic's Flowing"
date: '2024-11-08'
id: 'xdp-figure-out-which-way-the-traffic-s-flowing'
---

```c
// This program demonstrates how to determine the direction of network traffic using XDP.

#include <linux/bpf.h>
#include <linux/if_ether.h>
#include <linux/ip.h>
#include <linux/udp.h>
#include <linux/tcp.h>
#include <stdbool.h>

// Define a macro to simplify checking the direction of traffic
#define INBOUND_TRAFFIC(ctx) (bpf_get_current_ifindex(ctx) == bpf_get_local_ifindex(ctx))

// The XDP program
struct bpf_prog_type {
    // Insert your XDP program code here.
};

// This program does not handle outgoing traffic at this time.
int xdp_program(struct xdp_md *ctx)
{
    // Check if the packet is inbound
    if (INBOUND_TRAFFIC(ctx)) {
        // Handle the incoming packet
        // ...
    }

    // Return a verdict
    return XDP_PASS;
}
```
