---
title: "how to inspect incoming tcp and ip headers ebpf?"
date: "2024-12-13"
id: "how-to-inspect-incoming-tcp-and-ip-headers-ebpf"
---

Okay so you wanna get down and dirty with inspecting TCP and IP headers using eBPF right been there done that plenty of times its like trying to understand a really messed up network conversation in real time but super fascinating once you get the hang of it

Alright so first things first eBPF Extended Berkeley Packet Filter its the cool kid on the block for doing all sorts of kernel level networking magic its not your grandma's packet filtering this is deep dives into packet data without even needing to copy it to user space which is insane efficiency for our use case and very good if you are dealing with low-latency stuff and you are basically working on baremetal environments

Now how do we actually do this inspection you ask well we need to write some eBPF code load it into the kernel and tell it to watch network events. We are going to use socket filters since its what we want in this case and we want to trace on the socket level

Let's get some code going I remember working on a very low level monitoring system back in 2016 where I had to pinpoint the source of some weird packets that were causing latency issues took me a while before I actually had to inspect the packet content using ebpf because i tried everything else first its always the last choice to trace on the packet level with the tcpdump I mean just the basic tcpdump tool would have worked but I wanted more context and more low level stuff. I even considered tracing the network interfaces directly but that meant that we were going to do more processing than needed and the bottleneck would be on the data extraction not on the networking stack.

Here is our very basic C code snippet that we will compile into eBPF bytecode:

```c
#include <linux/bpf.h>
#include <linux/pkt_cls.h>
#include <linux/ip.h>
#include <linux/tcp.h>
#include "bpf_helpers.h"

struct data_t {
    __u32 saddr;
    __u32 daddr;
    __u16 sport;
    __u16 dport;
    __u8 protocol;
};

BPF_PERF_OUTPUT(events);


int xdp_prog(struct xdp_md *ctx) {
    void *data = (void *)(long)ctx->data;
    void *data_end = (void *)(long)ctx->data_end;

    struct ethhdr *eth = data;
    if (data + sizeof(struct ethhdr) > data_end)
        return XDP_PASS;
    
    if (eth->h_proto != htons(ETH_P_IP))
        return XDP_PASS;
    
    struct iphdr *iph = data + sizeof(struct ethhdr);
    if (data + sizeof(struct ethhdr) + sizeof(struct iphdr) > data_end)
        return XDP_PASS;
    
    if (iph->protocol != IPPROTO_TCP)
        return XDP_PASS;
    
    struct tcphdr *tcph = data + sizeof(struct ethhdr) + sizeof(struct iphdr);
    if (data + sizeof(struct ethhdr) + sizeof(struct iphdr) + sizeof(struct tcphdr) > data_end)
        return XDP_PASS;

    struct data_t event = {};
    event.saddr = iph->saddr;
    event.daddr = iph->daddr;
    event.sport = tcph->source;
    event.dport = tcph->dest;
    event.protocol = iph->protocol;

    events.perf_submit(ctx, &event, sizeof(event));

    return XDP_PASS;
}


```
This code grabs the IP and TCP headers extracts the source and destination IP addresses and ports and the protocol. After this it sends this data to user space through the perf ring buffer that will enable us to read it in the user space application. I had to debug that `data_end` pointer for 3 days straight because i messed up the pointer arithmetic and the application kept crashing due to out-of-bounds issues. Its the kind of problem that you look at and say "well I'm definitely doing something wrong here". You can get the includes from the kernel headers or compile it with libbpf-tools.

Now we need a user-space application to load this BPF program and read the perf buffer data which is the output from the kernel part and we want to see the data that we are tracing. Lets create a simple example in python I usually prefer using Go but for this example Python is more simple:

```python
from bcc import BPF
import socket
import struct

# Load the BPF code
bpf_text = """
#include <linux/bpf.h>
#include <linux/pkt_cls.h>
#include <linux/ip.h>
#include <linux/tcp.h>
#include "bpf_helpers.h"

struct data_t {
    __u32 saddr;
    __u32 daddr;
    __u16 sport;
    __u16 dport;
    __u8 protocol;
};

BPF_PERF_OUTPUT(events);


int xdp_prog(struct xdp_md *ctx) {
    void *data = (void *)(long)ctx->data;
    void *data_end = (void *)(long)ctx->data_end;

    struct ethhdr *eth = data;
    if (data + sizeof(struct ethhdr) > data_end)
        return XDP_PASS;
    
    if (eth->h_proto != htons(ETH_P_IP))
        return XDP_PASS;
    
    struct iphdr *iph = data + sizeof(struct ethhdr);
    if (data + sizeof(struct ethhdr) + sizeof(struct iphdr) > data_end)
        return XDP_PASS;
    
    if (iph->protocol != IPPROTO_TCP)
        return XDP_PASS;
    
    struct tcphdr *tcph = data + sizeof(struct ethhdr) + sizeof(struct iphdr);
    if (data + sizeof(struct ethhdr) + sizeof(struct iphdr) + sizeof(struct tcphdr) > data_end)
        return XDP_PASS;

    struct data_t event = {};
    event.saddr = iph->saddr;
    event.daddr = iph->daddr;
    event.sport = tcph->source;
    event.dport = tcph->dest;
    event.protocol = iph->protocol;

    events.perf_submit(ctx, &event, sizeof(event));

    return XDP_PASS;
}
"""

b = BPF(text=bpf_text)
fn = b.load_func("xdp_prog", BPF.XDP)

# Attach the filter to an interface (replace with your interface)
interface = "eth0"
b.attach_xdp(interface, fn, flags=0)

# Function to handle perf event data
def handle_event(cpu, data, size):
    event = b["events"].event(data)
    print(f"Source IP: {socket.inet_ntoa(struct.pack('I', event.saddr))}")
    print(f"Destination IP: {socket.inet_ntoa(struct.pack('I', event.daddr))}")
    print(f"Source Port: {socket.ntohs(event.sport)}")
    print(f"Destination Port: {socket.ntohs(event.dport)}")
    print(f"Protocol: {event.protocol}")
    print("---")


b["events"].open_perf_buffer(handle_event)


try:
    while True:
        b.kprobe_poll()

except KeyboardInterrupt:
    pass
finally:
    b.remove_xdp(interface)
```
This script loads the C code into the kernel attaches the xdp filter to a network interface and it reads the data from the kernel using the `events` perf ring buffer that we declared in the C code with the `BPF_PERF_OUTPUT(events);` macro. I once spent an afternoon figuring out why I was not receiving any data and the reason was just a typo on the perf event name I called it `event` instead of `events` and it just returned nothing very annoying to debug.

Okay so that’s like the basic concept we are using XDP because it's faster than tc filters because it is more low level but for inspecting TCP headers with ebpf we could also use socket filters.

Here is how a socket filter version may look like:

```c
#include <linux/bpf.h>
#include <linux/pkt_cls.h>
#include <linux/ip.h>
#include <linux/tcp.h>
#include "bpf_helpers.h"

struct data_t {
    __u32 saddr;
    __u32 daddr;
    __u16 sport;
    __u16 dport;
    __u8 protocol;
};

BPF_PERF_OUTPUT(events);

int socket_prog(struct __sk_buff *skb) {
    void *data = (void *)(long)skb->data;
    void *data_end = (void *)(long)skb->data_end;

    struct ethhdr *eth = data;
    if (data + sizeof(struct ethhdr) > data_end)
        return BPF_DROP;
    
    if (eth->h_proto != htons(ETH_P_IP))
        return BPF_DROP;
    
    struct iphdr *iph = data + sizeof(struct ethhdr);
    if (data + sizeof(struct ethhdr) + sizeof(struct iphdr) > data_end)
        return BPF_DROP;
    
    if (iph->protocol != IPPROTO_TCP)
        return BPF_DROP;
    
    struct tcphdr *tcph = data + sizeof(struct ethhdr) + sizeof(struct iphdr);
    if (data + sizeof(struct ethhdr) + sizeof(struct iphdr) + sizeof(struct tcphdr) > data_end)
        return BPF_DROP;

    struct data_t event = {};
    event.saddr = iph->saddr;
    event.daddr = iph->daddr;
    event.sport = tcph->source;
    event.dport = tcph->dest;
    event.protocol = iph->protocol;

    events.perf_submit(skb, &event, sizeof(event));

    return BPF_OK;
}
```
This is basically the same code with only one key difference that is how it is attached to the network stack. We are going to use socket filters instead of XDP filters and also we are going to use the `__sk_buff` struct instead of `xdp_md` but the overall logic is the same. The advantage of socket filters is that you have more context on the kernel and the network operations and you can attach to a specific socket instead of all the traffic coming to the network interface.

For resources you should check out "BPF Performance Tools" by Brendan Gregg its like the bible of eBPF stuff. It goes deep into the details and he provides a lot of examples. You can also check out the "Linux Kernel Networking" book by Rami Rosen for a very good explanation on the networking subsystem of Linux which is the fundamental knowledge that you need to really start understanding how to use BPF at the lowest levels. You can also take a look at the linux kernel documentation or the bpftrace tools source code for very good examples. I always had an issue when a colleague kept using `trace-bpf` and I had to explain to him that is `bpftrace` it's not that hard to remember.

eBPF is a powerful tool and there is so much that you can do with it like modifying packets inspecting application level data on real time with low latency and very good performance. There are new things being developed all the time so the possibilities are really endless. Just remember to always double-check your pointer arithmetic and always use the correct struct on the BPF program and the correct output for the ring buffer or you'll have a bad time debugging
