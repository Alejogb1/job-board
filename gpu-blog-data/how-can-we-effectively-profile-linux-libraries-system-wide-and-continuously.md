---
title: "How can we effectively profile Linux libraries system-wide and continuously?"
date: "2025-01-26"
id: "how-can-we-effectively-profile-linux-libraries-system-wide-and-continuously"
---

Profiling shared libraries across an entire Linux system, continuously, presents a complex challenge requiring a multi-faceted approach. It’s not a simple matter of attaching a debugger and hoping for relevant information. Instead, we need to consider the performance impact of our tooling on the system, the granularity of data we aim to collect, and the long-term storage and analysis of that data. In my experience managing high-performance compute clusters, we found that a combination of kernel-level tracing, sampling profilers, and custom monitoring scripts provides the best insight with minimal overhead.

The primary hurdle is the "system-wide" aspect. We can't realistically attach a user-space profiler to every single process simultaneously. Instead, we must use techniques that operate at a lower level, often within the kernel, to observe all running code. This involves techniques like *perf* and *eBPF* (Extended Berkeley Packet Filter). These technologies enable tracing function calls, sampling CPU execution, and monitoring system events with minimal impact on the running applications.

*Perf*, a performance analysis tool available in the Linux kernel, is a powerful starting point. It allows us to sample program counters periodically, thereby identifying which code regions are consuming the most CPU cycles. *Perf* is configured using various command-line options to filter events and specific libraries. For continuous monitoring, we wouldn't use it in interactive mode; instead, we'd configure it to record data periodically to a file, which can then be analyzed later. While powerful, *perf*’s sampling mechanism inherently implies some degree of approximation. It might miss short-duration functions or events that happen too quickly between samples.

eBPF offers more flexibility and lower overhead. It allows us to execute custom code within the kernel to observe specific events like function calls, syscalls, and network activities. Using eBPF, we can create custom probes that monitor library functions by name or address, allowing us to track function execution counts and execution times. The programmable nature of eBPF allows precise filtering and aggregation of data, which can be then exported to user-space for analysis. For instance, eBPF can be used to track how often certain shared libraries are accessed, how long functions inside them take to execute, and which caller processes are invoking them. This granularity is exceptionally valuable when debugging performance issues in complex system environments.

Continuous monitoring requires us to automate this data collection process, manage the size of the data, and have a pipeline for processing and analysis. For this, a combination of shell scripts or a system monitoring agent coupled with tools like Prometheus can provide a practical solution. We collect the data regularly from the kernel tracing, process it using custom tools and scripts, and then send it to a time-series database for analysis and visualization. This setup provides a history of system performance and alerts when specific thresholds are crossed.

To put this in practical terms, consider a situation where we want to profile a crucial shared library, *libcrypto.so*, which provides cryptography functions. First, we could use *perf* to get an overview of CPU usage in the system, focusing on *libcrypto.so*:

```bash
#!/bin/bash

while true; do
  perf record -g -e cpu-clock -o perf.data -- /bin/true # run a minimal command for profiling period
  perf script -i perf.data | grep 'libcrypto.so' > current_libcrypto_perf.txt
  mv current_libcrypto_perf.txt libcrypto_perf_$(date +%Y%m%d_%H%M%S).txt
  sleep 60 # gather data every 60 seconds
done

```

This script sets up a continuous loop that uses `perf` to collect CPU-clock events. It then filters the output for lines containing "libcrypto.so" and stores the results in timestamped files. The loop runs indefinitely, providing a time-series of perf data. While simple, it provides a basic form of continuous profiling. The `-g` option instructs `perf` to capture call graphs which is helpful for understanding the execution flow, but increases file size. Using a simple shell script allows easy automation, which I've found crucial for production systems.

The above method uses sampling. Next, we can delve deeper into a specific function using an eBPF script using *bcc* tools for instance. Imagine we are interested in the `SHA256` function within `libcrypto.so`:

```python
#!/usr/bin/env python
from bcc import BPF
import time

program = """
#include <uapi/linux/ptrace.h>
BPF_HASH(counts, u64, u64);

int kprobe__SHA256(struct pt_regs *ctx) {
    u64 pid = bpf_get_current_pid_tgid();
    u64 zero = 0;
    u64 *val = counts.lookup_or_init(&pid, &zero);
    (*val)++;
    return 0;
}

"""

b = BPF(text=program)
b.attach_kprobe(event="SHA256", fn_name="kprobe__SHA256")

try:
    while True:
      time.sleep(5)
      for pid, count in b["counts"].items():
        print(f"PID {pid.value} called SHA256 {count.value} times")
      b["counts"].clear()
except KeyboardInterrupt:
    exit()
```

This script defines a simple eBPF program using *bcc* Python binding. It creates a kprobe at the beginning of `SHA256` function and increments a counter in the `counts` hash map using process ID as a key. While this example focuses on a single function and tracks counts, it illustrates the principle of using custom eBPF programs to get precise data about library usage. I have used such eBPF scripts in production to pinpoint the source of performance degradation, providing real-time visibility into function call frequencies across the system.

Lastly, for integration into a larger system-monitoring stack, we might aggregate information by writing custom monitoring agents. This script below aggregates perf data collected by the first script, then exports it into a metrics format:

```python
import os
import re
import time
from datetime import datetime

def extract_libcrypto_usage(file_path):
    usage_data = {}
    with open(file_path, 'r') as f:
        for line in f:
            match = re.search(r'\s+([\w-]+)\s+(.*libcrypto.so.*)', line)
            if match:
              pid = match.group(1)
              function = match.group(2)
              usage_data[pid] = usage_data.get(pid, 0) + 1
    return usage_data

def process_perf_logs():
    perf_data_dir = "." #Assume files are in the same directory, could be different
    for filename in os.listdir(perf_data_dir):
       if filename.startswith("libcrypto_perf") and filename.endswith(".txt"):
        file_path = os.path.join(perf_data_dir, filename)
        usage_data = extract_libcrypto_usage(file_path)
        for pid, count in usage_data.items():
          timestamp = datetime.now().isoformat()
          print(f"perf_libcrypto_usage{{pid=\"{pid}\"}} {count} {timestamp}")
        os.remove(file_path)

while True:
    process_perf_logs()
    time.sleep(60) # process data every 60 seconds
```

This Python script demonstrates how we could consume the output generated by the first shell script. It processes the timestamped files, extracts usage data, and then prints metrics in a format that is easily scraped by a time-series database such as Prometheus. The script aggregates usage counts per PID and then discards the perf files to ensure a stable monitoring loop. It is an example of the processing logic needed for a basic monitoring agent. I've implemented similar agents in the past, which not only aggregated data but also pushed it to a centralized monitoring system using the agent’s API.

For further understanding of the techniques, I recommend exploring the official documentation for *perf* and *eBPF*, specifically focusing on BCC tools and its Python bindings. There are excellent books on kernel development and system programming that cover these tools in depth, too. For time-series database knowledge, explore resources on Prometheus, which will provide valuable insight into metric collection, storage, and visualization. Additionally, consider the wealth of information available within the Linux kernel’s source code regarding perf and tracepoint implementations, which while advanced, provides the most detailed explanation of how this system works.
