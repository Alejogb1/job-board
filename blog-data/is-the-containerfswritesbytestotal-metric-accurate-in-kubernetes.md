---
title: "Is the container_fs_writes_bytes_total metric accurate in Kubernetes?"
date: "2024-12-23"
id: "is-the-containerfswritesbytestotal-metric-accurate-in-kubernetes"
---

, let's talk about `container_fs_writes_bytes_total` in Kubernetes. This is a metric that, like many others, appears straightforward on the surface but can present some nuances if you're not careful in your interpretation. I've spent a fair bit of time debugging performance issues related to persistent volumes and storage in Kubernetes, and this specific metric has definitely popped up as a point of investigation more than once.

The short answer? Accuracy is relative. The metric itself, as exposed by the kubelet via cAdvisor, aims to measure the total number of bytes written to the container’s filesystem. However, how accurately it reflects the *application's* writes is where things get interesting. There are a few layers involved, and each can introduce complexities.

First, let's establish what's actually being measured. `container_fs_writes_bytes_total` tallies the raw bytes written by *processes within the container* at the filesystem level. This means it captures writes that go through the operating system's virtual file system (vfs) layer. It's not measuring writes that might be buffered in application memory or cached before a write syscall is made. It also doesn't account for storage-level replication or compression that might happen below the filesystem. Furthermore, it tracks *all* writes, regardless of where in the container filesystem they're directed – this includes mounted volumes, any temp directories, and the container's base writable layer. This is critical, especially when dealing with persistent volumes that could be on network-attached storage. I remember a particularly tricky case where an application was thrashing on a temp directory, filling it with logs and impacting overall storage performance. It wasn't immediately obvious because most monitoring focused on the mounted persistent volume.

Secondly, keep in mind that these metrics are being collected and exposed by cAdvisor, which runs as part of the kubelet agent on each node. cAdvisor relies on kernel interfaces and its own internal mechanisms to gather resource usage data. While it's generally reliable, it’s not a perfect representation of underlying hardware. There can be minute delays or slight inaccuracies based on the sampling intervals and the efficiency of the kernel instrumentation. This is something I’ve seen when attempting to correlate container-level disk metrics with system-level disk metrics -- you often won’t see a perfect one-to-one correspondence. The granularity of the collected data and the chosen scrape interval also impact how accurately this metric represents the actual underlying filesystem writes. For instance, if your scrape interval is infrequent (let's say 60 seconds), a significant burst of writes within that 60 second window will appear as a single cumulative value which can obscure any finer-grained trends.

Finally, and perhaps most importantly, think about what's "within the container's filesystem". If the container writes to a path that maps to a persistent volume, those writes are accounted for within this metric. However, the underlying mechanics of those writes depend entirely on the storage provider. So, if you have an EBS volume, for example, the actual write operation might involve network transfer and replication that's not visible at the container level. The metric simply registers the byte-count of the operation as it’s issued *from* the container. It says nothing about any downstream network-level or storage-level overhead. In another case I troubleshot, a pod was performing a high volume of writes to an NFS volume, and while the pod showed increasing bytes-written, the performance bottleneck was entirely within the NFS server and network, not within the container itself.

Let’s illustrate this with a few code examples. These aren't Kubernetes code snippets, but rather examples of common scenarios using python and shell scripting that will demonstrate what a container’s process can do to impact the metric.

First, a simple python script that writes a file:

```python
import os

def write_large_file(filename, size_mb):
    chunk_size = 1024 * 1024
    with open(filename, 'wb') as f:
        for i in range(size_mb):
            f.write(os.urandom(chunk_size))

if __name__ == '__main__':
    write_large_file("test.dat", 500) # write 500 MB file.
```

This simple python script, when executed within a container, would directly increase the `container_fs_writes_bytes_total` counter, reflecting a 500MB increase. Nothing surprising there. Now, suppose a similar operation, done in batch, using a loop. Let’s look at a shell example:

```bash
#!/bin/bash
for i in $(seq 1 5); do
  dd if=/dev/urandom of=test$i.dat bs=1M count=100
done
```

This script generates 5 separate 100MB files. Each iteration would increment the metric, so you will have cumulative bytes of 500MB added to the counter. The key here is that each file will be a new, distinct file at the filesystem level. If a persistent volume is mounted at the path where these files are created, then these writes will be reflected on that volume as well as in the container metric.

Finally, let's consider a case involving a temporary file, often used by applications. This is frequently a location of high writes. Consider this slightly adapted python script:

```python
import os
import tempfile

def write_to_temp(size_mb):
    with tempfile.NamedTemporaryFile() as tmp_file:
      chunk_size = 1024 * 1024
      for i in range(size_mb):
        tmp_file.write(os.urandom(chunk_size))

if __name__ == '__main__':
    write_to_temp(100) # Write 100 MB to temp.

```

This script, when run in the container, writes 100MB to a temporary file location. Even though these files are automatically cleaned up by the python library, they do count against the metrics during their existence. It is a commonly used pattern by applications and is important to monitor because temporary file operations can cause unintended filesystem performance issues, especially when these operations occur at high frequencies.

So, to return to your core question about accuracy, `container_fs_writes_bytes_total` is accurate in its measurement of raw bytes written by processes within the container's vfs context, but it does not account for the myriad complexities that happen below that level. To understand total write volume and the potential impact of an application, one must also examine metrics from the storage layer itself (e.g. iops and throughput for ebs volumes). Think of the metric as a starting point for investigation, not necessarily a definitive gauge of the true amount of disk activity.

If you're looking to dive deeper into this, I recommend a thorough reading of “Understanding the Linux Kernel,” by Daniel P. Bovet and Marco Cesati. This will give you a very firm understanding of kernel interactions involved when a write operation is issued. For a more practical take on monitoring in Kubernetes, “Kubernetes in Action” by Marko Lukša is an excellent resource. Additionally, the official Kubernetes documentation on cAdvisor is vital for understanding how these metrics are collected and exposed; you’ll find the information under metrics and monitoring documentation. Understanding the layers will help interpret this metric with necessary caution. It’s not wrong; it’s just not the entire story.
