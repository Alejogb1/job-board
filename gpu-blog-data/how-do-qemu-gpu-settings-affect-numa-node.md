---
title: "How do QEMU GPU settings affect NUMA node allocation?"
date: "2025-01-30"
id: "how-do-qemu-gpu-settings-affect-numa-node"
---
Directly impacting system performance, QEMU’s GPU configuration significantly influences how virtual machines (VMs) interact with Non-Uniform Memory Access (NUMA) nodes. Understanding this interaction is crucial for optimizing memory bandwidth and minimizing latency, especially in environments with multi-socket or multi-core processors. My experience optimizing high-performance compute workloads on virtualized infrastructure has consistently demonstrated that improper configuration leads to bottlenecks and significant performance degradation. Specifically, the placement of the emulated GPU's memory, and the mechanisms used to access it, directly correlates with the VM’s observed memory access patterns.

The core issue stems from how QEMU models hardware. A virtual GPU, regardless of the guest OS driver, requires memory allocation within the host. When configuring QEMU, the `-device` parameter, used to define the virtualized graphics card, doesn't inherently tie itself to a particular NUMA node. By default, the memory for the virtual GPU and other devices is typically allocated on the node where the QEMU process itself is running. In multi-NUMA environments, the processor core executing the VM's workload is often on a different NUMA node than the one hosting the GPU's memory. This disparity creates an implicit NUMA hop for every memory transfer initiated by the GPU within the VM. This situation significantly impacts performance. It increases latency and reduces effective bandwidth, especially in scenarios requiring frequent GPU-to-CPU or CPU-to-GPU data movement, such as in graphics-intensive tasks, scientific simulations, or even simple windowing compositing.

This issue isn’t limited to the device's memory alone. Certain QEMU GPU emulation modes can influence the behavior of DMA operations and can, inadvertently, make the NUMA problem more severe. I've observed that when using virtual devices that rely on emulated MMIO (Memory-Mapped I/O), every interaction requires QEMU to translate guest addresses to host addresses, further amplifying delays resulting from poor NUMA affinity. Conversely, technologies that attempt to pass through the physical hardware of the host into the VM (like vGPU passthrough) often bypass the problem altogether, but are complex and may have other trade-offs.

To address this problem, the most straightforward approach is to explicitly bind the virtual GPU’s memory allocation to the same NUMA node where the relevant virtual CPU (vCPU) threads are scheduled by the hypervisor. QEMU doesn't have a single flag that directly dictates NUMA affinity for every virtual device. Instead, it relies on a combination of libvirt XML configuration and command line options when the former is not possible. We can explicitly specify a NUMA node for a given virtual device.

Here's a demonstration of a configuration leveraging command line parameters. This strategy provides flexibility in scenarios where XML configurations are cumbersome or impossible, such as in bare-metal cloud environments.

```bash
qemu-system-x86_64 \
  -enable-kvm \
  -machine type=q35,accel=kvm \
  -cpu host,topoext=on \
  -smp 4,sockets=1,cores=4,threads=1 \
  -m 8G \
  -mem-path /dev/hugepages \
  -numa node,nodeid=0,cpus=0-3,mem=4G \
  -object memory-backend-file,id=mem-node1,prealloc=yes,mem-path=/dev/hugepages,size=4G,share=yes,policy=bind,host-nodes=0 \
  -numa node,nodeid=1,cpus=0-3,mem=4G \
  -object memory-backend-file,id=mem-node2,prealloc=yes,mem-path=/dev/hugepages,size=4G,share=yes,policy=bind,host-nodes=1 \
  -device virtio-gpu-pci,numa=0 \
  -drive file=my_disk.img,if=virtio,format=qcow2 \
  -net nic,model=virtio -net user,hostfwd=tcp::2222-:22 \
  -vnc :0
```

In this example, `-numa` explicitly defines NUMA nodes within QEMU and we allocate memory using hugepages to improve performance. The `-device virtio-gpu-pci,numa=0` binds the emulated GPU to the first defined NUMA node (node 0). The important aspect to notice is that the host-nodes property dictates the node which will contain the memory backing the memory-backend-file.  This node number refers to the host operating system’s NUMA nodes. The `numa=0` property of virtio-gpu-pci refers to the node inside the guest. Note that it may not map to the host directly.

The next example showcases a libvirt XML configuration that achieves a similar outcome. Utilizing libvirt provides a more structured and management-friendly approach.

```xml
<domain type='kvm'>
  <name>my-vm</name>
  <memory unit='GiB'>8</memory>
  <vcpu placement='static'>4</vcpu>
  <numatune>
    <memory mode='strict' nodeset='0,1'/>
  </numatune>
  <cpu mode='host-passthrough' check='partial'>
    <topology sockets='1' cores='4' threads='1'/>
  </cpu>
  <memoryBacking>
    <hugepages/>
  </memoryBacking>
  <devices>
      <graphics type='vnc' port='-1' autoport='yes' listen='0.0.0.0'>
            <listen type='address' address='0.0.0.0'/>
      </graphics>
      <video>
          <model type='virtio' ram='262144' vram='262144' vgamem='16384' heads='1'/>
      </video>
      <memballoon model='virtio'/>
        <interface type='network'>
          <source network='default'/>
          <model type='virtio'/>
      </interface>
      <disk type='file' device='disk'>
        <source file='/path/to/my_disk.img'/>
        <target dev='vda' bus='virtio'/>
      </disk>
        <graphics type='spice' port='5900' autoport='yes'>
            <listen type='address' address='0.0.0.0'/>
            <image compression='off'/>
        </graphics>
      <hostdev mode='subsystem' type='pci' managed='yes'>
        <source>
          <address domain='0x0000' bus='0x00' slot='0x02' function='0x0'/>
        </source>
        <address type='pci' domain='0x0000' bus='0x00' slot='0x02' function='0x0'/>
      </hostdev>

  </devices>
  <cpu>
     <numa>
          <cell id="0" cpus="0-1" memory="4294967296"/>
          <cell id="1" cpus="2-3" memory="4294967296"/>
      </numa>
  </cpu>
    <memory unit='GiB'  model='dimm'  access='shared'>
    <target nodeset='0'/>
     <address type='dimm'  slot='0'/>
   </memory>
   <memory unit='GiB'  model='dimm' access='shared'>
        <target nodeset='1'/>
        <address type='dimm' slot='1'/>
    </memory>
</domain>
```

This libvirt configuration explicitly defines two NUMA nodes within the `cpu/numa/cell` sections, assigning vCPUs and memory to each node. Crucially, the `<video>` model defines how the virtual graphics device is configured. Here it is a virtio-gpu, which will utilize host resources from the first NUMA node. Note that this is not an explicit tying of device memory to node, but more an implied placement through the configuration of vCPU threads. We could also have a situation like an NVidia vGPU being passed through, with resources coming from the host. In this case, we must also ensure the VM runs on the node with the dedicated host GPU.

The final code example uses the command line to perform vGPU pass-through, a more complex procedure but often required in high performance situations that also addresses the NUMA node issue:

```bash
qemu-system-x86_64 \
  -enable-kvm \
  -machine type=q35,accel=kvm \
  -cpu host,topoext=on \
  -smp 4,sockets=1,cores=4,threads=1 \
  -m 8G \
  -mem-path /dev/hugepages \
  -numa node,nodeid=0,cpus=0-3,mem=4G \
  -object memory-backend-file,id=mem-node1,prealloc=yes,mem-path=/dev/hugepages,size=4G,share=yes,policy=bind,host-nodes=0 \
  -numa node,nodeid=1,cpus=0-3,mem=4G \
  -object memory-backend-file,id=mem-node2,prealloc=yes,mem-path=/dev/hugepages,size=4G,share=yes,policy=bind,host-nodes=1 \
  -device vfio-pci,host=01:00.0 \
  -drive file=my_disk.img,if=virtio,format=qcow2 \
  -net nic,model=virtio -net user,hostfwd=tcp::2222-:22 \
  -vnc :0
```

Here, `-device vfio-pci,host=01:00.0` passes through the physical PCI device at address `01:00.0`. In this case we must ensure that the vCPU threads are running on the NUMA node that also hosts this PCI device. We have to perform other configurations on the host system, such as enabling IOMMU. In addition, we must ensure that the host driver for the device is not loaded, so it can be used by the guest directly.

For further exploration of the concepts outlined here, I recommend studying the official QEMU documentation, specifically the section on device assignment and NUMA configuration. Furthermore, delving into the libvirt documentation regarding XML definition for virtual machines will be beneficial. I also found it valuable to explore practical case studies and discussions from virtualized environment user groups, where detailed setups and debugging strategies are often presented. Resources covering the basics of CPU and memory architecture like those found at operating system and compiler resources can be beneficial.
