---
title: "Why does virt-install hang during GPU passthrough setup?"
date: "2025-01-30"
id: "why-does-virt-install-hang-during-gpu-passthrough-setup"
---
The most common cause of `virt-install` hanging during GPU passthrough configuration stems from an incomplete or incorrectly configured IOMMU group assignment.  My experience troubleshooting this issue across numerous production and development environments points consistently to this root cause.  While other factors can contribute to hangs, a faulty IOMMU setup almost always underlies the problem. This necessitates a rigorous examination of your system's IOMMU configuration before addressing other potential bottlenecks.


**1.  Explanation:**

The `virt-install` command relies on the kernel's ability to accurately isolate and assign PCI devices – in this case, the GPU – to the virtual machine.  This isolation is achieved through Input/Output Memory Management Units (IOMMU).  The IOMMU acts as a translator, mapping physical memory addresses to virtual addresses, thus preventing a guest VM from directly accessing the host's memory outside its assigned resources.  For GPU passthrough to work correctly, the GPU and any associated devices (like the framebuffer) must reside within a dedicated IOMMU group, and that group must be exclusively assigned to the virtual machine.  If the IOMMU configuration is flawed, `virt-install` may hang indefinitely as it attempts to allocate resources that are either unavailable or improperly mapped.  This can manifest in several ways, including a complete freeze, a seemingly unresponsive process, or erratic behavior during the installation process.

Several factors can lead to this faulty configuration:

* **Insufficient Kernel Support:** The host kernel must have IOMMU support enabled and correctly configured.  This involves enabling the appropriate kernel modules and ensuring the BIOS is configured for VT-d (Intel) or AMD-Vi (AMD).
* **Conflicting Device Assignments:**  Other virtual machines or system processes might be inadvertently accessing or claiming resources that are necessary for the GPU passthrough, causing resource conflicts that prevent `virt-install` from completing its allocation process.
* **Incorrect `virtio-gpu` Configuration:** While not directly causing the hang, improper configuration of the `virtio-gpu` driver within the virtual machine can lead to compatibility issues and ultimately impact the process.
* **BIOS Settings:**  Incorrect BIOS settings, such as disabling VT-d/AMD-Vi or misconfiguring memory mapping, can prevent proper IOMMU functionality.


**2. Code Examples and Commentary:**

The following examples demonstrate different aspects of correctly configuring IOMMU and GPU passthrough. Note that these examples are simplified for illustrative purposes and may need adjustments based on your specific hardware and distribution.  Always back up your system before making significant changes to kernel parameters or system configuration.

**Example 1: Checking IOMMU Status:**

```bash
lspci -nnk | grep -i "vga\|3d\|display" | grep -i "kernel"
```

This command lists PCI devices related to graphics and displays the kernel drivers associated with them.  It's crucial to confirm that the GPU is correctly recognized and that the kernel driver is loaded without errors.  A missing or incorrectly loaded driver indicates a potential problem.  Further investigation might require checking `dmesg` for error messages related to IOMMU or the specific GPU.


**Example 2: Verifying IOMMU Group Assignment (using `iproute2`):**

```bash
echo 0 > /sys/kernel/iommu/groups/<iommu_group_number>/devices/<pci_address>
```

This example (requires root privileges) demonstrates manipulating the IOMMU group assignment.  You need to identify the appropriate IOMMU group number (`<iommu_group_number>`) and the PCI address (`<pci_address>`) of the GPU.  Carefully examine the output from `lspci -vv` to find this information.  This command *assigns* the GPU to a specific IOMMU group.  Before executing it, you MUST ensure the chosen IOMMU group is not already in use and dedicated solely to the virtual machine.  Incorrect usage could lead to system instability.


**Example 3:  `virt-install` Command with Specific IOMMU Group Assignment (requires Libvirt):**

```xml
<domain type='kvm'>
  <name>my-gpu-vm</name>
  <memory unit='KiB'>1048576</memory>
  <currentMemory unit='KiB'>1048576</currentMemory>
  <vcpu placement='static'>2</vcpu>
  <os>
    <type arch='x86_64' machine='pc-q35'>hvm</type>
    <boot dev='cdrom'/>
  </os>
  <features>
    <acpi/>
    <apic/>
    <vmport state='off'/>
  </features>
  <devices>
    <emulator>/usr/bin/qemu-kvm</emulator>
    <disk type='file' device='disk'>
      <driver name='qemu' type='qcow2'/>
      <source file='/path/to/disk.qcow2'/>
      <target dev='vda' bus='virtio'/>
    </disk>
    <interface type='bridge'>
      <source bridge='br0'/>
    </interface>
    <hostdev mode='subsystem' type='pci' managed='yes'>
      <source>
        <address domain='0x0000' bus='0x01' slot='0x00' function='0x0'/>
      </source>
      <address type='pci' domain='0x0000' bus='0x01' slot='0x00' function='0x0'/>
    </hostdev>
    </devices>
</domain>
```

This XML snippet defines a virtual machine with GPU passthrough. The `<hostdev>` section is critical. Replace placeholders with your GPU's PCI address obtained from `lspci -vv`.  This configuration requires using `virsh` or a similar tool to create and manage the VM, rather than `virt-install` directly, to allow for finer grained control.  This method gives you maximum control over device assignment and eliminates ambiguities.


**3. Resource Recommendations:**

I would recommend consulting the documentation for your specific virtualization software (e.g., Libvirt, KVM), your motherboard's manual for IOMMU settings, and your distribution's kernel documentation for information about IOMMU configuration and PCI device management.  Thorough examination of the system logs (`dmesg`, `/var/log/kern.log`) is essential for diagnosing any kernel-level issues related to IOMMU or device drivers.  The official documentation for the `virtio-gpu` driver is vital for correct driver configuration within the virtual machine.
