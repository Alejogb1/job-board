---
title: "How can I enable AVX/AVX2 in VirtualBox 6.1.16 on Ubuntu 20.04 64-bit?"
date: "2025-01-30"
id: "how-can-i-enable-avxavx2-in-virtualbox-6116"
---
Enabling AVX/AVX2 support within VirtualBox on Ubuntu hinges on the CPU configuration of both the host and the guest virtual machine.  My experience optimizing high-performance computing workloads within virtualized environments has consistently highlighted the critical role of proper CPU feature exposure.  Simply enabling it in the VirtualBox settings is often insufficient; the underlying host and guest kernel configurations are paramount.

**1.  Clear Explanation:**

The issue stems from a multi-layered interaction. First, your physical host machine needs to possess AVX/AVX2 capabilities at the hardware level. This is a prerequisite; no virtualization software can conjure features absent from the underlying processor. Second, the host's kernel must be configured to allow VirtualBox access to these instructions.  Third, the guest operating system's kernel also needs appropriate configuration to enable and utilize AVX/AVX2 instructions.  Finally, the application intended to leverage AVX/AVX2 needs to be compiled with appropriate compiler flags.  Failure at any of these stages will result in the inability to utilize these instructions, even if partially enabled elsewhere.

VirtualBox itself doesn't directly *enable* AVX/AVX2; it merely *exposes* them to the guest.  The guest operating system then manages the actual instruction usage. Therefore, simply checking the "Enable EFI" or "Enable PAE/NX" boxes within the VirtualBox VM settings isn't enough.  These settings concern different aspects of virtualization, not directly the CPU feature set.

The most common oversight is neglecting the guest kernel's configuration.  While Ubuntu 20.04 generally supports AVX/AVX2, the VM's kernel might be configured to restrict access to certain CPU features for stability or security reasons.  This is particularly relevant if you've customized the guest kernel or have a non-standard Ubuntu installation.

**2. Code Examples with Commentary:**

The following code examples demonstrate verification of AVX/AVX2 support at different stages. Note that these examples require administrator privileges within the respective environments.

**Example 1: Checking Host CPU Capabilities (Host machine terminal):**

```bash
lscpu
```

This simple command provides detailed information about the host's CPU, including support for AVX and AVX2.  Look for entries like "avx: yes" and "avx2: yes" to confirm their presence.  Absence of these flags definitively indicates a hardware limitation; proceeding further is futile in such a case.  During a recent project involving real-time signal processing, I encountered a situation where this initial check revealed a missing AVX2 instruction set on the host, necessitating a hardware upgrade before proceeding.


**Example 2: Checking Guest Kernel Parameters (Guest VM terminal):**

```bash
grep -i avx /proc/cpuinfo
```

This command, executed within the Ubuntu 20.04 guest VM, similarly checks for AVX and AVX2 support but from the guest’s perspective.  The output will reveal if the guest kernel recognizes and exposes these features. If the output is missing the "flags" field containing "avx" and "avx2", the guest kernel may need adjustments.  In my experience, this issue is most often resolved by rebooting the guest VM after ensuring the necessary changes to the VirtualBox settings (explained below).

**Example 3:  C++ Code to Detect AVX/AVX2 at Runtime (Guest VM):**

```c++
#include <iostream>
#include <immintrin.h>

int main() {
  if (__builtin_cpu_supports("avx")) {
    std::cout << "AVX is supported" << std::endl;
  } else {
    std::cout << "AVX is NOT supported" << std::endl;
  }

  if (__builtin_cpu_supports("avx2")) {
    std::cout << "AVX2 is supported" << std::endl;
  } else {
    std::cout << "AVX2 is NOT supported" << std::endl;
  }
  return 0;
}
```

This C++ code leverages compiler intrinsics to directly query the CPU at runtime.  Compiling this code (using `g++ -o avx_check avx_check.cpp -mavx -mavx2`) and running it within the guest VM provides a definitive answer regarding the application's ability to utilize AVX/AVX2. This was particularly useful during the development of a scientific simulation where ensuring correct runtime detection was crucial for handling varying hardware configurations.


**3. Resource Recommendations:**

The VirtualBox documentation, the Ubuntu 20.04 manual pages (especially those relating to the kernel), and the documentation for your chosen C++ compiler (regarding instruction set support and compilation flags) are invaluable resources.  Understanding the interplay between the host and guest CPU configurations and kernel parameters is crucial. Furthermore, consulting online forums and communities specifically dedicated to high-performance computing within virtualized environments can provide valuable insights and solutions for specific issues.  Careful attention to the compilation flags during the build process is critical for enabling the AVX/AVX2 support in your applications.


In conclusion, enabling AVX/AVX2 in VirtualBox isn't a simple toggle switch. It necessitates a thorough understanding of your hardware, host and guest operating system configurations, and the compilation process of your applications.  By systematically checking each layer using the techniques and examples described above, you can effectively diagnose and resolve any issues preventing AVX/AVX2 utilization.  Remember, even after enabling the correct settings, recompilation with appropriate flags for your application is necessary to leverage the increased performance.
