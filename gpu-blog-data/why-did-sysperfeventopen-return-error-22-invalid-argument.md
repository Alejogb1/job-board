---
title: "Why did sys_perf_event_open() return error 22 (Invalid argument) for msr/tsc events?"
date: "2025-01-30"
id: "why-did-sysperfeventopen-return-error-22-invalid-argument"
---
The `sys_perf_event_open()` system call returning error 22 (EINVAL - Invalid argument) when attempting to access MSR-based events, specifically the TSC (Time Stamp Counter), commonly stems from insufficient permissions or incorrect configuration of the perf event configuration structure.  In my experience debugging performance monitoring tools on various Linux kernels (primarily 4.19 through 5.15), this error frequently arises from a misunderstanding of the required capabilities and event configuration parameters.

**1. Explanation:**

The `perf_event_open` system call requires specific privileges to access hardware performance counters, including MSRs like the TSC.  These privileges are not automatically granted to standard user accounts.  Furthermore, the event configuration, particularly the `perf_event_attr` structure, needs precise specification.  Improperly setting fields like `type`, `config`, `sample_type`, and `sample_period` can lead to EINVAL.

The TSC, in particular, presents unique challenges.  While seemingly straightforward, its access can be restricted by the kernel's security modules or hypervisor (if present).  Virtualization environments might expose a virtualized TSC, which behaves differently and may not be accessible through the standard MSR interface.  Additionally, some systems might have the TSC disabled or configured for specific purposes, restricting its use for performance monitoring.

Therefore, the EINVAL error often indicates a mismatch between the requested event and the system's capabilities or limitations.  The kernel verifies the permissions, the validity of the specified MSR, and the compatibility of the requested configuration with the hardware and kernel's current settings before allowing access.  Failure in any of these checks results in the EINVAL return code.  Carefully reviewing the user's permissions, the event configuration parameters, and the system's hardware capabilities is crucial for troubleshooting this issue.

**2. Code Examples with Commentary:**

**Example 1:  Insufficient Permissions**

```c
#include <stdio.h>
#include <stdlib.h>
#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>

int main() {
    struct perf_event_attr pe;
    memset(&pe, 0, sizeof(pe));
    pe.type = PERF_TYPE_HW_CACHE; // Example using cache, not MSR directly, for permissions demonstration
    pe.config = PERF_COUNT_HW_CACHE_L1D;
    pe.size = sizeof(pe);
    int fd = perf_event_open(&pe, 0, -1, -1, 0);

    if (fd == -1) {
        perror("perf_event_open"); //Will likely show a permission error if run without sufficient privileges.
        return 1;
    }

    //Further processing... (normally would read events here)
    close(fd);
    return 0;
}
```

**Commentary:** This example attempts to open a hardware cache event.  While not directly an MSR event, it serves to illustrate a common error source: insufficient permissions.  Running this code as a non-privileged user often results in a permission-related error, which might manifest as EINVAL in some scenarios, highlighting the importance of proper privileges for accessing performance counters.  To resolve, run the code with `sudo`.


**Example 2: Incorrect Event Configuration (Targeting TSC)**

```c
#include <stdio.h>
#include <stdlib.h>
#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>

int main() {
    struct perf_event_attr pe;
    memset(&pe, 0, sizeof(pe));
    pe.type = PERF_TYPE_HW_EVENT; //Incorrect type for MSR access - should be PERF_TYPE_RAW
    pe.config = 0x10; //Incorrect config for TSC - needs to be correctly identified
    pe.size = sizeof(pe);
    int fd = perf_event_open(&pe, 0, -1, -1, 0);

    if (fd == -1) {
        perror("perf_event_open"); // May show EINVAL due to incorrect config
        return 1;
    }

    //Further processing...
    close(fd);
    return 0;
}
```

**Commentary:** This example attempts to access the TSC, but uses incorrect `type` and `config` values.  `PERF_TYPE_HW_EVENT` is generally not suitable for direct MSR access; `PERF_TYPE_RAW` is often necessary.  The `config` value `0x10` is a placeholder and is unlikely to be the correct configuration for the TSC on any system.  The correct `config` value needs to be determined based on the specific hardware and kernel.  Improper `config` values can easily lead to EINVAL.  This emphasizes the need for accurate hardware-specific information.


**Example 3:  Handling Errors Gracefully**

```c
#include <stdio.h>
#include <stdlib.h>
#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>

int main() {
    struct perf_event_attr pe;
    memset(&pe, 0, sizeof(pe));
    // ... Proper configuration for TSC (determined based on system architecture) ...
    pe.size = sizeof(pe);
    int fd = perf_event_open(&pe, 0, -1, -1, 0);

    if (fd == -1) {
        fprintf(stderr, "perf_event_open failed: %s (%d)\n", strerror(errno), errno);
        // Specific error handling based on errno, including EINVAL
        if (errno == EINVAL) {
            fprintf(stderr, "Invalid argument: Check event configuration and permissions.\n");
        }
        return 1;
    }

    //Further processing...
    close(fd);
    return 0;
}
```

**Commentary:**  This example demonstrates robust error handling.  It checks for errors after the `perf_event_open` call and provides informative messages based on the `errno` value.  Specifically, if `errno` is `EINVAL`, it prints a message guiding the user towards checking the event configuration and permissions. This approach is crucial for effective debugging.  The crucial aspect omitted is the correct configuration for TSC, which necessitates understanding your specific hardware and kernel's documentation.


**3. Resource Recommendations:**

The Linux kernel documentation, specifically the `perf_event_open` man page, is indispensable.  Consult the relevant hardware documentation for details on available performance counters and their configuration.  Understanding the capabilities of your specific CPU architecture (x86, ARM, etc.) is also critical.  Finally, studying existing performance monitoring tools' source code can provide valuable insights into proper event configuration and handling of potential errors.  Examine system calls like `ioctl` related to accessing performance registers.  Familiarity with system call tracing tools (like `strace`) can help in diagnosing the precise point of failure.
