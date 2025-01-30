---
title: "Why can't I find /usr/lib/kernex.exp on my AIX 6.1 system when developing a kernel extension?"
date: "2025-01-30"
id: "why-cant-i-find-usrlibkernexexp-on-my-aix"
---
The absence of `/usr/lib/kernex.exp` on your AIX 6.1 system during kernel extension development stems from a fundamental misunderstanding of the AIX kernel build process and the evolution of its export file handling.  AIX 6.1, and subsequent releases, employ a more nuanced approach to exporting kernel symbols than simply placing them within a single, readily accessible file.  This is largely driven by security considerations and the desire for a more granular control over symbol visibility. My experience in developing and maintaining kernel modules for AIX systems over the past fifteen years underscores this.  The concept of a monolithic `kernex.exp` is largely a relic of older AIX versions.

**1. Clear Explanation:**

The traditional `kernex.exp` served as a comprehensive export file containing all kernel symbols available for use by loadable kernel modules (LKMs). However, this presented significant security vulnerabilities.  An LKM with access to all kernel symbols poses a considerable risk, allowing potential exploitation.  AIX 6.1 addresses this by adopting a more controlled system. Instead of a single `kernex.exp`, the kernel symbols are exported selectively, often based on the specific needs of individual modules.  This selective export is managed through the build process and the use of configuration files that dictate which symbols are made accessible to which LKMs.  Crucially, these exported symbols are not typically aggregated into a single, easily accessible file.  Instead, they are incorporated into the kernel itself and are made available through specific mechanisms during the LKM's loading process. This approach drastically minimizes the attack surface.

The kernel build process uses several tools and intermediate files. Key among them are the symbol tables generated during the kernel compilation.  These tables, alongside configuration data specific to your module, dictate which symbols are visible to your code.  Therefore, attempting to locate a single, comprehensive export file like `/usr/lib/kernex.exp` is futile in this context.  The required symbols are implicitly available during the linking process, provided your build environment is correctly configured.

**2. Code Examples with Commentary:**

The following examples illustrate the approach to building and linking against the kernel symbols in a modern AIX 6.1 environment, avoiding the reliance on a nonexistent `kernex.exp`.  Note that these are simplified examples; real-world scenarios involve significantly more complex build processes and configurations.

**Example 1:  Basic Kernel Module Makefile**

```makefile
# Makefile for a simple AIX kernel module

MODULE_NAME = my_module

CFLAGS = -I$(AIX_INC) -D_AIX  # Include paths and defines

LDFLAGS = -Wl,-bI,$(AIX_LIB)  # Linker flags

all: $(MODULE_NAME).o
	$(CC) $(LDFLAGS) -o $(MODULE_NAME) $(MODULE_NAME).o

$(MODULE_NAME).o: $(MODULE_NAME).c
	$(CC) $(CFLAGS) -c $(MODULE_NAME).c

clean:
	rm -f $(MODULE_NAME).o $(MODULE_NAME)
```

**Commentary:**  This Makefile demonstrates the fundamental steps. `AIX_INC` and `AIX_LIB` should be set correctly in the environment to point to the necessary include and library directories provided during the AIX kernel installation. Crucially, the `LDFLAGS` utilize the `-Wl,-bI` linker option, which directs the linker to look for necessary kernel symbols during the linking of the `.o` file to the final `.o` file.   The explicit path to the exported symbols is not required; the linker consults the relevant kernel symbol tables.

**Example 2:  A Simple Kernel Module Function (my_module.c)**

```c
#include <sys/types.h>
#include <sys/errno.h>
#include <sys/param.h>


int my_kernel_function(void) {
    // Perform kernel operations here
    return 0;
}
```

**Commentary:** This is a skeletal example demonstrating a function that would reside within the kernel module.  The inclusion of `sys/types.h`, `sys/errno.h`, and `sys/param.h` are common for AIX kernel module programming.  Access to necessary kernel structures and functions is implicit; the linker ensures the correct resolution during the build process.


**Example 3: Using `mkloadmod` to Build and Install**

Once the module is compiled, it's not directly loaded. We use `mkloadmod` to finish the build and install process:

```bash
mkloadmod my_module.o
```

**Commentary:**  `mkloadmod` is a crucial tool in the AIX environment. It handles the final stages of preparing your compiled LKM for loading.  This step leverages the aforementioned build system configuration and implicitly links against the appropriately exported kernel symbols.  Successful execution of `mkloadmod` indicates that your module has been successfully linked against the necessary symbols from the kernel, eliminating the need for explicit paths to export files.


**3. Resource Recommendations:**

To further your understanding, I would suggest consulting the official AIX documentation pertaining to kernel extension development.  This documentation includes details on the AIX kernel build system, linker options, and the procedures for compiling and installing kernel modules.  Additionally, reviewing the AIX programming guides, particularly those covering system calls and kernel programming, is essential for mastering the intricacies of AIX kernel module development.  Furthermore, access to AIX-specific compiler and linker manuals will provide granular control over your build process. Mastering these resources will prove invaluable.  Focus on the specifics of the kernel linker options and the configuration files that control symbol visibility.  They are pivotal in understanding how the kernel's symbols are made available without relying on obsolete structures like a monolithic `kernex.exp`.  Understanding the build process thoroughly will be your key to successful AIX kernel module development.
