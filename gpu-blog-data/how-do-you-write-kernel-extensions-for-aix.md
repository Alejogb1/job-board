---
title: "How do you write kernel extensions for AIX?"
date: "2025-01-26"
id: "how-do-you-write-kernel-extensions-for-aix"
---

Kernel extensions on AIX, often referred to as loadable kernel modules (LKMs), offer a powerful mechanism to extend the operating system's functionality. I've spent considerable time debugging and developing these, and a core understanding revolves around their direct access to the kernel's address space. This power, of course, comes with responsibility and risks, as improperly implemented extensions can destabilize the entire system. This contrasts significantly with user-level programming.

The development of an AIX kernel extension fundamentally involves crafting C code that interfaces with the kernel's internal structures and APIs. These APIs are primarily accessed through header files provided by IBM, specifically those found in the `/usr/include/sys` directory, supplemented by other system-specific headers. Unlike a typical user application, your extension will execute with kernel privileges and is loaded into the kernel's address space using the `loadext` command. A successful extension interacts seamlessly with existing kernel services, potentially adding new ones or modifying existing behaviors. The lifecycle of a kernel extension is also different; it's loaded, initialized, runs, potentially deinitialized, and unloaded – a sequence managed by the kernel.

The initial stage involves creating the core C source file. This file will include a number of essential components: an entry point function, usually named `extinit()`, which is automatically called by the kernel upon loading; optional initialization and deinitialization routines; and the main functionality of the module. For any direct interaction with the hardware, you must exercise considerable care and adhere to the AIX Device Driver Programming manual's specific advice. The code also needs to be compiled using a suitable compiler chain, specifically the AIX-specific compiler (`xlc` or `gcc`) and associated toolchain, and linked to create an object file. Finally, `genext` is used to package the compiled object into a loadable form (.ext file).

Let's look at a simplified example:

```c
#include <sys/types.h>
#include <sys/errno.h>
#include <sys/lockl.h>
#include <sys/sleep.h>
#include <sys/sysmacros.h>
#include <sys/param.h>
#include <sys/proc.h>
#include <sys/systm.h>
#include <sys/xmem.h>
#include <sys/uio.h>

int my_global_counter = 0;

int extinit() {
  printf("My first kernel extension loaded.\n");
  return 0;
}

int my_kernel_function(int value) {
    lock_t my_lock;
    int old_pri;

    // Kernel locking is critical - never operate on shared data without
    // appropriate locking, else race conditions lead to a kernel panic.
    // This uses a simple spin lock, consider more sophisticated options as
    // appropriate for your application.
    old_pri = lockl(&my_lock, LOCK_SHORT);

    my_global_counter += value;

    unlockl(&my_lock, old_pri);
    
    return my_global_counter;
}

int extproc() {
    // This is the deinit routine - resources should be freed here.
    printf("My first kernel extension unloaded.\n");
    return 0;
}
```

This rudimentary example provides a starting point. The `extinit` function simply prints a confirmation message. The `my_kernel_function` function, demonstrates the crucial use of kernel spin locks using `lockl` and `unlockl` to protect a shared resource, `my_global_counter`. This function adds the given `value` to the counter. Proper locking is non-negotiable within kernel code to prevent race conditions. The `extproc()` is the deinitialization function that executes when the module is unloaded. Note that a kernel module that doesn't unload cleanly can cause issues and the function itself can be called in error conditions as well. We will return 0 in case of a successful process. This simple code would be compiled with the command:
`xlc -c my_extension.c`
and `genext -o my_extension.ext my_extension.o`
resulting in the loadable kernel extension, `my_extension.ext`.

Now consider a module that interacts with user-space. You often need to transfer data in or out of your extension. It's vital to avoid directly referencing user-space memory as it’s outside the kernel’s protected address space; you must use specialized functions. The following example showcases memory transfer using `copyin` and `copyout`:

```c
#include <sys/types.h>
#include <sys/errno.h>
#include <sys/lockl.h>
#include <sys/sleep.h>
#include <sys/sysmacros.h>
#include <sys/param.h>
#include <sys/proc.h>
#include <sys/systm.h>
#include <sys/xmem.h>
#include <sys/uio.h>


int extinit_mem_transfer() {
    printf("Memory Transfer extension loaded.\n");
    return 0;
}

int my_transfer_function(char *user_input, int len, char *user_output) {
    char kernel_buffer[128];
    int err;

    if(len > 127) {
       return EINVAL; // User input buffer too big
    }

    // Copy data from user-space
    err = copyin(user_input, kernel_buffer, len);
    if (err != 0)
        return err;

    // Modify the data in the kernel
    for(int i=0; i < len; i++){
      kernel_buffer[i] = kernel_buffer[i] + 1;
    }

    // Copy data back to user-space
    err = copyout(kernel_buffer, user_output, len);
    if (err != 0)
        return err;

    return 0; // Success
}

int extproc_mem_transfer() {
    printf("Memory Transfer extension unloaded.\n");
    return 0;
}
```
In `my_transfer_function`, the user's input `char *user_input` of specified length `len` is transferred into the kernel's memory using `copyin` into the `kernel_buffer`, which is allocated on the stack. `copyin` handles the required checks to avoid the problems associated with directly referencing a user-space address, returning 0 on success. Subsequently, the content is processed, in this case, incrementing each byte by one. The processed data is returned to the user via `copyout` into the `user_output` location also of length `len`. Again, error checking is mandatory to handle invalid operations.

Finally, consider the inclusion of a system call. To introduce a new system call, you need to modify the kernel's system call table. This requires careful registration of your new call and is a relatively complex process. Here is a very simplified version:

```c
#include <sys/types.h>
#include <sys/errno.h>
#include <sys/lockl.h>
#include <sys/sleep.h>
#include <sys/sysmacros.h>
#include <sys/param.h>
#include <sys/proc.h>
#include <sys/systm.h>
#include <sys/xmem.h>
#include <sys/uio.h>
#include <sys/syscall.h>


int extinit_syscall() {
  printf("System Call extension loaded.\n");
  return 0;
}

// This would be a unique number which may not be already assigned to a different system call.
#define MY_CUSTOM_SYSCALL 300

int my_syscall_handler(int arg1, int arg2, int arg3) {

    // Kernel locking is critical for global resources.
    lock_t my_syscall_lock;
    int old_pri;
    old_pri = lockl(&my_syscall_lock, LOCK_SHORT);

    printf("Custom System Call handler invoked: %d, %d, %d\n", arg1, arg2, arg3);
    unlockl(&my_syscall_lock, old_pri);

    return arg1 + arg2 + arg3;
}

int extproc_syscall() {
   printf("System Call extension unloaded.\n");
   return 0;
}

// In a real-world extension, you would need to properly add this to the system call table
// The system call registration is significantly more involved than this example
// Consider the system call table to be a global shared resource, locking of this would be crucial
// during add/removal of calls. In addition, the handler would need to be added to a dispatcher.
// This example is just for illustrative purposes only.
// This simplified version would need to be added to the sysent.c system call dispatch table
// during kernel compilation which is not normally what you would do for an LKM.
// Example only: sysent[MY_CUSTOM_SYSCALL].sy_call = (caddr_t) my_syscall_handler
// Example only: sysent[MY_CUSTOM_SYSCALL].sy_narg = 3
```
In this snippet, `MY_CUSTOM_SYSCALL` represents a system call number. The `my_syscall_handler` function would process the arguments passed to the system call. The important note here is that the mechanism for actually registering this system call is not shown and is much more complicated than a simple assignment, involving more complex kernel-specific techniques. This needs to be added to the system call table, the `sysent` structure, and there is a lot more error checking involved. This simplified example only attempts to highlight the code required for the syscall handler itself. As a module writer, you would likely need to utilize a set of macros to add the system call dynamically at module load time.

Developing AIX kernel extensions is an advanced topic. The IBM documentation is the best resource. Specifically, I recommend the *AIX Device Driver Programming Guide* for general development information and *AIX Kernel Extensions and Device Support* for additional technical detail. Thorough testing in a non-production environment before deploying to any production system is essential. IBM has extensive support materials online as well. Remember that mistakes can result in a system crash, which can require a reboot to resolve.
