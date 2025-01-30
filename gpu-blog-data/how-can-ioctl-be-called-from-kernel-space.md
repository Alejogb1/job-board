---
title: "How can ioctl be called from kernel space in Linux kernel 5.10?"
date: "2025-01-30"
id: "how-can-ioctl-be-called-from-kernel-space"
---
The `ioctl` system call, fundamentally a conduit for userspace applications to interact with device drivers, presents a distinct challenge when invoked directly from within kernel space. Unlike typical system calls, it doesn't have a direct kernel-space analogue. Instead, we manipulate the underlying file structure and use internal kernel APIs to achieve similar functionality.

My experience from developing a custom network driver for an embedded system, a project utilizing the 5.10 kernel, highlighted the need for kernel-initiated `ioctl`-like operations. In our case, the driver needed to communicate specific configuration changes to a subordinate hardware module, which was controlled via a custom interface abstracted by the driver itself. We couldn't rely on a userspace application constantly sending commands, as the logic was intertwined with the driver's internal state. This necessitated exploring how to emulate an `ioctl` operation from within the kernel module.

The core challenge stems from the fact that `ioctl` relies on a file descriptor, typically obtained through an `open` system call in userspace. Kernel modules generally don't have file descriptors that map directly to devices in the same way. To circumvent this, we must access the underlying file structure associated with the device. This involves directly working with the `struct file` object, which encapsulates the file descriptor information.

The process begins by obtaining a `struct file` pointer associated with the device the driver manages. If your driver is primarily handling device operations, you likely already have a context in which the underlying file structure can be accessed. If not, you must locate the `struct file` through other means, for example, by iterating through registered devices. Once you have that pointer, the next crucial step is invoking the device’s file operations’ `unlocked_ioctl` or `compat_ioctl` method, rather than the system call. This operation is where the magic happens, providing a kernel-side access point to the same entry point a userspace application would trigger via `ioctl`.

Below are three code examples demonstrating this principle, using hypothetical scenarios to illustrate different aspects:

**Example 1: Direct `unlocked_ioctl` call**

This code snippet demonstrates the most basic scenario: invoking the `unlocked_ioctl` method associated with the device.

```c
#include <linux/fs.h>
#include <linux/ioctl.h>
#include <linux/module.h>
#include <linux/device.h>

static int my_kernel_ioctl(struct device *dev) {
    struct file *file_p = NULL;
    int ret = -ENODEV;
    
    if(!dev) {
        printk(KERN_ERR "No device provided to my_kernel_ioctl.\n");
        return ret;
    }

    // Assumes the file structure is accessible via a device struct field;
    // Replace with the actual way your driver access the file structure.
    file_p = (struct file*)dev_get_drvdata(dev);
    if (!file_p) {
        printk(KERN_ERR "No file structure associated with the device.\n");
        return ret;
    }


    if(file_p->f_op->unlocked_ioctl) {
        int cmd = _IOW('M', 0x1, int); // Hypothetical command
        int arg = 10;
        ret = file_p->f_op->unlocked_ioctl(file_p, cmd, (unsigned long)&arg);
        if(ret) {
            printk(KERN_ERR "Error during unlocked_ioctl, error code %d\n", ret);
            return ret;
        } else {
            printk(KERN_INFO "Unlocked_ioctl command successful.\n");
        }

    } else {
        printk(KERN_ERR "No unlocked_ioctl operation in the file structure.\n");
        ret = -ENOTTY;
    }
    
    return ret;
}

static int __init my_module_init(void) {
	struct device *test_device; // Assuming this device is defined
    printk(KERN_INFO "Initiating Kernel IOCTL test\n");

	// Normally your driver module would obtain a device pointer during initialization or
	// via a device matching system, here I simply pass the same pointer I used
    // when setting up the file operations on the device.
	test_device = get_device(); // A fictional function that gives back a pointer to the device struct

	if (test_device)
		my_kernel_ioctl(test_device);

    return 0;
}

static void __exit my_module_exit(void) {
    printk(KERN_INFO "Exiting Kernel IOCTL test\n");
}

module_init(my_module_init);
module_exit(my_module_exit);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Your Name");
```

This example shows the essential parts: locating the `struct file` pointer, checking the file operations structure for `unlocked_ioctl`, and finally invoking it with a command and argument. The `_IOW` macro creates the `ioctl` command, and we pass the command and the address of an argument. This assumes the device’s file operations contain a valid `unlocked_ioctl` handler.

**Example 2: Using `compat_ioctl` for 32-bit compatibility**

When your kernel module might interact with 32-bit userspace applications using `ioctl` (for example in a 64-bit kernel environment), you should handle compatibility using `compat_ioctl`.

```c
#include <linux/fs.h>
#include <linux/ioctl.h>
#include <linux/module.h>
#include <linux/device.h>
#include <linux/compat.h>

static int my_kernel_compat_ioctl(struct device *dev) {
    struct file *file_p = NULL;
    int ret = -ENODEV;

    if(!dev) {
        printk(KERN_ERR "No device provided to my_kernel_compat_ioctl.\n");
        return ret;
    }

    // Assumes the file structure is accessible via a device struct field.
    // Replace with the actual way your driver access the file structure.
    file_p = (struct file*)dev_get_drvdata(dev);
    if (!file_p) {
        printk(KERN_ERR "No file structure associated with the device.\n");
        return ret;
    }

    if (file_p->f_op->compat_ioctl) {
        int cmd = _IOW('N', 0x2, int);
        int arg = 25;
        ret = file_p->f_op->compat_ioctl(file_p, cmd, (unsigned long)&arg);
         if(ret) {
            printk(KERN_ERR "Error during compat_ioctl, error code %d\n", ret);
            return ret;
        } else {
            printk(KERN_INFO "Compat_ioctl command successful.\n");
        }
    } else {
         printk(KERN_ERR "No compat_ioctl operation in the file structure.\n");
        ret = -ENOTTY;
    }

    return ret;
}


static int __init my_module_init(void) {
	struct device *test_device; // Assuming this device is defined
    printk(KERN_INFO "Initiating Kernel COMPAT IOCTL test\n");

	// Normally your driver module would obtain a device pointer during initialization or
	// via a device matching system, here I simply pass the same pointer I used
    // when setting up the file operations on the device.
	test_device = get_device(); // A fictional function that gives back a pointer to the device struct

	if (test_device)
		my_kernel_compat_ioctl(test_device);

    return 0;
}

static void __exit my_module_exit(void) {
    printk(KERN_INFO "Exiting Kernel COMPAT IOCTL test\n");
}

module_init(my_module_init);
module_exit(my_module_exit);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Your Name");
```

This example demonstrates the usage of `compat_ioctl`. In most modern kernel builds, both `unlocked_ioctl` and `compat_ioctl` often perform the same or very similar actions, but using `compat_ioctl` when interacting with a device potentially accessed by 32-bit userspace processes is advisable for robustness. The logic is similar, but we access `f_op->compat_ioctl` rather than `f_op->unlocked_ioctl`.

**Example 3: Passing Complex Data Structures**

`ioctl` calls are often used for passing complex data structures. The kernel side must handle the memory transfer with care.

```c
#include <linux/fs.h>
#include <linux/ioctl.h>
#include <linux/module.h>
#include <linux/device.h>
#include <linux/slab.h>

// Example data structure to pass
struct my_ioctl_data {
    int value1;
    char name[32];
};

static int my_kernel_ioctl_struct(struct device *dev) {
   struct file *file_p = NULL;
    int ret = -ENODEV;
    struct my_ioctl_data data;

    if(!dev) {
        printk(KERN_ERR "No device provided to my_kernel_ioctl_struct.\n");
        return ret;
    }


    // Assumes the file structure is accessible via a device struct field
    // Replace with the actual way your driver access the file structure.
    file_p = (struct file*)dev_get_drvdata(dev);
    if (!file_p) {
        printk(KERN_ERR "No file structure associated with the device.\n");
        return ret;
    }

    if(file_p->f_op->unlocked_ioctl) {
       int cmd = _IOW('P', 0x3, struct my_ioctl_data);
        data.value1 = 123;
        strncpy(data.name, "Example", sizeof(data.name)-1);
        data.name[sizeof(data.name)-1]='\0';

        ret = file_p->f_op->unlocked_ioctl(file_p, cmd, (unsigned long)&data);
         if(ret) {
            printk(KERN_ERR "Error during unlocked_ioctl with structure, error code %d\n", ret);
            return ret;
        } else {
            printk(KERN_INFO "Unlocked_ioctl command with structure successful.\n");
        }
    } else {
        printk(KERN_ERR "No unlocked_ioctl operation in the file structure.\n");
        ret = -ENOTTY;
    }


    return ret;
}


static int __init my_module_init(void) {
	struct device *test_device; // Assuming this device is defined
    printk(KERN_INFO "Initiating Kernel IOCTL STRUCT test\n");


	// Normally your driver module would obtain a device pointer during initialization or
	// via a device matching system, here I simply pass the same pointer I used
    // when setting up the file operations on the device.
	test_device = get_device(); // A fictional function that gives back a pointer to the device struct

	if (test_device)
		my_kernel_ioctl_struct(test_device);

    return 0;
}

static void __exit my_module_exit(void) {
    printk(KERN_INFO "Exiting Kernel IOCTL STRUCT test\n");
}


module_init(my_module_init);
module_exit(my_module_exit);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Your Name");
```

This example shows how to pass a `struct` using `ioctl`. The crucial step is properly constructing your `_IOW` macro to include the type of your complex data structure. The kernel will then interpret the `unsigned long` passed as the address of the provided `struct`. Note that memory management within the kernel for this data structure remains the kernel module's responsibility; there is no automatic userspace to kernel memory mapping.

For those seeking a deeper understanding of these concepts, I recommend delving into the following resources: "Linux Device Drivers" by Jonathan Corbet, Alessandro Rubini, and Greg Kroah-Hartman provides a comprehensive overview of device driver development, including detailed discussion of file operations. Furthermore, the Linux kernel documentation itself, particularly the sections detailing file system and driver interactions, offers the most authoritative information. Finally, studying the source code of existing device drivers within the Linux kernel can also serve as an invaluable learning experience. Always remember to check the API documentation for your specific kernel version as the interfaces may change.
