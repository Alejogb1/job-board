---
title: "Why can't the performance monitor use a symbol defined in the kernel module?"
date: "2025-01-30"
id: "why-cant-the-performance-monitor-use-a-symbol"
---
The fundamental disconnect preventing a performance monitor from directly accessing symbols defined within a kernel module stems from the separation of address spaces and the inherent design of the operating system's memory management. Kernel modules, while integral extensions of the kernel itself, operate within the kernel address space, a protected region inaccessible to user-space applications, where performance monitors typically reside. Direct memory access from user-space to kernel-space is strictly prohibited for system stability and security reasons.

Let's consider a scenario I encountered while developing a custom I/O scheduler for a high-performance storage system. I had embedded performance counters within my kernel module – counters tracking the number of read and write operations issued by the module. Initially, I envisioned using a straightforward performance monitoring application written in C to read these counters directly from their memory addresses. My user-space application, even when running with elevated privileges (via `sudo`), would trigger a segmentation fault attempting to access the kernel's memory locations. This is a hard failure at the operating system's core and reflects the expected behavior when breaching address space boundaries.

The crux of the problem lies in the fact that the virtual address space mappings used by the kernel and user-space processes are distinct. Even if a kernel module defines a global variable (symbol) with an address, that address is *virtual* within the kernel's context. User-space applications operate in a separate virtual address space, and the kernel's virtual address map is not accessible to them. Trying to dereference a memory address generated within the kernel, using user-space pointer operations, inevitably leads to an attempt to access an invalid memory region within the user's address space, thus resulting in a segmentation fault, or an equivalent access violation.

Moreover, the kernel module might not even have a symbol table accessible outside the kernel. The symbol table, which holds the association between symbolic names and memory addresses, is primarily intended for debugging purposes within the kernel environment itself. Kernel code compilation and loading can dynamically manage symbols, and these are often not exported for external consumption. This is by design for protection and flexibility within the kernel environment.

To bridge this communication gap and monitor a kernel module's internal metrics, an established mechanism is required to transfer the needed data to user-space through a controlled, secure method. Common methods used for this data transfer often include, but are not limited to: exposing debugfs entries, registering character devices, utilizing ioctl interfaces, or, on modern systems, exploiting eBPF facilities. Each provides a mechanism for the kernel module to "expose" its data for user space consumption.

Here are three code examples illustrating different approaches for a kernel module to provide its data to a user-space performance monitor:

**Example 1: Using `/proc` or `/sys/` (debugfs)**

```c
// Kernel Module (example_module.c)
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/fs.h>
#include <linux/debugfs.h>

static struct dentry *my_dir, *my_file;
static u64 counter = 0;

static ssize_t my_file_read(struct file *file, char __user *buf, size_t len, loff_t *offset) {
    char temp_buf[32];
    int temp_len = snprintf(temp_buf, 32, "%llu\n", counter);
    return simple_read_from_buffer(buf, len, offset, temp_buf, temp_len);
}

static const struct file_operations my_file_ops = {
    .read = my_file_read,
};

static int __init example_init(void) {
    my_dir = debugfs_create_dir("my_module_debug", NULL);
    if (!my_dir) return -ENOMEM;
    my_file = debugfs_create_file("counter", 0444, my_dir, NULL, &my_file_ops);
    if (!my_file) {
        debugfs_remove_recursive(my_dir);
        return -ENOMEM;
    }

    printk(KERN_INFO "Module Loaded\n");
    return 0;
}

static void __exit example_exit(void) {
    debugfs_remove_recursive(my_dir);
    printk(KERN_INFO "Module Unloaded\n");
}

module_init(example_init);
module_exit(example_exit);
MODULE_LICENSE("GPL");
MODULE_AUTHOR("Example Author");

// In your module's operational code:
void increment_counter(void) {
     counter++;
}
EXPORT_SYMBOL(increment_counter);

```

**Explanation:**

This example uses the `debugfs` virtual file system (usually mounted at `/sys/kernel/debug`) to expose the counter. The module creates a directory "my_module_debug" and a file "counter" inside it. A user-space application can then open `/sys/kernel/debug/my_module_debug/counter` and read the current value of the counter. The `simple_read_from_buffer` function provides a safe mechanism for transferring data to user-space. Note the counter is increased by a function `increment_counter`, which would be part of your modules main operation. The function is declared `EXPORT_SYMBOL` to make it accessible to other parts of the kernel that your module might interact with.
**Example 2: Character Device Driver**

```c
// Kernel Module (example_char_device.c)
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/fs.h>
#include <linux/cdev.h>
#include <linux/uaccess.h>

#define DEVICE_NAME "my_char_dev"
#define CLASS_NAME "my_chardev_class"

static int major_number;
static struct class*  my_char_class  = NULL;
static struct device* my_char_device = NULL;

static struct cdev my_cdev;

static u64 char_counter = 0;

static int my_char_open(struct inode *inode, struct file *file) {
    return 0;
}

static int my_char_release(struct inode *inode, struct file *file) {
    return 0;
}

static ssize_t my_char_read(struct file *file, char __user *buf, size_t len, loff_t *offset) {
    char temp_buf[32];
    int temp_len = snprintf(temp_buf, 32, "%llu\n", char_counter);
    if (copy_to_user(buf, temp_buf, temp_len))
        return -EFAULT;
    return temp_len;

}


static const struct file_operations my_char_ops = {
    .open    = my_char_open,
    .release = my_char_release,
    .read    = my_char_read,
};

static int __init my_char_init(void) {
    major_number = register_chrdev(0, DEVICE_NAME, &my_char_ops);
    if (major_number < 0) {
        printk(KERN_ALERT "Failed to register char device\n");
        return major_number;
    }

   my_char_class = class_create(THIS_MODULE, CLASS_NAME);
	if (IS_ERR(my_char_class)){
		unregister_chrdev(major_number, DEVICE_NAME);
		printk(KERN_ALERT "Failed to register char device class\n");
		return PTR_ERR(my_char_class);
	}

    my_char_device = device_create(my_char_class, NULL, MKDEV(major_number, 0), NULL, DEVICE_NAME);
    if (IS_ERR(my_char_device)) {
        class_destroy(my_char_class);
        unregister_chrdev(major_number, DEVICE_NAME);
        printk(KERN_ALERT "Failed to create char device\n");
        return PTR_ERR(my_char_device);
    }


    printk(KERN_INFO "Char Device Initialized with major num:%d\n",major_number);
    return 0;
}

static void __exit my_char_exit(void) {
    device_destroy(my_char_class, MKDEV(major_number, 0));
    class_unregister(my_char_class);
    class_destroy(my_char_class);
    unregister_chrdev(major_number, DEVICE_NAME);
    printk(KERN_INFO "Char Device Unloaded\n");
}

module_init(my_char_init);
module_exit(my_char_exit);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Example Author");

// In your module's operational code:
void increment_char_counter(void) {
     char_counter++;
}
EXPORT_SYMBOL(increment_char_counter);

```
**Explanation:**

This example registers a character device (typically accessible through a device file such as `/dev/my_char_dev`). User-space programs can open and read from this file, thereby obtaining the kernel module's counter value. The function `copy_to_user` handles data transfer safely between the kernel and user-space. Again, the counter can be increased by a function `increment_char_counter` in your module's main operational code.

**Example 3: Using a Simple ioctl Interface**

```c
// Kernel Module (example_ioctl.c)
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/fs.h>
#include <linux/cdev.h>
#include <linux/uaccess.h>
#include <linux/ioctl.h>

#define DEVICE_NAME "my_ioctl_dev"
#define CLASS_NAME "my_ioctldev_class"

#define MY_IOCTL_GET_COUNTER _IOR('K', 1, u64*)

static int major_number;
static struct class*  my_ioctl_class  = NULL;
static struct device* my_ioctl_device = NULL;
static struct cdev my_ioctl_cdev;
static u64 ioctl_counter = 0;


static int my_ioctl_open(struct inode *inode, struct file *file) {
    return 0;
}

static int my_ioctl_release(struct inode *inode, struct file *file) {
    return 0;
}
static long my_ioctl_ioctl(struct file *file, unsigned int ioctl_num, unsigned long ioctl_param)
{
    u64 *user_ptr = (u64 *)ioctl_param;

	switch(ioctl_num)
	{
		case MY_IOCTL_GET_COUNTER:
             if (copy_to_user(user_ptr, &ioctl_counter, sizeof(u64)))
                 return -EFAULT;
		 break;
        default:
            return -ENOTTY;
	}
	return 0;
}

static const struct file_operations my_ioctl_ops = {
    .open = my_ioctl_open,
    .release = my_ioctl_release,
    .unlocked_ioctl = my_ioctl_ioctl
};

static int __init my_ioctl_init(void) {
    major_number = register_chrdev(0, DEVICE_NAME, &my_ioctl_ops);
    if (major_number < 0) {
        printk(KERN_ALERT "Failed to register ioctl device\n");
        return major_number;
    }
    my_ioctl_class = class_create(THIS_MODULE, CLASS_NAME);
    if (IS_ERR(my_ioctl_class)){
        unregister_chrdev(major_number, DEVICE_NAME);
        printk(KERN_ALERT "Failed to register ioctl device class\n");
        return PTR_ERR(my_ioctl_class);
    }

    my_ioctl_device = device_create(my_ioctl_class, NULL, MKDEV(major_number, 0), NULL, DEVICE_NAME);
    if (IS_ERR(my_ioctl_device)) {
        class_destroy(my_ioctl_class);
        unregister_chrdev(major_number, DEVICE_NAME);
        printk(KERN_ALERT "Failed to create ioctl device\n");
        return PTR_ERR(my_ioctl_device);
    }

    printk(KERN_INFO "IOCTL Device Initialized with major num:%d\n",major_number);
    return 0;
}

static void __exit my_ioctl_exit(void) {
    device_destroy(my_ioctl_class, MKDEV(major_number, 0));
    class_unregister(my_ioctl_class);
    class_destroy(my_ioctl_class);
    unregister_chrdev(major_number, DEVICE_NAME);
    printk(KERN_INFO "IOCTL Device Unloaded\n");
}

module_init(my_ioctl_init);
module_exit(my_ioctl_exit);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Example Author");

// In your module's operational code:
void increment_ioctl_counter(void) {
   ioctl_counter++;
}
EXPORT_SYMBOL(increment_ioctl_counter);
```

**Explanation:**

This example utilizes the `ioctl` mechanism on a character device. Here a specific command (defined by `MY_IOCTL_GET_COUNTER`) is set to return the counter's value to user space. The `copy_to_user` function ensures safe data transfer between the kernel and user-space. The `ioctl_counter` can be increased by the function `increment_ioctl_counter`.

In summary, direct access of a kernel module’s symbols from user-space is not permissible due to the fundamental address space separation between user and kernel contexts. Communication between these domains requires established mechanisms which act as intermediaries. These mechanisms, such as debugfs, character devices, or ioctl interfaces, are designed to transfer data securely and controllably between these protected domains.

For further information, I recommend consulting texts on Linux kernel internals and driver development. Look for materials that cover:
*   Linux kernel memory management
*   Device drivers
*   The /proc and /sys file systems
*   ioctl interfaces
*   The debugfs file system
*   eBPF

These resources will provide a more comprehensive understanding of the underlying principles and mechanisms involved in kernel module development and performance monitoring.
