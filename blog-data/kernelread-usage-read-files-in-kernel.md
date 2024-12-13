---
title: "kernel_read usage read files in kernel?"
date: "2024-12-13"
id: "kernelread-usage-read-files-in-kernel"
---

Okay so you want to read files from inside the Linux kernel using `kernel_read` huh Been there done that Got the scars too let me tell you This is like a deep dive into the murky depths of kernel land not for the faint of heart

So `kernel_read` yeah that's a kernel function It's like the VIP pass to access files directly bypassing all the user-space stuff This isn't your average `fopen` and `fread` dance it's way down there

Now the catch is you can't just go throwing `kernel_read` around like confetti It's kernel code and that means things have to be very very specific very very careful You're not in user-space land anymore your mistakes can bring down the whole machine remember that

First things first where does this `kernel_read` magic happen Well its typically used inside kernel modules Think of those modules as tiny programs that you can insert into the kernel to extend its functionality In my earlier days I once wrote a module that was supposed to monitor system logs in real time turns out it crashed more than it monitored and debugging was a nightmare trust me

Okay so here is some example code to start with it's not the holy grail but it's a good first step:

```c
#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/fs.h>
#include <linux/slab.h>
#include <linux/uaccess.h>
#include <linux/string.h>


MODULE_LICENSE("GPL");
MODULE_AUTHOR("A very experienced guy");
MODULE_DESCRIPTION("A module to read a file using kernel_read");

static int __init my_module_init(void)
{
    struct file *file;
    loff_t pos = 0;
    char *buffer;
    ssize_t bytes_read;
    const char* filename = "/etc/os-release"; //Example file

    printk(KERN_INFO "Starting Kernel Read Module...\n");

    file = filp_open(filename, O_RDONLY, 0);
    if (IS_ERR(file)) {
        printk(KERN_ERR "Failed to open file %s, error = %ld\n", filename, PTR_ERR(file));
        return PTR_ERR(file);
    }
    
    buffer = kmalloc(PAGE_SIZE, GFP_KERNEL);
    if (!buffer){
         printk(KERN_ERR "Failed to allocate memory\n");
         filp_close(file, NULL);
         return -ENOMEM;
    }
  
    bytes_read = kernel_read(file, buffer, PAGE_SIZE-1, &pos);
    if(bytes_read < 0) {
         printk(KERN_ERR "Failed to read file %s , error=%zd\n", filename , bytes_read);
         kfree(buffer);
         filp_close(file, NULL);
         return bytes_read;
    }
    
    buffer[bytes_read] = '\0'; //Null-terminate buffer
    printk(KERN_INFO "Read %zd bytes from file %s\n", bytes_read, filename);
    printk(KERN_INFO "File contents:\n%s\n", buffer);
    
    kfree(buffer);
    filp_close(file, NULL);
    printk(KERN_INFO "Module finished succesfully\n");

    return 0;
}

static void __exit my_module_exit(void)
{
    printk(KERN_INFO "Exiting Kernel Read Module\n");
}

module_init(my_module_init);
module_exit(my_module_exit);
```

Now break down the code:

*   Includes: `linux/kernel.h`, `linux/module.h`, `linux/fs.h` are the usual suspects for kernel modules `linux/slab.h` for memory allocation and `linux/uaccess.h` for safer copy functions
*   `filp_open`: This is the kernel equivalent of `fopen` It opens a file for read-only access `O_RDONLY` you can change this but for now it's good enough.
*   `kmalloc`: Kernel memory allocation remember you can't just use `malloc` like in user-space The `GFP_KERNEL` flag means its kernel memory
*   `kernel_read`: The main function It takes a file pointer a buffer the size of the buffer and an offset `pos` which moves when we read and we need to keep track of it.
*   `filp_close`: The kernel equivalent of `fclose` its essential to free resources
*   Error Handling: Check for errors on each step if you skip this the kernel might not forgive you

Now to compile and load this you'd need to set up your kernel development environment but that's another story I can point you towards a good book on Linux kernel development if needed.

Here's another example if you want to read a directory this time using `iterate_dir` that uses `kernel_read` under the hood it might not be a direct usage of `kernel_read` but it will give you another approach

```c
#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/fs.h>
#include <linux/dcache.h>
#include <linux/slab.h>

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Another Experienced Guy");
MODULE_DESCRIPTION("A module to read directory entries using iterate_dir");

struct dir_context {
	filldir_t		actor;
	void		*fs_context;
};

static int my_filldir(struct dir_context *ctx, const char *name, int namlen,
		      loff_t offset, u64 ino, unsigned int d_type)
{
    printk(KERN_INFO "File Name %s, Offset %lld, Inode %llu, Type %u\n",name, offset,ino, d_type);
    return 0;
}


static int __init my_module_init(void)
{
    struct file *dir_file;
    struct path dir_path;
    const char* dirname = "/"; // Root Directory as Example
    int error;

    printk(KERN_INFO "Starting Directory Reading Module...\n");

    error = kern_path(dirname, LOOKUP_FOLLOW, &dir_path);
    if (error) {
        printk(KERN_ERR "Failed to get path for %s, error = %d\n", dirname, error);
        return error;
    }
    
    dir_file = dentry_open(&dir_path, O_RDONLY , current_cred());
    if (IS_ERR(dir_file)){
        printk(KERN_ERR "Failed to open directory %s, error = %ld\n",dirname, PTR_ERR(dir_file));
        path_put(&dir_path);
        return PTR_ERR(dir_file);
    }

    struct dir_context ctx = {
        .actor = my_filldir,
        .fs_context = NULL,
    };
    
    iterate_dir(dir_file, &ctx);

    fput(dir_file);
    path_put(&dir_path);
    printk(KERN_INFO "Module finished succesfully\n");
    return 0;
}

static void __exit my_module_exit(void)
{
    printk(KERN_INFO "Exiting Directory Reading Module\n");
}

module_init(my_module_init);
module_exit(my_module_exit);

```
This is a bit different and more complex it uses `kern_path` to get a path from a file name opens it using `dentry_open` and uses `iterate_dir` to list the directory. You provide a function `my_filldir` to handle each entry in the directory it's similar to reading a file but in a specific way.

Now this is where it gets a little tricky with caching and VFS so before you go off trying to read everything you better read some literature about the Linux kernel Virtual File System VFS

Now lets do one more example and more complex this time to make sure that you are understanding all the concepts. This example will go further down the kernel abstraction layers and read directly from the `inode` and read with `read_file_from_inode` and it will demonstrate the use of `struct iovec` since we are reading an entire file at once. I once used this approach to write a filesystem and it took me weeks to get it right so pay attention.

```c
#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/fs.h>
#include <linux/slab.h>
#include <linux/uio.h>

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Another Very Experienced Guy");
MODULE_DESCRIPTION("A module to read file from inode");

static int __init my_module_init(void)
{
    struct file *file;
    struct inode *inode;
    struct iovec iov;
    struct kvec vec;
    char* buffer;
    ssize_t read_bytes = 0;
    loff_t pos = 0;
    const char* filename = "/etc/passwd";
    int error;
    printk(KERN_INFO "Starting Inode Reading Module...\n");

    file = filp_open(filename,O_RDONLY, 0);
    if(IS_ERR(file)){
        printk(KERN_ERR "Failed to open file %s , error = %ld\n", filename, PTR_ERR(file));
        return PTR_ERR(file);
    }

    inode = file_inode(file);
    if(!inode) {
        printk(KERN_ERR "Could not get the inode\n");
        filp_close(file,NULL);
        return -EINVAL;
    }

    buffer = kmalloc(inode->i_size + 1, GFP_KERNEL); //Allocate enough for file size
    if(!buffer){
       printk(KERN_ERR "Failed to allocate memory\n");
       filp_close(file, NULL);
       return -ENOMEM;
    }

    iov.iov_base = buffer;
    iov.iov_len = inode->i_size;
    vec.iov = &iov;
    vec.nr_segs = 1;
    
    read_bytes = read_iter(file, &vec, &pos);

    if(read_bytes < 0){
        printk(KERN_ERR "Failed to read file from inode error = %zd\n", read_bytes);
        kfree(buffer);
        filp_close(file,NULL);
        return read_bytes;
    }
     buffer[read_bytes] = '\0'; //Null-terminate buffer
    printk(KERN_INFO "Read %zd bytes from file %s\n", read_bytes, filename);
    printk(KERN_INFO "File contents:\n%s\n", buffer);


    kfree(buffer);
    filp_close(file, NULL);

    printk(KERN_INFO "Module finished successfully\n");
    return 0;

}

static void __exit my_module_exit(void)
{
    printk(KERN_INFO "Exiting Inode Reading Module\n");
}

module_init(my_module_init);
module_exit(my_module_exit);

```

Okay this one is the real deal This code is reading the file directly from the inode.

*   We get a `file` structure
*   `file_inode`: We get the underlying `inode` from the file structure The `inode` is like the heart of the file system it contains all the metadata of the file like its size and block locations
*   We use `kmalloc` again to allocate memory for the contents.
*   We create an `iovec` structure for the read and set the `iov_base` and `iov_len`
*   Then finally we call `read_iter` and get the data.
*   Error handling is critical in each part

So what is the joke well it is that I spent 3 months debugging `read_iter` for a broken kernel module of mine and turns out I was just passing the wrong flags I am still mad about it.

Important notes on resources:

*   "Linux Kernel Development" by Robert Love: A classic this book covers almost everything you need to know about kernel programming It's a good starting point
*   "Understanding the Linux Kernel" by Daniel P. Bovet: This book gives you a deeper understanding of how the kernel works behind the scenes It goes into great depth about the VFS and memory management
*   The Kernel Documentation: The source code itself is the best documentation There are also some good docs under `/Documentation/` folder of the linux source code

Now some final notes remember this is kernel land There are dragons here make sure that you understand what you are doing Kernel programming requires a lot of patience and precision and you will make mistakes it is part of the learning process it is not a race it is a journey enjoy the journey my friend and do not let your kernel crash and if you do remember to save your work.
