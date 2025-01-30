---
title: "How to call system functions from AIX native code?"
date: "2025-01-30"
id: "how-to-call-system-functions-from-aix-native"
---
Direct system interaction on AIX, particularly when invoked from native code, necessitates a profound understanding of the operating system's Application Binary Interface (ABI) and system call mechanisms. Over my years developing performance-critical applications on AIX, I've repeatedly engaged with these low-level interfaces, revealing both their power and their potential pitfalls.

The core concept revolves around the `syscall` function, a C library interface that acts as a gateway to the operating system kernel. Unlike functions in standard libraries, which often implement complex functionality through other library calls, `syscall` directly executes a system call within the kernel. System calls provide fundamental services, such as file I/O, memory management, and process control. Each system call is identified by a unique integer, conventionally referred to as the system call number. Furthermore, these system calls accept a defined set of arguments, which must be marshalled and passed correctly to ensure successful execution.

To use `syscall`, you must first include the `unistd.h` header file, which provides the function prototype. The general format is `long syscall(long number, ...);`. The initial `long number` argument is the aforementioned system call number. Following this, an arbitrary number of arguments may be passed as required by the specific system call. Crucially, the return value of `syscall` is a `long`, which represents the kernel’s return code. A negative return typically signifies an error condition and the actual error details are set in the global `errno` variable, as defined in `<errno.h>`.

The system call numbers themselves are not portable, they differ between operating systems and even between versions within an operating system family. On AIX, system call numbers can be found in the header file `/usr/include/sys/syscall.h`, and a more human-readable version with function prototypes can be found in `/usr/include/sys/syscalls.h`. While you could use the raw numbers directly from these files, it is generally recommended that you use the symbolic constants (e.g. `SYS_open` instead of the literal number). This will make your code more readable and easier to maintain if AIX system call numbers should ever be updated (though it has been relatively stable over time).

The correct arguments to be passed to each system call are critically dependent on the definition of that system call within the kernel, often involving pointers, integers, and structures passed by reference or by value. Improper argument handling will typically result in program crashes or unpredictable behavior. Therefore, consulting the `man` pages for the system call in question is not optional, but an imperative step before invoking `syscall`. For instance, `man 2 open` will detail the parameters required by the `open` system call and their order. The `errno` variable, in the event of an error, should always be checked and can be interpreted by referring to `errno.h`.

Here are three illustrative code examples to demonstrate system call usage on AIX:

**Example 1: Using `open` and `close` to Create a File**
```c
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <stdio.h>
#include <string.h>

int main() {
    int fd;
    const char *filename = "testfile.txt";

    fd = syscall(SYS_open, filename, O_CREAT | O_WRONLY, 0644);
    if (fd < 0) {
        fprintf(stderr, "Error opening file: %s, errno: %d, %s\n", filename, errno, strerror(errno));
        return 1;
    }

    const char *message = "Hello from AIX system call!\n";
    ssize_t bytes_written = syscall(SYS_write, fd, message, strlen(message));
    if (bytes_written < 0){
        fprintf(stderr, "Error writing to file: %s, errno: %d, %s\n", filename, errno, strerror(errno));
         syscall(SYS_close, fd);
         return 1;
    }

    if(syscall(SYS_close, fd) < 0){
        fprintf(stderr, "Error closing file: %s, errno: %d, %s\n", filename, errno, strerror(errno));
        return 1;
    }

    printf("File '%s' created and written to successfully.\n", filename);
    return 0;
}
```
This first example illustrates a common operation: creating and writing to a file. The `SYS_open` system call is employed with the `O_CREAT` and `O_WRONLY` flags, creating the file `testfile.txt` if it does not exist, or truncating it if it does exist. The `0644` argument specifies the file's permissions (read/write for the user, read-only for the group and others). If the file descriptor `fd` is negative, it signals an error, and we use `strerror` to convert the `errno` to human readable form. After writing the message and closing the file, we check for errors again. This demonstrates the need to handle errors after *every* system call.

**Example 2: Reading System Time Using `times`**
```c
#include <unistd.h>
#include <sys/times.h>
#include <errno.h>
#include <stdio.h>
#include <time.h>

int main() {
    struct tms tbuf;
    clock_t ticks;

    ticks = syscall(SYS_times, &tbuf);
    if (ticks == (clock_t)-1) {
       fprintf(stderr, "Error retrieving times, errno: %d, %s\n", errno, strerror(errno));
        return 1;
    }
    printf("User CPU time: %ld ticks\n", tbuf.tms_utime);
    printf("System CPU time: %ld ticks\n", tbuf.tms_stime);
    printf("Clock ticks: %ld\n", ticks);

    return 0;
}
```
The second example illustrates a system call that requires a structure passed by reference as an argument (`struct tms`). The `SYS_times` system call populates this structure with process-related timing information. The return value of `times` is the clock ticks since an arbitrary point in the past. Again, proper error checking is crucial, as a failure here might suggest a serious operating system fault. This illustrates marshaling a specific structure for system call.

**Example 3: Using `getpid` to Obtain Process ID**
```c
#include <unistd.h>
#include <stdio.h>
#include <errno.h>
#include <string.h>

int main() {
  pid_t pid = syscall(SYS_getpid);

  if (pid < 0) {
        fprintf(stderr, "Error getting pid: errno: %d, %s\n", errno, strerror(errno));
        return 1;
  }
  printf("Process ID: %d\n", pid);
  return 0;
}
```
This third example demonstrates a very simple system call that does not require any arguments. `SYS_getpid` returns the process ID of the current running program. Again, error handling is paramount, although a failure here is highly unlikely under normal circumstances. Even the simplest calls require careful checking.

When working with system calls, several caveats should be noted. It's critical to be mindful that incorrect use can result in the program crashing or, more concerning, causing operating system instability. It’s vital to carefully read the documentation for each system call.

I also advise testing low-level system calls in a carefully controlled development environment before integrating them into a production system. Use of debugging tools such as `gdb` can be beneficial in these cases. A deep understanding of AIX kernel architecture is also paramount for more complex scenarios, where low level system calls must be combined.

For further learning, I recommend the following resources:
1. **AIX Operating System Documentation**:  The official documentation from IBM provides detailed descriptions of all aspects of AIX, including the kernel and system calls.
2. **Advanced Programming in the UNIX Environment**: A widely regarded text covering the detailed low-level concepts required when working with operating system interfaces.
3. **AIX specific header files:** `/usr/include/sys/syscall.h` and `/usr/include/sys/syscalls.h` which are critical for understanding AIX specific system calls and numbers.

In summary, invoking system calls directly on AIX from native code is achievable through the `syscall` function. However, this approach demands precision and careful attention to detail, particularly concerning system call numbers, parameter passing conventions, and error handling. This interaction should be employed with full awareness of the potential risks and only when absolutely necessary, having carefully explored higher-level library alternatives.
