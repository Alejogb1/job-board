---
title: "How can I identify called functions in a multi-process program without altering the source code?"
date: "2025-01-26"
id: "how-can-i-identify-called-functions-in-a-multi-process-program-without-altering-the-source-code"
---

Operating within a complex, multi-process environment often necessitates understanding the runtime behavior of individual processes, especially when source code modification is infeasible or undesirable. Identifying the exact functions called during execution, without direct code instrumentation, presents a significant challenge requiring leveraging operating system facilities and debugging techniques. My experience in performance analysis and reverse engineering has shown that this is most effectively achieved using a combination of dynamic tracing tools and system call interception.

The core principle involves observing the system calls made by the process; these calls often precede or are intimately associated with function execution, particularly those functions performing significant I/O operations or interacting with the operating system. In many systems, function calls involving shared libraries or OS functionality rely heavily on system calls. Therefore, capturing these calls provides a proxy for identifying the higher-level function execution path. This approach avoids the need for recompilation or direct modification of the target executable.

Specifically, one strategy involves utilizing system call tracing tools like `strace` on Linux-based systems or `dtruss` on macOS or similar tracing mechanisms on other platforms. These utilities allow us to monitor the system calls a particular process invokes during execution. Analyzing the output of such traces can reveal critical patterns and the sequence of operations that can correlate to called functions, especially when combined with information about shared object loading. The system call output is often noisy, filled with routine process interactions, but careful filtering, particularly by identifying calls to `open`, `read`, `write`, `connect`, `send`, `recv`, and `mmap`, is critical to correlate low-level system behavior to specific function calls. The analysis is often an iterative process of identifying system calls, comparing with library exports and then finding patterns corresponding to specific functional paths in the program.

Another technique involves more advanced dynamic tracing tools such as eBPF (Extended Berkeley Packet Filter) on Linux or similar mechanisms. eBPF programs can be attached to specific kernel events, including function entry and exit points, without modifying the target application’s source code. This approach allows for a more targeted and fine-grained level of instrumentation, enabling precise identification of function calls as they occur.

Here are several examples demonstrating practical implementations:

**Example 1: Using `strace` to identify file I/O function usage**

Suppose I have a binary called `process_data` that I suspect is opening and processing data files. We can use `strace` to identify the specific files that are opened:

```bash
strace -e open,read,write ./process_data data_file.txt
```

The `-e` flag specifies which system calls to trace, limiting the output to only `open`, `read`, and `write` syscalls, which are most directly associated with file I/O. This filtering is crucial since unfiltered `strace` output can be overwhelming. The `data_file.txt` is an arbitrary data file we supply to the program so that it executes a file I/O path.

**Commentary:**

*   This example focuses on a practical scenario: understanding file operations by a program. The filter helps to drastically reduce noise, leading to more relevant insights.
*   The `open` syscall will tell us what files were accessed, `read` will indicate if data is being read from these files, and `write` will reveal which files are modified.
*   The results from `strace` will not reveal the specific function that made these syscalls directly, however, observing that `open("data_file.txt")` is followed by a series of `read` system calls from the same file descriptor implies that some file reading function is executing. Further analysis would require correlating the address ranges of the function calls with symbols of the loaded libraries or executable itself.

**Example 2: Utilizing eBPF for function call tracing on Linux**

While full eBPF usage is complex, here’s a simplified concept using the `bpftrace` tool. Imagine a shared library `libdata.so` that contains a function `process_data`. We want to trace its execution:

```bash
bpftrace -e 'uprobe:/path/to/libdata.so:process_data { printf("Function process_data called\n"); }' ./process_data
```

**Commentary:**

*   This command leverages eBPF's uprobe functionality to place a breakpoint on the start of the `process_data` function inside the shared library `libdata.so`.
*   When the process `process_data` executes and calls `process_data`, the attached eBPF program prints the message "Function process_data called," which reveals the precise entry to the given function.
*   This approach requires knowledge of the target library and symbol name. However, we are not modifying source code of the program, or the shared library.
*   This method provides the most explicit identification of function execution and allows for precise measurements of call frequency, duration etc., which are more difficult to obtain using only syscall tracing.

**Example 3: Combining `strace` with shared library information**

Let's say `process_data` uses a networking library which contains the `send_data` function. First, we use `ldd` to see which libraries the program uses, then we use `strace` with network system calls:

```bash
ldd ./process_data
strace -e connect,sendto,recvfrom ./process_data
```

The output from `ldd` will provide a list of shared libraries used by the program, including those associated with networking, like `libc.so` or libraries providing network functionality.

**Commentary:**

*   The `ldd` command provides the shared libraries loaded by the process. Knowledge of the networking libraries will allow for selection of appropriate system calls to trace.
*   The `strace` command is filtered for `connect`, `sendto` and `recvfrom` which directly reveal network interactions of the program.
*   By observing system calls related to socket operations, alongside shared library loading information, one can often infer which library-level functions are being called to initiate such activity; often, calls to a library-level `send_data` function will be followed by corresponding network syscalls.

These examples illustrate a spectrum of methods to identify called functions without altering source code. Starting with `strace` provides a foundation to understand system-level interactions, while tools like eBPF offer deeper insights into specific function execution within loaded libraries and even inside the target process itself, if debug symbols are available. Combining these approaches allows for more informed and precise analysis of dynamic program behavior. A deep understanding of how system calls relate to high level functional execution is required, and debugging experience is often needed to bridge this conceptual gap. The analysis itself is often an iterative process, focusing first on system call interactions and then drilling down using function tracing once a potential functional path of interest is found.

For further learning on this topic, one should consult documentation on dynamic tracing methodologies and tools. Specific books or documentation pertaining to `strace`, `dtruss`, eBPF, and operating system concepts on process tracing provide invaluable information. Additionally, studying system call documentation of particular operating systems will help to understand their relationships with library and application-level functions. Textbooks on system programming that cover interprocess communication and operating system interaction will also prove valuable. It is imperative to learn about the specific instrumentation methods available for each target operating system as well as any debug tools that exist.
