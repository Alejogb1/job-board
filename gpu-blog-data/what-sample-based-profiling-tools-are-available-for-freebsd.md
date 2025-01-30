---
title: "What sample-based profiling tools are available for FreeBSD?"
date: "2025-01-30"
id: "what-sample-based-profiling-tools-are-available-for-freebsd"
---
On FreeBSD, analyzing performance bottlenecks typically necessitates a combination of kernel-level and user-space profiling tools, with `dtrace` standing out as the premier solution for sample-based analysis due to its flexibility and minimal overhead. My experience over several years optimizing high-performance network applications and databases on FreeBSD environments has heavily relied on a methodical approach, and `dtrace` is a cornerstone of that method, but we also leverage other tools at different points.

Sample-based profiling, unlike instrumentation-based approaches, doesn't require code modification. Instead, it periodically interrupts the execution of a process or the entire system, records the call stack at each interrupt, and then aggregates this data to show where time is spent. This makes it non-intrusive and ideal for production environments, where modifications can be costly.

`dtrace`’s power lies in its dynamic tracing capabilities. It allows you to insert probes into arbitrary locations, kernel and user space included, without requiring recompilation of the traced software or the kernel. These probes trigger actions when hit, such as recording a timestamp, register values, or the function call stack. The aggregated data provides a view into performance hot spots. `dtrace` utilizes a scripting language called D, which is somewhat similar to C, which provides the capability to perform complex analysis.

Beyond `dtrace`, FreeBSD offers other tools useful for performance investigations, though they are not strictly sample-based in the same way. `gprof`, a traditional profiler, utilizes compilation instrumentation. It’s useful during the development phase, though intrusive, while tools like `ktrace` and `truss` record system call execution and arguments; these are useful for understanding system interactions but don’t provide a direct measure of CPU time spent. The `perf` tool, available through the `sysutils/perf` port, offers similar sampling capabilities as `dtrace` but is not as tightly integrated or as versatile in FreeBSD. While `perf` has made improvements, my experience is that `dtrace` remains the more prevalent and dependable option.

Here are three illustrative examples of how `dtrace` can be employed for performance analysis.

**Example 1: Identifying Hot Functions in a User Process**

Suppose an application exhibits unexpected latency. I'd use `dtrace` to pinpoint the functions consuming most of the CPU time. The following script, executed as root, would be a starting point.

```d
#!/usr/sbin/dtrace -s

#pragma D option quiet

profile-99 /pid == $target/
{
	@func[ustack()] = count();
}

END
{
	printa(@func);
}
```

*Commentary:*

*   `#!/usr/sbin/dtrace -s`: This line designates the interpreter as `dtrace` and indicates it should use the D scripting language.
*   `#pragma D option quiet`: This directive suppresses the verbose output of `dtrace` and only displays the results.
*   `profile-99 /pid == $target/`: This is the core of the probe. The `profile-99` provider samples the target application, identified using the `-p` flag on the command line to specify the process ID which will become the target, at approximately 100 times per second; if the predicate, `pid == $target`, matches, the action associated with this probe is executed. `profile-99` provides stack samples.
*   `@func[ustack()] = count();`: For each sample, this line records the user-space call stack obtained via `ustack()`. These stack traces are keys of the associative array, `@func`, where the number of samples is counted using `count()`. This effectively accumulates how many times execution happened in each part of a call chain.
*   `END { printa(@func); }`: Once `dtrace` finishes, this block prints the associative array `@func` to the standard output, showing the aggregated stack samples and their counts.

To execute this, assuming the process ID of interest is 12345:

```bash
sudo ./profile_hot_functions.d -p 12345
```
The output would list call stacks and the number of times they were encountered. This data helps identify code sections where execution spends most of its time. If a significant proportion of samples land in a particular function or call stack, it indicates a potential bottleneck.

**Example 2: Analyzing System Call Latency**

System calls are the primary interface between an application and the kernel. If an application seems to be spending a lot of time in kernel space, analyzing the time spent in various system calls is important.

```d
#!/usr/sbin/dtrace -s

syscall:::entry
{
    self->ts = timestamp;
}

syscall:::return
/self->ts/
{
    @syscall_latency[probefunc] = quantize(timestamp - self->ts);
    self->ts = 0;
}

END
{
    printa(@syscall_latency);
}
```

*Commentary:*
*   `syscall:::entry`: This probe fires on the entry to any system call. It captures the timestamp and saves it in a thread-local variable named `ts` on entry. `timestamp` provides nanosecond precision.
*   `syscall:::return`: This probe triggers when a system call returns. The predicate `/self->ts/` ensures it’s triggered only if there is a saved timestamp in `ts`.
*   `@syscall_latency[probefunc] = quantize(timestamp - self->ts);`: This line calculates the latency by subtracting the entry timestamp from the current timestamp, and then accumulates a histogram into the `@syscall_latency` array, using the system call name, `probefunc`, as the key.
*   `self->ts = 0;`: This resets the thread-local timestamp, avoiding capturing latency from a different system call.
*   `END { printa(@syscall_latency); }`: This block, executed after `dtrace` has finished gathering data, displays the collected histogram.

Executing this script:

```bash
sudo ./syscall_latency.d
```

The output would present a breakdown of system call latency, illustrating which system calls contribute most to the execution time of the applications on the system. In practice, I often use the `-n` option to filter for specific system calls of interest.

**Example 3: Kernel Function Profiling**

Sometimes, the problem lies not in the user-space application, but in the kernel. This is especially relevant when dealing with network or storage I/O.

```d
#!/usr/sbin/dtrace -s

#pragma D option quiet

profile-100 /execname == "mykernelmodule"/
{
    @kernel_function[stack()] = count();
}

END
{
   printa(@kernel_function);
}
```

*Commentary:*

*   `profile-100 /execname == "mykernelmodule"/`: This probe samples execution of the kernel module "mykernelmodule" at about 100 times a second and only when running. I often find this better than doing blanket sampling and filtering results because the performance impact is minimized.
*   `@kernel_function[stack()] = count();`: The `stack()` function obtains the kernel call stack and, for each sample, counts the number of times this call stack was present. This is analogous to the user space example, except now examining the kernel.
*  `END { printa(@kernel_function); }`: After data collection, this block prints the collected data.

To execute this, ensure `mykernelmodule` is a loaded kernel module or a particular driver:

```bash
sudo ./kernel_hot_functions.d
```

The output will list the kernel call stacks which are accumulating samples, helping identify bottlenecks within the kernel. Identifying bottlenecks in the kernel can also indicate an opportunity to improve existing driver performance or adjust kernel configuration.

In summary, `dtrace` provides a dynamic and versatile approach to sample-based profiling on FreeBSD, with its scriptability enabling detailed investigations into both user and kernel space performance issues. However, it's not the only tool. `gprof` can be used during development, `perf` can also be used, and system call tracers such as `ktrace` and `truss` can also play a role in profiling different aspects of applications.

For further learning, the following resources are valuable:
*   The FreeBSD handbook is an excellent starting point for understanding system-level concepts.
*   The `dtrace` documentation is critical, although it's not always comprehensive. Exploring online forums and blogs may be necessary for more advanced use cases.
*   The source code of the FreeBSD kernel itself can also be instructive. Specifically, observing how other drivers and kernel modules are implemented will provide insight into potential problem areas.

Mastering these techniques has been essential in my own efforts to build and maintain high-performance systems.
