---
title: "Why isn't the NVIDIA Nsight Systems CLI reporting memory statistics?"
date: "2025-01-30"
id: "why-isnt-the-nvidia-nsight-systems-cli-reporting"
---
NVIDIA Nsight Systems command-line interface (CLI) not reporting memory statistics, despite successful profiling otherwise, typically stems from inadequate configuration during the collection phase or an incompatibility between the system's memory subsystem and the tool's sampling mechanisms. I’ve encountered this numerous times, often after moving between different GPU architectures or driver versions. Specifically, memory sampling through Nsight Systems relies on hardware counters and mechanisms that might not be universally available or enabled by default.

The first, and perhaps most common cause, is the lack of specific memory-related events being selected for collection. Nsight Systems, by default, captures a basic set of performance metrics. Memory metrics, particularly those concerning GPU memory allocation and usage, are not included in this default set. Therefore, the user must explicitly specify the events relevant to memory profiling through the `--sample` option during the `nsys profile` invocation. For instance, using only `nsys profile --gpu-metrics=all <application>` will not reliably provide memory statistics because `all` for GPU metrics does not equate to enabling all available metrics, but rather a predefined subset. Similarly, if profiling on a system with very specific configurations, the appropriate hardware counters for memory need to be enabled. This directly influences the ability of Nsight Systems to retrieve this particular set of performance data.

Another potential reason lies in the operating system or driver configurations. On Linux systems, permissions issues with accessing necessary kernel resources for memory sampling can arise. Nsight Systems requires permissions to probe these resources, and if the user or process does not possess these permissions, data collection fails silently. This often involves using the `sudo` command or setting up appropriate file permissions for the user running the profiler. The driver itself might also have limitations or bugs that hinder the proper reporting of memory-related counters. Upgrading to a more recent or stable driver version may resolve such issues. Further, on systems utilizing Virtual Machines or containers, the memory available to the application being profiled and the memory visible to Nsight Systems might not be congruent. This difference can skew the reported statistics or prevent any sampling at all. The profiler, operating within the context of the host machine, might not see the memory as it's perceived by the containerized application, especially when memory limits are set on the container.

Lastly, the sampled application's memory usage patterns may not be conducive to the sampling frequency or methodology employed by Nsight Systems. For example, a highly transient memory allocation pattern, where memory is frequently allocated and deallocated within short timeframes, might be missed if the sampling rate is not high enough or if the collection duration is too short. While Nsight Systems provides aggregation for these metrics, the underlying sampling frequency dictates the detail, and hence the observability, of the data. The `--sampling-interval` option can impact how much of these very short duration events are caught. Choosing an interval that is too high might lead to important events being skipped. Additionally, some application architectures may use custom memory management routines that are not easily tracked by the default memory sampling probes in Nsight Systems. For very specific memory usage patterns, custom Nsight Systems plug-ins or manually inserted instrumentation might be necessary.

To illustrate, here are three code examples demonstrating common scenarios, their resolution, and their effect.

**Code Example 1: Missing Memory Events:**
This example shows a basic `nsys` command that omits specific memory events, thus failing to record any memory statistics.

```bash
# Incorrect: Omits memory events
nsys profile -o memory_no_events --gpu-metrics=all ./my_application
```
This command only specifies generic GPU metrics without focusing on memory-related events. After this command is executed, inspecting the resulting report with `nsys stats memory_no_events.qdrep` will show limited memory statistics, if any, because the necessary sampling events were not triggered. The output will largely report kernel times and other basic GPU activity. No specific memory allocation details are present.

The solution involves specifying relevant memory sampling events with the `--sample` option.

```bash
# Correct: Includes memory events
nsys profile -o memory_with_events --gpu-metrics=all --sample="gpu_memory,gpu_memory_access,gpu_memory_page_faults"  ./my_application
```
Here, `--sample="gpu_memory,gpu_memory_access,gpu_memory_page_faults"` explicitly tells Nsight Systems to capture detailed information about memory usage, including allocations and accesses. Executing the same command and examining the report with `nsys stats memory_with_events.qdrep` now reveals detailed memory statistics. We are now seeing allocations, and page fault statistics for GPU memory usage during the application's execution.

**Code Example 2: Permission Issues on Linux:**

This example showcases a typical permission-related issue when profiling on Linux.
```bash
# Incorrect: Running without sudo or proper permissions
nsys profile -o permission_fail --gpu-metrics=all --sample="gpu_memory,gpu_memory_access" ./my_application
```

In this scenario, if the current user doesn’t have proper permissions, Nsight Systems might fail to collect detailed memory statistics, despite requesting them via the `--sample` option. The profiler will often complete without generating error messages, but the resulting report will lack memory data. The standard output stream will usually not indicate that a permissions issue was the reason behind the failure to collect memory data.

To resolve this, use `sudo` to elevate privileges:
```bash
# Correct: Running with sudo to obtain system access
sudo nsys profile -o permission_success --gpu-metrics=all --sample="gpu_memory,gpu_memory_access" ./my_application
```
This ensures that Nsight Systems has the necessary access to the kernel resources required for memory sampling. After profiling is complete and the report is generated, `nsys stats permission_success.qdrep` will now reveal detailed memory information as expected. Often, setting the necessary file permissions on relevant kernel control groups can also enable execution without the use of `sudo`, although this requires careful consideration for overall system security.

**Code Example 3: Sampling Frequency Adjustments:**

This example demonstrates the effect of inadequate sampling frequencies when dealing with highly dynamic memory usage.

```bash
# Incorrect: Default sampling frequency misses transient allocations
nsys profile -o default_frequency --gpu-metrics=all --sample="gpu_memory,gpu_memory_access"  ./my_application
```

This default setting might not capture short-lived memory allocations or deallocations, leaving gaps in the observed memory usage. This results in an inaccurate view of the application's true memory behavior.

Adjusting the sampling interval can capture this behavior.
```bash
# Correct: Lower sampling interval for transient allocations
nsys profile -o low_frequency --gpu-metrics=all --sample="gpu_memory,gpu_memory_access" --sampling-interval=100 ./my_application
```
Here, `--sampling-interval=100` specifies a more frequent sampling rate (here, every 100 microseconds), which enables Nsight Systems to capture more rapidly occurring memory events. When the resulting report is reviewed using `nsys stats low_frequency.qdrep`, a far more detailed picture of the applications memory footprint is visible. This is particularly important for applications with frequent allocation and deallocation. However, be cautious about setting the sampling rate to very high values because the profiler overhead might negatively impact the performance of the application. Experimentation is often needed to find the correct balance.

For further learning, I would highly recommend consulting NVIDIA's official Nsight Systems documentation. Several guides outline specific configurations for different operating systems and GPU architectures, along with example use cases. The documentation also covers advanced aspects like custom plugins. The developer forums are also a solid resource, where developers share their solutions to a wide range of profiling-related problems. There are also numerous publicly available articles on GPU profiling and memory analysis with Nsight Systems, many of which delve into nuances that may not be obvious from reading the standard documentation. Understanding the underlying hardware counters and performance monitoring units (PMUs) on the GPU can also help to refine the profiling process.
