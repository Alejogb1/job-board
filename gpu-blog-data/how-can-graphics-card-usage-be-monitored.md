---
title: "How can graphics card usage be monitored?"
date: "2025-01-26"
id: "how-can-graphics-card-usage-be-monitored"
---

Efficient monitoring of graphics processing unit (GPU) utilization is crucial for optimizing application performance, diagnosing bottlenecks, and ensuring efficient resource management. Over my tenure working on high-performance computing clusters, I’ve encountered numerous scenarios where subtle shifts in GPU workload had profound impacts on application run times. Understanding the various metrics and tools available for GPU monitoring is, therefore, essential.

The core of GPU monitoring lies in accessing hardware performance counters and exposing them through operating system interfaces. These counters provide granular details about various aspects of GPU operation, including utilization rates, memory usage, power consumption, and temperature.  These metrics are dynamic; they vary substantially depending on the computational intensity of the running workload and the design of the applications themselves. No single metric reveals the whole story; rather, a comprehensive analysis relies on several related measurements.

One important metric is GPU utilization, usually expressed as a percentage. This indicates the fraction of time the GPU’s processing cores are actively engaged in computation.  However, high utilization does not automatically imply optimal performance. The GPU might be waiting on data transfers, or experiencing memory bottlenecks, which appear as high utilization but do not indicate productive computation.  Another key metric is memory usage, measured in bytes, which shows how much of the GPU’s onboard memory is being used. This is important because exceeding memory capacity often results in significant performance penalties due to data swapping between GPU memory and system memory. Monitoring temperature is also critical, especially in multi-GPU configurations, because thermal throttling can drastically reduce performance, particularly in prolonged high-load scenarios. Finally, power consumption reveals the energy draw of the GPU, which is essential for managing overall system power budgets and costs.

Software tools bridge the gap between hardware counters and human-interpretable data. The specific method used to access and present these metrics depends largely on the operating system and the GPU vendor. For instance, on Linux, NVIDIA GPUs expose their metrics through the NVIDIA Management Library (NVML) and can be queried using command-line tools like `nvidia-smi`. AMD GPUs on Linux utilize the ROCm (Radeon Open Compute) stack and can be monitored using `rocm-smi`.  Windows exposes metrics via the Windows Performance Counters (WPC) framework and vendor-specific tools.

I’ll now present three code examples demonstrating different approaches to GPU monitoring. The first illustrates the basic use of `nvidia-smi` command-line interface, the second uses the Python `pynvml` library for dynamic metric gathering, and the third uses bash to analyze system metrics.

**Example 1: `nvidia-smi` CLI Monitoring**

```bash
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits --loop=1
```

This command, executed directly in a terminal, provides a timestamped snapshot of GPU metrics every second. The `--query-gpu` argument specifies which metrics to retrieve. `utilization.gpu` shows the GPU core utilization, `memory.used` and `memory.total` display the amount of used and total memory, respectively, `temperature.gpu` provides the GPU temperature, and `power.draw` shows the current power consumption. The `--format` argument specifies that the output be in comma-separated values, without a header or units, to enable easier parsing. The `--loop=1` ensures the information is refreshed every second. While simple, this is not well-suited to programmatic access; its strength lies in rapid debugging and live visual inspections.

**Example 2: `pynvml` Dynamic Monitoring**

```python
import pynvml
import time

pynvml.nvmlInit()
try:
    deviceCount = pynvml.nvmlDeviceGetCount()
    for i in range(deviceCount):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        name = pynvml.nvmlDeviceGetName(handle)
        print(f"GPU {i}: {name.decode()}")
        while True:
           util = pynvml.nvmlDeviceGetUtilizationRates(handle)
           mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
           temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
           power_usage = pynvml.nvmlDeviceGetPowerUsage(handle)
           print(f"  GPU Util: {util.gpu}%, Mem Used: {mem_info.used/1024**2:.2f}MB, Temp: {temp}°C, Power: {power_usage/1000:.2f}W")
           time.sleep(1)
except pynvml.NVMLError as error:
    print(f"Error: {error}")
finally:
    pynvml.nvmlShutdown()

```

This Python script demonstrates how to use the `pynvml` library to programmatically monitor NVIDIA GPUs. The script initializes the NVML library, iterates through all available GPUs, and continuously displays GPU utilization, memory usage, temperature, and power consumption.  The `pynvml.nvmlDeviceGetHandleByIndex(i)` function retrieves a handle to each GPU. Then `pynvml.nvmlDeviceGetUtilizationRates`, `pynvml.nvmlDeviceGetMemoryInfo`, `pynvml.nvmlDeviceGetTemperature`, and `pynvml.nvmlDeviceGetPowerUsage` retrieve the required metrics. The `try...except...finally` block ensures proper error handling and library shutdown. The output is formatted for readability, showing GPU utilization as a percentage, used memory in megabytes, temperature in degrees Celsius, and power in watts, refreshed once a second. This technique is valuable when developing more complex applications or dashboards for real-time monitoring.

**Example 3: Bash and system statistics**

```bash
while true; do
  echo "--- $(date) ---"
  sensors | grep 'GPU'
  free -m | grep 'Mem'
  echo "Processes with high GPU Usage (top 5)"
  nvidia-smi --query-compute-apps=pid,process_name,gpu_memory_usage --format=csv,noheader,nounits | sort -k3 -n -r | head -n 5
  sleep 5
done
```

This bash script provides a wider picture of GPU monitoring by combining several system tools. It utilizes `sensors` to monitor hardware temperatures (including, but not limited to, the GPU). The command `free -m` displays system RAM usage and focuses the grep filter to only include the total and used RAM numbers. `nvidia-smi` is then used to identify the top 5 processes using the most GPU memory and their respective process IDs. The results are combined with date information for a timestamped output. The `sleep 5` command results in these metrics being displayed every five seconds. This script, while not as granular as the previous `pynvml` example, offers a broader view of system resource usage alongside GPU-specific information. This is often useful when pinpointing overall bottlenecks and not just GPU issues.

In conclusion, while each example presented offers a slightly different focus, the underlying principle remains the same: accessing hardware performance counters to understand the dynamic state of GPU operation. The specific approach used—command-line tools, scripting languages, or system utilities—depends upon the context of the monitoring needs. For detailed per-application analysis, a programming library like `pynvml` would be preferred, for quick checks or simple logging, `nvidia-smi` suffices, and for general system awareness, the combination of `sensors`, `free` and `nvidia-smi` in bash is helpful.

For resources beyond what I’ve presented here, I recommend exploring the official documentation for the NVIDIA Management Library (NVML), AMD’s ROCm documentation, and the system-specific performance analysis tools available in both Linux and Windows.  Additionally, textbooks on high-performance computing and GPU architecture often have detailed descriptions of the various metrics available and their interpretations. Consulting vendor documentation for specific cards will reveal detailed parameter definitions and their relative importance for various application types. Deep understanding of such resources offers a more nuanced view of GPU utilization than any single metric or tool can provide.
