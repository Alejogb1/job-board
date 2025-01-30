---
title: "How can disk access be profiled?"
date: "2025-01-30"
id: "how-can-disk-access-be-profiled"
---
Disk access profiling is crucial for identifying performance bottlenecks in applications reliant on persistent storage. Through detailed analysis, I've been able to pinpoint inefficiencies stemming from both coding choices and inherent limitations of the underlying storage subsystem. This involves not just measuring raw I/O throughput, but also understanding patterns of access – sequential versus random, read versus write ratios, and the impact of different caching strategies.

The fundamental goal of disk access profiling is to uncover where I/O operations are slowing down an application. These slowdowns can manifest as high latency, low throughput, or excessive CPU utilization spent waiting on I/O. Profiling helps reveal whether these issues are due to poorly designed algorithms, suboptimal data access patterns, or the physical constraints of the storage hardware itself. The methodology hinges on intercepting or monitoring the communication channel between the application and the disk, often at the operating system level.

Several techniques facilitate this monitoring. Operating systems often provide tools for observing I/O activity per process. On Linux, `iotop` displays real-time I/O usage by process, offering an aggregated view of read and write rates. For more detailed, file-specific profiling, `strace` allows system call tracing, including `read`, `write`, and `open` calls, enabling analysis of exactly which files are accessed and how often. Windows offers similar capabilities through Performance Monitor, capturing various disk counters and I/O performance metrics. These OS-level tools are essential for getting a broad overview and identifying troublesome processes.

Within an application, custom instrumentation is frequently required for more fine-grained analysis. This involves embedding code to measure the duration of specific I/O operations or the latency of specific data requests. This application-level profiling reveals bottlenecks related to logical access patterns or specific components. The best approach combines both OS-level and application-level profiling techniques to gain a comprehensive understanding of the disk access behaviors.

Let's explore three code examples, focusing on areas where disk access can often be inefficient and how to profile them:

**Example 1: Simple File Reading (Python)**

This demonstrates a basic scenario, reading an entire file. I’ve used this example often to simulate simple read operations in my tests.

```python
import time
import os

def read_file_full(filename):
    start_time = time.perf_counter()
    with open(filename, 'rb') as f:
        data = f.read()
    end_time = time.perf_counter()
    return end_time - start_time

if __name__ == '__main__':
    filename = "large_file.bin"  # Assume this file exists
    if not os.path.exists(filename):
      with open(filename,"wb") as f:
        f.seek(1024*1024*100 -1 )
        f.write(b"\0") # creates a 100 MB file
    duration = read_file_full(filename)
    print(f"Time to read {filename}: {duration:.4f} seconds")
```

**Commentary:**

The python code measures the entire file read duration using `perf_counter()`. This simple method offers a basic way to track the total time a read operation takes. It is useful, but it does not pinpoint any specific issues other than the overall duration. If this operation takes an unexpectedly long time, OS-level profiling using `iotop` or similar tools can help to confirm if the delay is specific to this operation or if there are other contending operations impacting I/O. The approach is basic but effectively demonstrates the timing of a single read operation.

**Example 2: Chunked File Reading (Python)**

This shows a situation where a file is processed in smaller parts. I’ve repeatedly encountered cases where the naive approach of reading a whole file at once results in performance problems, especially with larger files.

```python
import time
import os

def read_file_chunked(filename, chunk_size=4096):
    start_time = time.perf_counter()
    total_bytes = 0
    with open(filename, 'rb') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            total_bytes += len(chunk)
    end_time = time.perf_counter()
    return end_time - start_time, total_bytes

if __name__ == '__main__':
    filename = "large_file.bin"
    if not os.path.exists(filename):
      with open(filename,"wb") as f:
        f.seek(1024*1024*100 -1 )
        f.write(b"\0") # creates a 100 MB file
    duration, size = read_file_chunked(filename)
    print(f"Time to read {size} bytes with chunking: {duration:.4f} seconds")
```

**Commentary:**

Here, the file is read in chunks, which is more memory-efficient for large files. The `chunk_size` can be adjusted to analyze its influence on performance. The code measures the total time taken for the entire chunked read operation. In addition to the timing, `strace` can be used to check the size and number of `read` calls made by this code during execution which can be correlated with the chunking. If the chunk size is extremely small, the performance might degrade from overhead relating to the high number of system calls, highlighting a trade-off between memory usage and performance. Similarly, extremely large chunks can defeat the purpose of memory efficiency.

**Example 3: Logging I/O Operations (Java)**

This Java example focuses on logging information about disk access when dealing with persistent data storage. This is a common scenario in server-side applications. I use this kind of logging approach extensively for database interactions.

```java
import java.io.*;
import java.nio.file.*;

public class LoggedFileAccess {

    public static void writeData(String filename, String content) {
        long startTime = System.nanoTime();
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(filename))) {
            writer.write(content);
        } catch (IOException e) {
            e.printStackTrace();
        }
        long endTime = System.nanoTime();
        System.out.println("Write to " + filename + ": " + (endTime - startTime) / 1_000_000.0 + " ms");
    }


  public static void readData(String filename) {
        long startTime = System.nanoTime();
        try (BufferedReader reader = new BufferedReader(new FileReader(filename))) {
            String line;
            while ((line = reader.readLine()) != null) {
                // Process each line
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        long endTime = System.nanoTime();
        System.out.println("Read from " + filename + ": " + (endTime - startTime) / 1_000_000.0 + " ms");
    }


    public static void main(String[] args) throws IOException{
      String filename = "test.txt";
      if(!Files.exists(Paths.get(filename))){
        Files.createFile(Paths.get(filename));
      }
      writeData(filename,"This is some text\n");
      readData(filename);

    }
}

```

**Commentary:**

This Java code measures the time for file read and write operations in milliseconds using `System.nanoTime()`. It logs the duration of both types of operations, making it easier to track individual operations. In scenarios where the application interacts with files frequently, this detailed timing can help to identify which read or write calls are slow. I’ve used similar logging for database interactions within Java, as well as when working with file storage for processing, to pinpoint slow individual read or write calls. It does not only gives overall timing, but also helps to find performance variance between different read/write calls to identify bottlenecks.

When choosing tools for disk access profiling, some standard operating system utilities and programming language libraries are typically useful. For Linux environments, beyond `iotop` and `strace`, `blktrace` allows for extremely detailed tracing of block device operations, which can be necessary when dealing with custom storage solutions. `perf` can be used to sample system-wide performance events, providing an aggregated view of I/O along with CPU and memory usage.  For Windows, as previously mentioned, Performance Monitor, along with Resource Monitor, provides comprehensive information regarding disk access, process I/O, and overall system load. Within applications, most languages offer timing libraries, like those shown in the examples, or more specialized profiling packages that can integrate with application-specific contexts.

While specific libraries and tools depend heavily on the operating system and programming language, the core principles behind disk access profiling remain consistent. Whether you're working with Python, Java, C++, or any other language, profiling disk access requires a combination of system-level monitoring and application-level instrumentation. This combined approach provides a comprehensive understanding of how data is being accessed and where potential bottlenecks reside. The objective is not merely to quantify the overall I/O performance but to understand the intricate interactions between the application, operating system, and the disk subsystem. These insights then allow for informed decisions to enhance performance, either through algorithm adjustments or more strategic use of storage resources.
