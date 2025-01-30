---
title: "Why is `os.path.getsize()` slow on a network drive in Python (Windows)?"
date: "2025-01-30"
id: "why-is-ospathgetsize-slow-on-a-network-drive"
---
The inherent latency associated with network file system operations contributes significantly to the performance bottleneck experienced when using `os.path.getsize()` on a Windows network drive. Specifically, this is compounded by the way Windows handles requests for file attributes over the network, typically through protocols like SMB/CIFS. Unlike local storage where file size retrieval is a rapid metadata lookup, querying a network file's size necessitates a round trip over the network to the server hosting the file share.

Fundamentally, the function `os.path.getsize()`, even though seemingly trivial, involves a chain of operating system-level calls to retrieve the file's size. On Windows, using Python's `os` module essentially translates to a call to the Windows API function, `GetFileAttributesExW`, or an equivalent API. This low-level function retrieves various attributes of a file, including its size. When invoked for a locally stored file, these attributes are directly accessible, usually within the file system's metadata structures, and the process is swift. However, when the targeted file is located on a network share, this operation undergoes a more complex process.

The Windows operating system recognizes the network location and, instead of directly accessing the storage, sends an SMB (Server Message Block) request to the remote server hosting the file share. This request essentially asks the remote server, “What are the attributes of this specific file?” The server then needs to access its local file system, retrieve the requested file attributes, and transmit them back across the network. This network transmission, irrespective of network speed, introduces significant latency. Even with gigabit Ethernet connections, the latency inherent in network communication, plus the overhead of protocol handling, server resource allocation, and file access on the server’s storage medium all accumulate. This entire exchange adds noticeable delays, especially when attempting to retrieve sizes for multiple files or for large numbers of small files. Every file size call initiates a separate network round trip, further escalating the overall processing time. In essence, the `os.path.getsize()` function is not slow in itself, but rather the underlying network operation causes the delay. The function is merely revealing the consequence of this latency.

Moreover, the perceived latency can be impacted by several factors not directly related to the code. The network’s condition, server load, quality of cabling and routing, and the specific network protocols involved can all contribute to variations in the speed of `os.path.getsize()` on network drives. Furthermore, antivirus software and other security services may add processing time as they intercept and analyze these file attribute requests.

Over my years managing file processing systems, I encountered a specific project where we were parsing through numerous log files stored on a network share. Initial testing using `os.path.getsize()` to determine if files were empty resulted in intolerably slow performance. We refactored the code after identifying the network bottleneck. The first example demonstrates the initial inefficient code:

```python
import os
import time

def check_file_size_slow(filepaths):
    empty_files = []
    start_time = time.time()
    for file in filepaths:
        if os.path.getsize(file) == 0:
            empty_files.append(file)
    end_time = time.time()
    print(f"Time taken using os.path.getsize: {end_time - start_time:.4f} seconds")
    return empty_files

# Assume filepaths is a list of network file paths
file_list = [r"\\server\share\file1.txt", r"\\server\share\file2.txt", r"\\server\share\file3.txt"] #Sample paths
check_file_size_slow(file_list)
```

This example highlights the problem. For each file path in the list, a separate call to `os.path.getsize()` is made. This triggers a round trip over the network for every single file, resulting in cumulative delays. The larger the file list, the greater the overhead incurred due to network latency.

To overcome this limitation, we employed a bulk file attribute retrieval approach, leveraging the `os.scandir()` function. This function allows for retrieving file attributes including the file size, from a directory. It is significantly more efficient when retrieving information about numerous files in the same directory, because it can make a single query to the server for the directory's contents, reducing the total network traffic. The modified code is shown below:

```python
import os
import time

def check_file_size_fast(filepaths):
    empty_files = []
    start_time = time.time()

    directories_to_check = {}
    for file in filepaths:
       directory = os.path.dirname(file)
       if directory not in directories_to_check:
           directories_to_check[directory] = []
       directories_to_check[directory].append(os.path.basename(file))

    for directory, files_in_dir in directories_to_check.items():
      for entry in os.scandir(directory):
        if entry.name in files_in_dir and entry.is_file():
            if entry.stat().st_size == 0:
                empty_files.append(os.path.join(directory,entry.name))

    end_time = time.time()
    print(f"Time taken using os.scandir: {end_time - start_time:.4f} seconds")
    return empty_files

#Assume filepaths is a list of network file paths (same as before)
file_list = [r"\\server\share\file1.txt", r"\\server\share\file2.txt", r"\\server\share\file3.txt"] #Sample paths
check_file_size_fast(file_list)
```

In this modified code, I grouped files by their directory, then used `os.scandir()` to access all file information within each directory. After retrieving the directory entries, I filtered out file paths that were relevant to the original list. Inside of this loop, the `stat().st_size` attribute retrieves the file size efficiently without requiring additional round trips per file. This change demonstrated a marked improvement in processing speed, as the number of network requests was reduced significantly.

In scenarios where the file paths do not reside in a single directory, we can use a combination of the two approaches. Specifically, we can use `os.stat()` to retrieve file size if a path is located in a local directory or `os.scandir()` on a network path. Below is an example of this:

```python
import os
import time
def check_file_size_hybrid(filepaths):
   empty_files = []
   start_time = time.time()
   directories_to_check = {}
   for file in filepaths:
      directory = os.path.dirname(file)
      if directory.startswith(r"\\"):
         if directory not in directories_to_check:
             directories_to_check[directory] = []
         directories_to_check[directory].append(os.path.basename(file))
      else:
         if os.path.getsize(file) == 0:
            empty_files.append(file)


   for directory, files_in_dir in directories_to_check.items():
     for entry in os.scandir(directory):
        if entry.name in files_in_dir and entry.is_file():
           if entry.stat().st_size == 0:
             empty_files.append(os.path.join(directory, entry.name))

   end_time = time.time()
   print(f"Time taken using os.path.getsize and os.scandir: {end_time - start_time:.4f} seconds")
   return empty_files
#Assume filepaths is a list of network file paths and local file paths
file_list = [r"\\server\share\file1.txt", r"C:\test\local_file.txt", r"\\server\share\file3.txt"] #Sample paths
check_file_size_hybrid(file_list)
```
This implementation showcases a mixed strategy. We handle local file paths with the regular `os.path.getsize()` and utilize `os.scandir()` for network paths. This method allows to effectively retrieve file sizes across both local and network storage systems, optimizing for each environment.

For further learning, exploring the documentation for the Python `os` module is highly recommended. Additionally, understanding the principles of network file system protocols, such as SMB, can offer deeper insight into these performance limitations. Consulting operating system-specific documentation regarding file system access, particularly related to network drives, will aid in optimization. Also, researching techniques in concurrent processing, like multithreading or multiprocessing, can help improve overall throughput when working with large numbers of files, allowing parallel retrievals of file attributes.
