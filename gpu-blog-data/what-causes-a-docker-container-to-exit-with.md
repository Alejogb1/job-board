---
title: "What causes a Docker container to exit with code 139?"
date: "2025-01-30"
id: "what-causes-a-docker-container-to-exit-with"
---
Exit code 139 in a Docker container almost invariably indicates a segmentation fault.  My experience debugging production systems over the past decade has consistently shown this to be the case, despite the lack of explicit error messaging within the container itself.  Understanding this requires a nuanced grasp of process management within Linux, Docker's underlying architecture, and common coding pitfalls.

**1.  Explanation:**

A segmentation fault (SIGSEGV) occurs when a program attempts to access memory that it's not permitted to access. This can manifest in numerous ways, ranging from dereferencing a null pointer to attempting to write to read-only memory or accessing memory beyond allocated bounds.  When this happens within a Docker container, the container's process receives the SIGSEGV signal. The default behavior of the signal handler in most Linux distributions is to terminate the process, resulting in the exit code 139.  It's crucial to understand that 139 is not a Docker-specific code; it's a standard Linux exit code representing a signal termination, specifically SIGSEGV (11) shifted left by 8 bits (11 << 8 = 28672), plus 128 (128 + 28672 = 28800 which is 0x7000), and then typically represented as 139 in decimal.  This calculation reflects how Linux represents signals in exit codes.

Debugging this requires a systematic approach.  First, you must identify the process within the container that is causing the crash. This can be challenging if the application is complex, but tools like `strace` and `gdb` become invaluable, as demonstrated below. Second, meticulous code review focusing on memory allocation, pointer manipulation, and array bounds checking is necessary.  Third, careful consideration must be given to the container’s environment, including library versions and system calls. Inconsistent library versions between the host and container environment, as well as faulty system calls due to unexpected operating system behavior or insufficient permissions, can also trigger segmentation faults.

**2. Code Examples and Commentary:**

**Example 1: C++ Null Pointer Dereference**

```c++
#include <iostream>

int main() {
  int* ptr = nullptr;
  *ptr = 10; // Dereferencing a null pointer
  return 0;
}
```

This simple C++ example demonstrates a classic cause of segmentation faults.  The `ptr` variable is initialized to `nullptr`, which means it doesn't point to any valid memory location. Attempting to dereference it using `*ptr` leads to a segmentation fault.  When this code is run within a Docker container, the container will exit with code 139.  To prevent this, thorough null checks are vital:


```c++
#include <iostream>

int main() {
  int* ptr = nullptr;
  if (ptr != nullptr) {
    *ptr = 10;
  } else {
    std::cerr << "Error: Null pointer encountered." << std::endl;
    return 1; // Return a non-zero code to indicate an error
  }
  return 0;
}
```

**Example 2:  Python Array Index Out of Bounds**

```python
my_list = [1, 2, 3]
print(my_list[3]) # Accessing an index beyond the list's bounds
```

Python's dynamic typing often masks these issues during development, but accessing an index beyond the allocated size of a list or array will result in a segmentation fault in the underlying C implementation of the Python interpreter.  This, too, will manifest as a 139 exit code in Docker.  Robust bounds checking is essential:

```python
my_list = [1, 2, 3]
try:
    print(my_list[3])
except IndexError:
    print("Error: Index out of bounds.")
    exit(1) #Explicitly handle error
```

**Example 3:  Go Buffer Overflow**

```go
package main

import "fmt"
import "unsafe"

func main() {
	buf := make([]byte, 4)
	copy(buf, []byte("This is more than 4 bytes")) //Buffer Overflow Attempt
	fmt.Println(string(buf))
}

```

Go's memory management, while generally safe, is susceptible to buffer overflows if not handled carefully. This example attempts to copy more data into `buf` than it can hold, leading to a memory overwrite and potentially a segmentation fault. The `unsafe` package provides low-level access, making such vulnerabilities easier to introduce.  Similar to the previous examples, this will likely trigger a 139 exit code.  Careful consideration of buffer sizes and the use of safer string manipulation functions should be implemented to address this.

```go
package main

import "fmt"

func main() {
	buf := make([]byte, 100) //Increase buffer size to accommodate the data.
	copy(buf, []byte("This is more than 4 bytes"))
	fmt.Println(string(buf))
}
```

**3. Resource Recommendations:**

For advanced debugging, I recommend consulting the documentation for `gdb` (the GNU debugger), `strace` (system call tracer), and `valgrind` (memory debugger). Mastering these tools is crucial for efficient and comprehensive debugging of segmentation faults. For efficient memory management and handling of data structures in your application code, the official documentation for your chosen programming language will provide guidelines for best practices and safe coding techniques. Finally, reviewing the Docker documentation on container troubleshooting and examining its logging capabilities will aid in tracking issues stemming from containerized environments.  These resources, combined with diligent code review and testing practices, offer the most effective strategy for preventing and resolving segmentation faults leading to exit code 139.
