---
title: "How can C++ programs leverage POSIX AIO on Linux?"
date: "2025-01-30"
id: "how-can-c-programs-leverage-posix-aio-on"
---
POSIX Asynchronous I/O (AIO) on Linux represents a powerful mechanism for performing non-blocking input/output operations, significantly enhancing application performance, especially when dealing with multiple concurrent I/O requests. Unlike traditional blocking I/O, where a thread pauses while waiting for an operation to complete, POSIX AIO allows a thread to initiate an I/O request and then continue executing other tasks. The operating system handles the actual data transfer in the background, notifying the application upon completion. This is particularly beneficial for I/O-bound applications, allowing for greater throughput and responsiveness. My experience building high-performance network servers and data processing pipelines has made me deeply familiar with its intricacies and benefits.

To effectively leverage POSIX AIO, one must understand the core functions and data structures involved. The primary interface revolves around the `aio_read`, `aio_write`, `lio_listio`, and related functions, all defined in `<aio.h>`. The fundamental data structure is the `aiocb` structure, which represents an asynchronous I/O control block. This structure holds all the pertinent information about a particular I/O request, including the file descriptor, buffer address, transfer size, and notification mechanism. It’s through the manipulation of these `aiocb` structures that you instruct the kernel to perform asynchronous operations.

Before initiating any AIO operations, the program must set up the necessary infrastructure. This generally involves several steps: Firstly, you need to create `aiocb` structures, filling in members such as `aio_fildes` (the file descriptor), `aio_offset` (the file offset), `aio_buf` (the data buffer), `aio_nbytes` (the transfer size), and `aio_lio_opcode` (the operation type - LIO_READ, LIO_WRITE, or LIO_NOP). Subsequently, you need to specify how the application wants to receive notification of operation completion. This can be done using signals (`aio_sigevent`) or by polling.  Using signals can be efficient as the kernel directly interrupts the process when an I/O is complete. Polling, achieved with functions like `aio_suspend`, can be suitable when tight control over the notification cycle is required.  Finally, after initialising the `aiocb`, use functions like `aio_read`, `aio_write` or `lio_listio` to initiate the async I/O.

After the asynchronous operation has been initiated, the application can perform other work. The crucial part is handling the notification of completion. Whether a signal was used or the application is actively polling, this is where it will know the request is complete. You can check the status of I/O requests using `aio_error`.  This function returns an error code corresponding to the operation identified by `aiocb`.  A return value of `EINPROGRESS` indicates the operation is still pending.  A return value of zero means the operation was successful. Once an operation is complete, you can retrieve the actual number of bytes transferred using `aio_return`. It's essential to handle the return value from `aio_return` appropriately as errors can still occur even after the `aio_error` check returns 0.

Now, let’s examine some specific C++ code examples:

**Example 1: Asynchronous File Read using Signals**

This example demonstrates reading a file asynchronously using signal-based notifications.

```c++
#include <iostream>
#include <fcntl.h>
#include <unistd.h>
#include <signal.h>
#include <errno.h>
#include <aio.h>
#include <cstring>
#include <cstdint>

#define BUFFER_SIZE 1024

struct aiocb aio_request;
char buffer[BUFFER_SIZE];

void signal_handler(int sig, siginfo_t *si, void *uc) {
    if (si->si_code != SI_ASYNCIO) {
        return; // Not an AIO signal
    }

    if (aio_error(&aio_request) == 0) {
        ssize_t bytes_read = aio_return(&aio_request);
        if (bytes_read > 0) {
            std::cout << "Read " << bytes_read << " bytes asynchronously" << std::endl;
            // Process the data in 'buffer'
        } else {
            std::cerr << "Error in aio_return after signal " << std::endl;
        }

    } else {
        std::cerr << "AIO Error: " << strerror(errno) << std::endl;
    }
    exit(0); //Exit after handling the signal.
}

int main() {
    int fd = open("example.txt", O_RDONLY);
    if (fd == -1) {
        perror("Error opening file");
        return 1;
    }
   std::memset(&aio_request, 0, sizeof(aio_request));
   aio_request.aio_fildes = fd;
    aio_request.aio_offset = 0;
    aio_request.aio_buf = buffer;
    aio_request.aio_nbytes = BUFFER_SIZE;
    aio_request.aio_sigevent.sigev_notify = SIGEV_SIGNAL;
    aio_request.aio_sigevent.sigev_signo = SIGUSR1;
    aio_request.aio_sigevent.sigev_value.sival_ptr = &aio_request;


    struct sigaction sa;
    sa.sa_sigaction = signal_handler;
    sa.sa_flags = SA_SIGINFO;
    sigemptyset(&sa.sa_mask);

    if (sigaction(SIGUSR1, &sa, nullptr) == -1) {
      perror("Error setting signal handler");
      close(fd);
      return 1;
    }


    if (aio_read(&aio_request) == -1) {
        perror("Error starting asynchronous read");
        close(fd);
        return 1;
    }

    std::cout << "Asynchronous read initiated." << std::endl;

    while(true){ //Wait for signal.
      pause();
    }

    close(fd);
    return 0;
}
```

In this example, `aio_read` is initiated. The main thread does not block, proceeding with other work, represented by the `std::cout` output. When the read operation completes, a `SIGUSR1` signal is generated, which invokes `signal_handler`. The signal handler checks for errors and reads data once the operation has completed. This exemplifies the core concept of asynchronous operation completion via signals.

**Example 2: Asynchronous File Write using Polling**

This example illustrates how to perform an asynchronous file write using polling.

```c++
#include <iostream>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <aio.h>
#include <cstring>

#define BUFFER_SIZE 1024
struct aiocb aio_request;
char buffer[BUFFER_SIZE] = "Hello, asynchronous world!";

int main() {
    int fd = open("output.txt", O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd == -1) {
        perror("Error opening file");
        return 1;
    }
    std::memset(&aio_request, 0, sizeof(aio_request));
    aio_request.aio_fildes = fd;
    aio_request.aio_offset = 0;
    aio_request.aio_buf = buffer;
    aio_request.aio_nbytes = std::strlen(buffer);

    if (aio_write(&aio_request) == -1) {
        perror("Error starting asynchronous write");
        close(fd);
        return 1;
    }

    std::cout << "Asynchronous write initiated." << std::endl;

    while (aio_error(&aio_request) == EINPROGRESS) {
      //Do other work here.
      usleep(100000);
    }

    if (aio_error(&aio_request) == 0) {
        ssize_t bytes_written = aio_return(&aio_request);
        if (bytes_written > 0) {
            std::cout << "Wrote " << bytes_written << " bytes asynchronously." << std::endl;
        } else {
          std::cerr << "Error in aio_return after polling" << std::endl;
        }
    }else {
      std::cerr << "Error during asynchronous write: " << strerror(errno) << std::endl;
    }


    close(fd);
    return 0;
}
```

Here, a simple text string is written to a file using `aio_write`. The `while` loop checks the status of the request using `aio_error`. The loop performs other work, represented by the `usleep` function to simulate activity. Once completed, the program uses `aio_return` to retrieve the number of written bytes and checks for error. This shows an alternative method of tracking completion with polling.

**Example 3: List I/O using lio_listio**

This example uses the `lio_listio` function, which allows multiple I/O operations to be initiated in a single call.

```c++
#include <iostream>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <aio.h>
#include <cstring>
#include <vector>
#include <algorithm>

#define NUM_REQUESTS 2
#define BUFFER_SIZE 1024
int main() {
    int fd1 = open("input1.txt", O_RDONLY);
    int fd2 = open("input2.txt", O_RDONLY);

    if (fd1 == -1 || fd2 == -1) {
      perror("Error opening input files");
      return 1;
    }

    std::vector<aiocb> aiocb_list(NUM_REQUESTS);
    std::vector<char> buffer1(BUFFER_SIZE);
    std::vector<char> buffer2(BUFFER_SIZE);

    std::memset(&aiocb_list[0], 0, sizeof(aiocb));
    aiocb_list[0].aio_fildes = fd1;
    aiocb_list[0].aio_offset = 0;
    aiocb_list[0].aio_buf = buffer1.data();
    aiocb_list[0].aio_nbytes = BUFFER_SIZE;
    aiocb_list[0].aio_lio_opcode = LIO_READ;
    std::memset(&aiocb_list[1], 0, sizeof(aiocb));
    aiocb_list[1].aio_fildes = fd2;
    aiocb_list[1].aio_offset = 0;
    aiocb_list[1].aio_buf = buffer2.data();
    aiocb_list[1].aio_nbytes = BUFFER_SIZE;
    aiocb_list[1].aio_lio_opcode = LIO_READ;

    aiocb* list_ptr = aiocb_list.data();

    if (lio_listio(LIO_WAIT, &list_ptr, NUM_REQUESTS, nullptr) == -1) {
        perror("Error starting list I/O");
        close(fd1);
        close(fd2);
        return 1;
    }
  
    for(auto const & aio : aiocb_list) {
       if (aio_error(const_cast<aiocb*>(&aio)) == 0) {
          ssize_t bytes_read = aio_return(const_cast<aiocb*>(&aio));
          if (bytes_read > 0) {
            std::cout << "Read " << bytes_read << " bytes asynchronously" << std::endl;
          }else{
             std::cerr << "Error in aio_return after listio" << std::endl;
          }

       } else{
          std::cerr << "AIO Error during list I/O: " << strerror(errno) << std::endl;
       }
    }

    close(fd1);
    close(fd2);
    return 0;
}
```

This code opens two files and attempts to read from them using `lio_listio`. The `LIO_WAIT` flag makes `lio_listio` block until all operations are complete. After completion, it iterates through the requests, checks for errors, and reads the result bytes. The significant advantage here is initiating multiple requests simultaneously, although this example does use blocking with `LIO_WAIT` to keep it simple. You can replace that with `LIO_NOWAIT` if necessary, similar to the previous examples.

For further exploration of POSIX AIO, I recommend consulting operating systems programming textbooks.  Also consider manuals dedicated to the topic, specifically those focusing on Linux system calls. Furthermore, studying the source code of widely used software leveraging AIO, particularly network servers like Nginx or Redis, can provide practical insight into real-world usage patterns and optimisation techniques.  The 'man' pages for `aio.h`, `lio_listio`, `aio_read`, `aio_write`, etc, are also excellent resources, providing comprehensive documentation for each function and data structure involved in AIO operations. Remember to verify your specific kernel version's implementation as there might be small differences in implementation. This will also give you a comprehensive overview of using POSIX AIO in Linux.
