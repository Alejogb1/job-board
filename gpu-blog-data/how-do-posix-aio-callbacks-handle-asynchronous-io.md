---
title: "How do POSIX AIO callbacks handle asynchronous I/O operations?"
date: "2025-01-30"
id: "how-do-posix-aio-callbacks-handle-asynchronous-io"
---
POSIX Asynchronous I/O (AIO) callbacks, specifically when used with functions like `aio_read` or `aio_write`, provide a mechanism for a program to initiate an I/O operation and continue processing, without blocking while waiting for completion. This non-blocking behavior is achieved by the kernel managing the I/O in the background and notifying the application via a callback function when the operation finishes. The crucial point is that the application does not actively poll or wait; instead, it’s notified asynchronously.

My experience working on a high-throughput network monitoring tool highlighted the necessity for this approach. Traditional blocking I/O would have significantly reduced the application's responsiveness, especially when dealing with multiple data streams simultaneously. Switching to AIO with callbacks dramatically improved performance by allowing the application's main thread to focus on processing data while the kernel handled the I/O.

The mechanism involves a few core components. First, an `aiocb` (AIO control block) structure is populated with information about the I/O operation, including the file descriptor, buffer, length, offset, and most importantly, a pointer to the callback function. This structure is then passed to functions like `aio_read` or `aio_write`, initiating the asynchronous operation. The system call returns immediately, even though the I/O is still in progress. The kernel then monitors the status of the requested I/O operation. Upon completion (success, failure, or cancellation), the kernel executes the callback function provided in the `aiocb` structure. It's important to note that this callback is executed within a kernel context, so any code executing within the callback function should be carefully crafted to be lightweight and avoid blocking calls.

The callback function itself receives an `siginfo_t` structure which contains information about the completed operation. Within this structure, the `si_value` member holds the `aio_sigevent.sigev_value` field from the original `aiocb`, offering a way to pass contextual data to the callback. This allows the callback to know which I/O operation has completed and act accordingly. This mechanism allows the application to maintain a low CPU usage while waiting for I/O events and enabling the main program flow to progress independently.

Now, let's consider some specific code examples that showcase this interaction.

**Example 1: Simple Asynchronous Read with Callback**

```c
#include <aio.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <signal.h>

#define BUFFER_SIZE 1024

void read_callback(sigval_t sigval) {
  struct aiocb *cb = (struct aiocb *)sigval.sival_ptr;
  ssize_t bytes_read = aio_return(cb);

  if (bytes_read > 0) {
    printf("Asynchronous read completed: %zd bytes\n", bytes_read);
    printf("Data: %.*s\n", (int)bytes_read, (char *)cb->aio_buf);
  } else {
     printf("Asynchronous read failed.\n");
  }

  free(cb->aio_buf);
  free(cb);
}

int main() {
  int fd = open("test.txt", O_RDONLY);
  if (fd == -1) {
    perror("open");
    return EXIT_FAILURE;
  }

  struct aiocb *cb = malloc(sizeof(struct aiocb));
  if (!cb) {
      perror("malloc");
      close(fd);
      return EXIT_FAILURE;
  }

  char *buffer = malloc(BUFFER_SIZE);
  if (!buffer) {
      perror("malloc");
      free(cb);
      close(fd);
      return EXIT_FAILURE;
  }
  memset(cb, 0, sizeof(struct aiocb));

  cb->aio_fildes = fd;
  cb->aio_buf = buffer;
  cb->aio_nbytes = BUFFER_SIZE;
  cb->aio_offset = 0;
  cb->aio_sigevent.sigev_notify = SIGEV_THREAD;
  cb->aio_sigevent.sigev_notify_function = read_callback;
  cb->aio_sigevent.sigev_value.sival_ptr = cb;

  if (aio_read(cb) == -1) {
     perror("aio_read");
     free(buffer);
     free(cb);
     close(fd);
     return EXIT_FAILURE;
  }


  printf("Asynchronous read initiated...\n");
  sleep(1);
  printf("Main thread continues processing...\n");
  while(aio_error(cb) == EINPROGRESS);

  close(fd);
  return EXIT_SUCCESS;
}

```
This example initiates a read from a file "test.txt" using `aio_read`. The `read_callback` function is set to be executed when the read operation completes. Importantly, after calling `aio_read`, the main thread is able to continue its execution, demonstrated by the `printf` call and the simulation of other tasks with the `sleep` statement. The `aio_error` call at the end ensures that the program waits until the AIO operation has completed.

**Example 2: Multiple Asynchronous Reads with a shared callback**

```c
#include <aio.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <signal.h>

#define BUFFER_SIZE 1024
#define NUM_READS 2

struct aio_context {
  struct aiocb *cb;
  int id;
};


void read_callback(sigval_t sigval) {
  struct aio_context *context = (struct aio_context *)sigval.sival_ptr;
  struct aiocb *cb = context->cb;
  ssize_t bytes_read = aio_return(cb);

  if (bytes_read > 0) {
    printf("Asynchronous read %d completed: %zd bytes\n", context->id, bytes_read);
    printf("Data: %.*s\n", (int)bytes_read, (char *)cb->aio_buf);
  } else {
    printf("Asynchronous read %d failed.\n", context->id);
  }

  free(cb->aio_buf);
  free(context->cb);
  free(context);
}

int main() {
  int fd = open("test.txt", O_RDONLY);
    if (fd == -1) {
        perror("open");
        return EXIT_FAILURE;
    }

  struct aio_context* contexts[NUM_READS];

  for (int i = 0; i < NUM_READS; i++) {
    contexts[i] = malloc(sizeof(struct aio_context));
    if (!contexts[i]) {
        perror("malloc");
        close(fd);
        for (int j = 0; j < i; j++){
            free(contexts[j]->cb->aio_buf);
            free(contexts[j]->cb);
            free(contexts[j]);
        }
      return EXIT_FAILURE;
    }
    contexts[i]->cb = malloc(sizeof(struct aiocb));

    if (!contexts[i]->cb) {
        perror("malloc");
        close(fd);
        for (int j = 0; j < i; j++){
            free(contexts[j]->cb->aio_buf);
            free(contexts[j]->cb);
            free(contexts[j]);
        }
        free(contexts[i]);
      return EXIT_FAILURE;
    }

     char *buffer = malloc(BUFFER_SIZE);
        if(!buffer) {
             perror("malloc");
             close(fd);
            for (int j = 0; j < i; j++){
                free(contexts[j]->cb->aio_buf);
                free(contexts[j]->cb);
                free(contexts[j]);
            }
            free(contexts[i]->cb);
            free(contexts[i]);
           return EXIT_FAILURE;
        }
    memset(contexts[i]->cb, 0, sizeof(struct aiocb));

    contexts[i]->cb->aio_fildes = fd;
    contexts[i]->cb->aio_buf = buffer;
    contexts[i]->cb->aio_nbytes = BUFFER_SIZE;
    contexts[i]->cb->aio_offset = i * BUFFER_SIZE;
    contexts[i]->cb->aio_sigevent.sigev_notify = SIGEV_THREAD;
    contexts[i]->cb->aio_sigevent.sigev_notify_function = read_callback;
    contexts[i]->cb->aio_sigevent.sigev_value.sival_ptr = contexts[i];
    contexts[i]->id = i;


    if (aio_read(contexts[i]->cb) == -1) {
        perror("aio_read");
        free(buffer);
        free(contexts[i]->cb);
        free(contexts[i]);
        close(fd);
        for (int j = 0; j < i; j++){
            free(contexts[j]->cb->aio_buf);
            free(contexts[j]->cb);
            free(contexts[j]);
        }
        return EXIT_FAILURE;
    }
    printf("Asynchronous read %d initiated\n",i);
  }

  printf("Main thread continues processing...\n");
  sleep(1);

    for(int i = 0; i < NUM_READS; i++)
        while(aio_error(contexts[i]->cb) == EINPROGRESS);

   close(fd);
  return EXIT_SUCCESS;
}

```

This example showcases how to manage multiple concurrent reads. Rather than using a global variable for the `aiocb`, we associate each AIO operation with an `aio_context` struct, enabling us to pass specific parameters to the callback function. The callback is the same, but it now accesses the `aio_context` to print the specific read id. This demonstrates a common pattern for managing callbacks in complex scenarios.

**Example 3: Asynchronous Write with Callback**

```c
#include <aio.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <signal.h>

#define BUFFER_SIZE 256

void write_callback(sigval_t sigval) {
  struct aiocb *cb = (struct aiocb *)sigval.sival_ptr;
  ssize_t bytes_written = aio_return(cb);

  if (bytes_written > 0) {
    printf("Asynchronous write completed: %zd bytes\n", bytes_written);
  } else {
      printf("Asynchronous write failed\n");
  }

  free(cb->aio_buf);
  free(cb);
}

int main() {
  int fd = open("test_write.txt", O_CREAT | O_WRONLY, 0644);
    if (fd == -1) {
    perror("open");
    return EXIT_FAILURE;
  }

  struct aiocb *cb = malloc(sizeof(struct aiocb));
  if (!cb) {
    perror("malloc");
    close(fd);
    return EXIT_FAILURE;
  }

  char *buffer = malloc(BUFFER_SIZE);
  if (!buffer) {
      perror("malloc");
      free(cb);
      close(fd);
      return EXIT_FAILURE;
  }
  memset(buffer, 'A', BUFFER_SIZE);

  memset(cb, 0, sizeof(struct aiocb));
  cb->aio_fildes = fd;
  cb->aio_buf = buffer;
  cb->aio_nbytes = BUFFER_SIZE;
  cb->aio_offset = 0;
  cb->aio_sigevent.sigev_notify = SIGEV_THREAD;
  cb->aio_sigevent.sigev_notify_function = write_callback;
  cb->aio_sigevent.sigev_value.sival_ptr = cb;

  if (aio_write(cb) == -1) {
    perror("aio_write");
    free(buffer);
    free(cb);
    close(fd);
    return EXIT_FAILURE;
  }

  printf("Asynchronous write initiated...\n");
  sleep(1);
  printf("Main thread continues processing...\n");

  while(aio_error(cb) == EINPROGRESS);

  close(fd);
  return EXIT_SUCCESS;
}
```
This example demonstrates asynchronous writing to a file "test_write.txt". The structure mirrors the read example, highlighting the consistency of the AIO approach. Again, the main thread continues to execute while the write operation is completed asynchronously.

To further improve your understanding of POSIX AIO callbacks, I suggest consulting the following resources:

*   The official POSIX standard documentation for AIO. This provides the most authoritative description of the underlying mechanisms.
*   Advanced Programming in the UNIX Environment (APUE) by W. Richard Stevens. This book provides in-depth explanations of system calls like AIO.
*  Various Linux manual pages, specifically those for `aio.h`, `aio_read`, `aio_write`, `aio_error`, and `aio_return`. These man pages offer granular information on each function's usage and parameters.

By exploring these resources and practicing with code examples like the ones provided, you can develop a solid understanding of how POSIX AIO callbacks work, which will enable the creation of more performant asynchronous applications. Remember to handle error conditions and cleanup resources properly for robust functionality.
