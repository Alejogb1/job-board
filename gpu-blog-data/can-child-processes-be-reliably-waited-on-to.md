---
title: "Can child processes be reliably waited on to exit on macOS?"
date: "2025-01-30"
id: "can-child-processes-be-reliably-waited-on-to"
---
On macOS, while the `wait` family of system calls provides the core mechanism for monitoring child process termination, the reliability of this mechanism can be subtly undermined by factors not always immediately obvious, particularly concerning signal handling and process groups. My experience developing cross-platform process management utilities highlights some specific areas where vigilance is required to ensure consistent and reliable child process waiting on macOS, specifically when compared to simpler, Linux-centric implementations.

The primary challenge arises from the interaction of signals and the `wait` system calls, specifically `wait()`, `waitpid()`, and `waitid()`. While these calls inherently block the parent process until a child process changes state (typically, termination), the delivery of signals to either the parent or child process can interrupt this blocking behavior. Specifically, if the parent process receives a signal while blocking in a `wait` call, the call may be prematurely terminated with an `EINTR` error, necessitating explicit handling of this interruption. Furthermore, the signal delivery to a child may, if not handled, lead to the child's termination through the default signal action. This termination, however, is not always the expected outcome from a process management viewpoint.

A crucial aspect often overlooked is the concept of process groups. When launching child processes, they often belong to the same process group as their parent. Consequently, signals directed to the process group (e.g., from the shell) can affect both the parent and the children. In particular, signals like `SIGINT` or `SIGTERM` sent to the process group will affect all its members and might terminate children unexpectedly, thereby interfering with the parent's attempt to gracefully wait for them, and possibly before the parent has registered their pid with the `wait` function.

To reliably wait on child processes, several steps are essential:

First, consistently handle `EINTR` errors returned by the `wait` calls. This ensures that the parent process continues to wait for the child process to terminate even if it is interrupted by a signal. Second, consider using `waitpid()` with a specific child process ID rather than `wait()`. This offers better control when waiting on multiple children and prevents the parent process from inadvertently picking up the state change of an unrelated child process. Third, consider using `sigaction` to block specific signals before entering the wait loop and to subsequently re-enable them after the wait loop ends. This prevents the parent from prematurely terminating due to signals that it should not process directly. Finally, establish a mechanism to capture and act upon the return code from the children. Without this, errors in the child process may not be detectable, leading to further complications later on in program execution.

Here are three examples illustrating these concepts, using C:

**Example 1: Basic `waitpid` Implementation with Error Handling**

This example focuses on handling the `EINTR` error from `waitpid`. It launches a child process and waits for its termination, including loop to handle interrupted system calls:

```c
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>
#include <errno.h>

int main() {
  pid_t pid = fork();

  if (pid == 0) {
      // Child process
      printf("Child process started (PID: %d).\n", getpid());
      sleep(2);
      exit(EXIT_SUCCESS);
  } else if (pid > 0) {
      // Parent process
      int status;
      pid_t child_pid;

        do{
            child_pid = waitpid(pid, &status, 0);
        } while (child_pid == -1 && errno == EINTR);

      if (child_pid == -1) {
        perror("waitpid failed");
        exit(EXIT_FAILURE);
      }

      if (WIFEXITED(status)) {
          printf("Child process exited with status: %d\n", WEXITSTATUS(status));
      } else if (WIFSIGNALED(status)) {
          printf("Child process terminated by signal: %d\n", WTERMSIG(status));
      }
  } else {
      perror("fork failed");
      return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
```

*   This code uses `waitpid()` with the specific PID of the child.
*   The `do...while` loop ensures that if `waitpid` is interrupted by a signal (resulting in `errno == EINTR`), the loop will continue waiting for the child process to terminate.
*   It uses `WIFEXITED` and `WEXITSTATUS` to get the exit code and `WIFSIGNALED` and `WTERMSIG` to determine a termination cause due to signal, if applicable.
*   A failure of fork or waitpid is handled and indicated to the user via stderr and an exit code.

**Example 2: Signal Masking Before and After Wait**

This example utilizes `sigaction` to block specific signals (`SIGINT` and `SIGTERM`) before entering a wait loop, to prevent interruption and handles the signal masking to the child.

```c
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>
#include <errno.h>
#include <signal.h>

int main() {
  pid_t pid = fork();
  sigset_t mask, oldmask;

    sigemptyset(&mask);
    sigaddset(&mask, SIGINT);
    sigaddset(&mask, SIGTERM);

    if (pid == 0) {
        // Child process
        printf("Child process started (PID: %d).\n", getpid());
        sleep(2);
        exit(EXIT_SUCCESS);
    } else if (pid > 0) {
        // Parent process
        int status;
        pid_t child_pid;

        if (sigprocmask(SIG_BLOCK, &mask, &oldmask) < 0){
            perror("sigprocmask - block");
            exit(EXIT_FAILURE);
        }

        do{
            child_pid = waitpid(pid, &status, 0);
        } while (child_pid == -1 && errno == EINTR);

        if (sigprocmask(SIG_SETMASK, &oldmask, NULL) < 0){
          perror("sigprocmask - unblock");
          exit(EXIT_FAILURE);
        }

        if (child_pid == -1) {
          perror("waitpid failed");
          exit(EXIT_FAILURE);
        }

        if (WIFEXITED(status)) {
          printf("Child process exited with status: %d\n", WEXITSTATUS(status));
        } else if (WIFSIGNALED(status)) {
          printf("Child process terminated by signal: %d\n", WTERMSIG(status));
        }
    } else {
        perror("fork failed");
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
```
*   Here, `sigemptyset`, `sigaddset` are used to build a signal set containing `SIGINT` and `SIGTERM`.
*   `sigprocmask` is used to block these signals before the `waitpid` loop.
*   The old signal mask is stored and restored after the wait loop, allowing the parent process to handle signals again later on.
*   Note: the child process is not impacted by this `sigprocmask` call as it has its own context.

**Example 3: Handling Children in a Loop**

This example shows how to reliably wait for multiple children and track them when more than one child is launched:

```c
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>
#include <errno.h>
#include <signal.h>

#define NUM_CHILDREN 3

int main() {
    pid_t children[NUM_CHILDREN];
    int i;

    sigset_t mask, oldmask;
    sigemptyset(&mask);
    sigaddset(&mask, SIGINT);
    sigaddset(&mask, SIGTERM);


    for (i = 0; i < NUM_CHILDREN; i++) {
        pid_t pid = fork();

        if (pid == 0) {
            // Child process
            printf("Child process %d started (PID: %d).\n", i, getpid());
            sleep(rand() % 3);
            exit(EXIT_SUCCESS);
        } else if (pid > 0) {
            children[i] = pid;
        } else {
            perror("fork failed");
            return EXIT_FAILURE;
        }
    }

    if (sigprocmask(SIG_BLOCK, &mask, &oldmask) < 0){
        perror("sigprocmask - block");
        exit(EXIT_FAILURE);
    }

    for(i=0; i<NUM_CHILDREN; i++){
      int status;
      pid_t child_pid;

        do{
            child_pid = waitpid(children[i], &status, 0);
        } while (child_pid == -1 && errno == EINTR);

        if (child_pid == -1) {
            perror("waitpid failed");
        } else {
            if (WIFEXITED(status)) {
                printf("Child process (PID %d) exited with status: %d\n", (int)children[i], WEXITSTATUS(status));
            } else if (WIFSIGNALED(status)) {
                printf("Child process (PID %d) terminated by signal: %d\n", (int)children[i], WTERMSIG(status));
            }
        }
    }

    if (sigprocmask(SIG_SETMASK, &oldmask, NULL) < 0){
        perror("sigprocmask - unblock");
        exit(EXIT_FAILURE);
    }

    return EXIT_SUCCESS;
}
```

*   This example launches multiple children in a loop and stores their PIDs in an array.
*   It iterates through the `children` array to `waitpid` on each child.
*   Error handling for `waitpid` and the return status checking are still present.
*   Signal masking is also performed to ensure that signals are not processed before we can reliably wait on all the children

To further investigate reliable process waiting on macOS, I would recommend focusing on the following areas through consultation with the relevant documentation:

*   **Operating System Concepts:** Textbooks such as "Operating System Concepts" by Silberschatz, Galvin, and Gagne provides a detailed understanding of process management and signals in operating systems.
*   **Advanced Programming in the UNIX Environment (APUE):** The textbook by Stevens, Rago, and Maurer offers a thorough explanation of system calls relevant to process management, including `fork`, `exec`, and the `wait` family.
*   **macOS System Documentation:** Consulting the official macOS documentation, specifically the manual pages for `wait`, `waitpid`, `sigaction`, and other related system calls is essential.
*   **Unix Signal Handling:** Books and resources detailing POSIX signal handling, especially the interaction of signals with system calls, such as `waitpid` are valuable.

By focusing on these technical aspects and using appropriate error handling and signal management techniques, one can reliably wait on child processes on macOS. This has been my practical experience, and has led to robust process management code.
