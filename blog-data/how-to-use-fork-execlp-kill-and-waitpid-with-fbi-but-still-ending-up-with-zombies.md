---
title: "How to use fork, execlp, kill and waitpid with fbi but still ending up with zombies?"
date: "2024-12-15"
id: "how-to-use-fork-execlp-kill-and-waitpid-with-fbi-but-still-ending-up-with-zombies"
---

alright, so, you're running into the classic zombie process problem when trying to use fork, execlp, kill, and waitpid, particularly with fbi involved. i've been down this road, seen this movie a few times, it's not a fun one. let me unpack what's likely happening and how i've tackled similar issues.

first, the basics: you're forking a process, then in the child process, you're using execlp to run fbi (the framebuffer image viewer i presume). the parent process, meanwhile, is supposed to use waitpid to clean up after the child process finishes, preventing it from becoming a zombie. the problem, as you’ve noticed, is that sometimes, and seemingly at random, you get these lingering zombie processes.

i recall having this exact issue way back when i was working on an embedded system project. i was trying to get a small slideshow to display on a little lcd panel, using fbi as the image viewer. i forked off a process to run fbi on a schedule. everything seemed fine, until the board started randomly freaking out when processes started piling up with the `defunct` flag. memory was obviously getting wasted, and the board's stability would dramatically reduce. i learned quickly that zombies, while technically inactive, are still a nuisance.

the core problem here isn’t necessarily with `fork` or `execlp` by themselves. they’re behaving as designed. the issue typically lies with how you handle the parent process's waitpid call, how it's synchronized with the child process's exit.

let's walk through some code examples, and i'll highlight some of the common pitfalls.

here’s a simple version that will, very likely, give you zombies:

```c
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <signal.h>

int main() {
  pid_t pid;

  pid = fork();

  if (pid == 0) { // child process
    execlp("fbi", "fbi", "-noverbose", "-T", "1", "-a", "/path/to/your/image.jpg", (char *)NULL);
    perror("execlp failed");
    exit(1);
  } else if (pid > 0) { // parent process
      // problematic wait, not catching all exit conditions, or signal interrupts
      waitpid(pid, NULL, 0);
      printf("child process finished\n");
  } else {
    perror("fork failed");
    return 1;
  }
  return 0;
}
```

this first example is what you might expect, right? it forks, the child executes fbi and the parent waits for its completion. the problem here is that the `waitpid` in the parent process isn't fully robust. if the child exits unusually, or if the parent process receives a signal (like sigint) *while* `waitpid` is blocking, it can exit before `waitpid` returns. you end up with a child process that exits correctly, but without a parent process to clean it up, becoming a zombie.

a better approach is to add a loop around `waitpid`, along with non-blocking behavior and checks for the return values of waitpid, and handling for interrupted system calls:

```c
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <errno.h>
#include <signal.h>

int main() {
  pid_t pid;
  int status;

  pid = fork();

  if (pid == 0) { // child process
    execlp("fbi", "fbi", "-noverbose", "-T", "1", "-a", "/path/to/your/image.jpg", (char *)NULL);
    perror("execlp failed");
    exit(1);
  } else if (pid > 0) { // parent process
      
      do {
        pid_t w = waitpid(pid, &status, WNOHANG); 
        if (w == -1) {
            if (errno == EINTR) {
                //system call was interrupted try again
                continue;
            } else {
                // error other than system call interruped
                perror("waitpid failed");
                break;
            }
        }
        if(w == 0) {
            //child is still running
            sleep(0.1);
            continue;
        }

        if (WIFEXITED(status)) {
            printf("child process exited with status: %d\n", WEXITSTATUS(status));
        } else if (WIFSIGNALED(status)) {
            printf("child process was terminated by signal: %d\n", WTERMSIG(status));
        }
        break;

    } while(1); //loop until the waitpid either failed or we got the process return info
    printf("child process finished\n");

  } else {
    perror("fork failed");
    return 1;
  }
  return 0;
}
```

in the second example, `waitpid` is inside a loop, that includes checks for system call interruption using `errno` and it uses the `WNOHANG` option, making it non-blocking. it checks for various exit conditions, and signals, and continues waiting until the child exits properly or it reaches an error. This approach makes sure that you handle different cases better. i've used this quite extensively across different systems, and it's a good foundation.

one subtle thing that used to trip me up is when the parent process exists before `waitpid` is finished, the solution then is to use a signal handler to wait for the child exits. a signal handler for the `sigchld` signal can be quite useful here, you know... a tiny bit of extra defensive programming against unexpected behavior. here’s the approach using `sigchld` handler:

```c
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <errno.h>
#include <signal.h>


void sigchld_handler(int s) {
    int status;
    pid_t pid;
    while ((pid = waitpid(-1, &status, WNOHANG)) > 0) {
        if(WIFEXITED(status)) {
            printf("child process pid %d exited with status: %d\n", pid, WEXITSTATUS(status));
         } else if(WIFSIGNALED(status)){
             printf("child process pid %d was terminated by signal: %d\n", pid, WTERMSIG(status));
         }
    }
}

int main() {
  pid_t pid;
  struct sigaction sa;

  sa.sa_handler = sigchld_handler;
  sigemptyset(&sa.sa_mask);
  sa.sa_flags = SA_RESTART | SA_NOCLDSTOP;
  if (sigaction(SIGCHLD, &sa, NULL) == -1) {
      perror("sigaction");
      return 1;
  }


  pid = fork();

  if (pid == 0) { // child process
    execlp("fbi", "fbi", "-noverbose", "-T", "1", "-a", "/path/to/your/image.jpg", (char *)NULL);
    perror("execlp failed");
    exit(1);
  } else if (pid > 0) { // parent process
        printf("parent continuing doing its stuff\n");
        sleep(2); // simulate some work
        printf("parent finished\n");

  } else {
    perror("fork failed");
    return 1;
  }
  return 0;
}
```

in the third example, we establish a signal handler for `sigchld` signal. inside the handler we loop waiting for all the child processes (in case multiple are forked). the signal handler is responsible for cleaning up the zombies. The parent process continues doing its task and exists.

that third example was how i managed a persistent background process on my embedded system – it would fork for the display, then do other things. i had one little bug when i forgot the `sa_restart`, that caused many strange things to happen. debugging this took longer that i'd like to admit. it was so bad that at one point i told my wife i'm having a zombie apocalypse at home but on a computer.

regarding resources, i found “advanced programming in the unix environment” by w. richard stevens incredibly helpful. it goes deep into process management, signaling, and all the low-level aspects, like in this case, the interaction of processes, and signals. another book that is also helpful is "operating system concepts" by silberschatz, galvin, and gagne, the chapter on process management is excellent to understand the fundamentals behind process execution. the man pages, of course, are crucial too. specifically, look at `fork(2)`, `execlp(3)`, `waitpid(2)`, `signal(2)`, `sigaction(2)` and their error handling sections in detail. understand them well and you will be in very good shape.

one final tip. make sure that you are looking into error conditions, specially on the return values of waitpid. always check what's being returned and use the macros (like `wifexited` and `wtermsig`). this is the biggest mistake i see people doing, myself included, in the past. don't just ignore the return values or the possible errors.

that's it. dealing with zombie processes can be tricky, but a systematic approach like the ones i described and being careful when coding the parent and child process, should get you past it. let me know how it goes, maybe i could give you some more tips on specific situations.
