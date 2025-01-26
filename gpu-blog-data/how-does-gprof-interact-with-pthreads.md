---
title: "How does gprof interact with pthreads?"
date: "2025-01-26"
id: "how-does-gprof-interact-with-pthreads"
---

The interaction between `gprof`, the GNU profiler, and POSIX threads (`pthreads`) is often a source of confusion because `gprof` was primarily designed for single-threaded applications. Its sampling-based approach inherently struggles with the concurrent execution model of multi-threaded programs, leading to inaccurate or misleading profile data when directly applied. This arises because `gprof` relies on timer-based interrupts that are delivered to the process as a whole, not specific threads, leading to an inconsistent mapping of time spent. I've observed this limitation firsthand in a large-scale simulation project where we transitioned from a single-threaded implementation to a multi-threaded one using `pthreads`, and our initial profiling results became largely nonsensical.

The core issue stems from how `gprof` samples the program counter (PC). In a single-threaded application, each sample directly correlates to time spent in the currently executing function. However, with multiple threads, a sample taken might fall within a function executed by any of the threads currently running on the CPU cores. If thread A happens to be performing the most work and thus consuming the most CPU time, a disproportionate number of samples might fall within its execution context while other threads might not receive sufficient sampling, even if their operations are equally important to the overall performance. This leads to a skewed representation of time consumption and potentially misguided optimization efforts.

Furthermore, `gprof` typically only reports times as “self” time and “cumulative” time, which aggregates all time spent within a function, including calls to other functions. These metrics become difficult to interpret when multiple threads are concurrently executing functions, potentially including the same function, concurrently, and in an overlapping fashion. This means it's difficult to discern which thread spends time in a particular function, and whether that time is indeed valuable.

To illustrate these points, I'll consider a practical scenario: a parallel computation where two threads process parts of a dataset. The first thread calculates a sum, and the second thread does a square root for each element. For simplicity, the processing is artificially slowed down with a `sleep` function to simulate work. This allows the sampling profiler to have some data in its execution window.

**Example 1: Basic pthreads with gprof**

```c
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>

void *sum_thread(void *arg) {
    int *data = (int*)arg;
    int sum = 0;
    for (int i = 0; i < 10000; ++i) {
        sum += data[i];
        usleep(10); // Simulate some work
    }
    return NULL;
}

void *sqrt_thread(void *arg) {
    int *data = (int*)arg;
    for (int i = 0; i < 10000; ++i) {
        sqrt(data[i]);
        usleep(15); // Simulate some work
    }
    return NULL;
}

int main() {
    int data[10000];
    for (int i = 0; i < 10000; ++i) {
        data[i] = i + 1;
    }

    pthread_t thread1, thread2;
    pthread_create(&thread1, NULL, sum_thread, data);
    pthread_create(&thread2, NULL, sqrt_thread, data);

    pthread_join(thread1, NULL);
    pthread_join(thread2, NULL);

    printf("Done!\n");
    return 0;
}
```

When compiled with `-pg` to enable `gprof` and run, the generated `gmon.out` profile will show aggregate time spent in `sum_thread` and `sqrt_thread`, but with no indication of how that time was divided between different threads. We might see an aggregate of time associated with the `sqrt` function, however, we cannot tell how much time was spent inside `sqrt` specifically by `sqrt_thread`. The time is merely reported as having happened inside the function, not which thread invoked it. The sampling process does not differentiate between threads. This makes it exceedingly difficult to identify bottlenecks or areas of heavy CPU usage within the individual threads.

**Example 2: The Illusion of High Time Spend**

Now, let's slightly modify the previous example by removing the sleep statements from one of the threads to see how skewed `gprof` becomes.

```c
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>

void *sum_thread(void *arg) {
    int *data = (int*)arg;
    int sum = 0;
    for (int i = 0; i < 10000; ++i) {
        sum += data[i];
    }
    return NULL;
}

void *sqrt_thread(void *arg) {
    int *data = (int*)arg;
    for (int i = 0; i < 10000; ++i) {
        sqrt(data[i]);
        usleep(15); // Simulate work on sqrt thread
    }
    return NULL;
}

int main() {
    int data[10000];
    for (int i = 0; i < 10000; ++i) {
        data[i] = i + 1;
    }

    pthread_t thread1, thread2;
    pthread_create(&thread1, NULL, sum_thread, data);
    pthread_create(&thread2, NULL, sqrt_thread, data);

    pthread_join(thread1, NULL);
    pthread_join(thread2, NULL);

    printf("Done!\n");
    return 0;
}
```

In this example, since the `sum_thread` runs quickly to completion due to the absence of the `sleep`, the majority of the profiling samples are now going to be in the `sqrt_thread` even though it might not be the most "important" thread in terms of overall application behavior. If `sum_thread` had other work to do besides the `sum` operation, `gprof` might give a skewed perspective of its relative CPU usage. It will heavily focus on `sqrt` and less on any other functions executed by the `sum_thread` as these happen very quickly without many samples. The relative weight `gprof` gives functions does not always correlate to the thread contribution to the overall process time.

**Example 3: The Impact of Lock Contention (Conceptual)**

Let’s consider a conceptual example of how `gprof` struggles with lock contention. Assume both our `sum_thread` and `sqrt_thread` require access to a shared resource, protected by a mutex. In such a case, the thread that acquires the lock will proceed while other thread might be blocked. When blocked, the sampling events may not even register the time spent. When samples do land, they might register as waiting for the mutex instead of the original computation task the thread set out to do. `gprof` will then record time spent in the system calls managing the lock or in the critical region, but it won't explicitly show us the lock contention directly on a per-thread basis. The overall CPU time is spent in some function in either thread, but the delay associated with the lock will often be attributed to the function itself or to the underlying system libraries implementing mutex management. Identifying bottlenecks arising from lock contention requires tools that are aware of thread execution context.

To summarize, while `gprof` *can* provide a basic overview of which functions are being executed in a multi-threaded application, it lacks the necessary granularity to accurately profile individual thread behavior. The timer-based sampling mechanism does not map well to the concurrent execution model.

For more accurate and granular profiling of multi-threaded applications using `pthreads`, several alternatives exist. I recommend exploring tools like `perf` which can provide both system-wide and process-specific performance data, including thread-specific metrics. It can collect more robust data even during concurrent execution, and is less reliant on timers. Similarly, Valgrind's Callgrind tool is a worthwhile option, although it incurs a higher performance overhead. It traces functions calls and provides a call graph that can be invaluable for discovering bottlenecks in a complex code base. Another approach is to implement custom logging or monitoring mechanisms within your code to capture timing and resource usage on a per-thread basis. While this method requires more development overhead, it can provide the most precise and tailored information for your specific application requirements. Finally, profiling tools that rely on hardware performance counters are also worth investigation. These tools typically incur minimal overhead, and can be used to profile at various levels of abstraction.

Ultimately, using `gprof` in multi-threaded programs is akin to using a blunt instrument; it might give some information, but its insights must be interpreted with extreme caution, and in many cases should be avoided in favor of tools that have been designed for concurrent program analysis.
