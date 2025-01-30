---
title: "Why is `Future.delayed` faster than a direct `Future` for small delays?"
date: "2025-01-30"
id: "why-is-futuredelayed-faster-than-a-direct-future"
---
Directly constructing a `Future` with an immediate completion using `Future(() => value)` incurs an overhead that `Future.delayed` circumvents for small durations, specifically due to the execution context and scheduling of work. My experience, particularly within asynchronous game engine development, highlights this performance distinction. In performance-critical code paths, even microsecond differences are significant, demanding careful consideration of `Future` constructors.

The core issue stems from how the Dart VM handles these two `Future` creation methods. A simple `Future(() => value)` immediately attempts to execute the provided function (`() => value` in this instance) synchronously. The resulting value is then used to complete the `Future`, but this synchronous execution can be inefficient when the function is very short or when all we care about is a small delay before some future work. Conversely, `Future.delayed(Duration(milliseconds: N), () => value)` leverages the Dart event loop’s timer mechanism. This mechanism essentially places the completion of the `Future` on a queue that will execute the function once the specified delay has elapsed. For short durations, this deferral via the event loop provides a significantly more performant approach.

The synchronous path of a standard `Future` instantiation, even if the provided function is effectively instantaneous, still involves more internal steps. The Dart runtime needs to evaluate the function expression, enter a stack frame, and then return a result. This involves both internal state management and memory access. `Future.delayed`, on the other hand, registers a callback with the event loop's timer system. This registration involves a relatively low-overhead operation of adding a timer event and associated callback to the loop’s queue. This is a lightweight scheduling process. Consequently, for very short delays, even in the range of milliseconds, the timer-based `Future.delayed` is significantly quicker than the immediate execution via a typical `Future` constructor.

It is crucial to understand that this performance characteristic is mainly observed with small delay durations, likely under one or two milliseconds based on observed profiles in my use cases. Beyond this duration, the overhead of either path becomes less significant, and the specific workload associated with the function becomes more important. However, for situations requiring brief delays, such as pacing animation transitions or managing UI interactions, selecting `Future.delayed` over a direct `Future` instantiation can improve overall system responsiveness.

Here are three code examples to illustrate this difference, along with commentary:

**Example 1: Direct `Future` Instantiation**

```dart
import 'dart:async';

void main() async {
  Stopwatch stopwatch = Stopwatch()..start();
  for (int i = 0; i < 100000; i++) {
    await Future(() => null);
  }
  stopwatch.stop();
  print('Direct Future: ${stopwatch.elapsedMicroseconds} microseconds');
}
```

*   **Commentary:** This code snippet creates and awaits a large number of `Future` objects. Each future immediately returns `null` on its creation. This approach highlights the overhead associated with repeatedly invoking the standard `Future` constructor. We’re not actually waiting on anything substantial; the measured time is the time spent creating the future and completing it. The absence of real work inside the callback makes it a good example to demonstrate the impact of the framework’s internal overhead. Running this code repeatedly will showcase how each synchronous function invocation adds up. The time taken will be relatively long for creating that number of future objects.

**Example 2: Using `Future.delayed`**

```dart
import 'dart:async';

void main() async {
  Stopwatch stopwatch = Stopwatch()..start();
  for (int i = 0; i < 100000; i++) {
      await Future.delayed(Duration.zero, () => null);
  }
  stopwatch.stop();
  print('Future.delayed: ${stopwatch.elapsedMicroseconds} microseconds');
}
```

*   **Commentary:**  This code, similar to the first example, also creates and awaits a large number of future objects. The crucial difference is that this example utilizes `Future.delayed` with a `Duration.zero` delay. By employing `Future.delayed` with zero delay, we instruct the Dart VM to leverage the event loop to schedule the future’s completion. Though seemingly equivalent to an immediate `Future` completion, the utilization of the timer mechanism makes a noticeable difference in performance because it avoids stack frame management and other overhead. The elapsed time should be significantly less than that measured in example one. Zero delay means that the provided work will be put in the event loop, which executes it the next time the loop has nothing else to do.

**Example 3: Comparing with a Slightly Longer Delay**

```dart
import 'dart:async';

void main() async {
  Stopwatch stopwatchDirect = Stopwatch()..start();
  for (int i = 0; i < 100000; i++) {
    await Future(() => null);
  }
  stopwatchDirect.stop();

  Stopwatch stopwatchDelayed = Stopwatch()..start();
  for (int i = 0; i < 100000; i++) {
    await Future.delayed(Duration(microseconds: 10), () => null);
  }
  stopwatchDelayed.stop();

  print('Direct Future: ${stopwatchDirect.elapsedMicroseconds} microseconds');
  print('Future.delayed (10 microseconds): ${stopwatchDelayed.elapsedMicroseconds} microseconds');
}
```

*   **Commentary:** This example directly compares the timings of both approaches. One part of the code uses a direct `Future` as in the first example. The other part creates `Future.delayed` objects with a slightly longer delay of 10 microseconds. The measurement from this example illustrates a few points. The direct `Future` construction takes about the same time as in the previous example because, again, we are doing almost nothing in the body. In the second part, we are delaying for 10 microseconds. For 10 microseconds, the time taken will be noticeably higher than for a `Duration.zero`, but still much faster than the direct future constructor. Importantly, beyond a certain threshold (a few milliseconds, experimentally), the two approaches approach closer to equal. This exemplifies the specific performance advantage that `Future.delayed` offers for very short delays.

Based on my professional experience and the aforementioned performance characteristics, I recommend the following for deeper understanding:

*   **Dart VM Internals:** Familiarize yourself with the Dart Virtual Machine’s architecture, especially concerning its event loop and timer mechanism. Understanding how tasks are scheduled and executed within the VM helps demystify the performance differences.
*   **Concurrency and Asynchrony Patterns:** Exploring various concurrency patterns, particularly those involving asynchronous operations, is beneficial. This will allow more informed decisions regarding scheduling logic and the trade-offs between immediate and deferred execution.
*   **Profiling Tools:** Utilize the Dart SDK’s profiling tools, such as the DevTools. Profiling concrete use cases is crucial for identifying performance bottlenecks and making data-driven decisions about utilizing synchronous vs. asynchronous methods.
*   **Performance Benchmarking:** Regularly conduct micro-benchmarks to empirically validate performance assumptions, as environmental factors and the specific version of the Dart SDK can influence the measured timings. Create benchmarks similar to the above code examples.

In summary, the performance discrepancy between a simple `Future` constructor and `Future.delayed` for short durations is rooted in the Dart runtime's scheduling and execution strategy. The standard `Future` constructor introduces synchronous overhead, whereas `Future.delayed` leverages the event loop’s timer functionality for optimized deferral. This distinction is particularly relevant in scenarios demanding fine-grained control over execution timing and responsiveness, necessitating careful consideration of `Future` construction methods, especially for short delays.
