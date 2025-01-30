---
title: "How can I profile misbehaving Emacs Lisp code?"
date: "2025-01-30"
id: "how-can-i-profile-misbehaving-emacs-lisp-code"
---
Emacs Lisp, while often perceived as inherently fast due to its nature as an interpreted language, can still exhibit performance bottlenecks that significantly degrade responsiveness. Identifying these issues requires a methodical approach, leveraging built-in profiling tools and understanding common performance pitfalls. Based on my experience optimizing numerous Emacs configurations, effective profiling begins with the `profiler` package, which is included in standard Emacs installations.

The primary profiling function, `profiler-start`, initiates the collection of execution statistics. Crucially, this function accepts arguments specifying which resources to track: `cpu`, `gc`, and `memory`. Each of these offers different insights. `cpu` profiling reveals time spent in function execution, useful for pinpointing computational bottlenecks. `gc` profiling tracks garbage collection activity, which can be a significant source of pauses. `memory` profiling monitors memory allocation, which can highlight resource leaks.

To use the profiler effectively, a targeted approach is essential. Rather than profiling the entire Emacs session, focus on the problematic area. This involves enabling the profiler before executing the code in question and then stopping it afterward using `profiler-report`. This function presents the collected data. The format of this report is a hierarchical display of function calls, showing how much time was spent in each function, including its descendants, and how many times each function was called. This breakdown of resource consumption allows you to pinpoint exactly where performance is being lost.

After generating the profiler report, analyzing it critically is the next step. Look for functions with high ‘total’ time values, particularly those with a disproportionately high 'exclusive' time. Total time includes time spent within that function and its callees. Exclusive time, conversely, only includes the time spent within that function itself. If a function exhibits high total time and moderate exclusive time, its bottlenecks might lie in its callees. Conversely, high exclusive time points to internal performance issues within the function. Functions called frequently, even with low time consumption per call, can also be performance issues when accumulated.

A critical but often overlooked consideration when analyzing profile data is the effect of byte compilation. If a function has been byte-compiled, Emacs runs a more optimized version, therefore differences between compiled and non-compiled code should also be kept in mind while analyzing profiles. During development and debugging, it is advisable to ensure code is evaluated in its non-compiled form. I have personally observed instances where functions that are slow when interpreted show vastly improved performance after byte compilation, significantly altering the profile. You can compile a specific file using `byte-compile-file`.

Now, let’s consider a few practical examples of how to use the profiler.

**Example 1: Identifying a computationally expensive function**

Assume there's a function to manipulate strings that seems slow.

```lisp
(defun slow-string-manipulation (str count)
  (let ((result str))
    (dotimes (_ count)
      (setq result (concat result "a")))
    result))

(profiler-start 'cpu)
(slow-string-manipulation "start" 10000)
(profiler-report)
(profiler-stop)
```

In this example, we start the profiler tracking CPU usage. We then invoke `slow-string-manipulation` with a reasonably large count to exaggerate the performance impact. `profiler-report` displays the collected data. The report will show that the time spent in `slow-string-manipulation` is high, and specifically that `concat` (due to repeated string concatenation) is consuming a substantial portion of the time. This reveals a clear bottleneck: repeated string concatenation is an inefficient process; constructing the string with a more efficient method, such as accumulating characters in a list and using `mapconcat` at the end, would be faster.

**Example 2: Observing garbage collection overhead**

Consider a scenario where excessive object creation occurs.

```lisp
(defun create-many-objects (count)
  (let ((objects nil))
    (dotimes (_ count)
      (push (make-vector 1000 0) objects))
    objects))

(profiler-start 'gc)
(create-many-objects 5000)
(profiler-report)
(profiler-stop)
```

Here, the profiler is set to track garbage collection ('gc').  The function `create-many-objects` creates numerous vectors, leading to significant memory allocation. The profiler report, in this instance, should demonstrate a considerable amount of time spent on garbage collection. This indicates that the code creates many short-lived objects. This can highlight inefficient memory usage.  In such scenarios, it might be possible to avoid creating such a large amount of vectors, reuse existing objects, or change the data structures altogether.

**Example 3: Diagnosing memory leaks**

Although the previous example highlights garbage collection issues, let's modify it to simulate an actual memory leak, albeit contrived.

```lisp
(defvar leaky-objects nil)

(defun leak-memory (count)
  (dotimes (_ count)
    (push (make-vector 1000 0) leaky-objects)))

(profiler-start 'memory)
(leak-memory 5000)
(profiler-report)
(profiler-stop)
```
This version uses a global variable `leaky-objects` to accumulate allocated vectors. The profiler is started with the `memory` argument. The profiler report should now show an increase in total memory allocated, and likely that `push` and `make-vector` are the functions responsible for the allocations. In a real application, a memory leak can manifest when you keep adding elements to a variable that is not cleaned up, resulting in unbounded growth. This is an important signal to identify and fix. Unlike the previous example, where the objects are allocated and quickly discarded, the use of `leaky-objects` demonstrates how to create a memory leak, showing a constant increase in the memory footprint.

To further enhance your proficiency in profiling Emacs Lisp, I recommend consulting the following resources: The Emacs Manual, specifically the section on "Profiling." It is essential reading for understanding the nuances of the profiler. "Programming in Emacs Lisp" by Robert Chassell offers valuable insights into writing efficient Emacs Lisp code and understanding the underlying language mechanics. I also suggest studying well-regarded open source Emacs packages. By examining the techniques used by experienced Emacs Lisp developers in real-world scenarios, you can learn effective patterns for optimization. While these are books or offline material, always refer to the most up to date documentation available in Emacs via `C-h i m Emacs RET` or via the help command `C-h f` followed by the specific function you wish to know more about (e.g. `C-h f profiler-start`). Experimenting with these tools and techniques is crucial to developing a keen eye for performance bottlenecks.
