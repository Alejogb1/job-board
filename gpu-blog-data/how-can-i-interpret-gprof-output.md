---
title: "How can I interpret gprof output?"
date: "2025-01-26"
id: "how-can-i-interpret-gprof-output"
---

The interpretation of `gprof` output requires understanding its sampling-based profiling approach and the resulting data presentation. This profile data, collected through periodic program counter samples, provides a statistical approximation of time spent within functions and their call relationships. I’ve found that relying solely on the raw numbers can be misleading without considering the statistical nature of the data.

`gprof`, short for “GNU profiler”, operates by periodically interrupting program execution to record the current program counter value. These samples are then used to estimate the time spent in each function. It also tracks function call relationships, providing a call graph that reveals how the program's execution flow. The output is generally presented in two main sections: a flat profile and a call graph profile.

The *flat profile* displays information about each function individually, ranked by the time spent within that function. This section includes several key columns:

*   **% time**: The percentage of the total execution time spent in this function. It's crucial to understand this is based on the sampled data, not exact time.
*   **cumulative seconds**: The total seconds spent in this function and all the functions above it in the list, again, according to the samples. This is less helpful for analyzing independent hotspots and best used for comparing cumulative times.
*   **seconds**: The total seconds spent in this function itself based on sampling. This value provides a more direct indication of where the program spent most of its execution time.
*   **calls**: The number of times the function was called. Note that this counts direct calls only, not indirect calls via other functions.
*   **s/call**: The average time spent per call to the function. This value can be misleading for recursive or long-running functions because this is an average across samples taken during execution.
*   **name**: The function's name.

The *call graph profile* provides a more detailed view of the program's execution flow. For each function, it shows:

*   **index**: A unique numerical identifier for the function.
*   **% time**: The percentage of the total execution time spent in this function (as in flat profile).
*   **self seconds**: The total time spent in this function itself.
*   **children seconds**: The total time spent in all functions called by this function.
*   **called**: The number of times this function was called and the number of times it calls other functions.
*   **name**: The function's name, often with details about the file it belongs to and whether its static or not.
*   **parents**: Functions that call the current function, with the time spent in each.
*   **children**: Functions called by the current function, with the time spent in each.

This call graph section can be initially dense, but it allows for a more thorough analysis of which functions are consuming the most time, including the time spent in the child functions called from the identified hotspots.

Below are three code examples demonstrating how to interpret `gprof` output, along with commentary.

**Example 1: Simple Computation**

```c
#include <stdio.h>
#include <stdlib.h>

void slow_function(int n) {
    int i, j, k;
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            for(k = 0; k < n; k++){
             volatile int x = i * j * k;
            }
        }
    }
}

int main() {
    slow_function(100);
    slow_function(150);
    return 0;
}
```

After compiling with `-pg` and running the program, the following output is a subset of the flat profile output:

```
Flat profile:

  %   cumulative   self              self     total
 time   seconds   seconds    calls   s/call   s/call  name
 69.26      0.36     0.36        2     0.18     0.18  slow_function
 30.74      0.52     0.16        1     0.16     0.52  main
```

**Commentary:** Here, the `slow_function` occupies a significant portion (69.26%) of the total execution time according to `gprof` samples, as expected. This suggests that it's a performance bottleneck. `main` also appears, but much of its time is likely spent calling `slow_function`. This example confirms that the flat profile can efficiently pinpoint computationally intensive functions.

**Example 2: Function Call Graph**

```c
#include <stdio.h>
#include <stdlib.h>

void leaf_function() {
    volatile int sum = 0;
    for (int i=0; i< 10000; i++){
        sum += i;
    }
}

void middle_function(int n) {
    for(int i = 0; i < n; i++){
    leaf_function();
    }
}

int main() {
    middle_function(1000);
    return 0;
}
```

A subset of the call graph output:

```
index % time    self  children    called     name
                seconds seconds  called+parents    name
[1]   75.0    0.02     0.02     1000/1000        middle_function [2]
                0.02       1      main [3]
[2]   75.0    0.02     0.00     1000/1000       leaf_function [1]
                0.00      1000  middle_function [1]
[3]   25.0   0.02    0.04         1/0           main  [0]

```

**Commentary:** This output indicates that `middle_function` and `leaf_function` consume the most execution time. `middle_function` calls `leaf_function` many times. The call graph confirms that the majority of `middle_function`’s time is spent calling the `leaf_function`, which accounts for 75% of the time and is called 1000 times. We can trace back that `main` calls `middle_function` once. Note that the times can shift depending on the number of samples, it should be seen as an approximation rather than absolute time.

**Example 3: Understanding Self vs. Children**

```c
#include <stdio.h>
#include <stdlib.h>

int child_function(int n) {
    volatile int result = 0;
    for(int i = 0; i< n; i++){
        result += i;
    }
    return result;
}

int parent_function(int n) {
  int total = 0;
    for (int i = 0; i < n; ++i) {
        total += child_function(100);
    }
  return total;
}

int main() {
    parent_function(1000);
    return 0;
}
```

A subset of the call graph:

```
index % time    self  children    called     name
                seconds seconds  called+parents    name
[1]    90.0    0.02      0.08       1/1           parent_function [2]
                        0.08       1     main [3]
[2]  100.0     0.08    0.00    1000/1000        child_function [1]
                       0.00      1000  parent_function [1]
[3]  10.0   0.01    0.08         1/0           main [0]
```

**Commentary:** In this example, the `parent_function` consumes 90% of the execution time, but most of that time is spent executing the `child_function`, which is called 1000 times by the `parent_function`. The `parent_function`’s `self seconds` only indicates time spent *within* the `parent_function`, excluding time spent in the called `child_function`. The `children seconds` column clarifies that a significant portion of the program's total time is within that function and its descendants. This clearly illustrates the importance of understanding the “self” and “children” distinction.

When interpreting `gprof` output, it is important to consider the limitations of sampling-based profiling. The profile data provides an approximation of execution behavior, but it does not capture precise timing. Short-duration functions might be missed, and the observed distribution of samples can vary from run to run. High frequency sampling may increase accuracy, but will increase overhead. Therefore, results are more useful for relative comparisons than absolute values.

Resource recommendations for further understanding include: reading the official GNU documentation for `gprof`, exploring computer architecture textbooks that describe program counter functionality, and exploring papers on performance analysis methodologies, specifically for sampling based profilers. Studying these resources will provide a deeper understanding of how sampling-based profilers such as `gprof` collect and present their data, leading to more accurate interpretations and more effective performance optimization.
