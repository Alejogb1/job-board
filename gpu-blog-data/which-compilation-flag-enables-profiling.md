---
title: "Which compilation flag enables profiling?"
date: "2025-01-30"
id: "which-compilation-flag-enables-profiling"
---
Profiling, a critical technique for performance analysis, is frequently activated during compilation through specific flags targeting instrumentation. Having spent considerable time optimizing application performance, particularly in resource-constrained embedded systems, I’ve found that the `-pg` flag, commonly used with GCC and Clang, is instrumental in enabling profiling with gprof. This flag modifies the compilation process to insert instrumentation code into the generated executable, which allows the profiler to track execution time and function call counts.

**Explanation of the -pg Flag and its Impact**

The `-pg` flag’s primary function is to prepare a program for profiling using gprof (GNU profiler). When this flag is present during both compilation and linking, the compiler injects specific code into the compiled object files. This injected code operates as a hook into the operating system's profiling mechanism. Every time a function is entered or exited, these hooks record the current program counter along with other contextual information.

Essentially, `-pg` adds instructions that increment counters associated with each function. During program execution, a log of these increments is maintained. Upon program termination, this log is translated into a `gmon.out` file, containing raw profile data. The `gprof` utility then processes this file, interpreting the raw data and providing a human-readable report including the call graph, execution time, and call counts for every function.

The `-pg` flag has some performance implications. Since it adds instrumentation, the resulting executable becomes slower compared to its non-profiled counterpart. The overhead includes both function entry and exit logging, plus the storage of call history data during runtime. This slowdown can be significant, especially in code with high function call density or tight loops. Consequently, profiling with `-pg` is not typically used in production environments; it is employed solely for performance analysis and debugging.

Furthermore, the `-pg` flag requires the operating system kernel to support the recording of program counter data. In some embedded systems or highly specialized platforms, this support might be absent or require additional kernel configuration.

**Code Examples and Commentary**

The following code examples will demonstrate the use of the `-pg` flag in a simple C program. The provided code is for explanatory purposes and may need adaptation for specific toolchain and compiler setup.

*Example 1: Basic Function Call Tracking*

```c
#include <stdio.h>

void function_a() {
  int i;
  for (i=0; i < 10000; i++);
}

void function_b() {
  int j;
    for (j=0; j < 5000; j++);
  function_a();
}

int main() {
    function_b();
    return 0;
}
```

**Compilation and Analysis Process:**

1.  **Compilation with -pg:** `gcc -pg -o example1 example1.c`
    This command compiles `example1.c` and creates an executable file named `example1`. The `-pg` flag enables profiling instrumentation within this generated executable.
2.  **Running the Executable:** `./example1`
    Running the executable will create the `gmon.out` file in the same directory, containing the raw profile data.
3.  **Profiling with gprof:** `gprof example1 gmon.out`
    The `gprof` command analyses the output file `gmon.out` against the executable `example1`, and produces a profile summary report in the console. This report details the timing and call information for functions `function_a`, `function_b`, and `main`. Specifically, it will show how much time was spent in each function, and how many times functions where called.

*Example 2: Recursive Function Call Tracking*

```c
#include <stdio.h>

int factorial(int n) {
  if (n <= 1) {
    return 1;
  } else {
    return n * factorial(n - 1);
  }
}

int main() {
  factorial(5);
  return 0;
}
```

**Compilation and Analysis Process:**

1.  **Compilation with -pg:** `gcc -pg -o example2 example2.c`
    This compiles the code with instrumentation for profiling, generating the `example2` executable.
2.  **Running the Executable:** `./example2`
    This generates the `gmon.out` file in the current directory.
3.  **Profiling with gprof:** `gprof example2 gmon.out`
    Running gprof against the executable and profiling data will demonstrate detailed insights into the recursive calls made by the `factorial` function, showing how much time is spent within each level of the recursive calls.

*Example 3: Interaction with Library Functions*

```c
#include <stdio.h>
#include <math.h>

void perform_calculations() {
  double x = 2.0;
    for(int i = 0; i < 1000; i++)
    {
        x = sqrt(x + i);
    }
}

int main() {
  perform_calculations();
  return 0;
}
```

**Compilation and Analysis Process:**

1.  **Compilation with -pg:** `gcc -pg -o example3 example3.c -lm`
    This compiles the code, including linking to the math library (`-lm`). The `-pg` flag enables the profiling instrumentation within the `example3` executable.
2.  **Running the Executable:** `./example3`
    The execution generates `gmon.out`.
3.  **Profiling with gprof:** `gprof example3 gmon.out`
    The gprof analysis will reveal the execution time taken both inside `perform_calculations` and the call to the library function `sqrt`. This demonstrates how gprof tracks time spent within library functions as well, given that the library itself was compiled with `-pg` support.

**Resource Recommendations:**

For further information regarding profiling techniques and tools, several resources are available. The following, while not a complete list, provide a strong foundation for understanding and using profiling within software development.

1.  **GNU Profiler (gprof) Manual:** The official gprof documentation, typically available as a man page or a standalone document included in the GNU development tools, is the definitive source for detailed information on gprof usage, options, and interpretation of results. It includes information on the `gmon.out` file format and the various reporting mechanisms.
2.  **GCC Compiler Documentation:** The GCC documentation contains a section detailing how `-pg` and other profile-related compiler flags influence the generated code. It offers a detailed explanation of compiler-side instrumentation and limitations. Understanding this documentation will provide a more comprehensive view of what occurs at compile time when profiling is enabled.
3. **Operating System Documentation:** Specific OS documentation may contain notes on kernel level support requirements, and the mechanisms used to collect profiling data. Understanding this aspect is crucial for developing within embedded or low-level systems.

In conclusion, the `-pg` compilation flag is essential for enabling profiling with gprof. Its usage adds instrumentation to the executable which enables collecting execution data, allowing for subsequent analysis. This comes at the cost of performance overhead during runtime, making it unsuitable for production environments, but invaluable for performance analysis and optimization during development. Utilizing this alongside the recommended resources provides a framework for developers to gain insights into application behavior.
