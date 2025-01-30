---
title: "Why are these function calls not optimized?"
date: "2025-01-30"
id: "why-are-these-function-calls-not-optimized"
---
The primary reason function calls are sometimes not optimized, even in supposedly optimizing compilers, lies in the intricate interplay between function scope, visibility, and the potential for side effects, which directly impacts a compiler's ability to make transformations safely. Compilers must adhere to the principle of 'as-if' behavior; any optimization cannot alter the observable outcome of the program, except for performance characteristics.

Function calls introduce several barriers to optimization, notably the potential for unpredictable behavior. Optimizations like inlining, loop unrolling, and register allocation become considerably more complex when functions are involved. Consider a scenario where I once struggled with a hot-spot in a physics simulation; the bottleneck centered on a function calculating a vector's magnitude. Even though the function itself was small, the number of calls within the simulation’s time-stepping loop made it a prime candidate for optimization, or so I initially thought.

The crux of the issue often boils down to what the compiler *knows* about a given function. If a function is declared in a separate compilation unit, or is part of a dynamically linked library, the compiler essentially treats it as a black box. The compiler cannot, without further information, make assumptions about the function’s behavior, what it modifies or what its inputs and outputs depend on. This lack of visibility prevents aggressive optimizations. Even when a function is defined within the same compilation unit, features like function pointers and virtual functions further complicate analysis, preventing techniques like inlining from being applied.

Let's delve into some specific scenarios where these restrictions manifest:

**Scenario 1: External Function Calls**

```cpp
// file1.cpp
extern double calculateForce(double mass, double acceleration);

double applyForce(double mass, double acceleration) {
    return calculateForce(mass, acceleration);
}
```

```cpp
// file2.cpp
double calculateForce(double mass, double acceleration) {
    return mass * acceleration;
}
```

In this setup, the `calculateForce` function is defined in `file2.cpp` and declared in `file1.cpp`. The compiler, while compiling `file1.cpp`, cannot "see" the actual implementation of `calculateForce`.  Therefore, even if `calculateForce` simply multiplies two doubles, the compiler cannot inline this function within `applyForce`, due to the lack of complete information at compile time. The linkage between the two compilation units occurs later in the linking process. Thus, the `applyForce` function will invoke `calculateForce` via a regular function call at runtime, incurring the overhead of the call stack setup and parameter passing. The compiler must treat the `calculateForce` call as a potential point of arbitrary side effects, as it has no information otherwise. This is further complicated by potential shared object libraries that might replace the implementation at run time. Compilers prioritize correctness over optimization under these circumstances.

**Scenario 2: Function Pointers**

```cpp
#include <iostream>

double add(double a, double b) { return a + b; }
double subtract(double a, double b) { return a - b; }

double operate(double a, double b, double (*func)(double, double)) {
  return func(a, b);
}

int main() {
  double result1 = operate(5, 3, add);
  double result2 = operate(5, 3, subtract);
  std::cout << "Result 1: " << result1 << std::endl;
  std::cout << "Result 2: " << result2 << std::endl;
}
```

Here, the `operate` function accepts a function pointer `func`. The specific function being called, `add` or `subtract`, is only determined at the call site within `main`.  The compiler cannot know at compile time which function will be invoked inside `operate`. Therefore, inlining of either `add` or `subtract` within `operate` is not feasible. The function pointer adds a layer of indirection, preventing the compiler from making decisions based on concrete implementations. This dynamic nature of function pointers inherently blocks certain optimization strategies. While runtime optimization through techniques like just-in-time compilation can partially address this, it also carries its own costs, including potential delays from the additional compilation step and an increased memory footprint.

**Scenario 3: Mutable Global State**

```cpp
#include <iostream>

int globalCounter = 0;

int increment() {
  return ++globalCounter;
}

void performOperations() {
  int value1 = increment();
  int value2 = increment();
  std::cout << "Value 1: " << value1 << ", Value 2: " << value2 << std::endl;
}

int main() {
  performOperations();
}
```

In this example, the `increment` function modifies a global variable, `globalCounter`. While seemingly simple, this mutable global state has ramifications for optimization. If the compiler tried to inline `increment` into `performOperations`, it would need to ensure that the side effect on `globalCounter` remains consistent with the non-inlined version. The compiler must, in essence, track and manage the state of the global variable through any inlining. This adds overhead, as the compiler would essentially need to perform similar steps as a normal function call but in-place. Due to the increased complexity, often compilers may forego inlining in favor of correct behavior, or limit inlining in an attempt to prevent incorrect or unintended behaviors. Furthermore, if the global variable is shared with other threads, inlining can become very problematic and potentially introduce data races or incorrect results.

These cases illustrate the core issues. Optimizing function calls is not always straightforward, and compilers make rational trade-offs between potential performance gains and ensuring that the program's behavior remains unchanged. More advanced techniques, such as link-time optimization (LTO) and profile-guided optimization (PGO) can sometimes alleviate these issues. LTO allows the compiler to see across compilation units, improving opportunities for inlining, and PGO uses runtime performance data to influence compile-time decisions, often producing more efficient code through informed choices. These can significantly boost performance but require further configuration and aren't always available or applicable.

To further explore this topic, consider reviewing compiler optimization textbooks and documentation. Resources detailing compiler architecture, intermediate representations, and optimization passes are very informative. Books focusing on advanced compilation techniques, such as link-time optimization and profile-guided optimization, provide additional insight. Exploring documentation for compilers such as GCC, Clang, and MSVC can give precise details about optimization strategies applied by particular systems.
