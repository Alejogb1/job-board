---
title: "How can compiled code size be reduced through refactoring?"
date: "2025-01-30"
id: "how-can-compiled-code-size-be-reduced-through"
---
Excessively large compiled code often stems from redundant computations, bloated data structures, or the inclusion of unnecessary libraries. Reducing this footprint requires a deliberate approach, focusing on optimizing both algorithmic efficiency and resource utilization during development. My experience across several large-scale embedded systems has consistently shown that refactoring plays a pivotal role in achieving more compact binaries.

I've found that code size reduction, beyond simple compiler optimizations, frequently necessitates reevaluating the fundamental structure and implementation of the software. This process isn’t just about removing dead code, although that's a basic step. It’s more about architecting the code such that it generates less machine code to begin with, while maintaining the desired functionality and, crucially, not introducing performance penalties. This often entails identifying patterns which, while seemingly functional, contribute to bloat due to repeated implementation or inefficient memory access patterns.

Here, I’ll outline several common strategies and provide examples that I have found effective in reducing code size via refactoring. These are not exhaustive but represent some of the most frequently encountered and successfully addressed situations.

**1. Function Duplication and Abstraction:**

A common source of code bloat is the presence of similar code sequences scattered throughout a project. Often, these sequences differ slightly but perform a fundamentally similar operation. Instead of repeating near-identical code blocks, I've found it exceptionally beneficial to abstract these into reusable functions. This minimizes redundant machine code by centralizing functionality.

**Example 1: Redundant Array Processing**

Suppose in a hypothetical audio processing project, I initially had the following snippets for adjusting gain on different audio buffers:

```c
// Initial inefficient code:
void adjustGainBuffer1(float buffer[], int size, float gain) {
  for (int i = 0; i < size; i++) {
    buffer[i] = buffer[i] * gain;
  }
}

void adjustGainBuffer2(float buffer[], int size, float gain) {
   for (int i = 0; i < size; i++) {
    buffer[i] = buffer[i] * gain;
  }
}
// ... several other very similar functions...
```

The above pattern, common in quickly developed codebases, creates a series of nearly identical functions, increasing binary size. The straightforward refactor is to extract the core gain adjustment into a generic function:

```c
// Refactored code:
void adjustGain(float buffer[], int size, float gain) {
    for (int i = 0; i < size; i++) {
        buffer[i] = buffer[i] * gain;
    }
}

void adjustGainBuffer1(float buffer[], int size, float gain) {
  adjustGain(buffer,size,gain);
}

void adjustGainBuffer2(float buffer[], int size, float gain) {
  adjustGain(buffer,size,gain);
}

// Other areas of code that use the old pattern now use the generic 'adjustGain' function

```

By implementing the `adjustGain` function, the repetitive multiplication logic is centralized. Each call to `adjustGainBuffer1`, `adjustGainBuffer2` (and others that can now be refactored) simply calls the generic `adjustGain` function, eliminating the redundant machine code, and substantially reducing the compiled binary’s size. The cost of function call overhead is minimal compared to the duplicated loop logic.

**2. Data Representation Optimization:**

Another major contributor to code size arises from inefficient data structures. This includes using overly general data types where more specific types could suffice or utilizing data structures that consume excess memory. Refactoring to utilize the minimum data size necessary reduces the instructions required to manipulate that data and lowers the demand on RAM, which can also indirectly lead to reduced code size due to less complicated memory management instructions.

**Example 2: Efficient Enumeration**

Let’s assume I initially used integers to represent the possible states of a device. This required 32 bits per state, though only a few states existed:

```c
//Initial inefficient enumeration

#define STATE_OFF 0
#define STATE_IDLE 1
#define STATE_RUNNING 2
#define STATE_ERROR 3

int deviceState = STATE_OFF;
```

While this works, it occupies a full integer, despite the fact only values 0-3 need to be represented. I typically refactor this with a more optimal approach that utilizes an enumeration, which automatically uses the minimal storage required for the values:

```c
// Refactored efficient enumeration
typedef enum {
  STATE_OFF,
  STATE_IDLE,
  STATE_RUNNING,
  STATE_ERROR
} DeviceState;

DeviceState deviceState = STATE_OFF;
```

Using an enumeration often results in the compiler using an 8-bit type under the hood (on many architectures). The size reduction isn’t always large, but when you have numerous such states throughout a project, the overall size reduction can be significant. More importantly, it encourages using correct data representations early in the project, preventing bloated designs from propagating throughout the project.

**3. Conditional Compilation and Code Stripping:**

Often, software may contain features that are required only in specific build configurations, such as debugging code or features only needed in particular hardware variants. Retaining this code in all builds unnecessarily increases the compiled binary's size. I utilize conditional compilation directives, most commonly preprocessor `#ifdef` blocks, along with code stripping techniques, to exclude unnecessary code.

**Example 3: Debug Code Exclusion**

Suppose a debugging print function is used frequently during the software development process:

```c
// Inefficient code containing debug prints
void processData(int data) {
  printf("Debug: Received data value %d\n", data); // Debug print
  //... other data processing logic...
}
```

While useful for development, these print statements add to the overall code size. To address this I would use conditional compilation:

```c
// Refactored code using conditional compilation

#ifdef DEBUG
  #define DEBUG_PRINT(fmt, ...) printf("Debug: " fmt, ##__VA_ARGS__);
#else
  #define DEBUG_PRINT(fmt, ...)
#endif

void processData(int data) {
    DEBUG_PRINT("Received data value %d\n", data);
    //...other data processing logic...
}

```

During development, I would compile the code with `DEBUG` defined. In production, `DEBUG` would not be defined, and the compiler effectively ignores all instances of `DEBUG_PRINT` due to its empty definition, preventing debug logic from making it into the final executable, significantly reducing code size. This can apply to entire functions as well, by encapsulating large blocks within `#ifdef DEBUG / #endif` directives.

**Further Considerations:**

Beyond these specific examples, other strategies can be employed for code size reduction. These include:

* **Linker Optimization:** Using linker flags to eliminate unused symbols and sections can reduce the final executable size. This is especially useful with large libraries.

* **Compiler Flags:** Using appropriate optimization flags (such as `-Os`) instructs the compiler to optimize for size, rather than speed. However, care must be taken when enabling certain optimization levels, as they can impact debug-ability.

* **Library Selection:** Choosing lighter libraries that perform specific tasks can be beneficial, as a more complex library may contain functions that are never used in the project.

* **Code Profiling:** Using code profilers can identify the areas of code that contribute most to the binary size. This can help focus refactoring efforts.

**Resource Recommendations:**

For further exploration of this topic I suggest: "Refactoring: Improving the Design of Existing Code" by Martin Fowler, although it’s not targeted at code size specifically it offers excellent design considerations that promote good practices which translate to smaller code size. Furthermore, many compiler documentation sets include details on optimization flags for binary reduction. It’s also worth looking at resources explaining linker behavior and how that impacts the final output size, which varies by compiler and toolchain. Finally, studying platform specific assembly outputs for target architectures can be useful in seeing the exact result of source changes.

In conclusion, refactoring for reduced code size is a crucial aspect of efficient software development. While compiler optimization is important, it’s the architecture and design choices that offer the most substantial gains. By focusing on eliminating redundancy through function abstraction, optimizing data representations, and strategically removing unnecessary code blocks, I’ve consistently been able to produce smaller and more efficient executables. The code examples provide just a few concrete illustrations of how I’ve applied this in real projects.
