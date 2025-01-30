---
title: "How long does linking take with GCC/G++ and LD?"
date: "2025-01-30"
id: "how-long-does-linking-take-with-gccg-and"
---
The time spent linking with GCC/G++ and LD is highly variable, frequently overshadowing compilation time for larger projects, and it's not uncommon to see link times range from a few seconds to several minutes, even hours. This variation is primarily due to project complexity, the size of the object files being processed, and the presence of external dependencies. The actual linking process involves resolving symbols across various object files and libraries, which can be computationally intensive and disk I/O bound depending on the chosen strategies. I’ve personally observed this phenomenon countless times, especially when working with large C++ simulation frameworks.

The linker, specifically `ld` (part of the GNU binutils package), is responsible for combining the compiled object files (.o files) into a final executable or a shared library. This entails a significant amount of work beyond simply concatenating file contents. This work can be summarized in three primary phases: symbol resolution, relocation, and output generation.

Symbol resolution involves identifying where each symbolic reference (e.g., a function call or a global variable access) is actually defined across all input object files. When an object file mentions a function `foo`, the linker needs to locate the actual machine code for `foo` among all other object files and libraries. This can involve scanning symbol tables within each file and dealing with variations in symbol decoration or mangling across different architectures. Undefined symbols, those referenced but not found anywhere, result in linking errors. For shared libraries, the linker might also perform lazy symbol binding, deferring the symbol resolution until the function is actually used during runtime. The search order across directories and provided libraries also drastically influences the duration of symbol resolution.

Relocation occurs after symbol resolution. At this stage, the linker updates the addresses within machine code instructions to point to the correct locations in the final executable. Object files generally don’t contain absolute addresses; they use relative offsets or placeholder addresses. These must be updated to the final loaded location within memory. Relocation can be particularly expensive if numerous memory segments and relocations are involved. This becomes even more involved with position-independent code (PIC), where the code must operate correctly regardless of its base address.

Finally, the linker generates the output executable file or shared library. This involves assembling all the resolved code and data sections into the final output format and writing these to disk. The output format varies, for example, ELF format for Linux systems. Different output formats have varying complexity, and writing large executable files is directly correlated to time spent linking.

Furthermore, the static linking versus dynamic linking significantly affects the total link time. Static linking, where all necessary libraries are merged into the final executable, tends to increase the overall linking duration, especially if numerous or large libraries are involved. Dynamic linking, on the other hand, generates smaller executables since it relies on shared libraries loaded at runtime. However, the dynamic linker still needs time to resolve these dependencies at program startup, which adds to the application’s overall load time. However, this occurs at runtime, not during the linking phase.

Below are three code examples demonstrating various scenarios that affect linking time. Each example has a brief explanation, which relates directly to the described linking stages.

**Example 1: Minimal Project with Static Linking**

```c++
// main.cpp
#include <iostream>

int main() {
    std::cout << "Hello, world!" << std::endl;
    return 0;
}
```

Compilation Command: `g++ main.cpp -o program`

Commentary: This example compiles and links quickly. The only library dependency is the C++ standard library, and this is usually statically linked by default. Symbol resolution is straightforward; the `std::cout` symbol is found within the system's standard C++ library. Relocation is minimal, as the standard library is typically compiled to allow for efficient reuse. The output generation is also small, leading to a quick build time.

**Example 2: Project with Multiple Source Files and Function Calls**

```c++
// math.h
int add(int a, int b);

// math.cpp
#include "math.h"

int add(int a, int b) {
    return a + b;
}

// main.cpp
#include <iostream>
#include "math.h"

int main() {
    int sum = add(5, 3);
    std::cout << "Sum: " << sum << std::endl;
    return 0;
}
```

Compilation Commands:
```bash
g++ -c math.cpp -o math.o
g++ -c main.cpp -o main.o
g++ math.o main.o -o program
```

Commentary: This case illustrates the linker's work in resolving symbols across multiple object files. When `main.cpp` calls `add`, the linker needs to locate the definition of `add` within `math.o`. It traverses the symbol table of each object file until it resolves the call. Relocation is now more complex as the address of the `add` function needs to be embedded into `main.o`. This takes significantly longer than the previous case because of the extra steps involved in resolving external functions across different object files and updating call instructions with final address locations.

**Example 3: Project with External Static Library**

```c++
// Assuming a static library libmymath.a containing add and subtract functions is present.
// main.cpp
#include <iostream>
#include "mymath.h"

int main() {
    int sum = add(10, 5);
    int difference = subtract(10, 5);
    std::cout << "Sum: " << sum << ", Difference: " << difference << std::endl;
    return 0;
}
```

Compilation Command: `g++ main.cpp -o program -L. -lmymath`

Commentary: This case introduces an external static library, `libmymath.a`. The linker now needs to search for symbols, `add` and `subtract`, not just in the provided object files but also within this static library archive. The `-L.` specifies the directory where the linker looks for libraries, while `-lmymath` links with the library file `libmymath.a`. If the library is large, with numerous functions and symbols, the process of searching for and linking those required symbols can be time consuming. Moreover, if `libmymath.a` contains numerous other unused symbols, they will still be included, contributing to increased overall executable size and potentially longer output generation times.

To mitigate long link times, several techniques are commonly used:

* **Use of Dynamic Libraries:** For large projects, shared libraries offer significant advantages. They reduce the final executable size and can be updated without requiring a complete relinking of the application. The linker performs less work at link time, deferring some to program startup.
* **Incremental Linking:** Some build systems support incremental linking, which relinks only the parts of the application that have changed, rather than relinking the entire project each time. This can drastically reduce linking time.
* **Link Time Optimization (LTO):** LTO optimizes code across object file boundaries during the link stage. While LTO can significantly improve the performance of the final executable, it also increases the total link time because of the additional analysis and optimization that must be performed. This technique may cause the linker to use more CPU time and RAM.
* **Linker Flags and Options:** Understanding specific linker flags, such as `-ffunction-sections` and `-Wl,--gc-sections`, which allow unused code sections to be eliminated, can optimize final binary size and indirectly the linking duration. Also understanding the order in which you pass in libraries, as the linker resolves them in order.

For further research into this topic, I suggest exploring the following resources:

* **The GNU LD documentation**: Specifically its section on linking and command-line options. Understanding the various options can directly impact build time.
* **Material on object file formats**: Resources explaining the structure of ELF files and how they are used by the linker offer a detailed perspective on the complexities involved.
* **Documentation related to build systems**: Learning more about how build systems like CMake or Make can be configured to optimize link times for different platforms is valuable.
* **Research on symbol resolution and relocation techniques**: Deeply understanding how symbols are managed and addresses are calculated will provide greater insights.
* **Books and articles on compiler design and implementation**: They frequently include sections on linking, often providing the most technical details about the linking process.
