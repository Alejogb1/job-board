---
title: "Why did the linker fail with undefined dynamic_lookup error in clang?"
date: "2025-01-26"
id: "why-did-the-linker-fail-with-undefined-dynamiclookup-error-in-clang"
---

The `undefined symbol` error during linking, specifically `dynamic_lookup`, in Clang, often stems from a misunderstanding of how dynamic linking interacts with template instantiation and visibility when utilizing C++'s dynamic loading features. This issue is not about traditional, static linkage failures, but rather about a lookup failure during runtime caused by a lack of exported symbol visibility from a dynamically loaded shared library. I've frequently encountered this in scenarios involving plugins or modular architectures, where a core application attempts to call a templated function defined within a plugin loaded at runtime.

The root cause is that while a template function might be defined in a shared library (.so or .dylib), it's not instantiated until it's *used* by a translation unit (compiled file). The compiler generates code only where the template is explicitly instantiated with concrete types, and typically only when that code generation occurs in the main application or when the template definition and its usage are in the same compilation unit. If the main application tries to call a templated function from the plugin, and there was *no explicit template instantiation* in any translation unit of the plugin, no corresponding symbol is generated in the shared library for the dynamic loader to find. The dynamic linker then cannot perform the 'dynamic lookup' – it cannot find the code it's supposed to execute.

This becomes especially tricky with dynamic loading because the application loads the library at runtime and the compiler’s static visibility rules do not apply in quite the same way. The library’s symbols, while potentially present, might not be visible for dynamic lookup if they haven’t been explicitly instantiated and exported. This contrasts with statically linked code, where all needed instantiations are typically resolved during the compile/link process.

Furthermore, the linker error is reported as `dynamic_lookup` rather than `undefined symbol`, because dynamic loading involves the application specifically requesting a function by name through an OS function call (like `dlsym` on Linux/macOS, or `GetProcAddress` on Windows). The lookup fails at this *runtime* request, hence the ‘dynamic’ nature of the error. The problem isn't that no symbol exists whatsoever, but rather that no *exported* symbol exists for the dynamic loader to retrieve.

Let’s examine some code examples to solidify this understanding.

**Example 1: Basic Templated Function in a Plugin**

Assume we have a basic templated function.

```cpp
// plugin.hpp (part of the plugin shared library)
#pragma once

template <typename T>
T add(T a, T b);
```
```cpp
// plugin.cpp (part of the plugin shared library)

#include "plugin.hpp"

template <typename T>
T add(T a, T b) { return a + b; }
```
```cpp
// main.cpp (main application)
#include <iostream>
#include <dlfcn.h>

int main() {
  void *handle = dlopen("./libplugin.so", RTLD_LAZY);
  if (!handle) {
    std::cerr << "Cannot open library: " << dlerror() << '\n';
    return 1;
  }

  typedef int (*AddInt)(int, int);
  AddInt addIntFunc = (AddInt)dlsym(handle, "add<int>");
    
  if (!addIntFunc) {
     std::cerr << "Cannot find symbol add<int>: " << dlerror() << '\n';
     dlclose(handle);
     return 1;
  }


  std::cout << "Result: " << addIntFunc(5, 3) << std::endl;

  dlclose(handle);
  return 0;
}
```
Compiling and running this without explicitly instantiating `add<int>` in the plugin will result in the `dynamic_lookup` error. The template is defined in `plugin.cpp`, but not instantiated for any specific type, which means no compiled code for `add<int>` is generated within the plugin's shared library. Therefore, when the main application uses `dlsym` to find the symbol `add<int>`, it cannot be found. The specific message might vary slightly depending on the system (e.g., "`dlsym(handle, "add<int>"): symbol not found`").

**Example 2: Explicit Template Instantiation in the Plugin**

To fix the previous issue, we explicitly instantiate the template in the plugin's source code.

```cpp
// plugin.cpp (modified)
#include "plugin.hpp"

template <typename T>
T add(T a, T b) { return a + b; }

// Explicit instantiation of add<int>
template int add<int>(int a, int b);
```
By adding `template int add<int>(int a, int b);`, we explicitly instruct the compiler to generate the code for `add<int>`. With this change, the shared library will now contain the compiled machine code for this specific instantiation. The application can now locate it via `dlsym`. No other change is required in the main app.

**Example 3: Using a Wrapper Function (more practical for complex templates)**

Explicit template instantiations can become cumbersome with many template parameters. An alternative is using a wrapper function that acts as a concrete interface to the templated function:

```cpp
// plugin.hpp (modified)
#pragma once

template <typename T>
T add(T a, T b);

int addIntWrapper(int a, int b); // wrapper
```
```cpp
// plugin.cpp (modified)
#include "plugin.hpp"

template <typename T>
T add(T a, T b) { return a + b; }


int addIntWrapper(int a, int b){
    return add<int>(a, b);
}
```
```cpp
// main.cpp (modified)
#include <iostream>
#include <dlfcn.h>

int main() {
  void *handle = dlopen("./libplugin.so", RTLD_LAZY);
  if (!handle) {
    std::cerr << "Cannot open library: " << dlerror() << '\n';
    return 1;
  }
   
  typedef int (*AddIntWrapper)(int, int);
  AddIntWrapper addIntFunc = (AddIntWrapper)dlsym(handle, "addIntWrapper");

    if (!addIntFunc) {
        std::cerr << "Cannot find symbol addIntWrapper: " << dlerror() << '\n';
        dlclose(handle);
        return 1;
    }


  std::cout << "Result: " << addIntFunc(5, 3) << std::endl;

  dlclose(handle);
  return 0;
}
```
Here, the main application now searches for `addIntWrapper`.  The template `add<int>` is implicitly instantiated because the wrapper calls it internally within the `plugin.cpp` file, ensuring the instantiation happens *within* the plugin's compilation unit. This avoids the need to directly instantiate the template in `plugin.cpp`.  This technique is usually better for complex types or template parameter lists.

In summary, a `dynamic_lookup` error with Clang during dynamic loading stems from a failure to generate and export the necessary code for a templated function in a shared library. The dynamic linker expects a concrete symbol, but a template alone does not inherently provide it; only instantiations do. You can address this through explicit template instantiations or, for more maintainability, through a concrete wrapper function. These methods ensure that the necessary code is generated within the shared library so that it can be discovered by the main application’s `dlsym` call at runtime.

For further exploration, I would recommend exploring the following resources:
*   C++ Template Metaprogramming textbooks. Understanding template instantiation rules deeply aids in preventing these types of problems.
*   Operating system documentation regarding shared library loading (e.g., `dlopen`, `dlsym` on POSIX systems or corresponding Windows APIs). This will provide a clearer picture of how dynamic linking works at a lower level.
*   The Clang compiler documentation, especially the sections on template instantiation and symbol visibility. This will detail how the compiler handles these scenarios and related compiler flags.
*   Platform-specific documentation regarding symbol export and import from shared libraries. This can vary on Linux, macOS, and Windows.
