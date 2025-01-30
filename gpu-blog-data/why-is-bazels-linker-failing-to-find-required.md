---
title: "Why is Bazel's linker failing to find required symbols?"
date: "2025-01-30"
id: "why-is-bazels-linker-failing-to-find-required"
---
During my tenure leading a team developing a complex service framework at Megacorp, we frequently encountered Bazel linker failures, particularly when migrating towards more modular architectures and utilizing shared libraries. These failures, often manifested as "undefined symbol" errors during the linking phase, typically stem from a mismatch between how Bazel manages dependency visibility and the actual runtime needs of the application, especially in situations involving dynamic linking. This response details the typical causes behind these errors, with code examples demonstrating effective resolutions.

The primary reason a Bazel linker might fail to find required symbols is that the build graph, meticulously constructed by Bazel based on `BUILD` files, doesn't fully capture the dynamic linkage requirements. While Bazel excels at tracking compile-time dependencies, dynamic linking introduces a wrinkle. At link time, the linker must be explicitly told where to find the symbols – typically, within shared libraries (.so on Linux, .dylib on macOS, .dll on Windows). If these libraries, or the symbols they contain, aren't correctly exposed to the linking process, failures occur. It is crucial to understand that Bazel's dependency system dictates *compile-time* visibility; linker visibility is a separate consideration.

Bazel achieves symbol resolution through a combination of:

1.  **Linking Libraries:** Using the `deps` attribute in `cc_binary`, `cc_library`, or other relevant rules, Bazel passes necessary `.a` (static libraries) or `.so` (shared libraries) files to the linker. However, simply listing these files isn't sufficient.
2.  **Linkflags:**  The `linkopts` attribute allows for specifying arbitrary flags to the linker, including options for library search paths (`-L`), library names (`-l`), and other linker-specific directives.
3.  **Symbol Visibility:**  Bazel needs to be informed when a particular library needs its symbols explicitly exported, beyond just being available for compile-time inclusion. This often involves configuring visibility attributes, particularly `linkstatic=0` which enables shared library usage.

Let's consider a hypothetical scenario within our Megacorp project. Imagine we have a utility library `//util/string_utils` that offers string manipulation routines and a service `//service/my_service` that uses this library.

**Example 1: Implicit Dependency Failure**

Initially, our `//service/my_service/BUILD` file looked like this:

```python
cc_binary(
    name = "my_service",
    srcs = ["my_service.cc"],
    deps = ["//util/string_utils"],
)
```

and `//util/string_utils/BUILD` file:

```python
cc_library(
  name = "string_utils",
  srcs = ["string_utils.cc"],
  hdrs = ["string_utils.h"],
)
```

`my_service.cc` included `string_utils.h` and made calls to functions defined within `string_utils.cc`. This setup worked fine when initially implemented. However, as we transitioned to creating shared libraries, this simple approach began failing during linking. The issue arose when `//util/string_utils` started producing a shared library. Even though the header was included during compilation and compilation succeeded, the linker didn’t know it was supposed to be looking for the implementation within a shared library. This resulted in `undefined symbol` errors related to the functions in `string_utils.cc`.

The fix involved explicitly telling Bazel we wanted a shared library. We revised our `//util/string_utils/BUILD` file to include:

```python
cc_library(
  name = "string_utils",
  srcs = ["string_utils.cc"],
  hdrs = ["string_utils.h"],
  linkstatic = 0, # Tells Bazel to create a shared library
)
```

While this made `string_utils` link dynamically, our original `//service/my_service/BUILD` file remained problematic. The `deps` attribute still only specified a compile-time dependency, not a dynamic linking dependency. We need to explicitly instruct the linker to link against the output shared object of `string_utils`. To do this, we added a `runtime_deps` attribute which explicitly tells Bazel that `//service/my_service` needs to link at runtime against `//util/string_utils`. The updated `//service/my_service/BUILD` file looks like this:

```python
cc_binary(
    name = "my_service",
    srcs = ["my_service.cc"],
    deps = ["//util/string_utils"], # still a compile-time dependency
    runtime_deps = ["//util/string_utils"], # now explicitly a runtime dependency
)
```

The addition of the `runtime_deps` attributes was crucial for ensuring correct symbol resolution at runtime, as it instructed the linker to link against the output of `string_utils`.

**Example 2: Incorrect Library Search Paths**

Another common cause of linking errors is an incorrect library search path. Let's say our `//service/my_service` relied on a third-party library, `libexternal.so`, located in a directory not known by default to the linker. Our initial `BUILD` file might look like:

```python
cc_binary(
  name = "my_service",
  srcs = ["my_service.cc"],
  deps = ["//util/string_utils"],
  runtime_deps = ["//util/string_utils"],
  linkopts = ["-lexternal"], # Incorrect, relying on global linker paths
)
```

If `libexternal.so` was not located within a path that was already part of the linker's default search paths, Bazel would generate a binary which couldn't find the external library at runtime. While the compiler knew of the library's existence during compile time (perhaps using an include path or explicit build setting), the linker, running separately, was unaware of its location.

We fixed this by specifying the full path to the library using `-L` and then the library name using `-l`:

```python
cc_binary(
    name = "my_service",
    srcs = ["my_service.cc"],
    deps = ["//util/string_utils"],
    runtime_deps = ["//util/string_utils"],
    linkopts = ["-L/path/to/external/lib", "-lexternal"],
)
```

Here, `-L/path/to/external/lib` added `/path/to/external/lib` to the linker's search path, enabling the linker to find `libexternal.so` when linking. While this solves the problem, hardcoding the absolute path was not ideal for portability. A more robust approach would involve passing the path as a build variable or by defining an alias.

**Example 3: Symbol Visibility Issues in Shared Libraries**

In complex projects, symbols might be present in shared libraries but still not be visible to the linker due to compiler optimizations or export directives. Assume that in `//util/string_utils`, the function `trimString` was accidentally marked as `static`, thus not exported from the shared library.

```cpp
// string_utils.cc
static std::string trimString(const std::string& input) { // Should be non-static
 // ... implementation ...
}
```

In this situation, even with the shared library properly linked, the linker would complain because `trimString` is not an exported symbol. The correction would involve removing the static keyword, or alternatively using compiler attributes to force visibility:

```cpp
// string_utils.cc
std::string trimString(const std::string& input) { // now non-static
    // ... implementation ...
}
```

or, depending on compiler,

```cpp
// string_utils.cc
__attribute__ ((visibility ("default"))) std::string trimString(const std::string& input) {
    // ... implementation ...
}
```

These scenarios, encountered during my time at Megacorp, underline that diagnosing linker errors in Bazel goes beyond simple dependency declarations. A firm understanding of the interplay between Bazel build artifacts, linker behavior, and shared library principles is crucial.

For further information on this topic, I recommend exploring the official Bazel documentation concerning `cc_library` and `cc_binary` rules, particularly focusing on `deps`, `runtime_deps`, `linkopts`, and visibility. In addition, consulting compiler documentation (GCC, Clang, MSVC) regarding symbol visibility modifiers will help diagnose similar issues. Understanding compiler specific options like `-fvisibility=hidden` can be critical. Furthermore, the linker documentation (e.g., the GNU linker's man page) provides valuable insights into linker behavior and its command-line interface.
