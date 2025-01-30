---
title: "Why can't TensorFlow find the `stats_calculator.cc` file?"
date: "2025-01-30"
id: "why-cant-tensorflow-find-the-statscalculatorcc-file"
---
The inability of TensorFlow to locate `stats_calculator.cc` during a build process, particularly when working with custom operations, often stems from a mismatch between the include paths specified in the build system (typically Bazel) and the actual physical location of the file within the project’s directory structure. This frequently occurs when developers adapt pre-existing TensorFlow code or create new operators without fully grasping the nuances of Bazel configuration and dependency management. I’ve personally encountered this issue multiple times while developing specialized image processing layers, specifically when incorporating statistical calculations directly into the TensorFlow graph.

The TensorFlow build system, heavily reliant on Bazel, uses a sophisticated mechanism for identifying and incorporating source files and headers during the compilation phase. This system employs the concept of “targets,” which are buildable units, and “rules,” which define how to build those targets. When you create a new custom TensorFlow operator that utilizes a C++ implementation (e.g., `stats_calculator.cc`), you must meticulously configure Bazel to understand where to find this source file and its associated headers. Errors such as “Cannot find `stats_calculator.cc`” typically indicate that the `cc_library` or `tf_custom_op_library` rule in your `BUILD` file does not accurately specify the path to this file, or that necessary dependencies are not correctly declared.

To elaborate, Bazel uses relative paths from the location of the `BUILD` file, not absolute paths on your filesystem. Incorrect paths within the `srcs` attribute of a `cc_library` rule are a major source of these errors. Further complexity can arise from differing configurations for include paths within the `copts` attribute or when dealing with nested subdirectories. In addition, implicit dependencies within the TensorFlow build graph can require specific include directories, and forgetting to declare these can lead to the compiler failing to locate the required headers.

Here are a few scenarios, represented through simplified code examples, to illustrate common causes:

**Example 1: Incorrect `srcs` Path**

Suppose the project structure is like this:

```
project/
├── custom_ops/
│   ├── BUILD
│   └── stats_calculator.cc
└── tensorflow/
    ... (TensorFlow source code) ...
```

And `custom_ops/BUILD` contains:

```python
load("@org_tensorflow//tensorflow:tensorflow.bzl", "tf_custom_op_library")

tf_custom_op_library(
    name = "stats_ops",
    srcs = ["stats_calculator.cc"], # Incorrect relative path
)
```

The `srcs` attribute, when set to `["stats_calculator.cc"]`, assumes that the source file exists in the same directory as the `BUILD` file. This is a very common mistake. Because in this case `stats_calculator.cc` exists under the same directory, and if you are building this custom op from `project/`, Bazel will report that it cannot locate the file. The correct `srcs` path, relative to the `BUILD` file, is `["./stats_calculator.cc"]` and it should be the correct method when the source file is in the same directory.

**Example 2: Missing Include Directory**

Assume this project structure now:

```
project/
├── custom_ops/
│   ├── include/
│   │   └── stats_calculator.h
│   ├── BUILD
│   └── stats_calculator.cc
└── tensorflow/
    ... (TensorFlow source code) ...
```

The `stats_calculator.cc` file contains:

```cpp
#include "stats_calculator.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

// ... Implementation ...
```

And `custom_ops/BUILD` contains:

```python
load("@org_tensorflow//tensorflow:tensorflow.bzl", "tf_custom_op_library")

tf_custom_op_library(
    name = "stats_ops",
    srcs = ["stats_calculator.cc"],
    copts = ["-I./include"], # Incorrect include path
    deps = ["@org_tensorflow//tensorflow/core:framework"]
)
```

While the `srcs` path is correct (relative to `BUILD` file), the `copts` include flag `-I./include` is also relative. So it will instruct the compiler to look for headers under the directory of the `BUILD` file itself. The location `stats_calculator.h` in `custom_ops/include/stats_calculator.h`. This will not work because Bazel compiles source files using an intermediate directory structure that is separate from the project directory. Instead, the include directory should be specified using `-Icustom_ops/include`. The correct path would be `-Icustom_ops/include` for the include path, where we specify a relative path relative to `project` folder. In practice, we should always define the include path in the root directory with something like `-I$(GENDIR)/include`. `GENDIR` is a predefined environment variable that resolves the path to root directory in the Bazel building process.

**Example 3: Implicit Dependency Issue**

Now, let us consider a different situation. Let's assume we are using `Eigen` library, and it is installed globally in our building environment. The structure remains the same as Example 2, and `stats_calculator.cc` is updated to:

```cpp
#include "stats_calculator.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "Eigen/Core"

// ... Implementation ...
```

`custom_ops/BUILD` is updated accordingly as well:

```python
load("@org_tensorflow//tensorflow:tensorflow.bzl", "tf_custom_op_library")

tf_custom_op_library(
    name = "stats_ops",
    srcs = ["stats_calculator.cc"],
    copts = ["-I./include"],
    deps = ["@org_tensorflow//tensorflow/core:framework",
            "@eigen//:eigen"]
)
```

Despite including `-I./include` for our header file and explicitly declaring the TensorFlow and Eigen dependencies, if you get `stats_calculator.cc` not found, it is because there is some other intermediate dependency required for building `stats_calculator.cc`. In some cases, even if we declare our header dependency correctly and provide the correct include path, if other intermediate libraries used by `stats_calculator.cc` are not declared in the `deps`, compiler will still fail to find the source file. This is because it fails to resolve the intermediate dependencies and cannot create the correct include paths for the compiler during the build. Therefore, you need to double check all other libraries it uses to make sure they are all specified in `deps`. This is a difficult problem to debug as there is no clear error message for missing intermediate dependencies.

**Resolution Strategy**

Based on my experience, to resolve these issues effectively, it is essential to approach debugging systematically:

1.  **Verify `srcs` paths:** Always double-check that the paths specified in the `srcs` attribute of your `cc_library` or `tf_custom_op_library` rules are relative to the location of the `BUILD` file itself and correctly reference the source files. This is usually the first point of failure.

2.  **Inspect `copts` include directories:** Ensure that the `-I` flags within the `copts` attribute of your rules point to the correct directories containing header files, specifically relative to the workspace root. Use the command line tool `bazel query` to see what include path Bazel uses during the building process, especially when using external libraries.

3.  **Declare all Dependencies:** Carefully declare all the dependencies, including intermediate libraries used by the source file, in `deps` attribute. When using external libraries, make sure the external dependencies are configured correctly by defining them in the `WORKSPACE` file, or they are pre-installed in the build environment and included in the `copts` flags.

4.  **Use a consistent project structure:** Maintain a clear, well-defined project structure where custom operations and their corresponding source files and include directories are organized logically and consistently, which helps reduce the complexity.

**Resource Recommendations**

To deepen your understanding of the TensorFlow build system and Bazel, I recommend consulting the following resources:

*   **TensorFlow Official Documentation:** The TensorFlow official documentation provides comprehensive information about building custom operations and using Bazel. Specific documentation related to the build system and `tf_custom_op_library` is invaluable.
*   **Bazel User Guide:** The official Bazel documentation details concepts like targets, rules, dependencies, and build configuration, which are fundamental for understanding how Bazel operates.
*   **TensorFlow Source Code:** Examining the TensorFlow source code, especially the `BUILD` files in the `tensorflow/core` and `tensorflow/ops` directories, provides practical examples of how to structure custom operations and define Bazel rules effectively.

In conclusion, the "Cannot find `stats_calculator.cc`" error highlights the importance of understanding the Bazel build system and correctly configuring paths to source files, include directories and dependencies. Employing a meticulous, systematic debugging process and consulting the resources above will dramatically improve the ability to avoid and resolve build problems.
