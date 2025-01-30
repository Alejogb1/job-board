---
title: "How do I resolve the 'tensorflow/core/framework/common_shape_fns.h: No such file or directory' error when creating a custom TensorFlow op?"
date: "2025-01-30"
id: "how-do-i-resolve-the-tensorflowcoreframeworkcommonshapefnsh-no-such"
---
The "tensorflow/core/framework/common_shape_fns.h: No such file or directory" error typically arises when building a custom TensorFlow operation using the TensorFlow C++ API. It signals that the compiler cannot locate a critical header file responsible for defining shape inference functions. This header, `common_shape_fns.h`, is not part of the publicly exposed TensorFlow include directories, requiring specific steps to incorporate it during the build process.

The core issue isn't that the file is missing from the TensorFlow source, but that the build system, particularly when using `tf_custom_op_library` in Bazel, needs explicit instructions on how to locate internal TensorFlow headers. I've encountered this exact problem multiple times while developing custom operations, notably when creating a highly optimized graph convolution operator, and the solution usually involves adjusting build configurations rather than directly modifying TensorFlow source files. The discrepancy stems from the design choice to delineate between public and internal APIs within TensorFlow.

To resolve this error effectively, it's crucial to understand that your build system must be informed where to find TensorFlow's internal headers and libraries. Let's dissect what typically goes wrong. The naive approach, which often leads to this error, is to assume that TensorFlow's public include paths are sufficient. However, many of the fundamental pieces for constructing custom operations, particularly relating to shape manipulation, reside in internal directories like `tensorflow/core`.

The resolution revolves primarily around the use of Bazel, TensorFlow's build system, with slight variations needed when other build tools are employed. When crafting a custom operation using `tf_custom_op_library`, Bazel needs explicit directives to locate the necessary internal headers. This is accomplished through configuring include paths in the `BUILD` file alongside declarations for linking the relevant TensorFlow libraries. Neglecting this aspect is the major cause of this error, as I have experienced first hand numerous times when introducing newly optimized matrix multiplication kernels.

Here is the first practical example, a skeletal `BUILD` file that illustrates the core concept.

```python
# BUILD file
load("@org_tensorflow//tensorflow:tensorflow.bzl", "tf_custom_op_library")

tf_custom_op_library(
    name = "my_custom_op",
    srcs = ["my_custom_op.cc"],
    visibility = ["//visibility:public"],
    deps = [
        "@org_tensorflow//tensorflow:framework",
        "@org_tensorflow//tensorflow:lib_headers",
    ],
    copts = [
        "-I",
        "$(@org_tensorflow//tensorflow:includes)",
    ],
)

```

Here's a breakdown of the provided `BUILD` file snippet:

*   `load("@org_tensorflow//tensorflow:tensorflow.bzl", "tf_custom_op_library")`: This line imports the `tf_custom_op_library` rule from TensorFlow's Bazel build definitions. It's what allows us to easily define our custom TensorFlow operations with the Bazel build system.
*   `name = "my_custom_op"`: This specifies the name of our Bazel target for the custom operation. We will use this to build our custom op in our shell using `bazel build`.
*   `srcs = ["my_custom_op.cc"]`: This dictates the source file containing the implementation of our custom operation. It is crucial to ensure that this file has the required implementations for register custom operations.
*    `visibility = ["//visibility:public"]`: This sets the visibility level for the target. Declaring it public allows other parts of our codebase to link and use it.
*   `deps = [...]`: The crucial dependency list specifies that our code depends on TensorFlow's core framework, found at `//tensorflow:framework`,  as well as TensorFlow lib headers found at `//tensorflow:lib_headers`. Failing to include the correct framework headers will cause issues during compilation and runtime.
*   `copts = ["-I", "$(@org_tensorflow//tensorflow:includes)"]`: This is the pivotal line. It adds an include path to the compiler's search directory. The variable  `$(@org_tensorflow//tensorflow:includes)` dynamically resolves to the correct TensorFlow include directory, effectively allowing the compiler to find internal header files, including `common_shape_fns.h`. The `-I` flag informs the C++ compiler about an additional header file location.

This foundational example addresses the common problem of lacking internal include paths. Without this `copts` entry, the compiler simply cannot resolve the path to `common_shape_fns.h`. My experience, particularly when implementing custom activation functions, reinforces the necessity of this seemingly minor detail.

Often, the need for specialized implementations require inclusion of additional libraries. A more involved example might need to include support for `eigen3` or more specific CPU related optimisations. Consider this extension of the previous example:

```python
# BUILD file
load("@org_tensorflow//tensorflow:tensorflow.bzl", "tf_custom_op_library")

tf_custom_op_library(
    name = "my_advanced_custom_op",
    srcs = ["my_advanced_custom_op.cc"],
    visibility = ["//visibility:public"],
    deps = [
        "@org_tensorflow//tensorflow:framework",
        "@org_tensorflow//tensorflow:lib_headers",
        "@org_tensorflow//third_party:eigen3",
    ],
    copts = [
        "-I",
        "$(@org_tensorflow//tensorflow:includes)",
    ],
    alwayslink = 1,
)
```

Here we have introduced the concept of a third party dependency to our custom operation.

*   `deps = [...]`: We have extended the dependencies to include Eigen3 with  `@org_tensorflow//third_party:eigen3`. This is a matrix manipulation library commonly used to provide speed-ups in many custom operations.
*  `alwayslink = 1`: This parameter ensures that this library is always linked in when building the target. This can sometimes resolve linking issues when it's hard to track down which library is the source of the error.

Lastly, it might be necessary to add dependencies to the specific TensorFlow libraries you need during compilation. Consider an example that makes use of more specialised TensorFlow functionality:

```python
# BUILD file
load("@org_tensorflow//tensorflow:tensorflow.bzl", "tf_custom_op_library")

tf_custom_op_library(
    name = "my_specialized_op",
    srcs = ["my_specialized_op.cc"],
    visibility = ["//visibility:public"],
    deps = [
       "@org_tensorflow//tensorflow:framework",
        "@org_tensorflow//tensorflow:lib_headers",
        "@org_tensorflow//tensorflow/core:core_cpu",
    ],
    copts = [
        "-I",
        "$(@org_tensorflow//tensorflow:includes)",
    ],
)
```

*   `deps = [...]`: We now depend on `core_cpu` which contains functions and resources designed for CPU related computations. This sort of dependency is very specific to the use case of the custom operation.

These variations highlight the importance of knowing exactly what resources are required by your custom operation. Incorrect configurations can lead to complex build errors and a failure to link correctly with TensorFlow.

For further exploration, the official TensorFlow documentation on custom operations provides valuable insight into the various aspects of building, testing and deploying them. In addition, reviewing the Bazel documentation is imperative to understanding how to configure your custom op properly. Understanding the use of `tf_custom_op_library` and dependency resolution is critical to successful custom operation development. In my experience, many of the less obvious errors stem from these types of configuration issues and not the code within the custom op itself. Finally, reviewing the TensorFlow source code, specifically under `tensorflow/core`, allows you to understand where to look for additional resources that you might need.
