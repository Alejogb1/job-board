---
title: "How can Bazel reference binaries from custom toolchains?"
date: "2024-12-16"
id: "how-can-bazel-reference-binaries-from-custom-toolchains"
---

Alright,  I recall a particularly sticky situation involving precisely this issue back when I was optimizing our build pipelines for a multi-platform project at an old company, so I've got some practical experience to draw from. Referencing binaries from custom toolchains in Bazel can initially feel like navigating a labyrinth, but it’s actually quite logical once you understand the underlying mechanisms. It boils down to ensuring that Bazel understands where your toolchain binaries reside and how to use them. The core concept is that a toolchain provides the necessary tools (compilers, linkers, etc.) to build for a particular target architecture, and these tools are themselves often executable binaries. Here's how we can approach it effectively:

First, understand that toolchains in Bazel are defined using `toolchain` rules which link to a `cc_toolchain` (or other language-specific) definition. These toolchain definitions are responsible for specifying where those crucial binaries are. The most common method is to use `path` attributes within the toolchain rule, which are often derived from `select()` statements to handle different architectures. We essentially create a map between build configurations and the correct toolchain binaries.

Now, let’s break down the steps with concrete examples. Assume we’re dealing with a hypothetical compiler we’ll call `my_special_compiler`.

**Example 1: Basic Toolchain Definition**

Let's start with a simple scenario where we have `my_special_compiler` and related binaries in a separate directory. We need to declare this toolchain so Bazel knows about it. Let's say that our compiler is located in `tools/compilers/my_arch/my_special_compiler`, and a linker, `my_special_linker` is in `tools/compilers/my_arch/my_special_linker`.

```python
# tools/build_defs/my_toolchain.bzl

def _my_toolchain_impl(ctx):
    my_arch_compiler = ctx.attr.my_arch_compiler
    my_arch_linker = ctx.attr.my_arch_linker

    toolchain_info = platform_common.ToolchainInfo(
      compiler_path = my_arch_compiler,
      linker_path = my_arch_linker,
     )

    return [toolchain_info]

my_toolchain = rule(
    implementation = _my_toolchain_impl,
    attrs = {
      "my_arch_compiler": attr.string(mandatory=True),
      "my_arch_linker": attr.string(mandatory=True)
    },
)

def my_cc_toolchain(name, cpu, compiler_path, linker_path):
    native.toolchain(
        name = name,
        toolchain_type = "@my_toolchain//:my_toolchain_type",
        exec_compatible_with = ["@platforms//os:linux", "@platforms//cpu:%s" % cpu],
        target_compatible_with = ["@platforms//os:linux", "@platforms//cpu:%s" % cpu],
        toolchain = ":my_toolchain_implementation",
    )
    my_toolchain(
        name = "my_toolchain_implementation",
        my_arch_compiler = compiler_path,
        my_arch_linker = linker_path,
    )

```

```python
# BUILD file

load(":tools/build_defs/my_toolchain.bzl", "my_cc_toolchain")

my_cc_toolchain(
    name = "my_arch_toolchain",
    cpu = "my_arch",
    compiler_path = "//tools/compilers/my_arch:my_special_compiler",
    linker_path = "//tools/compilers/my_arch:my_special_linker",
)

```

In this example, we define a `my_toolchain` rule that contains the necessary information about compiler and linker. The `my_cc_toolchain` macro sets the correct toolchain name, target/exec compatibilities, and maps the rule to the `my_toolchain_implementation`.

**Example 2: Using `select()` for Multiple Architectures**

Now, what happens when you need to support multiple target architectures? Here's where the power of `select()` comes into play. Imagine we need to handle `my_arch_a` and `my_arch_b` architectures. Let’s extend our example:

```python
# tools/build_defs/my_toolchain.bzl (modified)

def _my_toolchain_impl(ctx):
    my_compiler = ctx.attr.compiler_path
    my_linker = ctx.attr.linker_path

    toolchain_info = platform_common.ToolchainInfo(
      compiler_path = my_compiler,
      linker_path = my_linker,
    )
    return [toolchain_info]

my_toolchain = rule(
    implementation = _my_toolchain_impl,
    attrs = {
      "compiler_path": attr.string(mandatory=True),
      "linker_path": attr.string(mandatory=True)
    },
)

def my_cc_toolchain(name, cpu, compiler_paths, linker_paths):
  native.toolchain(
      name = name,
      toolchain_type = "@my_toolchain//:my_toolchain_type",
      exec_compatible_with = ["@platforms//os:linux", "@platforms//cpu:%s" % cpu],
      target_compatible_with = ["@platforms//os:linux", "@platforms//cpu:%s" % cpu],
      toolchain = ":my_toolchain_implementation",
  )
  my_toolchain(
      name = "my_toolchain_implementation",
      compiler_path = select(compiler_paths),
      linker_path = select(linker_paths)
    )
```

```python
# BUILD file (modified)

load(":tools/build_defs/my_toolchain.bzl", "my_cc_toolchain")

my_cc_toolchain(
  name = "my_arch_a_toolchain",
  cpu = "my_arch_a",
  compiler_paths = {
    "//conditions:default": "//tools/compilers/my_arch_a:my_special_compiler_a",
  },
  linker_paths = {
    "//conditions:default": "//tools/compilers/my_arch_a:my_special_linker_a",
  }
)

my_cc_toolchain(
    name = "my_arch_b_toolchain",
    cpu = "my_arch_b",
    compiler_paths = {
        "//conditions:default": "//tools/compilers/my_arch_b:my_special_compiler_b",
    },
    linker_paths = {
        "//conditions:default": "//tools/compilers/my_arch_b:my_special_linker_b",
    }
)
```

Here, the `compiler_paths` and `linker_paths` attributes to the `my_cc_toolchain` are dictionaries mapping conditions to specific paths, using `select()` to choose the correct paths based on the target CPU. This allows you to easily add new architectures by specifying paths in a `BUILD` file, and let Bazel make the correct choice during compilation.

**Example 3: Using `ctx.executable` and Data Dependencies**

Sometimes, the tool binaries are not just plain executables but might require support files. In such cases, you need to declare these files as dependencies and access them with `ctx.executable` inside the toolchain implementation. Consider, for example, a compiler that relies on a special configuration file:

```python
# tools/build_defs/my_toolchain.bzl (modified)

def _my_toolchain_impl(ctx):
  compiler = ctx.executable.compiler
  linker = ctx.executable.linker
  config_file = ctx.file.config_file

  toolchain_info = platform_common.ToolchainInfo(
    compiler_path = compiler.path,
    linker_path = linker.path,
    compiler_configuration_file = config_file.path,
  )

  return [toolchain_info]


my_toolchain = rule(
  implementation = _my_toolchain_impl,
  attrs = {
      "compiler": attr.label(executable=True, mandatory=True),
      "linker": attr.label(executable=True, mandatory=True),
      "config_file": attr.label(allow_files=True, mandatory=True)
   },
)

def my_cc_toolchain(name, cpu, compiler_label, linker_label, config_file_label):
  native.toolchain(
      name = name,
      toolchain_type = "@my_toolchain//:my_toolchain_type",
      exec_compatible_with = ["@platforms//os:linux", "@platforms//cpu:%s" % cpu],
      target_compatible_with = ["@platforms//os:linux", "@platforms//cpu:%s" % cpu],
      toolchain = ":my_toolchain_implementation",
  )

  my_toolchain(
    name = "my_toolchain_implementation",
      compiler = compiler_label,
      linker = linker_label,
      config_file = config_file_label,
  )

```

```python
# BUILD file (modified)

load(":tools/build_defs/my_toolchain.bzl", "my_cc_toolchain")


my_cc_toolchain(
  name = "my_arch_c_toolchain",
  cpu = "my_arch_c",
  compiler_label = "//tools/compilers/my_arch_c:my_special_compiler_c",
  linker_label = "//tools/compilers/my_arch_c:my_special_linker_c",
  config_file_label = "//tools/compilers/my_arch_c:config.ini"
)

```

In this setup, we’ve changed to use `attr.label` instead of strings to accept label references for the tool binaries and the config file, ensuring Bazel correctly handles dependencies and location information. The compiler and linker are marked as `executable=True` and the config file allows files. This way, the compiler and linker can be executable binaries which we build via a `cc_binary`, and are properly accessible from within the toolchain definition with `ctx.executable`, and config file with `ctx.file`.

**Key Takeaways & Further Learning**

The crucial aspect is that toolchain definitions must specify, unambiguously, the paths to the binaries that are required to do compilation/linking. Whether it’s through string paths resolved by `select()`, or by using label references and `ctx.executable`, the principle remains the same. This way, Bazel can correctly invoke these tools during the build process.

For deeper understanding, I’d recommend exploring these resources:

1.  **Bazel Documentation on Toolchains:** The official Bazel documentation provides comprehensive details on configuring and using custom toolchains. Pay special attention to the sections regarding toolchain types and provider definitions.

2.  **"Effective Bazel" by G.V. Vishwanath:** This book provides an excellent overview of Bazel concepts, including advanced topics like custom toolchains, and offers practical guidelines for using Bazel effectively.

3.  **Source code of Bazel rules related to cc_toolchain:** Reviewing the official implementations within Bazel's source tree gives a firsthand view on how the rules are implemented internally, which can be very helpful for understanding the intricate details.

My advice is to start with simpler configurations, progressively adding complexity as you become more comfortable with these concepts. Toolchains in Bazel are powerful once you understand the core mechanics, allowing you to build cross-platform and specialized software with ease. Remember to start small and build up, that’s how I mastered it, and it will serve you well.
