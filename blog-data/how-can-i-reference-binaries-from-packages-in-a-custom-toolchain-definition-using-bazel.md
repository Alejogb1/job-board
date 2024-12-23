---
title: "How can I Reference binaries from packages in a custom toolchain definition using Bazel?"
date: "2024-12-23"
id: "how-can-i-reference-binaries-from-packages-in-a-custom-toolchain-definition-using-bazel"
---

Okay, let's tackle this one. It’s a challenge I've faced numerous times, particularly when integrating legacy systems into a bazel-based build environment. Referencing binaries from packages within a custom toolchain is a surprisingly common pain point, and the solutions, while powerful, often aren't immediately obvious. Let me walk you through my experiences and what I've found works consistently.

The core problem lies in how bazel isolates build actions. Toolchains need access to external tools, but these tools are often themselves built by bazel, residing within its hermetically sealed execution environment. Directly referencing them via absolute paths will fail – the paths won't exist in the sandbox during the action’s execution. The solution involves employing bazel's mechanism for declaring dependencies between toolchain definitions and the artifacts they consume. Specifically, we're going to focus on using `ctx.executable`, and strategically defined `providers`.

In my previous work, I was tasked with migrating a complex C++ toolchain. We had custom preprocessors and linkers that weren't standard and weren't easily replaced. The key was mapping the locations of these tools built with bazel to our toolchain definition. Here’s how I approached it, and how you can too.

First, let's start by outlining what a toolchain typically looks like. You'll define a `toolchain()` rule that specifies a `toolchain_type`, declares your various tools (compilers, linkers, etc.) with custom rules, and then registers the toolchain via the `register_toolchains()` function. Inside these rules, we will need to access our built binaries.

The trick lies within the rule implementation, particularly when you’re declaring the execution action, which consumes the toolchain. Let's look at the `ctx.executable` attribute. This special attribute, exposed via the `ctx` object, allows us to map a declared dependency to its corresponding executable output, respecting bazel's sandbox.

Now, let's consider a concrete example. Imagine we have a simple tool called `my_custom_tool`, built using a `genrule`.

```python
# my_tools/BUILD.bazel

load("@bazel_skylib//rules:build_file.bzl", "build_file")

genrule(
    name = "my_custom_tool_impl",
    srcs = [],
    outs = ["my_custom_tool"],
    cmd = "echo '#!/bin/bash' > $@; echo 'echo \"Hello from custom tool\"' >> $@; chmod +x $@",
)

build_file(
  name = "my_custom_tool",
  srcs = [":my_custom_tool_impl"],
  out = "my_custom_tool",
  executable = True
)
```

This `genrule` generates a basic executable shell script. The key here is the `build_file`, marking our output as executable. This is crucial for `ctx.executable` to function properly. Now, let’s define a custom rule which would utilize this.

```python
# my_rules/my_custom_rule.bzl

def _my_custom_rule_impl(ctx):
  tool_executable = ctx.executable.my_custom_tool

  output_file = ctx.actions.declare_file(ctx.label.name + ".output")

  ctx.actions.run(
        outputs = [output_file],
        executable = tool_executable,
        arguments = [output_file.path],
    )

  return [
      DefaultInfo(files = depset([output_file])),
  ]


my_custom_rule = rule(
    implementation = _my_custom_rule_impl,
    attrs = {
      "my_custom_tool": attr.label(
        mandatory = True,
        executable = True,
        allow_files = False
      )
    }
)
```

Here, `_my_custom_rule_impl` takes the tool dependency via `ctx.executable.my_custom_tool`. It then executes it, passing the output file. The `my_custom_rule` definition, using the `attr.label` directive, declares the specific dependency of the tool, making sure it's an executable.

Now to tie this all together, we would define our toolchain and a usage example:

```python
# my_toolchain/toolchain.bzl

load("//my_rules:my_custom_rule.bzl", "my_custom_rule")
load("@bazel_skylib//lib:dicts.bzl", "dicts")

def _my_toolchain_impl(ctx):
    my_tool_executable = ctx.attr.custom_tool
    return platform_common.ToolchainInfo(
       my_custom_tool = my_tool_executable,
       my_custom_rule = my_custom_rule,
     )

my_toolchain = rule(
  implementation = _my_toolchain_impl,
    attrs = {
        "custom_tool" : attr.label(
          mandatory = True,
          executable = True,
          allow_files = False,
        ),
    }
)

def _my_toolchain_type_impl(ctx):
    return platform_common.ToolchainTypeInfo()

my_toolchain_type = rule(
  implementation = _my_toolchain_type_impl,
)

def _register_my_toolchains_impl(ctx):
  return [
    platform_common.ToolchainRegistrationInfo(
          toolchain = ctx.attr.my_toolchain,
          toolchain_type = ctx.attr.my_toolchain_type,
    )
  ]
register_my_toolchains = rule(
    implementation = _register_my_toolchains_impl,
    attrs = {
        "my_toolchain" : attr.label(mandatory=True),
        "my_toolchain_type" : attr.label(mandatory=True),
    }
)
```

```python
# BUILD.bazel

load("//my_rules:my_custom_rule.bzl", "my_custom_rule")
load("//my_toolchain:toolchain.bzl", "my_toolchain", "my_toolchain_type", "register_my_toolchains")
load("//my_tools:BUILD.bazel", my_custom_tool = "my_custom_tool")


my_toolchain(
    name = "my_custom_toolchain",
    custom_tool = ":my_custom_tool",
)


my_toolchain_type(
  name = "my_custom_toolchain_type",
)

register_my_toolchains(
    name = "register_toolchains",
    my_toolchain = ":my_custom_toolchain",
    my_toolchain_type = ":my_custom_toolchain_type",
)


my_custom_rule(
    name = "my_custom_target",
    my_custom_tool = ":my_custom_tool",
)
```

Now, the important thing to notice here is that the toolchain rule takes the `my_custom_tool` as a dependency, and is forwarding that to the `ToolchainInfo`. The custom rule is also taking the `my_custom_tool` as a dependency via its attribute. The magic here is that `ctx.executable` is aware of bazel's sandboxing system. No matter where the tool is ultimately placed within the build environment, `ctx.executable` will always provide the correct location.

For a deep dive, I’d recommend a careful read of the bazel documentation on toolchains, along with the source code of the `platform_common` module. Specifically, examine how `ToolchainInfo` and `ToolchainTypeInfo` function. Understanding these underlying mechanisms provides a clearer picture of how bazel handles toolchain resolution. "Effective Bazel" by Buildbarn is also an exceptional resource, covering many aspects of advanced bazel usage. There's also the paper "Bazel: A Correct, Scalable, and Maintainable Build System" that details the philosophical underpinnings and design decisions that motivate bazel's architecture and would provide further grounding in how it works internally.

This is, of course, a basic example. Real-world scenarios may involve more intricate toolchains, perhaps handling multiple tools or complex dependency graphs. However, the core principles remain the same: declare your dependencies as attributes, and use `ctx.executable` to reference the tool's executable path during execution. It’s this approach that will allow you to bridge the gap between bazel's hermetic build environment and the external tools your project relies on, avoiding paths that wouldn't be usable in the sandboxed actions. The biggest takeaway is to think in terms of dependencies, not just file paths, and use the tools bazel provides to make dependency management transparent and manageable within your build environment.
