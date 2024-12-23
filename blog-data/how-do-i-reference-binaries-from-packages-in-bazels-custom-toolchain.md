---
title: "How do I reference binaries from packages in Bazel's custom toolchain?"
date: "2024-12-23"
id: "how-do-i-reference-binaries-from-packages-in-bazels-custom-toolchain"
---

Okay, let’s talk about referencing binaries from packages within Bazel’s custom toolchains. This is a subject I've had to iron out a few times over the years, and it can initially feel a bit… roundabout, shall we say. It’s not always obvious how to navigate Bazel’s hermeticity while still accessing necessary tooling packaged within your project.

First, let's be clear on the core problem: Bazel aims for hermeticity. This means that build actions shouldn't depend on anything outside of the specified inputs. This is great for reproducibility and reliability, but it creates a challenge when your custom toolchain needs to execute binaries you’ve defined in your build. You can't simply rely on absolute paths or things floating around on your system. You need a reliable way to tell Bazel where those binaries are *within* its build graph, so they can be tracked, cached, and handled correctly.

The approach hinges on two key concepts: defining *toolchains* and using the `ctx.executable` attribute (in starlark, the configuration language for Bazel rules) within your toolchain’s execution action.

I remember a particularly frustrating incident involving a custom processor I had to support. The processor’s toolchain was a mess of hardcoded paths, and rebuilding the toolchain, or moving it to another machine, resulted in complete build failure. I spent a good chunk of a weekend fixing it, and I've been a proponent of properly using Bazel's mechanisms ever since.

So, let’s break this down with some concrete examples. Imagine we have a custom compiler and linker, and we’ve packaged them as binaries under the `//tools:my_compiler` and `//tools:my_linker` targets, respectively.

First, let's define a simple Starlark rule to represent our custom toolchain type. This rule, which might be in `toolchains.bzl`, could look something like this:

```python
# toolchains.bzl
def _my_toolchain_impl(ctx):
  return struct(
      compiler = ctx.attr.compiler,
      linker = ctx.attr.linker,
  )

my_toolchain = rule(
    implementation = _my_toolchain_impl,
    attrs = {
        "compiler" : attr.label(mandatory = True, executable = True),
        "linker"   : attr.label(mandatory = True, executable = True),
    },
)
```

Here we've defined a rule that simply collects the necessary executable labels, and it doesn’t do much beyond storing them. Notice the use of `executable = True`, which ensures that Bazel requires this attribute to be a build target that generates an executable file.

Now, let's define an actual toolchain instance. In a build file, such as `BUILD`, we’d define a concrete toolchain using the `my_toolchain` rule we created:

```python
# BUILD
load(":toolchains.bzl", "my_toolchain")

my_toolchain(
  name = "my_custom_toolchain_instance",
  compiler = "//tools:my_compiler",
  linker = "//tools:my_linker",
)
```

This creates a specific toolchain that points to the executables we defined elsewhere. Now the core part - how do you actually use these in a rule that executes a custom action? We need to consume this toolchain. Let's assume we are creating a rule to compile some source file. A very basic version of this could be as follows in a new `my_rules.bzl`:

```python
# my_rules.bzl
def _my_compile_impl(ctx):
  toolchain = ctx.toolchains["//:my_custom_toolchain_instance"]
  source_file = ctx.file.src
  output_file = ctx.actions.declare_file(ctx.label.name + ".o")

  ctx.actions.run(
      executable = toolchain.compiler,
      inputs = [source_file],
      outputs = [output_file],
      arguments = [
          "--output",
          output_file.path,
          source_file.path,
      ],
  )

  return [DefaultInfo(files = depset([output_file]))]

my_compile = rule(
  implementation = _my_compile_impl,
  attrs = {
      "src" : attr.label(mandatory=True, allow_single_file = True),
  }
  toolchains = ["//:my_custom_toolchain_instance"]
)
```

Let’s step through the most important lines:

1.  `toolchain = ctx.toolchains["//:my_custom_toolchain_instance"]`: This fetches our toolchain instance. Note that `//:my_custom_toolchain_instance` refers to the *label* of the toolchain, not a file path.
2. `executable = toolchain.compiler`: This is the crucial part. `toolchain.compiler` is not a string, but a Bazel `File` object, which represents the output of the target we previously defined (`//tools:my_compiler`). This is how Bazel ensures hermeticity; it has to track all the necessary files required to execute the build action.
3. The action itself is configured using `.run()` to invoke our compiler with the correct inputs and outputs.

This pattern is fundamental. By retrieving the toolchain using `ctx.toolchains` and accessing the executable files using the attributes of our toolchain's struct (in this case, `toolchain.compiler`), we are ensuring that Bazel knows about the dependencies and can schedule the actions correctly.

Now to apply our compile rule, here’s how it might look in your `BUILD` file:

```python
# BUILD (Continuing from previous example)
load(":my_rules.bzl", "my_compile")

my_compile(
    name = "my_source_obj",
    src = "my_source.c",
)

```
We've set up our custom compile rule to take `my_source.c`, execute our custom compiler defined in our toolchain instance, and finally output `my_source_obj.o`.

A few key considerations to mention:

*   **Toolchain registration**: While not directly part of this example, a `toolchain_type` should generally be used with a `target_compatible_with` attribute to ensure only rules that ask for it will actually use it. I've avoided that here to focus on the core mechanics. In a larger project, using toolchain types and constraints is highly recommended and adds additional layers of correctness.
*   **Error handling:** My examples are streamlined for clarity, but in production, proper error handling, logging, and input validation are essential for any Starlark code you write.
*   **Toolchain Selection:** When projects get complex, Bazel’s platform selection and toolchain resolution will come into play. You might have multiple toolchains for different architectures. The `toolchains` attribute on a rule indicates which toolchains it will use for compilation.

For further reading, I would recommend delving into Bazel's documentation for custom rules and toolchains. Specifically, review the concepts related to platform selection and toolchain resolution, which are crucial for large projects. I also suggest studying the source code for some of Bazel’s built-in rules to get a feel for how these are used in practice. Consider checking out *Bazel: Building Reliable Software Faster* by the Bazel team; it offers a comprehensive look at these concepts. Additionally, understanding how Bazel handles dependencies and its hermeticity principles, as outlined in papers detailing Bazel's underlying design philosophy, will give you a deeper appreciation for why these things are done a certain way. There’s a significant value in grasping the ‘why,’ not just the ‘how’.

In summary, referencing binaries in custom Bazel toolchains is all about aligning your needs with Bazel's model for dependency management and hermeticity. By correctly defining your toolchain, using the `ctx.toolchains` construct, and accessing the necessary files through `File` objects, you gain the full benefits of Bazel's build system, avoiding the pitfalls of relying on external environment configurations and increasing the reliability of your builds. Remember to always think in terms of build graph dependencies, and you’ll find these concepts much more straightforward.
