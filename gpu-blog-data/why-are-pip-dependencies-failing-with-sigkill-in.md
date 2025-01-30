---
title: "Why are pip dependencies failing with SigKill in Bazel builds?"
date: "2025-01-30"
id: "why-are-pip-dependencies-failing-with-sigkill-in"
---
The occurrence of `SIGKILL` during pip dependency resolution within a Bazel build, specifically manifested as a sudden and unceremonious process termination, typically indicates that the process invoked by Bazel to install these Python packages is consuming more system resources, particularly memory, than the build environment permits. My experience across several large-scale Python projects migrating to Bazel consistently highlighted this issue as a recurring point of friction, especially when dealing with complex dependency graphs.

The Bazel execution model, by design, operates within tightly controlled sandboxes, which limits resource usage per action. When Bazel attempts to resolve Python dependencies utilizing pip, it often constructs a temporary virtual environment and delegates the package installation to the pip executable. This pip process, when encountering a large number of dependencies or particularly resource-intensive packages during the build, can exceed the imposed resource limitations. The operating system, as an enforcement mechanism, then issues a `SIGKILL` signal, forcing the pip process to abruptly terminate. This differs from a more graceful `SIGTERM` signal, which would allow the process to clean up before shutting down.

This issue stems from two primary contributing factors. The first is the inherent behavior of pip itself, which may not always resolve dependencies efficiently, especially when dealing with multiple conflicting versions of packages or very large packages that undergo compilation. Second, and perhaps more critically, are the resource restrictions Bazel imposes on its action execution. Bazel aims to make builds reproducible and hermetic; it doesn't assume unlimited resources, and by default, often employs conservative resource limits per action. When the pip process exceeds these, `SIGKILL` is the inevitable consequence.

The problem is further exacerbated when using `requirements.txt` files that include transitive dependencies without explicitly pinning them. Uncontrolled version resolution during pip’s operation can lead to the download and processing of larger packages and more packages than intended. This significantly contributes to resource consumption. Similarly, any native extensions or compiled components within a Python package necessitate more computational resources during the pip install phase.

Therefore, to address the `SIGKILL` errors, multiple strategies can be used to fine-tune the pip resolution process and/or resource limits within Bazel. The strategy should be adopted based on a granular understanding of the dependency graph, the packages' sizes, and the complexity of the build process.

Here are three illustrative code examples along with commentary that can assist in resolving this issue.

**Example 1: Adjusting Bazel’s Action Resource Limits**

This approach involves increasing resource limitations imposed by Bazel on actions that involve pip package installation. Bazel provides command-line flags and build configurations to control memory and CPU. This example shows configuring memory limits for a specific `py_binary` target by using the `bazelrc`.

```bazel
# bazelrc file

# Increase memory limit for actions involving pip install
build --action_resource_limit=memory_mb=4000

# This applies to a single target
build --target_resource_limit=memory_mb=8000 -- //path/to:my_binary

```

*Commentary:*

This example demonstrates how to modify Bazel’s global action limits and override them for specific targets. The `build --action_resource_limit=memory_mb=4000` line specifies a global increase to 4 GB of memory for all build actions that use memory. While this may help with some `SIGKILL` errors, increasing the resource usage of *every* action is not always recommended; it can slow down your builds overall. The `build --target_resource_limit=memory_mb=8000 -- //path/to:my_binary` line shows that it is often better to target specific rules which are consuming too many resources. Here, `//path/to:my_binary` specifies the py_binary rule which is being affected. Note that you will probably need to change the target specification based on your Bazel rule. This is effective because it pinpoints the actual problem area, which allows a more targeted approach to address the resource exhaustion.

**Example 2: Utilizing a `requirements.txt` with Pinned Versions**

Pinning the versions in the `requirements.txt` can prevent pip from attempting to resolve many versions, and it also allows Bazel to cache the result of pip installation. This leads to faster and more repeatable builds.

```python
# requirements.txt
requests==2.28.1
numpy==1.23.5
pandas==1.5.2
```

```bazel
# BUILD file snippet
load("@rules_python//python:pip.bzl", "pip_parse")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_file")

http_file(
    name = "requirements_file",
    urls = ["//path/to/requirements.txt"],
    sha256 = "your_file_sha256_sum" # Replace with your actual sha
)

pip_parse(
    name = "pip_deps",
    requirements = ":requirements_file",
)

```

*Commentary:*

This example uses a `requirements.txt` file, specifying the exact version for each of your project’s dependencies. This avoids the dependency resolution process of pip selecting specific versions at build time and often decreases the build time significantly. This not only reduces uncertainty regarding the package versions used but also allows Bazel to cache the result of `pip install` more effectively. The Bazel snippet loads the required `pip_parse` rule, specifies the requirements file through `http_file`, and uses `pip_parse` to generate the relevant Bazel targets for each listed Python package. Importantly, a `sha256` check for the requirements file provides a deterministic installation process. The `requirements.txt` and its checksum guarantee consistent version resolution across different build environments and avoid unnecessary resource consumption.

**Example 3: Pre-installing dependencies with `pip_install` (Less Common)**

For some cases where the target architecture is different from the build platform, a pre-installed set of packages can improve the build process. Bazel allows the use of `pip_install`, but this is not generally recommended due to breaking the hermeticity of the builds. Use it with extreme care.

```bazel
# WORKSPACE file
load("@rules_python//python:pip.bzl", "pip_install")

pip_install(
    name = "pre_installed_deps",
    requirements = "//path/to/requirements.txt",
    python_interpreter = "//path/to/specific/python/interpreter", # Only needed if not in standard PATH
    # This is a location outside of the sandboxed execution
    output_path = "//external/prebuilt_py_deps",
    build_file_generation = "off", # Prevents Bazel from managing it.

)

```
```bazel
# BUILD File

load("@rules_python//python:defs.bzl", "py_binary")

py_binary(
    name = "my_binary",
    srcs = ["main.py"],
    deps = [
        "@pre_installed_deps//:requests",
        "@pre_installed_deps//:numpy",
        "@pre_installed_deps//:pandas",
    ]
)
```

*Commentary:*

In this example, the `pip_install` rule is employed. This command does not perform pip installation inside the Bazel sandboxed environment, but rather in the host system. The `output_path` in the example specifies a location outside the scope of Bazel's sandbox where the packages are installed. Note that the `build_file_generation` parameter is turned off in this particular scenario to prevent Bazel from managing its dependencies. This approach is only useful when you have pre-built dependencies that do not require further compilation, which can save significant time and reduce memory usage, and is used only as a very last resort when other methods are not possible. The `python_interpreter` parameter also specifies a specific Python executable to use when installing dependencies. Using `pip_install` is not ideal as it defeats the purpose of hermetic builds, and should be used with caution. The `py_binary` rule shows how you can declare this pre-installed dependency.

In conclusion, resolving `SIGKILL` errors during pip dependency resolution within Bazel requires a multi-pronged approach. It typically involves either optimizing pip dependency resolution by pinning versions or adjusting Bazel's resource limitations. While the `pip_install` method may occasionally be useful, it should be used with caution. Understanding Bazel's action execution model and its resource constraints is important in diagnosing these errors, and it allows a more tailored approach to each particular case.

For further investigation into Bazel and pip dependency management, consider exploring resources on Bazel’s official documentation website, particularly the sections on action resource limits and Python rules. Refer to pip’s documentation for strategies for requirements management and handling complex dependency trees. Additionally, articles and tutorials discussing common issues when working with Bazel and large Python projects can provide further context and insight.
