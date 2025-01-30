---
title: "Why does Bazel report 'no such package 'inception'' when building TensorFlow Serving?"
date: "2025-01-30"
id: "why-does-bazel-report-no-such-package-inception"
---
The root cause of "no such package 'inception'" errors when building TensorFlow Serving with Bazel typically stems from a mismatch between the WORKSPACE configuration and the dependencies required for the specific TensorFlow Serving model you are attempting to build. This error signals that Bazel cannot locate the necessary build files (BUILD files) within the designated source tree for the model named ‘inception’. My experience working on multiple distributed TensorFlow Serving deployments has shown this is a common, though often easily rectified, problem. Specifically, the ‘inception’ component, usually referring to the pre-trained Inception model, is not a core part of TensorFlow Serving's own repository and needs to be explicitly referenced in the workspace.

The Bazel build system relies on a top-level file named `WORKSPACE` to define external dependencies and the root of the project. It’s the equivalent of a `pom.xml` in Maven or `package.json` in Node.js ecosystems. When TensorFlow Serving encounters an instruction to build ‘inception’, which isn't defined within its own codebase, Bazel uses the `WORKSPACE` to look for its definition in an external repository. If this definition is missing, misconfigured or incorrectly placed, this ‘no such package’ error arises. Bazel doesn’t automatically infer external dependencies; they must be explicitly declared.

This error typically indicates one of the following situations:

1.  **Missing Inception Repository Definition:** The `WORKSPACE` file does not contain an appropriate `http_archive` or other repository rule to download or reference the repository containing the Inception model's build files.
2.  **Incorrect Path:** Even with a correctly defined repository rule, Bazel may not be configured to find the `BUILD` file for 'inception' within the downloaded repository. This could involve typos in labels, incorrect paths in `BUILD` files, or directory structure discrepancies.
3.  **Dependency Conflicts:** Other dependencies declared in the `WORKSPACE` may inadvertently cause conflicts or override expected path resolution. This becomes especially relevant with large dependency graphs.
4.  **Incorrect Build Configuration:** The `BUILD` file attempting to build the ‘inception’ package itself may be improperly written and not expose the targeted dependencies properly.

The `WORKSPACE` file is processed sequentially. Therefore, the order of declarations matters in some scenarios and debugging complex cases may require careful review of the file.

Let’s consider three common scenarios and how to address each with a configuration example.

**Code Example 1: Missing Repository Definition**

The most common cause is the absence of a definition for the `inception` repository in the `WORKSPACE` file. Here's a snippet demonstrating how to include a public Inception model repository as a `git_repository`. We assume such a repository exists for the purpose of this example; in reality, one would use the repository path containing the needed model definition.

```python
# WORKSPACE

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

# Existing TensorFlow Serving workspace rules...

git_repository(
    name = "inception_model",
    remote = "https://github.com/example/inception-model-repo.git", # Replace with a real repo if needed
    commit = "a1b2c3d4e5f678901234567890abcdef", # Pin to a specific commit
)

# Now we can refer to @inception_model in our BUILD files
```

**Commentary:** This code snippet adds a `git_repository` rule, named "inception\_model," which fetches the designated Git repository. The `remote` attribute specifies the URL of the repository, and `commit` pins it to a specific commit for reproducible builds. This approach is generally preferable as it avoids changes that can break the build. With this declaration, Bazel will fetch the code during the dependency resolution stage, and further `BUILD` files within that repo are usable as `@inception_model//path/to/buildfile`.

**Code Example 2: Incorrect Path and BUILD file reference**

After successfully adding the dependency, the `BUILD` file attempting to use `inception` might have an incorrect path reference. Let's assume inside the `inception_model` repository, we have a `BUILD` file under the path `models/inception/BUILD`. Let's see how we would specify that as a target in a TensorFlow serving `BUILD` file.

```python
# tensorflow_serving/BUILD

load("@bazel_tools//tools/build_defs/pkg:pkg.bzl", "pkg_tar")

# Other build rules..
pkg_tar(
    name = "inception_serving_bundle",
    srcs = [":serving_binary"],
    deps = ["@inception_model//models/inception:inception_model"], # Correct Reference
)

cc_binary(
    name = "serving_binary",
    deps = [
        "@org_tensorflow//tensorflow/core:lib", # Core TF deps
        "@org_tensorflow//tensorflow/cc:cc_ops",
        "//tensorflow_serving/core",  # Core Serving deps
        # ... other serving deps
    ],
)
```

**Commentary:** The crucial line here is `deps = ["@inception_model//models/inception:inception_model"]`. This tells Bazel that the target `inception_model` defined within the `models/inception/BUILD` file of the `inception_model` repo should be included as a dependency. Assuming the `BUILD` file in  `inception_model/models/inception` properly defines the `inception_model` target, this will correctly use the dependency. Errors here often arise from misspellings or an incorrect package path. If the target was instead defined in the BUILD file as 'my_inception_model' you'd use `@inception_model//models/inception:my_inception_model`. The full target path and its defined name must match.

**Code Example 3: Dependency Conflict/Override**

In rare cases, another dependency might define a package with the same name, resulting in Bazel resolving to the incorrect location. This example shows how to rename an external repository to avoid name clashes in complex environments.

```python
# WORKSPACE

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")

# Existing TensorFlow Serving workspace rules...

git_repository(
    name = "my_inception_model", # Renamed the repository
    remote = "https://github.com/example/inception-model-repo.git",
    commit = "a1b2c3d4e5f678901234567890abcdef",
)

# tensorflow_serving/BUILD
pkg_tar(
    name = "inception_serving_bundle",
    srcs = [":serving_binary"],
    deps = ["@my_inception_model//models/inception:inception_model"], #Correct renamed reference
)
```

**Commentary:** Here, I renamed the repository to `my_inception_model` to avoid any potential collisions with other similarly named external repos or internal packages. Then we reference this new name in the `deps` attribute of the target definition in the Serving `BUILD` file. This technique helps disambiguate and troubleshoot conflicts when multiple external dependencies are present.

To diagnose these problems, inspect the `WORKSPACE` file meticulously, comparing the paths defined there with the actual locations of the code on disk. Pay close attention to any error messages output by Bazel during build executions, and they often reveal where the build process is unable to resolve a target. Bazel also provides several debug options that can help in identifying more complicated problems, such as showing the dependency graph.

For further in-depth information, consult the Bazel documentation, which provides detailed explanations of WORKSPACE files, rules, and how Bazel handles dependencies. The TensorFlow Serving documentation often contains instructions and examples pertinent to particular model deployments. Also, the Bazel community is very active on various forums and online platforms.
