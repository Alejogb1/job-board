---
title: "Why does TensorFlow installation with Anaconda3 (Python 3.5) result in a 'Read timed out' error?"
date: "2025-01-30"
id: "why-does-tensorflow-installation-with-anaconda3-python-35"
---
The primary cause of "Read timed out" errors during TensorFlow installation using Anaconda3 (Python 3.5), specifically, stems from incompatibility between package versions managed by Anaconda's package manager, `conda`, and the TensorFlow distribution available through Python's package installer, `pip`. I've encountered this repeatedly while setting up environments for older deep learning projects relying on Python 3.5, and it almost invariably boils down to the way these two package management systems interact.

The default behavior of `pip`, even within a `conda` environment, will reach out to the Python Package Index (PyPI) for packages. When attempting to install TensorFlow with `pip install tensorflow`, it may try to fetch a version of TensorFlow that is not explicitly compatible with the version of Python or other dependent packages present in the Anaconda environment, especially one using an older Python version such as 3.5. While `conda` does have its own TensorFlow packages, they sometimes lag behind those on PyPI, or may have certain constraints related to `conda`-managed dependencies. This creates a situation where `pip` either downloads packages incompatible with the `conda` environment, or fails to resolve dependencies, leading to timeouts due to stalled downloads or internal dependency resolution failures. The timeout is a symptom, not the root cause.

Specifically for Python 3.5, official TensorFlow support has largely ceased. Newer versions of TensorFlow are not built or tested against that interpreter. Trying to install the newest `pip` versions in a Python 3.5 `conda` environment can easily lead to these kinds of failures. The `conda` environment itself may also not have up-to-date package information, which makes it difficult for `pip` to understand the compatibility and try to resolve from potentially incompatible sources or outdated indexes.

To mitigate this, the recommended approach is to primarily rely on `conda` to manage TensorFlow and its associated packages, and if using `pip`, use it with utmost care, specifically specifying versions that work with the environment and Python 3.5. This frequently means finding versions of TensorFlow that are older, and potentially even compiled by the community. We may also need to configure `conda`'s channels to include channels that might contain older packages suitable for the older Python environment.

Let me illustrate this with a few examples based on situations I have personally dealt with:

**Example 1: Naive installation attempt (failure)**

```python
# In a conda environment with Python 3.5 active
# This is the typical installation that often leads to the timeout
# failure:

(env35) user@machine:~$ pip install tensorflow
Collecting tensorflow
  ... [long download process that stalls and eventually results in a Read timed out error]
```

In this scenario, the `pip` command attempts to pull the latest version of TensorFlow, ignoring that this version likely doesn't work with Python 3.5 and also ignores the `conda` environment's package versions. This results in dependency resolution issues and subsequent timeout. The key here is that `pip` isn't automatically aware of the `conda` environment and acts as though it were installing in an isolated environment.

**Example 2: Using a conda specified channel (potentially successful)**

```python
# Assuming we know a particular older TensorFlow that works with 3.5
# Let's first check if conda itself provides this package
(env35) user@machine:~$ conda search tensorflow=1.15
# ...  displays available versions
# Let's say it shows a suitable version, 1.15.0
(env35) user@machine:~$ conda install tensorflow=1.15.0
Collecting package metadata (current_repodata.json): done
Solving environment: done
... [install process succeeds]
```

Here, the approach is to use `conda` to install a specific version of TensorFlow that has been known to work with Python 3.5, often hosted within a suitable `conda` channel that maintains older builds. This method avoids `pip`'s dependency problems and directly installs a compatible package. The key takeaway is searching with `conda` first before using `pip`.

**Example 3: Using pip with specific version and compatibility constraints (potential success, requires care)**

```python
# This example assumes a very specific version of TensorFlow 1.15 compatible with Python 3.5 and known compatible
# numpy version. We're using `pip` for a very specific version.

(env35) user@machine:~$ pip install numpy==1.16.4
(env35) user@machine:~$ pip install tensorflow==1.15.0
Collecting tensorflow==1.15.0
  ... [installation might succeed if specific dependencies are met]
```

Here, we are deliberately forcing `pip` to use a very specific TensorFlow and NumPy package version that is compatible with the Python 3.5 environment. This is a more risky approach than example 2 since the dependencies need to be exactly correct, and errors are still possible if other packages in the environment are incompatible. Careful documentation research is crucial when installing this way. The key aspect here is specificity, and having researched compatible versions and dependencies for Python 3.5.

It’s critical to note that Python 3.5 has aged significantly, and finding fully compatible packages via `pip` becomes increasingly challenging. The ideal approach, if feasible, is to transition projects to a newer Python version and TensorFlow release.

Several resources provide guidance for tackling such installation challenges. Anaconda's documentation outlines best practices for managing environments. The official TensorFlow documentation maintains a historical record of compatible versions with older Python interpreters, though it's not extensively maintained.  Community forums and older GitHub issues related to specific TensorFlow versions on Python 3.5 are also great sources of insight and alternative installation strategies. Stack Overflow itself is also filled with questions about the very issue we're exploring; searching for error messages similar to the "Read timed out" error is often fruitful. Additionally, the release notes of older TensorFlow versions often note specific Python compatibilities. These documentation sources, when combined with the strategies I outlined above, provide the best chance for resolving the timeout issue.
