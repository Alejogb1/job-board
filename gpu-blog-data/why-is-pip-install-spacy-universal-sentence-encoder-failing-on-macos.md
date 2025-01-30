---
title: "Why is `pip install spacy-universal-sentence-encoder` failing on macOS Monterey (M1)?"
date: "2025-01-30"
id: "why-is-pip-install-spacy-universal-sentence-encoder-failing-on-macos"
---
The `spacy-universal-sentence-encoder` package, often used to integrate Google’s Universal Sentence Encoder with the spaCy natural language processing library, frequently encounters installation failures on macOS Monterey systems equipped with Apple's M1 chip. This primarily stems from incompatibility issues between TensorFlow, a core dependency of the `universal-sentence-encoder` module, and the ARM64 architecture of M1 processors when using standard x86-64 builds. I encountered this problem extensively while setting up a local NLP pipeline for a text summarization project recently.

The root cause is that `pip`, when it encounters a dependency like TensorFlow, often pulls down pre-compiled binary wheels that are built for the x86-64 architecture (the standard Intel/AMD chip architecture) rather than the `arm64` architecture of M1 chips. These wheels, while readily available on PyPI, will not function correctly on M1 Macs, typically resulting in obscure errors or crashes. The issue manifests because TensorFlow’s pre-built binaries are often not immediately available for the latest ARM-based processors. Even if pip correctly selects an arm64 wheel, compatibility with the specific versions of TensorFlow required by `spacy-universal-sentence-encoder` might still present issues if these arm64 builds are not fully optimized, stable or are not available at all at the required version number.

Furthermore, `spacy-universal-sentence-encoder` typically has version-specific dependencies on particular spaCy and TensorFlow versions. Mismatches between the installed versions of these libraries and those expected by `spacy-universal-sentence-encoder` are another point of failure. When `pip` encounters a dependency constraint conflict, it may partially install packages or simply fail, leading to inconsistent or broken Python environments. These package conflicts often occur subtly during a bulk installation when several packages have dependencies that are not fully aligned.

A secondary cause, though less frequent, can arise from system-level configurations. Issues with the user’s environment, such as corrupted Python installations, misconfigured `PATH` variables, or conflicting environment variables, can interfere with the installation process. These scenarios introduce variables beyond the package itself and make debugging more complex.

To resolve these problems, I've found that several steps are required in combination. The most fundamental step is to use a Python virtual environment (e.g., using `venv` or `conda`) to isolate package installations for specific projects. This prevents conflicts with system-level Python or other project dependencies.

Next, one needs to install a TensorFlow version compatible with both the `spacy-universal-sentence-encoder` package and the ARM64 architecture. As direct compatible TensorFlow builds for M1 were not always readily available or are older than the version needed, this usually means installing TensorFlow with a specialized method, or sometimes an older version that is available as an `arm64` build. In practice, this often means using specific instructions provided by the TensorFlow team or through community resources, rather than relying solely on pip’s automatic selection. Specifically, when standard `pip install tensorflow` fails, `pip` should install tensorflow with the following command:

```python
   pip install --no-binary :all: tensorflow
```

This forces pip to compile the tensorflow package from source, utilizing the metal acceleration libraries that are available to mac systems. This process will still use pip and the user's Python environment, but the actual compilation process will occur on the target machine rather than using a pre-compiled binary wheel.  This usually results in a longer install time, but it bypasses the limitations of incompatible binary wheels.

After successfully installing a compatible TensorFlow build for `arm64`, we can then attempt to install `spacy` followed by `spacy-universal-sentence-encoder`.

```python
    pip install spacy
    pip install spacy-universal-sentence-encoder
```
If the installation still fails because of mismatches in versions of the packages, it becomes necessary to inspect the error messages carefully.  These typically provide a very clear indication of which packages are not matching up.  The error messages often recommend a particular version, in these cases, one must explicitly specify the required versions of spaCy and the `universal-sentence-encoder` compatible tensorflow packages, which can be done as follows:

```python
    pip install spacy==<required_spacy_version>
    pip install tensorflow==<required_tensorflow_version>
    pip install spacy-universal-sentence-encoder
```
Replace `<required_spacy_version>` and `<required_tensorflow_version>` with the actual version numbers indicated by the error message. This explicit specification can circumvent the dependency resolution issues and lead to a successful installation. One must also be aware that different versions of spaCy may have different language model dependencies, it is often worth installing the required language models at the same time to avoid any confusion.

In summary, the primary causes of installation failures of `spacy-universal-sentence-encoder` on macOS Monterey (M1) stem from architecture incompatibilities between the precompiled TensorFlow wheels and M1's ARM64 architecture, combined with potential version conflicts among package dependencies, and occasionally, user environment configurations. Successfully installing this combination of software typically requires one to enforce a compilation of tensorflow from source or locate and install compatible versions of the tensorflow and spaCy library and ensure these dependencies are met.

For further exploration, I would recommend reading the official TensorFlow documentation for macOS M1 installation, which provides current best practices and troubleshooting advice. The spaCy documentation and the relevant GitHub repositories also contain valuable information about package compatibility and common issues. Additionally, seeking out user communities related to the package in question, such as issues filed on GitHub repositories, can provide real-world solutions or insights into commonly encountered problems. Forums specific to TensorFlow and Python development are also useful resources for finding solutions and troubleshooting guidance.
