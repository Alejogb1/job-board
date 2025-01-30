---
title: "How can I install multiple Python versions on macOS M1?"
date: "2025-01-30"
id: "how-can-i-install-multiple-python-versions-on"
---
The core challenge when managing multiple Python versions on macOS M1 stems from the system Python interpreter, which is tightly coupled with the operating system and should ideally remain untouched. Directly modifying this system installation creates instability and can interfere with macOS functionality. My experience working on cross-platform development projects, specifically migrating existing Python 2.7 codebases to Python 3.9 on macOS, exposed me to the common pitfalls of relying solely on the default Python and underscores the need for a robust version management approach.

To effectively manage multiple Python versions, particularly on the ARM-based M1 architecture, employing a dedicated version manager is essential. While alternatives exist, I consistently found `pyenv` to be the most reliable and flexible solution. `pyenv` avoids directly modifying system Python. Instead, it intercepts Python commands and redirects them to the desired interpreter, creating isolated environments for different project needs. Crucially, `pyenv` handles both the x86-64 and arm64 architectures seamlessly, addressing a frequent hurdle encountered by M1 users.

Installation of `pyenv` typically involves using `brew`, which is a package manager for macOS. After installing `brew`, the following command installs `pyenv`:

```bash
brew install pyenv
```

Following installation, you must configure `pyenv` in your shell environment, usually the `.zshrc` or `.bashrc` file, depending on your chosen shell. This involves adding the necessary environment variables to instruct the shell on how to interact with `pyenv`. This step is crucial as it allows the shell to recognize and use `pyenv` commands correctly.

```bash
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
echo 'eval "$(pyenv init -)"' >> ~/.zshrc
```

The first line sets the root directory where `pyenv` will store its Python versions and related data. The second line ensures `pyenv`'s directory is added to the shell's search path, thus allowing the shell to find and execute the `pyenv` command. The final line initializes the `pyenv` shell integration, which allows it to intercept Python commands. After making these changes, it is necessary to either source the updated configuration file using `source ~/.zshrc` or open a new terminal window to activate the changes.

With `pyenv` configured, installing various Python versions becomes straightforward. The `pyenv install` command is used, specifying the version number desired. For example, to install Python 3.9.10:

```bash
pyenv install 3.9.10
```

`pyenv` downloads the specified Python version and compiles it within its designated directory structure, isolated from the system Python. This isolation prevents version conflicts between different projects. Installation times vary based on the complexity and architecture of the Python version being installed. Note that on M1 systems, some Python versions might require additional dependencies before compilation. `pyenv` usually indicates the necessity for these. A crucial aspect of `pyenv` is its flexibility in designating specific Python versions for use at the global level or per project directory.

To set a global Python version to be the default when no specific version is designated by a project, I utilize the `pyenv global` command. For instance, setting the previously installed Python 3.9.10 as the default across all projects that do not specify otherwise:

```bash
pyenv global 3.9.10
```

Executing this command results in any new Python shells, or existing ones that are not otherwise associated with a python version, defaulting to 3.9.10. This avoids having to specify each project’s python interpreter on every invocation. The value of this approach was clear during the aforementioned Python 2.7 to 3.9 migration, as I could easily toggle between versions while testing specific compatibility code without jeopardizing my other projects.

However, a common scenario requires a different Python version for a particular project. `pyenv` addresses this with the `pyenv local` command, which operates on a per-project-directory basis, automatically applying the selected version whenever you’re inside that directory, as indicated by the `.python-version` file that `pyenv` creates. For instance, let's assume we have a directory called `project_x`, and within this directory we require Python 3.11.2. First we must ensure that 3.11.2 is installed. If not, `pyenv install 3.11.2` needs to be called, then once the version has been installed, I would enter the `project_x` directory and execute the following:

```bash
cd project_x
pyenv local 3.11.2
```

This command generates a `.python-version` file inside `project_x` which contains the specification of which python version to invoke. Subsequent access to the project directory automatically switches to the specified Python version (3.11.2 in this case). `pyenv` intercepts the `python` executable and redirects to the correct version, giving a seemingly seamless transition.

The most significant benefit derived from using `pyenv` is the avoidance of direct manipulation of the system Python and isolation between various projects using different Python interpreters. While other solutions exist, including virtual environments or even containerization, I've found `pyenv` provides the best combination of versatility and convenience for managing Python versions in macOS, particularly with the M1 architecture’s specific needs.

Virtual environments can be employed within the context of a specific python version managed by `pyenv` to further manage dependencies. For example, if `project_x` needs particular libraries not required by the default python, I can use:

```bash
python -m venv venv_x
```

This will create a virtual environment named `venv_x` which can then be activated using:

```bash
source venv_x/bin/activate
```

Once activated, any installations using `pip` or other methods will be applied exclusively within that environment. It is also possible to create virtual environments from within `pyenv` to simplify activation. The combination of `pyenv` and virtual environments allows for very granular control over both the python interpreter and its packages.

Recommendations for further study include: Understanding package management in python using `pip` which will further enhance the user's ability to compartmentalize project dependencies. Investigation into shell scripting would also be beneficial as `pyenv` operations can be wrapped within automation scripts to simplify repetitive tasks. Lastly, consulting the official `pyenv` documentation will offer the most detailed description of the available features and customizations. The source code is also a great reference point. These three areas will solidify the techniques outlined and empower a more thorough grasp of the topic.
