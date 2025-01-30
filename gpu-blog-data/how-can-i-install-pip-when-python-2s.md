---
title: "How can I install pip when Python 2's `pip` command fails due to a missing module?"
date: "2025-01-30"
id: "how-can-i-install-pip-when-python-2s"
---
The problem of installing `pip` on legacy Python 2 environments where the built-in `pip` command is broken often stems from a corrupted or missing `setuptools` module. This situation typically arises in older, poorly maintained virtual environments or systems with outdated Python installations. My experience, particularly when dealing with legacy server migrations, has frequently highlighted the fragility of these older Python setups. Direct attempts to use the built-in `pip` command often result in `ImportError` or `ModuleNotFoundError`, indicating that `setuptools` itself is the root issue, not a problem with `pip` in isolation.

The strategy, therefore, is to bootstrap a functioning version of `setuptools`, and then use that to subsequently install a working `pip`. This requires a two-step process because we can't rely on an existing package manager. The primary tool for this is `ez_setup.py`, a script designed specifically for this purpose. It acts as a mini-installer for `setuptools`, which then allows `pip` to be installed correctly.

My first approach in such situations is always to attempt to download `ez_setup.py` directly, as it’s usually available through the PyPA (Python Packaging Authority) project’s repositories, albeit often in historical archives. This script can be executed using the Python 2 interpreter, and it downloads and installs the latest compatible version of `setuptools`. Once this succeeds, the path to installing `pip` becomes straightforward.

Here's a simple example of the process:

```python
# Example 1: Attempting to download and execute ez_setup.py directly

# 1. Download ez_setup.py from a reliable source. This is typically a mirror. 
#    For the purposes of this demonstration, assume that we've downloaded the file.
#    You would usually do this in a shell using `wget` or `curl`.

# 2. Execute the ez_setup.py script using Python 2:
#    python ez_setup.py

# 3.  This script performs an install of setuptools, ensuring it is available.
# 4. No specific code within the script is required here other than invoking it. 

# Note: This step assumes that your system has internet access.
```

This example showcases the most basic way to initiate `setuptools` installation. The script handles downloading and installing `setuptools`, and no further user interaction is required. A common pitfall here is a lack of internet access, in which case the script may fail to complete.

Once `setuptools` is installed, the `pip` installer can be downloaded and executed using an updated version of the `easy_install` command. This command is included with `setuptools`. The `easy_install` command can fetch and install from the Python Package Index (PyPI).

Here’s the code for installing `pip` using `easy_install`, after `ez_setup.py` is completed:

```python
# Example 2: Installing pip using easy_install from setuptools

#  1. Assuming setuptools has been installed via ez_setup.py.
#  2. Use the easy_install command to install pip
#  3. This uses the newly available easy_install from our setuptools installation

# The next step is normally executed as a command from your shell:
# easy_install pip

#  4.  No python specific code is required other than the command line execution.
#  5.  If this completes successfully, pip will be installed.

# Note: This command relies on the setuptools installation from the prior step and 
#        an active internet connection to download the pip installer.
```

This example demonstrates that the `easy_install` tool does all the heavy lifting. It retrieves the latest `pip` compatible with the installed `setuptools` and Python 2, and completes the installation. Successful completion results in a functioning `pip` command. In case of issues here, reviewing `easy_install` log outputs can help identify the root cause, often relating to network problems or access permissions.

After successfully installing `pip`, the final step is to upgrade it. While the previous install ensures `pip` is present, it may not be the latest or most secure version. Upgrading it via `pip` is trivial but must be done once the initial install is complete.

```python
# Example 3: Upgrading pip to the latest version using pip itself.

# 1. pip should now be installed at this point from the previous example.
# 2. Now use pip to upgrade itself:
# This is typically a shell command:
# pip install --upgrade pip

#  3. Again, no python specific code is required; just command line usage.
#  4.  This upgrades pip to the newest version available.
#  5. If this succeeds, pip is up to date.

# Note: This assumes pip is installed correctly, and an active internet connection exists.
```

This final example ensures `pip` is current, providing access to updated security features and bug fixes, which is crucial for reliable package management. Any problems during this upgrade suggest an issue with `pip`’s dependencies or configuration.

In summary, addressing a broken `pip` on Python 2 involves using `ez_setup.py` to install `setuptools`, subsequently using `easy_install` to get a basic version of `pip`, and finally using the newly installed `pip` to update itself. The core insight remains: address the root cause (`setuptools`) before directly attempting to fix `pip`.

For further resources, I would suggest consulting the Python Packaging User Guide, specifically the sections dealing with `setuptools` and `pip` installation. The documentation for both packages is also valuable. In addition, exploring resources related to virtual environments can assist in preventing similar problems in the future by providing isolated Python setups. Examining the changelogs of the respective packages is important to understand version-specific variations or potential incompatibilities that may arise while upgrading them. Lastly, consider exploring the official Python documentation site which hosts detailed information about both packages.
