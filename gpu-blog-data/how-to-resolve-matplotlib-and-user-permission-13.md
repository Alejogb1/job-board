---
title: "How to resolve 'matplotlib and user permission (13)' error when installing TensorFlow on Anaconda?"
date: "2025-01-30"
id: "how-to-resolve-matplotlib-and-user-permission-13"
---
The `matplotlib and user permission (13)` error encountered during TensorFlow installation within an Anaconda environment stems fundamentally from insufficient write permissions within the targeted installation directory. This isn't specific to TensorFlow; it's a broader issue arising from operating system security restrictions, frequently impacting Python package managers like `conda` and `pip`.  My experience troubleshooting this, spanning several years of developing and deploying machine learning models, indicates that the solution usually involves elevating privileges or altering the installation target.


**1.  A Clear Explanation**

The error message, while cryptic, hints at the core problem: the installer—whether `conda` or `pip`—lacks the necessary authorization to write files to a system directory.  Permission errors manifest as code 13 (Permission denied) within Unix-like systems (including macOS and Linux).  On Windows, the error might appear differently, but the root cause remains the same.  TensorFlow, being a substantial library with dependencies such as `matplotlib`, requires extensive file system access to install its components and associated data. When the installation process attempts to place files in a protected location where the user doesn't possess sufficient write privileges, the error arises.

Several factors contribute to this:

* **Installation Location:**  `conda` and `pip` default to system-wide directories which often necessitate administrator or root privileges.  Attempting an installation without such privileges will trigger the permission error.
* **User Account Type:**  Standard user accounts on most systems have restricted access to system directories.  Administrator or root accounts typically have unrestricted write access.
* **Antivirus/Firewall Interference:**  While less common, aggressive security software might interfere with the installation process by blocking file writes, simulating a permission error.
* **Corrupted Package Cache:**  A corrupted `conda` or `pip` cache can lead to seemingly arbitrary permission errors during installation.


**2. Code Examples with Commentary**

The solutions involve modifying the installation process to bypass the permission restriction. The following examples illustrate different approaches:

**Example 1:  Using a Virtual Environment with Administrator/Root Privileges**

This is the preferred method.  It isolates TensorFlow and its dependencies within a contained environment, preventing conflicts with other Python installations and avoiding potential system-wide permission issues.

```bash
# Activate your base conda environment (if not already activated)
conda activate base

# Create a new environment with Python 3.9 (adjust Python version as needed)
conda create -n tf_env python=3.9

# Activate the new environment
conda activate tf_env

# Install TensorFlow within the new environment (using pip or conda)
pip install tensorflow  # Or conda install -c conda-forge tensorflow

# Verify installation
python -c "import tensorflow as tf; print(tf.__version__)"
```

**Commentary:** The key here is creating the environment with administrator or root privileges.  This ensures that the `conda` commands have the necessary authority to write the files to the environment's directory, which is typically located within the user's home directory.


**Example 2: Installing with `sudo` (Linux/macOS)**

This should be used cautiously.  Using `sudo` grants root privileges, potentially leading to unintended consequences if not handled correctly.  It's generally recommended to avoid this approach unless the virtual environment method fails.

```bash
# Install TensorFlow using sudo and pip
sudo pip install tensorflow

# Or using conda:
sudo conda install -c conda-forge tensorflow
```

**Commentary:**  The `sudo` command precedes the installation command, temporarily elevating the user's privileges to root.  This allows the installer to write to protected directories.  However, this method carries security risks; misuse can compromise the system.  Therefore, using a virtual environment remains the safer option.


**Example 3:  Changing the Installation Directory (Advanced)**

This approach directly modifies the target installation path. It requires advanced knowledge and understanding of Python package management. It is less recommended than the previous methods.

```bash
# Set a custom installation path (replace /path/to/your/directory with an appropriate location)
export PYTHONUSERBASE=/path/to/your/directory

# Install TensorFlow using pip (ensure the path is writable)
pip install --user tensorflow
```


**Commentary:** This method uses environment variables to redirect the installation location to a user-writable directory. This avoids attempting to install into restricted system folders.  However, this can lead to configuration issues if not correctly managed, as it alters the standard Python environment setup.  Thorough understanding of Python's path configuration is essential before attempting this.


**3. Resource Recommendations**

I strongly recommend consulting the official documentation for both Anaconda and TensorFlow.  The Anaconda documentation provides comprehensive guidance on environment management, including creating and activating virtual environments.  Similarly, the TensorFlow documentation outlines the installation process and troubleshooting common issues.  Furthermore, reviewing the Python documentation on package management and the operating system's user permissions will prove invaluable. Understanding these core concepts will empower you to tackle similar problems independently in the future.  Finally, utilizing the search functionality on Stack Overflow, filtering by language (Python) and relevant keywords (e.g., "TensorFlow installation error", "conda permission error"), can expose many solutions to similar problems encountered by others. Remember to meticulously check the versions of all components involved, ensuring compatibility between TensorFlow, Python, and your system's libraries.  Incorrect versioning can easily lead to seemingly unrelated errors.
