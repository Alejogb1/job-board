---
title: "Why does pip install MySQL-python fail on Ubuntu 20.04?"
date: "2025-01-30"
id: "why-does-pip-install-mysql-python-fail-on-ubuntu"
---
The failure of `pip install MySQL-python` on Ubuntu 20.04 almost invariably stems from dependency issues related to the MySQL client library, specifically the absence of a correctly configured development package.  My experience troubleshooting this across numerous projects—from small-scale data analysis scripts to larger-scale production deployments—points to this root cause more often than not.  The `MySQL-python` package, now largely superseded by `mysqlclient`, relies on the MySQL client library's header files and libraries for compilation during the installation process. If these are not present or improperly configured, the build process will fail.

**1. Clear Explanation**

The `pip` package manager downloads source code or pre-compiled wheels for Python packages.  For `MySQL-python` (and `mysqlclient`), a compilation step is necessary. This step uses a compiler (like GCC) to build the extension module that bridges Python to the MySQL client library.  The compiler needs access to the MySQL client library's header files (`.h` files containing function declarations) and statically linked libraries (`.so` or `.a` files containing the actual code). These are usually provided by system-level packages, not Python packages.

On Ubuntu 20.04 (and other Debian-based systems), these essential components are packaged separately.  Simply having the MySQL server installed is insufficient.  You must also install the *development* packages for the MySQL client library.  These packages typically include the necessary header files and libraries needed by the `MySQL-python` build process.  Failure to install these development packages results in compilation errors, manifesting as `pip install` failures.

Further complicating matters is the evolution of MySQL connectors.  While `MySQL-python` was once prevalent, it's now considered legacy. `mysqlclient` is the recommended alternative.  However, similar dependency requirements apply.  Attempting to install `MySQL-python` on modern systems frequently runs into issues due to incompatibility and the general lack of maintenance for the older package.

**2. Code Examples with Commentary**

The following examples illustrate troubleshooting and successful installation strategies.  Remember, these are illustrative; specific package names might vary slightly depending on your Ubuntu version.

**Example 1: The Failure Scenario**

```bash
pip install MySQL-python
```

This command, without the necessary prerequisites, will likely result in an error resembling this (the exact error message might vary):

```
error: command 'gcc' failed with exit status 1
```
or

```
error: [Errno 2] No such file or directory: '/usr/include/mysql/mysql.h'
```

This indicates that the compiler couldn't find the required MySQL header files.


**Example 2: Successful Installation with `mysqlclient`**

This example demonstrates the preferred approach, using the modern `mysqlclient` package and explicitly installing the required development packages.

```bash
sudo apt-get update
sudo apt-get install libmysqlclient-dev
pip install mysqlclient
```

The `sudo apt-get install libmysqlclient-dev` command installs the necessary development files.  Then, installing `mysqlclient` proceeds smoothly because the required components are available.  This is the recommended and most robust approach.

**Example 3: Addressing Specific Compilation Errors (Hypothetical)**

Let's consider a hypothetical scenario where you encounter an error like this during the compilation phase of `mysqlclient` installation:

```
error: undefined reference to 'mysql_init'
```

This points to a linker error; the compiler has found the header files but cannot link against the actual MySQL library.  In such cases, double-check that `libmysqlclient-dev` is correctly installed and that there are no conflicts with other MySQL libraries.  Running `ldconfig` after installing the development package can help resolve some linker issues.  Moreover, meticulously reviewing the full compiler error message is crucial; it often provides specific clues about the missing library or header file.  If needed, you might need to specify the library path using environment variables passed to the compiler or linker (this is less common in modern systems with properly configured package managers).

```bash
sudo ldconfig
pip install mysqlclient
```



**3. Resource Recommendations**

I recommend consulting the official documentation for both `pip` and the `mysqlclient` package. The Ubuntu documentation regarding package management and the installation of development libraries is also invaluable.   Furthermore, thoroughly examine the error messages generated during the `pip install` process; they provide essential information for debugging.  Finally, searching through the archives of relevant mailing lists and forums can offer insights into solutions to specific issues and workarounds for unusual problems.  Remember to carefully review any external instructions or scripts found online before executing them on your system.  Always prioritize official and well-vetted resources.
