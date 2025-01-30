---
title: "How do I set an environment variable in Ubuntu 18.04?"
date: "2025-01-30"
id: "how-do-i-set-an-environment-variable-in"
---
The persistence of environment variables in Ubuntu 18.04, and indeed most Linux distributions, hinges on the location where the variable is set.  A temporary change within a single shell session differs significantly from a system-wide, persistent modification.  Understanding this fundamental distinction is critical to correctly managing environment variables.  My experience troubleshooting deployment scripts for large-scale applications across diverse Linux environments has repeatedly highlighted the importance of this distinction.

**1.  Understanding the Scope of Environment Variable Setting:**

Environment variables, in essence, are name-value pairs accessible by processes.  Their scope dictates their lifetime and visibility.  A variable set within a shell script, for instance, will only exist for the duration of that script's execution.  Conversely, variables added to system-wide configuration files will persist across reboots and be available to all users.

There are three primary ways to set environment variables in Ubuntu 18.04, each with a different scope:

* **Session-specific:** These variables are only available during the current shell session.  Closing the terminal or logging out eliminates them.  They're useful for temporary adjustments or testing.

* **User-specific:** These variables are persistent across sessions for a specific user.  They are loaded each time the user logs in.  This is ideal for personal configurations and application settings.

* **System-wide:** These variables are available to all users on the system, across all sessions.  They're typically used for system-level configurations impacting all processes.

**2. Code Examples and Commentary:**

**Example 1: Session-Specific Environment Variable:**

```bash
export MY_VARIABLE="Hello, world!"
echo $MY_VARIABLE
```

This code snippet demonstrates setting a session-specific variable. The `export` command makes the variable `MY_VARIABLE` available within the current shell session.  `echo $MY_VARIABLE` confirms its value.  Upon closing the terminal, this variable is lost.  This approach is useful for quick tests or one-off commands requiring specific environment configurations.  I've used this extensively during debugging phases of development, where altering the environment without persistent changes was crucial.

**Example 2: User-Specific Environment Variable:**

```bash
echo 'export MY_VARIABLE="Hello, user!"' >> ~/.bashrc
source ~/.bashrc
echo $MY_VARIABLE
```

This example shows setting a user-specific variable.  The command appends the `export MY_VARIABLE="Hello, user!"` line to the `.bashrc` file, which is executed upon login.  `source ~/.bashrc` immediately applies the changes within the current session.  Subsequent logins will automatically load this variable.  This method ensures persistent settings for a single user.  During my work on user-specific application configurations, I've relied on this technique for managing individual settings without affecting the broader system.


**Example 3: System-wide Environment Variable (using /etc/environment):**

```bash
sudo echo 'MY_VARIABLE="Hello, system!"' >> /etc/environment
sudo reboot  #or `source /etc/environment` but reboot ensures consistency.
```

Setting a system-wide variable requires root privileges (`sudo`).  We append the variable definition to `/etc/environment`.  This file is loaded during the system boot process.  Therefore, a reboot (or sourcing the file with appropriate privileges) is necessary for the change to take effect.  Directly modifying `/etc/environment` is generally preferred over other system-wide configuration files for its clarity and dedicated purpose.  In my experience, this is essential when deploying applications requiring system-wide environment configurations across multiple servers.  However, caution should be exercised as poorly managed system-wide variables can lead to unexpected system-wide issues.  The use of `sudo` should always be considered carefully, along with implementing proper version control and backup strategies.


**3.  Further Considerations and Resources:**

The choice of method depends entirely on the intended scope and longevity of the environment variable.  Temporary changes within a single shell should leverage the `export` command directly. User-specific configurations should modify `.bashrc` (or `.bash_profile`, depending on the shell).  System-wide settings demand careful consideration and should be managed through `/etc/environment` using appropriate privilege escalation.


While `/etc/environment` is a suitable option for system-wide variables, several other locations offer similar functionality.  `sudoers` (using `visudo`), and files within `/etc/profile.d/` provide alternative approaches, each possessing its own nuances regarding loading order and applicability.  Consulting the official Ubuntu 18.04 documentation and the `man` pages for `export`, `bashrc`, and related system configuration files provide a comprehensive understanding of these different approaches.


Furthermore, understanding the implications of variable scope on application behavior is paramount.  Applications often rely on specific environment variables for configuration and execution.  Improperly set variables can lead to application malfunctions or unexpected behavior. Thorough testing and a well-documented process for managing environment variables are essential for robust system administration and software deployment.  The use of configuration management tools, like Ansible or Puppet, can significantly improve the consistency and manageability of environment variables across multiple systems.  They provide version control, and repeatable deployment processes, mitigating the risks associated with manual configuration.



In summary, correctly setting environment variables requires a keen understanding of their scope and persistence.  Choosing the appropriate method—whether session-specific, user-specific, or system-wide—is vital for avoiding errors and ensuring the consistent behavior of applications and system services.  A methodical approach, combined with thorough documentation and the leverage of appropriate tools, significantly contributes to the reliability and maintainability of the system.
