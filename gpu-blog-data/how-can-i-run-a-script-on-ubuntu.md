---
title: "How can I run a script on Ubuntu startup?"
date: "2025-01-30"
id: "how-can-i-run-a-script-on-ubuntu"
---
The most reliable method for executing a script upon Ubuntu startup involves leveraging systemd, the system and service manager.  While other approaches exist, such as using cron or modifying shell profiles, they lack the robustness and control offered by systemd's sophisticated service management capabilities.  My experience working on large-scale server deployments has consistently demonstrated the superiority of this approach for its predictable behavior and integration with the overall system lifecycle.

**1. Clear Explanation of the Systemd Approach:**

Systemd utilizes unit files, configuration files written in a declarative language, to define services.  These unit files specify the script to be executed, dependencies, execution parameters, and various other settings governing the service's behavior.  The primary unit file type used for this purpose is a `*.service` file.  These files reside within the `/etc/systemd/system/` directory, and their names directly correspond to the service name used for management.

A typical `*.service` file contains several directives:

* **`[Unit]`:** This section defines metadata about the service, including descriptions and dependencies on other services.  Dependencies are critical for ensuring proper startup ordering; for example, a service requiring a network connection should depend on the networking service being active.

* **`[Service]`:** This section outlines the core functionality of the service, including the command to execute (`ExecStart`), user and group under which it should run (`User`, `Group`), working directory (`WorkingDirectory`), and other process-related settings.  Careful consideration of these settings is crucial to ensure security and correct execution.  Incorrectly setting the user or working directory can lead to permission errors or unexpected behavior.

* **`[Install]`:** This section specifies how the service should be managed during installation and removal.  It typically indicates which target (runlevel in older init systems) the service should be part of.  The `WantedBy` directive often points to `multi-user.target`, which represents the standard multi-user system state.

Incorrectly configuring these sections can lead to various problems; from the service failing to start to system instability.  I've personally debugged numerous instances where misplaced semicolons or missing directives caused considerable downtime. Therefore, meticulous attention to detail is imperative.


**2. Code Examples with Commentary:**

**Example 1: Basic Script Execution:**

Let's assume we have a simple script named `my_script.sh` located in `/home/user/scripts/`. This script simply prints "Hello from startup!" to the console.

```ini
[Unit]
Description=My Startup Script
After=network-online.target

[Service]
Type=simple
User=user
Group=user
WorkingDirectory=/home/user/scripts/
ExecStart=/bin/bash /home/user/scripts/my_script.sh
RemainAfterExit=no

[Install]
WantedBy=multi-user.target
```

* **`Type=simple`:**  Indicates a simple, one-shot execution.  The service exits after the script completes.
* **`RemainAfterExit=no`:**  The service is stopped after the script finishes.  Change to `yes` if the script should remain running.
* **`After=network-online.target`:** Ensures the script runs only after the network is up.  This dependency prevents errors if the script requires network access.  Careful selection of dependencies is essential for avoiding race conditions.


**Example 2: Script with Error Handling and Logging:**

This example enhances the previous one by adding error handling and logging.

```ini
[Unit]
Description=My Startup Script with Logging
After=network-online.target

[Service]
Type=simple
User=user
Group=user
WorkingDirectory=/home/user/scripts/
ExecStart=/bin/bash -c '/home/user/scripts/my_script.sh >> /var/log/my_script.log 2>&1'
StandardOutput=append:syslog
StandardError=append:syslog
RemainAfterExit=no

[Install]
WantedBy=multi-user.target
```

* **`ExecStart=/bin/bash -c ...`:**  Uses `bash -c` to execute the script and redirect both standard output (stdout) and standard error (stderr) to a log file.
* **`StandardOutput=append:syslog`**:  Directs stdout to the system log (syslog), ensuring comprehensive logging.
* **`StandardError=append:syslog`**:  Similar to `StandardOutput`, but for stderr.  This is crucial for debugging.  I've spent countless hours tracing issues down using precisely this technique.



**Example 3:  A Daemon Script:**

If your script needs to run continuously in the background, use `Type=forking` or `Type=notify`.

```ini
[Unit]
Description=My Long-Running Startup Daemon
After=network-online.target

[Service]
Type=forking
User=user
Group=user
WorkingDirectory=/home/user/scripts/
ExecStart=/usr/bin/python3 /home/user/scripts/my_daemon.py start
ExecStop=/usr/bin/python3 /home/user/scripts/my_daemon.py stop
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
```

This example requires `my_daemon.py` to handle `start` and `stop` signals appropriately.  This is a more advanced approach and requires careful consideration of process management within the Python script itself.  Failure to implement proper signal handling can result in orphaned processes.

**3. Resource Recommendations:**

The official Ubuntu documentation on systemd, particularly the section on service unit files.  A comprehensive guide on shell scripting and process management in Linux. A well-structured book on Linux system administration.


After creating the `*.service` file, enable and start the service using the following commands:

```bash
sudo systemctl enable my_script.service
sudo systemctl start my_script.service
```

Replace `my_script.service` with the actual name of your service unit file.  Checking the status with `sudo systemctl status my_script.service` provides valuable insights into the service's operational state, including logs and error messages, a critical step in troubleshooting any problems encountered during the implementation process.  Remember to always validate the correct execution user and permissions to prevent security vulnerabilities.  This systematic approach to managing startup scripts has been instrumental in my success with various projects over the years.
