---
title: "How can I run Airflow scheduler and webserver concurrently?"
date: "2025-01-30"
id: "how-can-i-run-airflow-scheduler-and-webserver"
---
The core challenge in concurrently running the Airflow scheduler and webserver lies in managing their distinct resource requirements and avoiding process conflicts.  Over the years, I've encountered numerous instances where improper configuration led to instability or outright failure, particularly in production environments with high concurrency.  My approach consistently prioritizes clean separation of processes, leveraging operating system features for robust management.

**1. Explanation:**

Airflow's scheduler and webserver are designed as independent processes, each with specific responsibilities.  The scheduler is responsible for monitoring DAGs, triggering tasks, and managing the overall workflow execution.  The webserver provides the user interface for monitoring, managing, and interacting with the Airflow environment. Running them concurrently is essential for a functional Airflow installation; otherwise, users cannot interact with the system, even while DAGs are scheduled.

To achieve concurrent operation, we must avoid interfering with their respective resource needs.  This primarily involves ensuring they bind to different ports and that the scheduler has sufficient resources to handle task scheduling without being impacted by webserver traffic.  Furthermore, employing a process supervisor like systemd (Linux) or Supervisor (cross-platform) provides a mechanism for monitoring, restarting, and managing both processes effectively, handling potential failures gracefully.  Improper resource allocation can lead to scheduler slowdowns, task queuing bottlenecks, and even complete system crashes.  Dedicated resource allocation, using tools like cgroups (Linux), is beneficial in production settings.

The configuration of Airflow itself is crucial.  The `airflow webserver` and `airflow scheduler` commands are the entry points, and their configurations, found in the `airflow.cfg` file, determine port numbers, database connections, and other critical settings.  Ensuring both processes point to the same metadata database is paramount for consistent operation. Mismatched configurations will lead to operational inconsistencies and data corruption.


**2. Code Examples with Commentary:**

**Example 1: Using Systemd (Linux)**

This example shows how to configure systemd service files for both the scheduler and the webserver.  This approach ensures both processes start at boot and are automatically restarted upon failure.  In my experience, this is the most reliable method for production deployment on Linux systems.

```ini
# /etc/systemd/system/airflow-scheduler.service
[Unit]
Description=Apache Airflow Scheduler
After=network.target

[Service]
User=airflow  # Replace with your Airflow user
Group=airflow # Replace with your Airflow group
WorkingDirectory=/opt/airflow  # Replace with your Airflow installation directory
ExecStart=/opt/airflow/airflow scheduler
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target


# /etc/systemd/system/airflow-webserver.service
[Unit]
Description=Apache Airflow Webserver
After=network.target

[Service]
User=airflow  # Replace with your Airflow user
Group=airflow # Replace with your Airflow group
WorkingDirectory=/opt/airflow  # Replace with your Airflow installation directory
ExecStart=/opt/airflow/airflow webserver -p 8081  #Note different port
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

After creating these files, run `sudo systemctl enable airflow-scheduler` and `sudo systemctl enable airflow-webserver` to enable them, followed by `sudo systemctl start airflow-scheduler` and `sudo systemctl start airflow-webserver` to start them. Remember to adjust paths and user/group accordingly.  Oversight in these details has caused numerous headaches for me in the past.


**Example 2: Using Supervisor (Cross-Platform)**

Supervisor offers a more portable solution, working effectively on Linux, macOS, and Windows. Its configuration file defines the processes to be managed.

```ini
[supervisord]
nodaemon=true

[program:airflow_scheduler]
command=/opt/airflow/airflow scheduler
autostart=true
autorestart=true
user=airflow
redirect_stderr=true
stdout_logfile=/var/log/airflow/scheduler.log  # Customize log path

[program:airflow_webserver]
command=/opt/airflow/airflow webserver -p 8081 #Note different port
autostart=true
autorestart=true
user=airflow
redirect_stderr=true
stdout_logfile=/var/log/airflow/webserver.log # Customize log path
```

This configuration file, typically named `supervisord.conf`, needs to be placed in the appropriate location for Supervisor and then Supervisor must be started.  Careful attention should be paid to logging, as detailed logs are crucial for debugging issues.  I've found inadequate logging to be a significant impediment to resolving operational problems.


**Example 3: Basic Shell Script (for testing only)**

This example should only be used for testing purposes in non-production environments.  It lacks the robustness and monitoring of systemd or Supervisor.

```bash
#!/bin/bash

# Start scheduler in background
airflow scheduler &

# Start webserver in background on a different port
airflow webserver -p 8081 &

wait
```

This script starts both processes in the background.  However, it lacks the essential features of process monitoring and automatic restarting offered by systemd or Supervisor.  Using this approach for production is strongly discouraged; I've seen many deployments suffer from unexpected crashes using similar simplistic methods.


**3. Resource Recommendations:**

* **Process Supervisors:** Systemd (Linux), Supervisor (cross-platform).  These are crucial for managing and monitoring the processes, ensuring reliability.

* **Configuration Management:** Ansible, Puppet, Chef.  These tools enable automated and consistent deployment of Airflow across multiple environments.

* **Monitoring Tools:** Prometheus, Grafana.  Monitoring system resource utilization and Airflow's health is paramount for identifying and resolving issues proactively.  Regular monitoring saved me countless hours of troubleshooting in the past.

* **Logging System:**  A centralized logging system such as ELK stack (Elasticsearch, Logstash, Kibana) provides consolidated logging for efficient troubleshooting.


Employing these recommendations and best practices for process management, resource allocation, and monitoring will ensure the robust and reliable concurrent operation of Airflow's scheduler and webserver, minimizing operational disruptions and maximizing system uptime.  Ignoring these aspects, based on my experience, invariably leads to operational difficulties in the long term.
