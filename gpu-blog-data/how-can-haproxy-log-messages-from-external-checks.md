---
title: "How can HAPROXY log messages from external checks be routed to rsyslog?"
date: "2025-01-30"
id: "how-can-haproxy-log-messages-from-external-checks"
---
The core challenge in routing HAProxy's external check logs to rsyslog stems from HAProxy's limited native logging flexibility regarding these specific events. Typically, HAProxy's logging configuration directs traffic and connection-related events, but external check messages, originating from scripts or services, often require a separate handling approach. I've encountered this situation numerous times while managing load balancers for high-availability applications. The key is realizing that HAProxy treats these check results as generic output from an external process and, by default, doesn't offer fine-grained control for their log destinations. We need to leverage system capabilities to intercept and redirect this information.

The most straightforward solution involves configuring the external health check script to explicitly write its output to a file or a named pipe, which rsyslog can then be configured to monitor and process. This is a workaround as HAProxy itself does not directly facilitate rsyslog integration for its external checks.

**Understanding the Flow**

First, HAProxy executes your external health check script or command. This script performs some predefined action like querying a service's HTTP endpoint or executing a database query. Upon completion, the script typically produces output indicating the health status of the checked resource. This output usually takes the form of a single line message indicating UP, DOWN or any other relevant status. HAProxy, based on the return code of this script, decides on the server's availability and status, but the script's *output* itself, isn't directly integrated into the standard HAProxy logging system. This is where the redirection comes into play.

**Configuring the External Check and Redirection**

Instead of letting the script output to standard output, we redirect it to a specific location, typically a named pipe. A named pipe acts as a kind of shared file, which is accessible to both the script and rsyslog.

**Code Example 1: Script Modification and HAProxy Configuration**

Here's an example of modifying an external check script and configuring HAProxy:

```bash
#!/bin/bash
# healthcheck.sh - simple http check
URL="http://localhost:8080/health"
RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" "$URL")

if [[ "$RESPONSE" -eq 200 ]]
then
  echo "UP" > /tmp/healthcheck.pipe
  exit 0
else
  echo "DOWN" > /tmp/healthcheck.pipe
  exit 1
fi
```

**Commentary:** This script sends an HTTP request. Instead of outputting to stdout, it redirects "UP" or "DOWN" to `/tmp/healthcheck.pipe`. This pipe is where rsyslog will pull the data from.

Now, in HAProxy's configuration file, we define the external check with the path to the above script:

```
frontend http-in
  bind *:80
  default_backend servers

backend servers
  server server1 127.0.0.1:8080 check check-type external /path/to/healthcheck.sh
```

**Commentary:** Notice the `check-type external` and path `/path/to/healthcheck.sh` which dictates the script used as an external check. The crucial part here is that we need to make sure that the script does redirect the output to the named pipe we established in the bash script `/tmp/healthcheck.pipe`

**Code Example 2: Rsyslog Configuration**

Now, we need to configure rsyslog to monitor and forward the data from the pipe to the desired log file. Below is an rsyslog configuration block that captures data from `/tmp/healthcheck.pipe` and stores it into `/var/log/haproxy_extchecks.log`:

```rsyslog
# module load imfile
module(load="imfile" PollingInterval="1")

# Configure input from the named pipe
input(type="imfile"
      File="/tmp/healthcheck.pipe"
      Tag="haproxy_extcheck"
      Severity="info"
      persistStateInterval="10"
      )

# Define action to write log to specific file
action(type="omfile"
       File="/var/log/haproxy_extchecks.log"
       template="<%timegenerated%> %HOSTNAME% %syslogtag% %msg%\n"
       )

```

**Commentary:**  This `rsyslog` configuration first loads the necessary `imfile` module to allow for reading files. The `input` module specifies the file to monitor, the tag to mark these specific log lines, and a severity of "info". Crucially, `persistStateInterval` is set to ensure rsyslog does not repeatedly reprocess old content of the pipe. The `action` module then writes log lines to `/var/log/haproxy_extchecks.log` using a custom template. The template adds timestamp, hostname, syslogtag, and message. This is a simplified template and it is recommended to tailor the template to specific logging standards.

**Code Example 3: Creating the Named Pipe**

Before executing the script for the first time, we need to ensure a named pipe exists:

```bash
mkfifo /tmp/healthcheck.pipe
```

**Commentary:** This command creates a named pipe which our health check script can output to and which rsyslog can read.

**Key Considerations**

1.  **Error Handling:** The script should have sufficient error handling. You must catch various issues including service failures, network glitches etc., and log informative messages to assist in debugging. The use of proper log level (like "error" instead of "info" on errors) is highly recommended.
2.  **Permissions:** Ensure that HAProxy's user has the appropriate permission to execute the external check script, that the script can write to `/tmp/healthcheck.pipe`, and that rsyslog has permissions to read from it. In secure production environments, you might need to adjust permissions or consider using dedicated log directories and different named pipes for each health check script.
3.  **Log Rotation:** Configure log rotation for `/var/log/haproxy_extchecks.log` to prevent the log file from growing indefinitely. Tools like `logrotate` are indispensable here.
4.  **Performance:**  While named pipes are efficient, extremely high-frequency checks may introduce a performance bottleneck. In high-load situations, you may need to monitor I/O performance and consider alternative logging methods or adjustments of health check frequency.
5.  **Clarity in Logs:** Employ meaningful messages in the script to clarify the health status (e.g. "Database connection failed" instead of simply "DOWN"). This can simplify debugging.
6.  **Centralized Logging:** For complex systems, consider centralizing log aggregation using systems like the ELK stack, Splunk, or similar tools. Rsyslog can be easily configured to forward data to such central logging platforms.
7.  **Security:** Secure the named pipe by restricting access, considering that sensitive data might be transferred through it depending on your script's outputs.

**Resource Recommendations**

For deeper understanding of the tools involved:

*   **The HAProxy documentation:** This is the definitive resource for the HAProxy configuration.
*   **The rsyslog documentation:** Explores the advanced input modules and output formats for rsyslog.
*   **Bash scripting resources:** Provides guidance on writing more robust and informative shell scripts, including error handling and best practices.
*   **Linux file system and permissions guides:** Helps with securing the named pipe and access to log files in production environments.
*   **General networking resources:** Explains TCP/IP protocol, which might be useful if the check involves a network component or service.

This method provides a practical workaround for logging external check messages from HAProxy into rsyslog, leveraging the existing system capabilities. While HAProxy might not offer direct logging integration for this specific feature, the above method allows for effective and controllable logging of these crucial operational events.
