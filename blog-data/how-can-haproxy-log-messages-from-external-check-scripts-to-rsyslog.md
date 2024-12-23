---
title: "How can HAPROXY log messages from external check scripts to rsyslog?"
date: "2024-12-23"
id: "how-can-haproxy-log-messages-from-external-check-scripts-to-rsyslog"
---

Alright, let's tackle logging external check script messages through HAProxy to rsyslog. It's a topic I've spent a good chunk of time on, having dealt with some gnarly load balancing configurations myself over the years. Back in my days maintaining a large-scale application platform, we relied heavily on HAProxy for routing and health checks. However, we quickly realized that the default HAProxy logging was insufficient for detailed health check analysis. This led us down the path of integrating external check scripts and figuring out how to effectively capture their output within our existing logging infrastructure.

The core challenge lies in the fact that HAProxy, by default, doesn’t directly capture the standard output or standard error streams of external check scripts. These scripts, typically executed via `option httpchk`, run outside the HAProxy process context. Therefore, we need a mechanism to funnel this information back into HAProxy, which can then forward it to rsyslog. My solution, and the one I’ll explain here, involves a bit of a workaround using custom log formats and HAProxy's log-output capabilities. We’ll need to ensure the external scripts themselves are designed to communicate the information in a structured way. Let's dive into the practical steps and rationale behind them.

First, the external check script must generate structured, easily parsable output. We're not dealing with unstructured text; we need key-value pairs or a similar format that HAProxy can later process. For instance, let's consider a simple Python script named `health_check.py`:

```python
import sys
import json

def main():
  # Simulate a health check - replace with your actual logic
  healthy = True  # Example: check database connection or service availability
  if len(sys.argv) > 1 and sys.argv[1] == 'fail':
     healthy = False
  
  status = "healthy" if healthy else "unhealthy"
  data = {
    "status": status,
    "latency_ms": 5, # Simulated latency, replace with actual
    "message": f"service health: {status}"
  }
  
  print(json.dumps(data))
  sys.exit(0) # Exit with 0 on success, or another code on failure for HAProxy checks


if __name__ == "__main__":
  main()
```

This script, when executed, produces a json string on standard out, which contains health status, latency and a descriptive message. Now, the crucial part: making HAProxy understand and log this information. In your HAProxy configuration (typically `haproxy.cfg`), you'd need to define a custom log format and utilize the `log-format` option within the `backend` section. I'll use a specific format to extract our json fields. In my example below, I used custom variables to parse the json string:

```
frontend http-in
    bind *:80
    acl healthy_check_fail path_beg /fail
    use_backend my_backend if healthy_check_fail
    default_backend my_backend

backend my_backend
   server app1 127.0.0.1:8080 check port 8080 inter 2s rise 2 fall 3
   http-check send meth GET uri /health
   option httpchk GET /health

   http-request set-var(req.healthcheck_result) str(0) #default failure

    # Check script with result processing:
    http-check expect rstatus (200|204|205|301|302|303|304|307|308) or status-code 404 # Check basic HTTP response

    http-request set-var(req.healthcheck_result) str(1) if status 200 # Mark as success

   http-check command /usr/bin/python3 /path/to/health_check.py set-var(req.healthcheck_output)
   http-request set-var(req.json_log) str(%[var(req.healthcheck_output)]) if { var(req.healthcheck_output) -m found }
   http-request set-var(req.json_status) str(%[var(req.json_log),json_path($.status)]) if { var(req.healthcheck_output) -m found }
   http-request set-var(req.json_latency) str(%[var(req.json_log),json_path($.latency_ms)]) if { var(req.healthcheck_output) -m found }
   http-request set-var(req.json_message) str(%[var(req.json_log),json_path($.message)]) if { var(req.healthcheck_output) -m found }

   log-format  "backend=%b,srv=%s,http_status=%H,check_result=%[var(req.healthcheck_result)],status=%[var(req.json_status)],latency_ms=%[var(req.json_latency)],message=%[var(req.json_message)],%t %T"
   log global
```
In this configuration we define a `http-check` which runs the python script and use custom variables to capture the json output, status, latency and message from our script. We can then log these variables through a custom log format.

Finally, we need to ensure HAProxy sends these logs to rsyslog. This is usually done in the global section of `haproxy.cfg`:

```
global
   log /dev/log local0
   log 127.0.0.1:514 local0 notice
   #other global settings ...
```

The first line directs logs to the system log (using the facility `local0`). The second sends logs via UDP to port 514 on the loopback address. This is typical, but it will depend on your rsyslog configuration.

On the rsyslog side, you need a configuration to capture these logs, typically in a file such as `/etc/rsyslog.d/haproxy.conf`. I used the same facility `local0` as defined in my `haproxy.cfg`:

```
if $programname == 'haproxy' and $syslogfacility-text == 'local0' then {
  action(type="omfile" file="/var/log/haproxy-healthchecks.log")
  stop
}
```
This configuration directs all messages from `haproxy` using the facility `local0` to `/var/log/haproxy-healthchecks.log`. This will be specific to each individual set up.

Now, a few crucial points to consider. First, the use of custom variables. These enable parsing the external output and making specific fields available in log statements. Second, error handling in external scripts is paramount. If your scripts throw an exception and exit abruptly, HAProxy might not log anything valuable. Implement robust error checking and logging within your scripts themselves. Third, performance is a concern. While custom logging formats are powerful, extensive processing can add overhead to HAProxy. It’s a balancing act between detailed logs and system resources. Always monitor the performance impact of your logging configurations. Fourth, always test your setup thoroughly before deploying to production. Start with basic cases and gradually add complexity. Log aggregation and monitoring is a critical component of production systems.

For a deeper dive, I recommend reading the HAProxy documentation thoroughly, specifically the sections on `http-check`, `log-format`, and variables. A good reference is the official HAProxy documentation, which I’ve found to be the most detailed resource. As for rsyslog, the official rsyslog documentation will be beneficial, particularly the parts about configuring input and output modules. It's important to understand the configuration options available on both sides to create the desired logging solution. If you want to dive deeper into HTTP and related topics, the book "HTTP: The Definitive Guide" by David Gourley is incredibly useful. Also, for logging and monitoring in distributed systems, you could check out “Distributed Systems: Concepts and Design” by George Coulouris. These resources offer detailed explanations that go beyond the scope of this response.

Integrating external check scripts with rsyslog through HAProxy requires a well-planned approach and a strong understanding of each component. By utilizing structured script output, custom log formats, and robust error handling, you can capture and analyze vital health check data. It's a challenge, but with a systematic method, it’s certainly manageable and can provide crucial insights into your application's overall performance.
