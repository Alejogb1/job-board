---
title: "Why is airflow web UI access denied after whitelisting IPs?"
date: "2024-12-23"
id: "why-is-airflow-web-ui-access-denied-after-whitelisting-ips"
---

,  I've seen this issue pop up more than a few times over the years, often leaving teams scratching their heads. The problem, at its core, isn't usually a flaw in the whitelisting mechanism itself, but rather a misunderstanding of how airflow, or more specifically, how its web server, handles these requests within its broader configuration context.

The scenario, as you've presented it, is common: you configure an airflow installation, set up your `webserver_config.py` or equivalent to whitelist specific ip addresses, and yet… access is still denied. Despite the seemingly straightforward approach of explicitly listing authorized IPs, the web ui consistently refuses connections originating from these exact locations. This generally stems from one of a few underlying configuration conflicts or misunderstandings of the request lifecycle within the webserver.

My past experiences dealing with similar cases have taught me that the devil is often in the detail. Let me walk you through the common culprits, and then I’ll demonstrate the solutions with some code examples.

**The Usual Suspects**

1. **Load Balancers and Proxies:** Quite frequently, the issue originates *before* the request even reaches airflow’s webserver. If there’s a load balancer or reverse proxy (like nginx or haproxy) in front of your airflow deployment, the *actual* source ip that airflow sees may not be the external ip that you've whitelisted. Rather, airflow sees the internal ip of the load balancer or proxy. The whitelisting, in this case, would be ineffective since it's attempting to match the wrong ip address. To correctly address this, it's necessary to ensure the `X-Forwarded-For` header, or a similar mechanism, is properly configured in the load balancer or proxy to preserve the originating client IP. Airflow’s webserver then needs to be configured to recognize and utilize this header for whitelisting decisions.
2. **Multiple `webserver_config.py` Files:** Another trap we've stepped into is having multiple configuration files in play. This can happen after upgrades or when dealing with multiple environments (dev, staging, prod) where different copies of `webserver_config.py` exist, accidentally overriding each other, or, sometimes, one or more copies of `airflow.cfg` with different configurations as well. The webserver can pick a config file that does not include your whitelisted ips. This is easy to overlook, especially in larger deployments. Double-check the exact path the airflow server is using to load configurations.
3. **Incorrect Format or Syntax:** The configuration entries themselves must follow the correct format. Airflow’s configuration is usually sensitive to syntax, which makes it imperative to get it right. A small error in listing the ip addresses, such as missing a comma or mixing ipv4 and ipv6 formats without proper syntax, can easily cause the entire whitelist to be ignored.
4. **Incorrect Configuration Section:** We’ve seen cases where the ip whitelisting configuration is placed in the wrong section of the configuration file or in an incorrect file. The most common location is `webserver.expose_config` or `webserver.filter_request_headers`, but it may be defined elsewhere in newer versions. Always reference the relevant version documentation to ensure you are modifying the expected settings.
5. **Insufficient Server Restart:** Finally, sometimes, after changes to the configuration files, a simple restart of the airflow scheduler or webserver may not be enough for the modifications to take effect. In some cases a complete service restart or sometimes restarting the whole vm/container might be necessary, especially in complex deployments using multiple docker containers or kubernetes pods.

**Code Examples: Illustrating the Fixes**

To make these points a bit more concrete, I'll present a few python code examples to demonstrate how to fix the most common issues. Note that these snippets are not directly executable but illustrate how you should construct configurations, keeping in mind these snippets depend on where in the file structure the airflow webserver is reading them from.

**Example 1: Configuring Airflow to Use `X-Forwarded-For`**

Assuming your load balancer or reverse proxy is passing the client's ip in the `X-Forwarded-For` header, you need to configure airflow's webserver to utilize it for ip whitelisting. In airflow 2.0 or later versions the configurations are typically within the `airflow.cfg` or `webserver_config.py` files. Here is how you would typically configure `airflow.cfg`:

```ini
[webserver]
filter_request_headers = X-Forwarded-For, X-Forwarded-Host, X-Forwarded-Proto
access_control_allow_origin =  * # or specify your domain
```
In this snippet, you are setting the `filter_request_headers` to instruct airflow to utilize the listed headers for access control, including the `X-Forwarded-For` header.

**Example 2: Correct Whitelist Syntax**

This example showcases the correct syntax when whitelisting ips in your `webserver_config.py` or equivalent. Remember that these settings may be configured in different configuration files depending on your airflow version:

```python
# webserver_config.py

import logging

ALLOWED_HOSTS = [
    "127.0.0.1",
    "192.168.1.100",
    "10.0.0.0/24", #CIDR Notation
    "2001:0db8:85a3:0000:0000:8a2e:0370:7334" #IPv6 example
]

def init_appbuilder(appbuilder):
    logging.info("Initializing security using ALLOWED_HOSTS: %s", ALLOWED_HOSTS)

    from airflow.providers.fab.auth_manager.security_manager.override_security_manager import (
        OverrideSecurityManager
    )
    appbuilder.sm = OverrideSecurityManager(appbuilder.sm, ALLOWED_HOSTS)
```
This snippet illustrates how to use the `ALLOWED_HOSTS` list to explicitly define allowed ips and ip ranges in both ipv4 and ipv6 formats. Notice the inclusion of CIDR notation for ip ranges.

**Example 3: Debugging Configuration Load Order**

This snippet demonstrates how to debug the airflow webserver to determine where it's actually reading configurations from. Adding logging statements to your configuration loading methods could help you determine the correct configuration file that is being used.

```python
#Inside airflow.cfg/webserver_config.py or the location it reads the configurations from
import logging
import os

logging.basicConfig(level=logging.INFO)
config_file_path = os.environ.get('AIRFLOW_CONFIG', 'default')
logging.info(f"Airflow config file path being used: {config_file_path}")


#Other configurations here.

```
This script prints the path from which airflow configuration is loaded, allowing you to verify that the correct files are being used and troubleshoot any discrepancies.

**Recommended Resources for Further Study**

To gain deeper knowledge into the topics we've discussed, I highly recommend the following resources:

*   **RFC 7239 (Forwarded HTTP Extension):** This IETF document provides the formal specification for the `Forwarded` and `X-Forwarded-*` headers used by proxies and load balancers. Understanding this document will greatly improve your ability to correctly handle IP whitelisting behind reverse proxies.
*   **"Networking for System Administrators" by Michael Lucas:** This book offers an excellent foundation in networking principles, including concepts such as CIDR notation and load balancing, which are highly relevant to this issue.
*   **Official Apache Airflow Documentation:** The most important resource is, of course, the official airflow documentation, which includes sections on security, configuration settings and troubleshooting. Pay attention to the documentation of the specific version of airflow you are using, as configuration options may differ across versions.

In conclusion, the issue of airflow web ui access being denied after ip whitelisting is seldom a bug in the core access control implementation but, more commonly, a misconfiguration due to load balancers, proxies, incorrectly formatted settings, file loading issues, or a misunderstanding of the server's request handling process. Always carefully examine the entire chain of request flow from the client to the airflow webserver, verifying that your intended source ips are indeed what the webserver is seeing and correctly configured in your configuration files. Pay attention to the format and syntax of your configurations, and make use of debug statements, while using authoritative documentation to guide your configurations.
