---
title: "How can fail2ban be configured for a new SSH port?"
date: "2024-12-23"
id: "how-can-fail2ban-be-configured-for-a-new-ssh-port"
---

Alright, let's tackle this one. I've spent my fair share of evenings troubleshooting rogue access attempts, so configuring fail2ban for a non-standard ssh port is a process I'm quite familiar with. It's more than just changing a single line; it involves understanding how fail2ban operates and adapting its configurations to correctly identify and block malicious actors targeting your specific setup.

The core challenge when deviating from the standard port 22 is that fail2ban, by default, is set up to monitor logs that primarily reflect activity on this particular port. If you've moved ssh to, say, port 2222, fail2ban's default filters and jail configurations will remain fixated on port 22, effectively blind to attacks on your actual active port. The beauty of fail2ban, however, is its flexibility, allowing you to adapt it to this kind of change.

The first step is to ensure your ssh service is indeed running on the new port. This typically involves modifying your sshd_config file, located usually at `/etc/ssh/sshd_config`. Look for the line containing `Port 22`, and change it to your desired port number, like so `Port 2222`. Remember to restart the ssh service after making this change for the new port to take effect (e.g., `sudo systemctl restart sshd`). After that check using `sudo ss -tlpn | grep ssh` to make sure ssh is running on the new port.

Now, let's turn our attention to fail2ban. It's structured around the concept of "jails," which define rules for identifying and banning malicious activity. Each jail is composed of a filter and an action. The filter defines the log pattern to match, and the action dictates what happens when a match is detected (usually, banning an IP address).

For a new ssh port, we often need to create or modify a filter. Fail2ban filters are typically defined in `.conf` or `.local` files under `/etc/fail2ban/filter.d/`. We’ll modify the `sshd.conf` or create a new one if it does not exist.

Here’s an example of a customized filter file, let's call it `sshd-custom.conf` or `sshd-newport.conf`, which you can place in `/etc/fail2ban/filter.d/`:

```
[Definition]
failregex = ^%(__prefix_line)sFailed password for (invalid user )?(?P<user>\S+) from <HOST>( port \d+)?\s*$
            ^%(__prefix_line)sReceived disconnect from <HOST>(:[\d]+)?(: (Bye bye|Too many authentication failures for \S+))?\s*$
            ^%(__prefix_line)sInvalid user \S+ from <HOST>( port \d+)?\s*$
            ^%(__prefix_line)sreverse mapping checking getaddrinfo for .* \[<HOST>\] failed - POSSIBLE BREAK-IN ATTEMPT!\s*$
ignoreregex =
```
This filter definition works on sshd logs with `Failed password`, `disconnect` due to failed attempts or `invalid user` attempts which are quite common, and it should function regardless of the ssh port used because it doesn't explicitly specify one. If it doesn’t detect attacks for your specific log format, you may have to analyze the logs with `grep` and adjust the failregex accordingly. If the new log contains port information, we may have to add the following line after `<HOST>`:
`^%(__prefix_line)sFailed password for (invalid user )?(?P<user>\S+) from <HOST> port <PORT>\s*$`

Next, we need to create or modify a corresponding jail configuration. Jail configurations are defined in `.conf` or `.local` files under `/etc/fail2ban/jail.d/`. Let's assume we’re modifying the default `jail.local`, or if it’s missing, you should add one. You need to add a section like this:

```ini
[sshd-custom]
enabled = true
port = 2222
filter = sshd-custom
logpath = /var/log/auth.log
maxretry = 5
bantime  = 3600
findtime = 600
backend = systemd
```

or

```ini
[sshd-newport]
enabled = true
port = 2222
filter = sshd-custom
logpath = /var/log/auth.log
maxretry = 5
bantime  = 3600
findtime = 600
backend = systemd
```

Here, `enabled = true` activates the jail. `port = 2222` specifies the port we're monitoring; this is the critical part for our custom port. The filter name needs to match the one we defined previously (`filter = sshd-custom`). `logpath = /var/log/auth.log` points to the location where the auth logs are stored (this may be different depending on your system, for example, it can be `/var/log/secure` on some distributions). `maxretry`, `bantime`, and `findtime` define the thresholds for banning. `maxretry = 5` means 5 failed attempts within `findtime=600` (10 minutes) will lead to a ban. `bantime = 3600` will ban the IP for 1 hour. Also, I have added the `backend = systemd` option which is the common logging mechanism in recent distros.

After making the changes, you need to restart fail2ban for the configurations to be reloaded:
`sudo systemctl restart fail2ban`.

Now that you have the basics covered, let's get a bit more practical. Imagine a scenario from one of my previous projects. I was securing a database server that had ssh access restricted to a non-standard port, say, 2222, like the example mentioned. In this instance, the initial filter provided by the operating system didn't adequately capture all failed login attempts, partly due to some variation in the log format. So, I had to modify the failregex as follows, to account for failed login attempts that included the port explicitly:
```
[Definition]
failregex = ^%(__prefix_line)sFailed password for (invalid user )?(?P<user>\S+) from <HOST> port \d+\s*$
            ^%(__prefix_line)sReceived disconnect from <HOST>(:[\d]+)?(: (Bye bye|Too many authentication failures for \S+))?\s*$
            ^%(__prefix_line)sInvalid user \S+ from <HOST> port \d+\s*$
            ^%(__prefix_line)sreverse mapping checking getaddrinfo for .* \[<HOST>\] failed - POSSIBLE BREAK-IN ATTEMPT!\s*$
ignoreregex =
```
This made sure that failed login attempts on port 2222, explicitly, were being picked up.

One thing to be aware of is the log level of your sshd configuration. Make sure the logging level is detailed enough so that fail2ban can pick up the log data. Typically, setting the level in `/etc/ssh/sshd_config` to `LogLevel VERBOSE` or `LogLevel DEBUG` is enough for capturing the events.

Now for some resources:

1.  **"Practical Packet Analysis" by Chris Sanders**: While not directly about fail2ban, understanding network protocols and log analysis, which this book thoroughly covers, is crucial for fine-tuning fail2ban filters, especially when encountering log format variations. The ability to understand the log formats and tcpdump captures is paramount to troubleshoot any firewall rule that is not working as expected.

2.  **"Linux Firewalls: Enhancing Security with nftables and Beyond" by Steve G. Shah**: This book delves into the intricacies of Linux firewalls. While fail2ban is more of a log analyzer, having a comprehensive grasp of firewalls in Linux environments helps understand how fail2ban integrates with them. It can guide you on using the iptables or nftables that are used by fail2ban on the backend to set rules and block the IPs.

3.  **The official fail2ban documentation:** The official documentation, found on the fail2ban website and its GitHub repository, remains the most authoritative source. It is detailed, up-to-date, and provides solutions to numerous situations, including examples of how to setup filters, and jail configurations, and includes advanced configuration setups.

In essence, configuring fail2ban for a non-standard ssh port is about properly configuring the filter and jail definitions to monitor your specific log format on the port you've defined. These adjustments ensure fail2ban isn't sitting idle while the bad guys are knocking on your door. With the example configurations and the mentioned resources, you should have a good starting point for securing your server.
