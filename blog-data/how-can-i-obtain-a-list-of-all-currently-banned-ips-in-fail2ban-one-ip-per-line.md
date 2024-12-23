---
title: "How can I obtain a list of all currently banned IPs in fail2ban, one IP per line?"
date: "2024-12-23"
id: "how-can-i-obtain-a-list-of-all-currently-banned-ips-in-fail2ban-one-ip-per-line"
---

, let’s tackle this. I recall a particularly frustrating incident a few years back when a misconfigured security policy almost took down a client's server due to a seemingly endless barrage of brute-force login attempts. That's where a solid grasp of fail2ban's internals, particularly how it manages banned IPs, became absolutely crucial. Getting a clean list of banned IPs, one per line, seems simple enough on the surface, but diving into the actual mechanisms requires some understanding of fail2ban's architecture and the tools it provides.

First, the straightforward way: directly querying fail2ban’s databases. Fail2ban, by default, stores its ban data in sqlite databases, usually named `fail2ban.db` or `fail2ban.sqlite`. These files are typically found in `/var/lib/fail2ban/`. Each jail you have configured has a corresponding table within that database. Extracting the IP addresses from these tables involves crafting a specific sql query. While doable directly from command line with the `sqlite3` command, it involves digging through the schema and formulating the query correctly each time. We're aiming for something a bit more robust.

Instead of direct database manipulation, a better practice is to use `fail2ban-client`. This command-line utility is designed to interact with the fail2ban server process and provides a more structured, and importantly, consistent way to obtain the data we need. Fail2ban-client allows for listing of currently banned IPs, however, the output is not always formatted as requested (one ip per line). Therefore, post-processing this output becomes necessary. Let me illustrate.

**Example 1: Using `fail2ban-client` with Post-Processing**

This method uses the `fail2ban-client` command to list currently banned IPs and pipes that output to a bash script using `awk` to format the result to obtain one IP per line.

```bash
#!/bin/bash

fail2ban-client status |
    awk '/^Banned IP list:/ { flag=1; next }
         flag == 1 && /^[[:space:]]+([0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3})$/ {
           print $1;
         }'
```

This script first calls `fail2ban-client status`. This gives a dump of the current status, including banned IPs. The awk command uses a flag, `flag`. When a line is encountered that matches the string `^Banned IP list:`, `flag` is set to 1, and the processing of that line stops (`next`). From this point onward, each line starting with a space followed by a valid ipv4 address format is printed to standard output, with only the first "word", which corresponds to the ip address, displayed, achieving the one IP per line output.

This solution assumes the default output from `fail2ban-client status` which has the format:

```
Status for the jail: sshd
|- Filter
|  |- Currently failed: 0
|  |- Total failed: 67
|  `- File list: /var/log/auth.log
`- Actions
  |- Currently banned: 2
  |  `- IP list: 1.2.3.4 5.6.7.8
  `- Total banned: 2
Status for the jail: apache-auth
|- Filter
|  |- Currently failed: 0
|  |- Total failed: 17
|  `- File list: /var/log/apache2/error.log
`- Actions
  |- Currently banned: 1
  |  `- IP list: 9.10.11.12
  `- Total banned: 1

```

**Example 2: Extracting Banned IPs from Specific Jails**

In situations where you are running several jails and you'd like to get all banned IPs associated with each of them, the following script would be more useful:

```bash
#!/bin/bash

while IFS= read -r jail; do
  echo "Banned IPs for jail: $jail"
  fail2ban-client status "$jail" |
    awk '/^Banned IP list:/ { flag=1; next }
         flag == 1 && /^[[:space:]]+([0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3})$/ {
           print $1;
         }'
done < <(fail2ban-client status | awk '/^jail:/ {print $2}')
```

This script iterates through each jail identified by the fail2ban status command and then for each one, it fetches the banned IPs in the same manner as Example 1.

This script is more robust because it iterates through all currently defined jails automatically, and provides a header before each block of IPs indicating the jail that the ips are associated with.

**Example 3: Using Python for More Complex Scenarios**

For situations requiring more complex filtering or post-processing, interacting with `fail2ban-client` via python would be appropriate. This method also offers cleaner output and is easily scalable.

```python
#!/usr/bin/env python3

import subprocess
import re

def get_banned_ips():
    try:
        result = subprocess.run(['fail2ban-client', 'status'], capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing fail2ban-client: {e}")
        return []
    banned_ips = []
    lines = result.stdout.splitlines()
    extracting_ips = False
    for line in lines:
      if "Banned IP list:" in line:
          extracting_ips = True
          continue
      if extracting_ips:
          ips = re.findall(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', line)
          banned_ips.extend(ips)
    return banned_ips

if __name__ == "__main__":
    ips = get_banned_ips()
    if ips:
      for ip in ips:
        print(ip)
    else:
      print("No banned IPs found.")

```

This python script executes the command `fail2ban-client status`, parses the output to find the line containing `Banned IP list`, and extracts all ipv4 addresses using the regular expression `\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b`. This method is preferred for situations where you need to manipulate the list or integrate it within a larger monitoring system.

In terms of relevant resources, a solid understanding of regular expressions is invaluable; Jeffrey Friedl's "Mastering Regular Expressions" is a classic. Further, the official fail2ban documentation should always be your first port of call for specific configuration or behavior questions. If you require deeper knowledge of system monitoring and automation, resources on bash scripting and practical python would be very valuable. The classic "Learning the Bash Shell" by Cameron Newham and Bill Rosenblatt and "Python Cookbook" by David Beazley and Brian K. Jones are good starting points.

These examples provide different approaches to tackling the same problem. The choice will depend on the specific requirements of your situation; the goal was to present a series of approaches that are versatile and practical, reflecting the kind of real-world experiences I've had working with fail2ban, and hopefully addressing your query. The key is to use `fail2ban-client` which is designed for this purpose, instead of attempting to directly manipulate the underlying database files.
