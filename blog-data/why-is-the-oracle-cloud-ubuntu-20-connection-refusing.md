---
title: "Why is the Oracle Cloud Ubuntu 20 connection refusing?"
date: "2024-12-23"
id: "why-is-the-oracle-cloud-ubuntu-20-connection-refusing"
---

Okay, let's tackle this. I've definitely seen my share of head-scratching connection refusals, particularly when dealing with cloud instances. The Oracle Cloud Infrastructure (oci) combined with Ubuntu 20 can occasionally present a few unique challenges, and 'connection refused' is often the symptom, not the root cause. Let's break down why this might be happening and, more importantly, how to diagnose and resolve it. From my experience, these issues usually boil down to a handful of suspects, which I'll detail here with examples.

First, let's discard the simple things: are you using the correct public ip address? I've certainly tripped myself up with outdated ip lists before. Also ensure you are using the proper ssh key. These sound like basic things, but it's essential to check before moving to more complex potential problems. This initial step often prevents a more prolonged investigation.

The most common culprit is a firewall misconfiguration, specifically around the port you're trying to connect through (typically port 22 for ssh). Both within the Oracle Cloud network and on the Ubuntu instance itself, firewalls are usually active by default. Let's examine how each can be problematic.

On the OCI side, your virtual cloud network (vnc) has security lists (formerly called network access control lists or nacls) that govern the traffic allowed in and out. If your vcn security list isn't configured to allow incoming tcp traffic on port 22 (or your custom ssh port), the connection will be blocked before it even reaches your instance. To check this, use the oci console or the oci cli and review the relevant security lists associated with your instance's subnet. I recall a particularly late night a few years back when a junior member of my team inadvertently created a new rule which only allowed icmp traffic and nothing else – very efficient at shutting things down!

Here's how a correct rule *should* look (as a representation of the concepts, not actual oci configuration):

```
{
  "direction": "ingress",
  "protocol": "tcp",
  "sourcePort": "any",
  "destinationPortRange": "22",
  "sourceCidr": "0.0.0.0/0" //allow from anywhere (for testing) - change in production.
}
```

The key part is that `destinationPortRange` is set to 22, and the `protocol` is `tcp`. The `sourceCidr` should be adjusted for production. If you were allowing, say, a range of ports, you would alter the rule to match.

Second, the ubuntu instance itself has a local firewall, often `ufw` (uncomplicated firewall). If ufw is enabled, and it isn't configured to allow ssh connections, that is also going to prevent incoming connections. `ufw` doesn't allow any traffic by default when enabled, so a careful configuration is required.

To check your ufw status on the ubuntu instance, you would usually use the following commands via a direct console connection (or an initial setup script):

```bash
sudo ufw status
```

This will show you if the firewall is active and what the current rules are. If it's inactive, that isn't the problem. If it's active and port 22 isn't explicitly allowed, that could be our culprit. To add a rule, you would use:

```bash
sudo ufw allow 22/tcp
```
followed by:
```bash
sudo ufw enable
```

This adds a rule that permits incoming traffic to port 22 using the tcp protocol, and then activates the firewall. It's important to note that rules can be more complex, allowing based on specific source addresses, which can be configured in the same way using different command line options. Remember that enabling the firewall without configuring the allowed rules usually leads to problems.

Another issue, though less common now with modern oci images, can be the ssh service not running. You would normally expect this service, `sshd`, to be active by default, but I have encountered situations where it's either stopped or not configured to start. You can confirm this via the console with the following command:

```bash
sudo systemctl status sshd
```
If the status is inactive or failed, you need to activate or correct the configuration. This can be done with the following:
```bash
sudo systemctl start sshd
```
and if you want to have the service start after every reboot:

```bash
sudo systemctl enable sshd
```

These commands will start the service and ensure it starts when the operating system boots.

Finally, if you are using a non-standard ssh port (not 22), ensure that all the firewall rules mentioned above are configured to reflect your custom port. For instance, if you changed your ssh port to 2222, the rules would refer to 2222/tcp instead of 22/tcp. There are legitimate reasons to change ssh port to something different, but remember that changing the port without updating all necessary security configuration often leads to further issues.

Debugging this scenario typically involves sequentially checking the following: 1) verify the correct ip and key are being used; 2) check the oci vcn security list for inbound port 22/tcp traffic; 3) check the ubuntu ufw configuration, enabling port 22/tcp; 4) verify the ssh service is running on the ubuntu instance.

To dive deeper into these networking principles, i would recommend two resources. First, “computer networking: a top-down approach” by james kurose and keith ross. it provides an excellent foundation in networking fundamentals, which is invaluable for troubleshooting any cloud connectivity issues. Second, for more context on cloud networking, i would suggest oracle's official cloud documentation, particularly the sections related to virtual cloud networks, security lists, and instance networking configurations, as those often change with updates to the platform. I personally found these two resources provided me with a solid understanding of network behavior that was invaluable during my projects and is the base for all troubleshooting of these connection issues.
Remember, ‘connection refused’ is merely a symptom. Methodically checking through these possibilities will invariably lead to the source of the issue, and it's often something relatively simple once it’s found.
