---
title: "Is exim4 experiencing a mail loop due to excessive 'Received' headers?"
date: "2024-12-23"
id: "is-exim4-experiencing-a-mail-loop-due-to-excessive-received-headers"
---

Let's tackle this intriguing situation with exim4 mail loops and excessive "received" headers. It's a problem I’ve seen surface more than once in my time, and it's rarely straightforward to pinpoint. The short answer is: yes, an excessive buildup of "received" headers *can* absolutely cause a mail loop, but it's usually a symptom of a larger configuration or infrastructure issue, not the root cause itself.

The "received" header, for those who might not be intimately familiar, is an essential part of the email delivery process. It acts as a breadcrumb trail, logging each hop an email takes between mail servers. Each mail server that handles the message adds its own "received" header, which includes details like the hostname, IP address, protocol used (e.g., esmtp), and timestamps. This information is invaluable for debugging routing problems and verifying mail authenticity. However, if an email finds itself bouncing back and forth between servers, each iteration adds another "received" header, and this accumulation can, in certain circumstances, lead to problems, ultimately resulting in a mail loop.

In my experience, what usually precipitates this header accumulation is a misconfigured mail server or a flawed interaction between multiple servers. Imagine two servers, both believing they are the final destination for a particular email. They both accept it, but instead of delivering it to a mailbox, they resubmit it to the other, creating an infinite loop. While exim4 has robust safeguards against simple loops, certain configurations or external factors can circumvent these protections.

Let’s get into why these excessive "received" headers are such a red flag. First, as the number of these headers grow, so does the overall size of the email message. Beyond a certain point, this will become problematic. Exim4 imposes limits on message sizes, and if the accumulation of headers causes the email to exceed this limit, it may be rejected, potentially triggering error messages, and potentially even contributing to the aforementioned loop. Second, parsing an email with an extremely large number of headers will consume system resources, potentially slowing down the server and affecting its capacity to handle new emails. Finally, and most importantly, the continuous back-and-forth and header addition is clearly an anomaly that needs investigation, and even if it doesn't immediately cause rejection, it's a symptom of a serious underlying routing issue.

Now, let's consider specific exim configurations and code snippets where this kind of problem can arise. Suppose you have an exim4 configuration that uses custom transport rules for certain domains. Let’s illustrate a simplified transport configuration:

```
# Transport definition
domain_specific_transport:
  driver = smtp
  transport_filter = "if \$domain eq 'example.com' then 'custom_forward'; else 'default_smtp';"
  route_data = "${lookup{${transport_filter}}lsearch{/etc/exim4/custom_routes}}
  protocol = smtp
  port = 25
```

And let's assume the custom routes file looks like this:

```
# /etc/exim4/custom_routes
custom_forward: 192.168.1.100
default_smtp:
```
This example demonstrates that if a message is destined for example.com, it gets routed to 192.168.1.100 via SMTP. If the server at 192.168.1.100 is misconfigured to forward the message back to the original server without delivering to a mailbox, a loop could be initiated. The original server would accept the forwarded message and resend it, accumulating “received” headers each time. The `transport_filter` is there to dynamically look up a route based on the target domain. This is a very common scenario that, if misconfigured, can lead to problems.

Now, let's consider a second example where the exim configuration may have problematic redirects related to a local alias.

```
# local delivery
local_user:
   driver = appendfile
   file = /var/mail/\${local_part}
   user = mail
   group = mail
```
And the alias configured like this:

```
# /etc/aliases
trouble_user:  trouble_user@example.com
```

If a message sent to `trouble_user@localhost` ends up being redirected to `trouble_user@example.com` and `example.com`’s server has an issue, then another loop can be created. Because the message is then received from `example.com` and then re-forwarded by this server, the `Received` headers will continue to accumulate. This highlights that not just the transport configurations, but also aliases can cause this loop if not configured properly.

For a third example, consider a configuration involving a milter (mail filter). Milters can be complex, and errors within them can cause looping behavior. Let’s consider a hypothetical milter in an `exim.conf` configuration:

```
# Milter setup
milter_macro_daemon_name = sendmail
milter_macro_from = ${sender_address}
milter_macro_to = ${recipients}
milters = "inet:localhost:8891"

# Basic routing rules
begin transports
remote_smtp:
  driver = smtp
  protocol = smtp

```

If this particular milter, on localhost port 8891, has a bug that causes it to resubmit an incoming mail to the mail server without modification, it can create a situation where every message goes through the same loop of milter and server resending the email. The milter is not changing the mail, so it will get stuck in the loop. And as it loops, it adds more headers. It’s a critical reminder that external services, like custom milters, also play a crucial role in the stability of your email flow.

What's the resolution? First, you’ll want to carefully examine your Exim configuration, specifically the transport rules, aliases, and any filtering or Milter setups. Use the exim log files. Look for email messages where the "received" header list is excessively long – they will usually stand out in the logs. The `exim -bt <email_address>` command will prove invaluable for tracing routing paths. Pay special attention to any rules or aliases that might be inadvertently redirecting the same messages back and forth.

I recommend reviewing the documentation included with your exim installation – often located in the `/usr/share/doc/exim4-base/` directory. For a comprehensive understanding of exim's mail routing and configuration, I'd also highly suggest taking a deep dive into "Exim: The Mail Transfer Agent" by Philip Hazel – it's the definitive guide. Similarly, “TCP/IP Illustrated, Volume 1: The Protocols” by W. Richard Stevens provides a thorough foundational understanding of the network protocols that undergird email delivery. For more practical analysis, the "Postfix Complete" book by Kyle Dent provides great examples of how other MTAs accomplish these same tasks, offering alternative perspectives that can be helpful.

Finally, consider implementing rate limiting and sanity checks on email sizes within your exim configuration to help mitigate these issues proactively. You can use the `message_size_limit` option and the various access control lists (ACLs) in Exim to further enhance your setup. Prevention through diligent configuration, careful testing, and a deep understanding of the technology at hand are the key to resolving and preventing issues like this. These situations often highlight underlying flaws in system configuration, emphasizing the need for thorough and methodical troubleshooting.
