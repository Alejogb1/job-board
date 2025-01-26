---
title: "How can I resolve a suspected mail loop causing excessive 'Received' headers in exim4?"
date: "2025-01-26"
id: "how-can-i-resolve-a-suspected-mail-loop-causing-excessive-received-headers-in-exim4"
---

When an Exim4 mail server encounters a mail loop, manifested as a spiraling accumulation of "Received" headers, it typically indicates a misconfigured routing scenario where a message is repeatedly processed and resent within the system, or between systems under your control. This condition, if left unchecked, rapidly exhausts resources and can lead to service degradation. I've encountered this specific problem several times while managing mail infrastructures for various clients and the resolution generally involves tracing the message flow and isolating the component responsible for the loop.

The core issue stems from Exim's processing logic, which uses transport rules to direct mail. If these rules contain a flawed configuration, they can inadvertently redirect a message back to an earlier point in its processing path, triggering the repeated cycle. Each "Received" header added signifies that Exim has handled the message and is attempting to deliver it. The build-up of these headers is a diagnostic indicator of the looping condition. The goal is to identify the source of the recursive routing and correct it.

To begin, diagnosing a mail loop requires inspecting the problematic message. Specifically, the "Received" headers provide a chronological record of each hop the message has made. By carefully scrutinizing these headers, I've been able to pinpoint where the loop originates. I typically look for repeating patterns, especially any instance where a message appears to be sent *from* my own server, *to* my own server, more than once. The time stamps and hostnames are critical here, helping establish the direction of the loop.

Once the loop is detected, I start my investigation by examining the Exim configuration files, focusing primarily on the transport configuration. The relevant files typically reside in `/etc/exim4/conf.d/`. I always utilize `grep` to search for potential misconfigurations. I look for rules that could cause a message to be routed back into a local delivery transport instead of being delivered externally. It is also important to inspect the retry settings; if a delivery fails, these settings can cause the message to be re-queued and thus be reintroduced into the loop.

Let's consider a few examples of common loop-inducing configurations and how to fix them.

**Example 1: Local Router Misconfiguration**

A common mistake is a local router that inadvertently redirects mail back into the local transport mechanism, creating an endless loop within the server. Suppose I have the following router in my configuration, typically residing in a file like `router/300_local_router`:

```exim4conf
local_router:
  driver = accept
  domains = mydomain.com
  local_parts = "*"
  transport = local_delivery
```

This router directs all mail for `mydomain.com` to the `local_delivery` transport, regardless of the validity of the local part. If there is another rule later in the process that also handles local delivery and directs mail back to this point, it will loop. The solution here is usually to be more specific in the router criteria, such as only routing to `local_delivery` for known existing users, or to modify the local transport to not re-queue under specific circumstances. To illustrate, let's assume I want to route only valid mailbox users to `local_delivery`, I would first define valid users and configure my router to utilize the user check:

```exim4conf
local_router:
  driver = accept
  domains = mydomain.com
  local_parts = lsearch;/etc/exim4/valid_users
  transport = local_delivery
```
This modification uses a text file to validate known valid users against `lsearch`, directing only messages destined for recognized local recipients to the `local_delivery` transport. It would require that I would populate `valid_users` with valid mailbox users.

**Example 2: Incorrect Aliasing or Forwarding**

Another situation I’ve encountered is where a mail forwarding rule or alias inadvertently points back to the original sending address, creating a closed feedback loop. Imagine that I have configured a forwarder for `support@mydomain.com` to a user `user@mydomain.com` and that user, in turn, has an autoresponder which resends the message back to the sender and in turn to the forwarder, creating an infinite loop. The initial forwarding rule would typically be stored in an alias file, `/etc/aliases` (or similar):

```
support: user@mydomain.com
```

And the problematic autoresponder configured by `user@mydomain.com` that responds to all incoming messages.

The solution here is to critically review all configured aliases, redirects, and autoresponders, looking for circular forwarding patterns. For instance, an auto-responder should not re-send the message *back to* the sender, but rather send a confirmation response without re-including the original message. Instead of resending the full original message, the correct auto-responder should only send the automated response to the original sender. The correction would lie in the auto-responder's logic, preventing the re-sending of the original mail to support and thus avoiding the loop.

**Example 3: Complex Transport Configuration**

In more complex cases, mail loops can originate from intricate transport configurations where different transports are chaining in a way that leads to message resending. Consider a scenario where I have a custom transport called `filtering_transport` which is meant to perform content filtering and is positioned *before* another transport called `smtp_delivery`. In my configuration, the `filtering_transport` is configured to not perform any action and always return a successful delivery, so every message passes through:

```exim4conf
filtering_transport:
  driver = pipe
  command = "/usr/local/bin/dummy_filter"
  return_output = true
  delivery_condition = true

smtp_delivery:
  driver = smtp
```

If I then configure a router to direct local mail to first `filtering_transport`, and then `smtp_delivery`, I would introduce a loop if, for example, `filtering_transport` has a misconfigured setting or a bug, that ends up directing the message back to the same router instead of to smtp delivery. The fix here would involve making sure that `filtering_transport` performs its action as expected and directs the message onward to `smtp_delivery`. If `filtering_transport` has the misconfiguration, correcting it would be the solution.

To summarize, resolving a mail loop typically involves these steps:

1. **Message Analysis:** Carefully review the "Received" headers in a looped message to understand its path. Look for recurring senders and receivers.
2. **Configuration Inspection:** Examine Exim's router and transport configurations for rules causing looping. Employ `grep` to isolate problematic settings.
3. **Router Refinement:** Adjust router conditions to be more specific and prevent unintended routing of local messages. Ensure messages are routed to the appropriate delivery transport.
4. **Alias and Forwarding Checks:** Scrutinize all forwarding rules and aliases, ensuring they do not create feedback loops. Autoresponders should only respond, not re-send.
5. **Transport Chain Analysis:** Ensure custom transports are functioning as intended and directing the message towards the proper output.

For further assistance and understanding of Exim4 configuration, I recommend consulting the official Exim documentation, focusing on sections covering routers, transports, and the overall message routing process. Also beneficial are resources covering mail server security, specifically regarding open relays, forwarders, and aliases. There are also excellent online tutorials and guides that focus on Exim troubleshooting and best practices. I have found that a deep understanding of mail routing protocols and message flow fundamentals is essential in tackling this type of issue. Finally, experimenting in a test environment is highly valuable before making changes to production systems.
