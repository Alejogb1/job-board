---
title: "How can I disable mail sending in Roundcube while retaining read-only access?"
date: "2024-12-23"
id: "how-can-i-disable-mail-sending-in-roundcube-while-retaining-read-only-access"
---

, let's dive into this. I’ve actually tackled this specific configuration with Roundcube more than once, usually in scenarios where we were decommissioning a mail server but wanted to preserve access to historical emails. The goal, as I understand it, is to prevent users from sending new mail, effectively rendering the client read-only, while still allowing them to browse their existing mailboxes. It's a delicate balance, and there are a few ways to approach it, each with its own nuances.

The most straightforward method, and the one I've found most reliable, involves manipulating the Roundcube configuration directly. Roundcube uses PHP, and its functionality is largely governed by settings in its main configuration file, `config/config.inc.php`. This file contains various parameters, and thankfully, there are specific options related to mail sending that can be disabled. We'll leverage those.

First, the key setting we need to focus on is `$rcmail_config['smtp_server']`. This parameter defines the SMTP server used by Roundcube to send emails. By setting this to an invalid or non-functional value, we essentially cut off the outbound mail flow. Crucially, we want to do this without introducing errors that might impede read operations.

Here’s a configuration snippet illustrating how I usually implement this:

```php
<?php

$rcmail_config['smtp_server'] = 'invalid.smtp.server';
$rcmail_config['smtp_port'] = 25; // Or any port really, it's irrelevant now.
$rcmail_config['smtp_user'] = '';
$rcmail_config['smtp_pass'] = '';
$rcmail_config['smtp_auth_type'] = '';

$rcmail_config['drafts_mbox'] = 'INBOX.Drafts'; // Keep drafts accessible
$rcmail_config['sent_mbox'] = 'INBOX.Sent'; // Keep sent mail visible
$rcmail_config['junk_mbox'] = 'INBOX.Junk';  // Keep junk visible

?>
```

The `invalid.smtp.server` value effectively disables email sending. We zero out the user, password and auth type, further ensuring sending is blocked. Note that I've also included settings for the drafts, sent, and junk mailboxes, ensuring users retain access to these folders as usual. Without specifying these, they might default or cause issues with display, even though no new mails can be sent. This is crucial for maintaining a consistent user experience.

While this method is effective, it’s not completely foolproof. A user, determined enough and with sufficient technical knowledge could potentially bypass this by modifying browser requests directly (though quite complicated), so it's essential to consider this as part of a wider security context rather than a singular, isolated measure.

Now, you might also consider using a plugin-based approach for increased flexibility. Roundcube supports plugins that can modify its behavior. While I haven’t used this in production, the principle is simple. A plugin could hook into the sending event and prevent emails from being sent.

Here’s a simplified example of what such a plugin, named for example `no_send`, might look like:

```php
<?php
class no_send extends rcube_plugin {

    public $task = 'mail';

    function init() {
        $this->add_hook('message_send', array($this, 'prevent_send'));
    }

    function prevent_send($args) {
        return false;
    }
}
?>
```

This hypothetical `no_send` plugin, once placed in the Roundcube plugins directory and enabled in `config/config.inc.php`, intercepts the 'message_send' event and prevents emails from being sent. I've used a similar pattern for adding extra functionality, like custom address book integrations. The main advantage is modularity. Plugin functionality can be toggled, whereas changing config values directly could lead to errors if not done correctly. The downside is added complexity, and a dependency on plugin upkeep (if any issues are found with that plugin over time).

Finally, a different method, and one that’s less ideal but worth mentioning, involves manipulating permissions within the underlying mail server itself. This can be done through the server's configuration files (e.g., for postfix or dovecot). For instance, you can disable SMTP authentication or restrict relaying for the specific user or user groups. I would caution against doing this, as it affects more than just Roundcube, and could impact other clients as well, which can result in unexpected consequences if not handled with great care. Further, it complicates maintenance significantly.

The code below shows a conceptual postfix configuration change, just to illustrate the point, although I would not recommend this as the primary method for disabling sending within Roundcube itself. It's more of a server-side restriction. It's provided *only* to highlight the option exists on the mailserver itself, and not as a proposed solution for the Roundcube problem at hand.

```
# /etc/postfix/main.cf

# Restrict relaying only for a group, if you know which group or user to restrict
# For example, if your Roundcube users share a specific group.
# I strongly advise against this type of configuration if you can avoid it.
# The preferred methods are those based in Roundcube configuration directly or via plugin
# smtpd_relay_restrictions = permit_mynetworks, reject_unauth_destination

# The above line can be used to restrict all relaying, making sending impossible
# on that server, for all clients. NOT a good solution for the roundcube problem,
# unless no one at all should send from that server.

```

This snippet showcases the concept of manipulating `smtpd_relay_restrictions` in `main.cf`. It's here for illustrative purposes, and not as part of your Roundcube solution.

To delve deeper into these topics, I recommend consulting the official Roundcube documentation, it’s invaluable. The postfix documentation is another critical resource for understanding server-side mail sending configurations, if you absolutely need to go that route as described previously. For more details on PHP plugin development, I recommend books like "PHP Cookbook" by David Sklar and Adam Trachtenberg or "Programming PHP" by Rasmus Lerdorf et al. These are well-regarded resources in the PHP community and provide a wealth of information on various programming techniques.

In summary, while there are multiple potential ways to disable mail sending in Roundcube, the simplest and safest involves configuring the smtp server settings within the `config.inc.php` file. Plugin development is another option but entails a bit more work. Server-side restriction, though technically feasible, is not the preferred approach for a Roundcube-specific problem. Remember, always approach system changes with caution and test configurations thoroughly in a non-production environment first.
