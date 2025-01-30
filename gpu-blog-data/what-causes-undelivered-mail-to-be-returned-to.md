---
title: "What causes undelivered mail to be returned to the sender?"
date: "2025-01-30"
id: "what-causes-undelivered-mail-to-be-returned-to"
---
The fundamental mechanism for returning undelivered email centers around Mail Transfer Agent (MTA) interactions and a specific set of error codes known as Delivery Status Notifications (DSNs). These codes, standardized in RFC 3463, provide a structured method for conveying why an email could not reach its intended recipient. I've encountered numerous variations of these issues during my years maintaining email infrastructure, from misconfigured DNS records to temporarily unavailable mail servers. The return process is not a singular event but a series of negotiations and decisions among mail servers governed by these DSNs.

An email’s journey typically begins with the sender’s Mail Submission Agent (MSA), often integrated into email client software. Upon receiving an email, the MSA forwards it to the sender's MTA. This MTA then performs a DNS lookup on the recipient's domain to identify the designated Mail Exchange (MX) server. The next step involves the sender's MTA establishing a Simple Mail Transfer Protocol (SMTP) connection with the recipient's MTA. This is where the crucial error reporting mechanism comes into play. If the recipient's MTA accepts the email, it proceeds with delivery to the local mailbox. However, if the delivery fails at any stage of the transaction or afterward, the recipient's MTA will generate a DSN and return it to the sender’s MTA.

The DSN encompasses several components. The most critical are the status codes, which are formatted as “x.y.z”. The 'x' signifies the class of the error, with ‘2’ indicating successful delivery, ‘4’ representing a temporary failure, and ‘5’ a permanent failure. The ‘y’ provides more detail within the class, and ‘z’ is the most granular level of information. For example, a code of "5.1.1" typically signifies an invalid recipient email address. The presence of a ‘5’ in the first position indicates the message will not be retried, prompting an immediate return. Conversely, a "4.x.x" code signifies a transient issue like a busy mail server, leading to retries according to the server's configuration. This retry mechanism is crucial, and servers often implement a back-off strategy, increasing the time between retry attempts. After repeated failures, a server will ultimately generate a permanent failure DSN and return the message. I’ve observed mail servers attempting redelivery for up to several days before giving up entirely.

Understanding these error codes and the entire process is imperative when troubleshooting email deliverability issues. Here are some examples of DSN responses and their interpretations:

**Example 1: Invalid Recipient Address**

```
Reporting-MTA: dns; mail.senderserver.com
Received-From-MTA: dns; [192.168.1.100]
Arrival-Date: Mon, 28 Aug 2023 10:00:00 -0500

Final-Recipient: rfc822; non-existent.user@example.com
Action: failed
Status: 5.1.1
Diagnostic-Code: smtp; 550 5.1.1 User unknown
Last-Attempt-Date: Mon, 28 Aug 2023 10:00:01 -0500
```

This example illustrates a "5.1.1" error, indicating the recipient email address, `non-existent.user@example.com`, is not valid. The `Diagnostic-Code` gives a more detailed explanation, stating "User unknown.” The sender’s mail server, `mail.senderserver.com`, reports the failure, and no further retries will occur because it’s a permanent error. In my experience, this is among the most common causes of return messages. Typically, this suggests a typo in the recipient's address or a deactivated email account.

**Example 2: Recipient Mail Server Unavailable**

```
Reporting-MTA: dns; mail.senderserver.com
Received-From-MTA: dns; [192.168.1.100]
Arrival-Date: Mon, 28 Aug 2023 10:00:00 -0500

Final-Recipient: rfc822; valid.user@example.net
Action: delayed
Status: 4.4.7
Diagnostic-Code: smtp; 450 4.4.7 Temporary server error. Please try again later
Last-Attempt-Date: Mon, 28 Aug 2023 10:01:00 -0500
```

Here, the `Status` is `4.4.7`, a temporary error. The `Action` is `delayed`, indicating retries are underway. The `Diagnostic-Code` confirms a transient server issue. The recipient’s mail server was temporarily unavailable when the sender's mail server attempted delivery. This scenario often involves short-lived server outages or high load. The sender's mail server would typically continue to attempt delivery periodically before ultimately returning a non-delivery message if the problem persists. I recall numerous times debugging this exact type of return and waiting for the external server to resolve itself, a task often beyond our direct control.

**Example 3: Email Exceeds Quota**

```
Reporting-MTA: dns; mail.senderserver.com
Received-From-MTA: dns; [192.168.1.100]
Arrival-Date: Mon, 28 Aug 2023 10:00:00 -0500

Final-Recipient: rfc822; valid.user@example.org
Action: failed
Status: 5.2.2
Diagnostic-Code: smtp; 552 5.2.2 Mailbox full
Last-Attempt-Date: Mon, 28 Aug 2023 10:00:01 -0500
```

This example indicates a `5.2.2` status code, again signaling a permanent failure. The `Diagnostic-Code` shows a "Mailbox full" error. This means that the recipient’s mailbox has exceeded its storage limit and is unable to accept new mail. As with the invalid recipient address error, no further delivery attempts are made. I often encounter such issues with users who have not been diligent in cleaning up their mailboxes.

In addition to these specific examples, other causes of undelivered mail include issues like DNS misconfigurations (specifically incorrect MX records, preventing the sender’s MTA from finding the correct destination server), emails being flagged as spam by the recipient's MTA due to content or sending reputation, security mechanisms such as SPF or DKIM failing verification, and network routing problems between the two mail servers. Furthermore, some recipient servers may have strict anti-relay policies, blocking delivery if the sender's server is not authorized to send email on behalf of a domain.

To gain a deeper understanding of the intricacies of email delivery, I recommend exploring several resources. First, the Internet Engineering Task Force (IETF) publishes the foundational RFC documents that dictate the standards for email communication and error reporting. Specifically, RFC 5321 defines the SMTP protocol, while RFC 3464 and related RFCs detail DSNs and their interpretation. Also, researching specific mail server implementations (e.g., Postfix, Sendmail, Exim) can provide a more in-depth technical perspective. Finally, I suggest a thorough review of the tools available within your operating system for analyzing mail server logs; for instance, `grep` for log searching and `dig` or `nslookup` for DNS lookups are often invaluable in the troubleshooting process. A practical understanding derived from hands-on experience, coupled with these resources, forms the foundation for successfully managing email deliverability.
