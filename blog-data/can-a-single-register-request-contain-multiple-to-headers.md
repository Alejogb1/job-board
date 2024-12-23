---
title: "Can a single REGISTER request contain multiple To headers?"
date: "2024-12-23"
id: "can-a-single-register-request-contain-multiple-to-headers"
---

Alright, let's tackle this one. I recall a particularly tricky SIP implementation I worked on years ago where this very issue cropped up. We were dealing with a complex system incorporating multiple user profiles under a single device, and the question of handling multiple 'to' headers in a register request became quite relevant. The short answer is yes, a single SIP register request *can* indeed contain multiple 'to' headers, though it's a practice that needs careful handling and doesn't conform to the standard intended use of the 'to' field. Let me explain.

The SIP standard (RFC 3261 and subsequent related RFCs) specifies that the 'to' header field is meant to indicate the logical recipient of the request. In the context of a register request, this would typically be the address-of-record (AOR) that is to be registered. So, intuitively, you would expect a single 'to' header containing this single AOR. However, the SIP syntax rules actually allow for multiple 'to' headers. This allowance is not generally intended for multiple registration endpoints within a single register request but is often leveraged for backward compatibility or in specific corner cases that I'll elaborate on.

The core issue here is that while the grammar permits it, processing multiple 'to' headers requires clarity about the intended registration procedure. If multiple 'to' headers specify completely different AORs, a SIP registrar handling this request would face ambiguity. It might choose to register only the first one encountered and ignore the rest, or simply reject the entire request. The behavior is therefore not deterministic without careful consideration and well-defined policies.

To illustrate, consider these scenarios and what a registrar might have to resolve:

**Scenario 1: Registration for Different AORs using multiple "To" headers**

Here's how it could look:

```
REGISTER sip:domain.com SIP/2.0
Via: SIP/2.0/UDP 192.168.1.100:5060;branch=z9hG4bKnashds7
Max-Forwards: 70
From: <sip:user1@domain.com>;tag=asdsf123
To: <sip:user1@domain.com>
To: <sip:user2@domain.com>
Call-ID: 123456789@192.168.1.100
CSeq: 1 REGISTER
Contact: <sip:user1@192.168.1.100:5060>
Expires: 3600
Content-Length: 0
```

In this case, a registrar is unlikely to register both `user1` and `user2` based on a single request. In practice, this would lead to unexpected behavior or outright rejection of the message. We've seen such messages during interoperability testing where devices tried (and failed) to consolidate multiple registrations, most commonly due to a faulty interpretation of what the standard allowed versus what makes practical sense.

**Scenario 2: Registration for the same AOR with different parameters (less common)**

This case is more subtle and perhaps more realistic:

```
REGISTER sip:domain.com SIP/2.0
Via: SIP/2.0/UDP 192.168.1.100:5060;branch=z9hG4bKnashds7
Max-Forwards: 70
From: <sip:user1@domain.com>;tag=asdsf123
To: <sip:user1@domain.com>
To: <sip:user1@domain.com>;param1=value1
Call-ID: 123456789@192.168.1.100
CSeq: 1 REGISTER
Contact: <sip:user1@192.168.1.100:5060>
Expires: 3600
Content-Length: 0
```
Here, the registrar will have to resolve which `To` header it intends to use for registration. Is it the first, the last, or a combination? The registrar's behavior would have to be explicitly defined and documented within your system. Generally, the safest approach is to avoid using duplicate 'to' headers entirely.

**Scenario 3: Use of Multiple To Headers during Backward Compatibility (the most probable case)**

Let's consider a case where old PBXs that rely on a non-standard usage of the 'to' header. In this scenario, some systems might have used it to denote the target URI in addition to the actual contact. In this case, a more nuanced approach must be used.

```
REGISTER sip:domain.com SIP/2.0
Via: SIP/2.0/UDP 192.168.1.100:5060;branch=z9hG4bKnashds7
Max-Forwards: 70
From: <sip:user1@domain.com>;tag=asdsf123
To: <sip:target@domain.com>
To: <sip:user1@domain.com>
Call-ID: 123456789@192.168.1.100
CSeq: 1 REGISTER
Contact: <sip:user1@192.168.1.100:5060>
Expires: 3600
Content-Length: 0
```

Here, a smart registrar would have to inspect the 'from' and the 'contact' headers to determine if the request must register the AOR provided by the 'from' header and using the 'contact' one. If the registrar supports backward compatibility with such edge cases, it must use the second 'To' header as the address to be registered. If not, the request should be rejected or an error log generated.

From these scenarios, it's clear that using multiple 'to' headers isn't a standard, recommended practice for registration, and can easily lead to confusion and system incompatibility. The core problem isn't that the grammar is broken, but that it breaks down when trying to enforce predictable behaviors.

If you are building a registrar, it's far cleaner and more compliant to require separate REGISTER requests for each user or address-of-record. This removes ambiguity and promotes interoperability.

Instead of relying on these kinds of multiple headers, consider using batch registration methods or more modern approaches to handle user profiles, if supported by the registrar. This often involves using dedicated extensions to the standard SIP procedures.

As for resources, I would recommend deeply exploring:

*   **RFC 3261:** This is the foundational document for SIP. Understanding the core definitions is critical.
*   **"SIP Demystified" by Gonzalo Salgueiro and Alan Johnston:** This is an excellent book for a comprehensive look at SIP, including how headers are processed.
*   **"The Session Initiation Protocol (SIP): Internet Signaling for Next-Generation Communications" by Alan B. Johnston:** This book provides a deep dive into SIP and is also a good resource.
*   **RFC 3263:** This document provides details about locating SIP servers, and is critical to understand how requests such as REGISTER are routed.

In conclusion, while the syntax allows it, sending multiple 'to' headers in a register request is generally ill-advised due to the potential for unpredictable outcomes. Adhering to the standard usage—a single 'to' header specifying the address-of-record—is crucial for creating robust and interoperable SIP implementations. Remember, standards are not just specifications of what is permissible, but also of what *should* be done for consistency and interoperability.
