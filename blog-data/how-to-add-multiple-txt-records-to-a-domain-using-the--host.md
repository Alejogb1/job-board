---
title: "How to add multiple TXT records to a domain using the @ host?"
date: "2024-12-23"
id: "how-to-add-multiple-txt-records-to-a-domain-using-the--host"
---

Alright, let's delve into the specifics of adding multiple TXT records to a domain at the root level (represented by the `@` host). This is a fairly common task, and although it seems straightforward on the surface, it can have some subtleties, especially when dealing with different DNS management interfaces. I've certainly had my share of late nights debugging DNS configurations, so I can speak from experience on this one.

The core issue lies in understanding that the `@` symbol in DNS zone files, or the equivalent field in various DNS management UIs, simply means 'this domain itself'. So, when we talk about adding TXT records for `@`, we are adding those records directly to the domain name (e.g., example.com). This is where we store records for things like SPF, DKIM, and DMARC. These records, often used for email authentication, require multiple TXT records. Crucially, DNS allows for multiple TXT records to exist for the same host, including `@`.

However, the *way* you achieve this differs depending on your DNS provider. Some providers have very intuitive interfaces, while others might require a bit of technical maneuvering. It's almost never a case of simply overwriting an existing record; that would cause problems. The DNS system expects a record to be added as an *additional* record, not replaced, when we need more than one TXT record for the same name.

Let me give you an example from a past project, a migration of an enterprise email system. We had to move from an old legacy platform to a new cloud provider, and this transition required meticulously setting up email authentication. So there was a need for several TXT records for `@`. We had to correctly create each of those new records, without accidentally deleting the others.

Here's how this typically breaks down into practice, with code snippets representing configuration steps or outputs (these are not direct live code, but rather illustrate the intended structure in configurations). Let’s explore three scenarios:

**Scenario 1: BIND Style Zone File:**

If you are dealing with a raw DNS zone file, such as with BIND, the setup is relatively straightforward. In a `example.com.zone` file, you'd add multiple TXT records for `@` like so:

```dns
$ORIGIN example.com.
@       IN      SOA     ns1.example.com. admin.example.com. (
                                       2023102701 ; Serial
                                         3600  ; Refresh
                                          600  ; Retry
                                      86400  ; Expire
                                        3600 ) ; Minimum
        IN      NS      ns1.example.com.
        IN      NS      ns2.example.com.

@       IN      TXT     "v=spf1 include:_spf.example.com ~all"
@       IN      TXT     "google-site-verification=xxxxxxxxxxxxxxxxxxxxxxxxx"
@       IN      TXT     "dmarc=v=DMARC1; p=reject; rua=mailto:mailauth@example.com"
```

Each `TXT` record starts with `@`, indicating it belongs to the root domain. You see, they don't overwrite each other; instead, they exist concurrently. When a DNS resolver requests TXT records for `example.com`, all three of these records will be returned. Pay close attention to how each record is on its own line.

**Scenario 2: Cloudflare DNS Management:**

Cloudflare has an intuitive interface, but you still need to understand how to add records correctly. In their DNS dashboard, you would:

1. Select the domain.
2. Navigate to the ‘DNS’ section.
3. Under 'DNS Records', select ‘Add Record’.
4. Choose ‘TXT’ as the type.
5. In the 'Name' field, enter `@`.
6. In the 'Content' field, add your TXT record value (e.g., `"v=spf1 include:_spf.example.com ~all"`), and save.
7. Repeat steps 3-6 for *each* additional TXT record. Importantly, you *don't* re-enter the name or try to comma-separate or concatenate TXT values here. Cloudflare, like most modern DNS providers, will handle multiple records correctly. Each will have the `@` in the name field and the data content field will hold individual text value.

This might look like the JSON structure beneath the web interface:

```json
[
  {
    "type": "TXT",
    "name": "@",
    "content": "v=spf1 include:_spf.example.com ~all",
    "ttl": 300
  },
  {
    "type": "TXT",
    "name": "@",
     "content": "google-site-verification=xxxxxxxxxxxxxxxxxxxxxxxxx",
    "ttl": 300
  },
  {
   "type": "TXT",
    "name": "@",
    "content": "dmarc=v=DMARC1; p=reject; rua=mailto:mailauth@example.com",
    "ttl": 300
  }
]
```

Notice how each object is distinct, but they all refer to the same name. This representation helps visualize what the system internally stores.

**Scenario 3: AWS Route 53:**

AWS Route 53 operates similarly to Cloudflare, but with a few AWS-specific nuances. To add multiple TXT records for `@` (represented as the zone name itself), you would:

1. Open the Route 53 console.
2. Navigate to your Hosted Zone.
3. Select ‘Create Record’.
4. Select ‘TXT’ as the record type.
5. Leave the ‘Record name’ field blank to target `@` (or manually input `@`).
6. In the ‘Value’ field, enter one of your desired TXT record values, like `"v=spf1 include:_spf.example.com ~all"`
7. Add a second record following steps 3-6 using a different text value like `"google-site-verification=xxxxxxxxxxxxxxxxxxxxxxxxx"`.
8. Add a third record following steps 3-6 using a different text value like `"dmarc=v=DMARC1; p=reject; rua=mailto:mailauth@example.com"`.

Here's a conceptual example of how this might be represented in the Route 53 API:

```json
{
  "ChangeBatch": {
    "Changes": [
        {
        "Action": "CREATE",
        "ResourceRecordSet": {
            "Name": "example.com",
            "Type": "TXT",
            "TTL": 300,
            "ResourceRecords": [
                {
                  "Value": "\"v=spf1 include:_spf.example.com ~all\""
                 }
              ]
          }
        },
         {
        "Action": "CREATE",
        "ResourceRecordSet": {
            "Name": "example.com",
            "Type": "TXT",
            "TTL": 300,
            "ResourceRecords": [
                {
                   "Value": "\"google-site-verification=xxxxxxxxxxxxxxxxxxxxxxxxx\""
                 }
               ]
            }
          },
          {
          "Action": "CREATE",
          "ResourceRecordSet": {
              "Name": "example.com",
              "Type": "TXT",
              "TTL": 300,
              "ResourceRecords": [
                    {
                      "Value": "\"dmarc=v=DMARC1; p=reject; rua=mailto:mailauth@example.com\""
                   }
                ]
              }
          }
    ]
   }
}
```

This structure shows that the same zone ("example.com") is the target for multiple separate `CREATE` actions, each with distinct content.

**Key Points & Resources:**

*   **Avoid overwriting:** Never replace an existing TXT record when you want to add another; that will lead to issues. Always add new records as independent entries, even if the ‘name’ or ‘host’ (@ in this case) is identical.
*   **Quoting matters:** Be meticulous about quoting your TXT record data. Many DNS providers require double quotes. However, remember to use backslashes to escape any double quotes already inside the text you're saving.
*   **Propagation Time:** Remember that DNS changes may take time to propagate fully across the internet, which can often be anywhere from a few minutes to up to 48 hours. Patience is key.

For authoritative information on DNS, I'd suggest starting with *DNS and BIND* by Paul Albitz and Cricket Liu – it's a classic and indispensable guide for a deeper understanding of DNS architecture. Another resource is RFC 1035, *Domain Names—Implementation and Specification*, which you can find on the Internet Engineering Task Force website (ietf.org). This details the underlying protocol. Also look at RFC 7208, *Sender Policy Framework (SPF) for Authorizing Use of Domains in Email*, and RFC 7489, *Domain-based Message Authentication, Reporting, and Conformance (DMARC)* for email related records. These will give you valuable insight into the purpose and structure of TXT records used for email authentication.

By keeping these principles in mind, adding multiple TXT records to the `@` host should become a far more predictable process, avoiding frustrating configuration headaches. Remember, paying close attention to the structure expected by your specific DNS provider is crucial.
