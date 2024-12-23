---
title: "Can Kaitai Struct be used to describe TLV data without creating new types for each field?"
date: "2024-12-23"
id: "can-kaitai-struct-be-used-to-describe-tlv-data-without-creating-new-types-for-each-field"
---

Alright, let’s tackle this. Funny, this reminds me of that project back at CloudGen, when we were dealing with a particularly nasty proprietary protocol; tons of tlv structures all over the place. We ended up using kaitai struct, and the core of the issue you've raised was absolutely central to our approach: could we define these things without exploding into a ridiculous number of types? The short answer is yes, absolutely, kaitai struct is indeed powerful enough to handle tlv data without necessarily needing a separate type for *every* field. The longer answer, as you might suspect, involves some nuances.

The key isn't to avoid new types altogether, because that's fundamentally how kaitai struct works – defining structures as types. Rather, it’s about structuring our definitions effectively to avoid redundant type declarations for similar tlv fields. We leverage kaitai struct’s flexibility, specifically its ability to use conditional logic and parameters, to handle the variation in tlv structures gracefully.

Let’s start with the fundamental concept: a tlv structure essentially consists of a ‘tag,’ a ‘length,’ and a ‘value.’ The ‘tag’ identifies the type of data in the ‘value,’ and the ‘length’ specifies how many bytes to read for the ‘value.’ The core issue then arises when these tag-value pairs are variable in both their tag and content type. This is where we often see the temptation to define a new type for each tag-value combination, which quickly leads to maintenance headaches.

Here's how I found we could address this more efficiently:

Instead of creating a type for each individual tag-value, we should design a general tlv structure. We could call it something like `generic_tlv`. Within this structure, we'd define the `tag` and `length` fields directly. For the `value` part, we use a `body` attribute. The most crucial thing is that we won't pre-determine the specific format of the `body`; instead, its interpretation is handled in different ways depending on the `tag` of the tlv record.

Here’s some simplified pseudocode for a basic tlv structure definition in ksy:

```yaml
  generic_tlv:
    seq:
      - id: tag
        type: u1
      - id: length
        type: u2
      - id: body
        size: length
```

This simple definition gives us the basic structure. The true power, and where we prevent type explosion, is how we consume the `body` based on the `tag` value. That's done through conditional type declarations using the `switch-on` and other related keywords of kaitai struct. For instance, if the `tag` equals 0x01, we expect an integer value, whereas if the `tag` equals 0x02, we might expect a string.

Let's consider a slightly more concrete example. Suppose we have some tlv data with these specifications:
* tag 0x01: 4-byte integer value (big endian)
* tag 0x02: a null-terminated string
* tag 0x03: 2-byte floating point value (big endian)

Here is the corresponding kaitai struct definition.

```yaml
  tlv_message:
    seq:
      - id: records
        type: generic_tlv
        repeat: eos
  generic_tlv:
    seq:
      - id: tag
        type: u1
      - id: length
        type: u2
      - id: body
        size: length
    instances:
      parsed_body:
        value: |
          switch (tag) {
            case 0x01:
              return _io.read_bytes(4).as_int(true);
            case 0x02:
              return _io.read_bytes_term(0, false, true, true).decode('utf-8');
            case 0x03:
              return _io.read_bytes(2).as_float(true);
            default:
              return body;
          }
```

In this version, we still have a generic `tlv_message` type for the overall container. But the real difference is in the `instances` block in our generic_tlv definition. We introduced an instance field called `parsed_body`. It uses a switch statement to determine how to parse the `body` field, depending on the `tag`. If `tag` is 0x01, the code reads four bytes as a big-endian integer. If it's 0x02, we read a null-terminated string. If it's 0x03, we read a two-byte big-endian float. This method avoids defining separate types for each of the fields, instead parsing the generic body dynamically, thereby reducing redundancy and improving clarity. The body itself is read as raw bytes.

Now, you might encounter situations where the value part is further structured, or contains sub-tlvs. In such a case, we can take this approach even further using custom types referenced through the instance `parsed_body` attribute. Consider an scenario where a specific tag (say 0x04) contains another tlv, nested within. Let’s add a hypothetical `embedded_tlv` type to our existing definition and parse the nested structure.

```yaml
  tlv_message:
    seq:
      - id: records
        type: generic_tlv
        repeat: eos
  generic_tlv:
    seq:
      - id: tag
        type: u1
      - id: length
        type: u2
      - id: body
        size: length
    instances:
      parsed_body:
        value: |
          switch (tag) {
            case 0x01:
              return _io.read_bytes(4).as_int(true);
            case 0x02:
              return _io.read_bytes_term(0, false, true, true).decode('utf-8');
            case 0x03:
               return _io.read_bytes(2).as_float(true);
            case 0x04:
              return new embedded_tlv(_io).parsed_embedded_body
            default:
              return body;
          }
  embedded_tlv:
    seq:
      - id: tag
        type: u1
      - id: length
        type: u2
      - id: body
        size: length
    instances:
       parsed_embedded_body:
        value: |
          switch (tag) {
            case 0x05:
              return _io.read_bytes(2).as_int(true)
            default:
              return body;
          }
```
Here, when `tag` is 0x04, we instantiate a new `embedded_tlv` using the current `_io` stream and then use it's own `parsed_embedded_body` instance to parse the nested tlvs recursively, allowing us to go multiple layers deep.

So, the key takeaways from my experience are that, instead of defining new types for every possible combination of tag and value, one should embrace the power of switch statements and the ability to call other structures through dynamically created `instances` within a more generalized `tlv` structure. This approach significantly simplifies definitions, and it mirrors how data is processed in many real-world applications.

For those seeking to delve deeper into this area, I'd highly recommend exploring the Kaitai Struct documentation, particularly the sections on instance fields, switch statements, and type construction. In addition, reading "ASN.1 Communication Between Heterogeneous Systems," by Olivier Dubuisson can offer further insights into handling data structure definitions that are flexible and widely applicable. The text "Understanding Network Hacks" by John Wiley & Sons delves into the realities of protocol analysis, which might be helpful in understanding how real-world tlv structures come about and why this approach is generally so practical. These resources collectively provide a comprehensive understanding of designing flexible data structure definitions, making your journey with Kaitai Struct easier and more effective. Hopefully, that adds some clarity.
