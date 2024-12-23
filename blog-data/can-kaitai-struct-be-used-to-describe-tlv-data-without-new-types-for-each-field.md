---
title: "Can Kaitai Struct be used to describe TLV data without new types for each field?"
date: "2024-12-23"
id: "can-kaitai-struct-be-used-to-describe-tlv-data-without-new-types-for-each-field"
---

Let's get straight into this. In my experience, particularly during a rather intense project involving network protocol analysis some years back, dealing with the ever-present challenge of Type-Length-Value (TLV) data streams became almost a daily ritual. The core issue, as you're touching upon, is whether a tool like Kaitai Struct can handle TLVs elegantly without having to define a custom structure for *every single field* you encounter. The answer, thankfully, is a resounding yes, albeit with some nuances that warrant explanation.

The beauty of Kaitai Struct, which I appreciated deeply while reverse-engineering that aforementioned protocol, lies in its declarative nature. It allows us to describe binary formats in a way that's both human-readable and machine-parsable. When it comes to TLVs, we're typically faced with a repeating sequence where each item consists of a type identifier, a length specification, and then the actual value data. Now, naively, you might think you’d need a separate struct definition for every ‘type’ you find, which quickly becomes unwieldy, especially with less documented or very dynamic formats. But the strength of Kaitai Struct is its ability to handle this through the intelligent use of conditional parsing and recursive structures.

To avoid this explosion of individual struct definitions, we can leverage Kaitai's ability to parse data based on context. The most powerful technique involves creating a base `tlv_record` struct. This structure would define only what’s common to all TLVs: the type, the length, and the value itself. The trick here is to defer the *interpretation* of the value until *after* the basic record is parsed. Then, using a combination of switch statements or conditional parsing inside Kaitai Struct, we can process the value based on its `type`. This way, we use *one* structure as the base and let logic determine what to do with value data based on the type field.

I will show three code snippets. The first is an example of the base `tlv_record` struct, which remains constant. The second and third are two examples of different value interpretations based on the type field.

**Snippet 1: Base `tlv_record` struct (Kaitai Struct YAML)**

```yaml
seq:
  - id: type
    type: u1
  - id: length
    type: u2
  - id: value
    size: length

```

This snippet is the foundation. Here: `type` is a one byte unsigned integer (u1), `length` is a two byte unsigned integer (u2), and `value` has the size dictated by the length field. The content of `value` is raw bytes at this stage.

Now, lets say we have two distinct type interpretations. Let's say type `0x01` is a simple ascii string and type `0x02` is an unsigned 32bit integer. Here are two different examples on how to handle these type specific interpretations:

**Snippet 2: Example of `type=0x01` interpretation (ascii string)**

```yaml
instances:
  parsed_value:
    pos: 0
    if: type == 1 # 0x01
    value: value.to_s

```

In the second snippet, we have an `instance` named `parsed_value`. The `if: type == 1` makes the parser execute this part only if the `type` field is `0x01`. The `value: value.to_s` converts the byte array into an ascii string.

**Snippet 3: Example of `type=0x02` interpretation (unsigned 32bit int little-endian)**

```yaml
instances:
  parsed_value:
    pos: 0
    if: type == 2 # 0x02
    type: u4le
    value: value

```

In the third snippet, again, the instance `parsed_value` only executes if type is `0x02`. This time however, we declare `type: u4le` which tells the parser to interpret value as a 4-byte little endian unsigned integer.

The combined approach of the base `tlv_record` struct and the conditional interpretation of the `parsed_value` via `instances` means that we are not stuck creating a struct for every variation. This can lead to huge savings on development effort, and, more importantly, maintainability. This approach, to my understanding, provides a clean and efficient way to manage complex and potentially evolving TLV structures.

When I encountered this problem initially, the temptation was indeed to create a different struct for each type, but this quickly became unmanageable. The breakthrough came when I dove into the Kaitai Struct documentation, in particular focusing on the sections describing "instances" and "conditional types". A resource I found particularly useful was the Kaitai Struct official documentation, specifically the sections discussing *conditional fields* and *instances*, along with some practical examples included in the repository. Also, "Understanding Network Protocols" by James Kurose and Keith Ross, despite not being specific to Kaitai Struct, gave me a great deal of understanding regarding how protocols use TLVs.

It's also worth pointing out that this approach isn’t only limited to simple data types. It extends to complex data structures too. You could have, for instance, another Kaitai Struct definition for one of the value interpretations based on the `type` field if the data structure is itself complex. In these cases, the value field in the base `tlv_record` would become a container for the entire structure, allowing complex TLV nestings. This is a very powerful technique for parsing arbitrarily complex data formats without cluttering your Kaitai Struct spec with a massive number of structs.

In situations where the `type` field values are not continuous or form a complex pattern, or where several interpretations share common data structures, you could create a separate `enum` or `switch` statement inside Kaitai Struct to handle a bigger variety of potential data interpretations. However, doing so will introduce complexity to your spec.

The key takeaway here is that while a naive approach might suggest that every TLV type demands its own custom struct, Kaitai Struct's capabilities allow for a much more streamlined approach. By focusing on a base `tlv_record` struct and using conditional parsing, we gain the flexibility to handle a wide variety of TLV data structures, greatly improving code readability and maintainability, especially when dealing with less than perfect documentation. The instances feature within Kaitai provides an elegant solution to interpret the raw byte data. This was my main learning when I worked with a particularly difficult TLV-based format. It's a more manageable solution than using a separate structure definition for each possible field.

In summary, Kaitai Struct is perfectly capable of handling TLV data effectively without demanding separate struct definitions for each individual field. By leveraging features like conditional parsing and instances, you can maintain a clean, maintainable, and scalable solution, and the key is to learn how to interpret raw bytes dynamically, based on the values of the type field. This approach, in my experience, has proven invaluable. I encourage further exploration of the Kaitai Struct documentation, specifically the parts dealing with "instances," "conditional types," and "enums." These resources will solidify these concepts, ultimately making you more proficient at handling any complex, binary data format.
