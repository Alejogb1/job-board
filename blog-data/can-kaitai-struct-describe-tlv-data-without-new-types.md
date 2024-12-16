---
title: "Can Kaitai Struct describe TLV data without new types?"
date: "2024-12-16"
id: "can-kaitai-struct-describe-tlv-data-without-new-types"
---

Alright, let's tackle this. It's a familiar scenario; I've spent countless hours parsing various binary formats, and TLV (Type-Length-Value) structures are a constant companion. The question of whether Kaitai Struct can handle them *without* resorting to creating new custom types is an interesting one, and thankfully, the answer leans towards 'yes,' albeit with some nuances. I recall a particularly challenging project a few years back, dealing with proprietary protocol data from a legacy system—a real test of patience and parsing skills. That project is, in a way, why I have some direct experience relevant to this discussion.

Essentially, Kaitai Struct shines at describing *structured* binary data. It operates using declarative descriptions, specifying how bytes should be interpreted into high-level data fields. The core strength is defining record-like structures with named fields of fixed or known size. This maps well to formats with relatively rigid layouts, but TLV introduces some dynamic elements through variable length fields, which requires us to think differently about how we represent it in Kaitai.

The crux of the problem isn’t that Kaitai can't process variable-length data —it certainly can. The constraint is the avoidance of declaring new types *specifically* for each possible ‘value’ type in our TLV stream. Instead of creating a unique struct definition for every possible type, we aim to describe how the TLV structure is organized without requiring explicit, custom, per-type definitions.

Here's the strategy: We leverage Kaitai's features for handling sequences and conditional parsing. Instead of declaring new types that pre-define how to interpret the ‘value’ part based on the ‘type’ field, we treat the ‘value’ as a raw byte sequence. Then, we can decide how to interpret these bytes *after* parsing, using the already-available type information. This interpretation can happen in code or even via expressions within the Kaitai spec itself but this will be limited in complexity.

Let’s start with a basic TLV example. Imagine a stream where:

*   The ‘type’ field is a 1-byte unsigned integer.
*   The ‘length’ field is a 2-byte unsigned integer representing the size of the ‘value’ field in bytes.
*   The ‘value’ field is simply a sequence of bytes with the length specified in the ‘length’ field.

Here’s how this would look in a Kaitai Struct specification:

```yaml
  seq:
    - id: type
      type: u1
    - id: length
      type: u2
    - id: value
      type: byte
      size: length
```

Notice how the value’s type is a simple byte array. The type itself isn’t further interpreted. We have a sequence of bytes, but no custom types. This works because we don’t *need* a specific type defined here. What’s important is parsing it according to its length; *how* we use it will be determined later, in the application code, by inspecting the 'type' field.

Consider another slightly more complex scenario, where the ‘type’ field specifies not just a type but also a subtype or version. Let's say that a 'type' of 0x01 means an integer, 0x02 means a string and 0x03 represents other structured data. We can handle this in our Kaitai definition:

```yaml
  seq:
    - id: type
      type: u1
    - id: length
      type: u2
    - id: value
      type: byte
      size: length
  instances:
    parsed_value:
      - if: type == 0x01
        value: value
        type: s4
      - if: type == 0x02
        value: value
        type: str
      - if: type == 0x03
        value: value
        type: byte #Or possibly a more specific type if known, and if it can be determined from the metadata.
```

In this case, I’ve used an instance with conditional values. We are not defining custom types, rather we are providing information which we can use when processing the data, if necessary. This is quite a bit different from trying to define a new type for every possible structure. The instance `parsed_value` becomes a dictionary that can be accessed from the instance. This still doesn't automatically *parse* the data. Instead, we describe how we could parse it if we wanted to. Kaitai uses a *lazy evaluation*, meaning the expression is only computed when it is accessed. We still don't need to define custom types. We use the 'type' field to determine how to treat 'value' within an expression.

Let's consider a scenario where the length field itself isn't fixed but rather variable-length encoded. I saw this on a project that used a custom protocol for embedded devices a long time ago. It used a 7-bit encoded integer for the length, with the most significant bit indicating continuation. This can also be easily handled:

```yaml
seq:
    - id: type
      type: u1
    - id: length
      type: vlq_base128_le #using kaitai built-in type
    - id: value
      type: byte
      size: length
```

In this example, the `vlq_base128_le` type is a predefined Kaitai type used to parse a variable-length quantity, in this case a little endian variant. We are not creating a custom type, but we leverage a pre-existing type that fits our needs. Again the 'value' is treated simply as a byte array, avoiding the need to create custom types.

So, the core message here is that you *can* effectively describe TLV structures in Kaitai Struct without declaring new types for every possible data format within the value, provided you are willing to handle some of the specifics of how the data is actually used outside of the Kaitai descriptor. The raw bytes, combined with information from the type and length fields, are sufficient for most use cases. The *interpretation* of the 'value' field's byte sequence is typically pushed to the application that reads the data, which is perfectly appropriate. You might need to use the Kaitai-generated object and write code to interpret each type, but you avoid defining countless types.

To dive deeper into handling binary structures efficiently, I recommend these resources:

*   **"Crafting Interpreters" by Robert Nystrom**: While primarily focused on language design, this book provides fundamental knowledge about parsing, which is directly relevant to understanding how Kaitai works.
*   **"Modern Compiler Implementation in C" by Andrew W. Appel**: Another compiler book, but again extremely valuable for understanding low-level parsing principles. If you know how a compiler is implemented, this kind of parsing will be much clearer.
*   **The official Kaitai Struct documentation and tutorials:** Specifically, pay close attention to the sections on built-in data types, instances, and expressions. The documentation is very thorough, and a deep dive into it is a requirement for using Kaitai effectively.

In short, by using byte sequences and applying conditional logic or processing the information in post-parsing processing, you can describe a wide range of TLV-based formats in Kaitai Struct without declaring a myriad of new types. This results in more maintainable and extensible parsers, which has been my consistent experience.
