---
title: "How do I write Kaitai structs?"
date: "2025-01-30"
id: "how-do-i-write-kaitai-structs"
---
Kaitai Struct, a declarative language, provides a method for defining binary file formats, enabling parsing into structured data. It is not a procedural programming language; rather, it defines the structure of binary data using a YAML-based format which then gets compiled into parser code for various target languages. This allows for a single specification to be reused across different platforms. My experience, particularly in reverse-engineering network packet captures and embedded firmware images, has demonstrated the profound utility of Kaitai Struct in managing complex binary data.

Writing a Kaitai Struct involves describing the layout and interpretation of data within a binary file. This includes specifying field names, data types, bit sizes, endianness, and other relevant parameters. A critical distinction is that we are describing the *what* rather than the *how*. We're declaring the format, not creating parsing logic directly. The Kaitai compiler then generates the necessary parsing code.

At its core, a Kaitai Struct definition consists of several key components:

*   **`seq`:** This sequence defines the order in which fields appear in the binary stream. Each entry within `seq` describes a single field and its associated data type.
*   **`types`:** These sections define reusable complex data structures which can contain nested fields. These types help encapsulate complex logical units within a file.
*   **`instances`:** These are computed fields – values derived from previously read fields. This enables, for instance, dynamically calculated lengths or values based on flags read earlier in the stream.
*   **`enums`:** This allows for the conversion of numeric values into human-readable names, improving the comprehensibility of the parsed data.
*   **`params`:** Parameters passed to the structure definition, useful when parsing multiple variations of data with slight differences in structure.

Let's illustrate with examples based on various common use-cases.

**Example 1: A Simple Fixed-Length Record**

Imagine a simple logging format I encountered while working with a proprietary sensor device. Each log entry contained a timestamp (32-bit unsigned integer), a sensor reading (16-bit signed integer) and a status code (8-bit unsigned integer). Here's how I defined this with Kaitai Struct:

```yaml
meta:
  id: simple_log_entry
  file-extension: log

seq:
  - id: timestamp
    type: u4
  - id: sensor_value
    type: s2
  - id: status_code
    type: u1
```

Here:

*   `meta.id` defines the unique identifier of this structure.
*   `meta.file-extension` provides a hint for recognizing files that might conform to this format.
*   `seq` dictates the order of fields: first, a 32-bit unsigned integer (`u4`) for the timestamp, then a 16-bit signed integer (`s2`) for the sensor value, and finally an 8-bit unsigned integer (`u1`) for the status code.

The Kaitai compiler, given this YAML file, will generate a class with corresponding attributes to access the `timestamp`, `sensor_value` and `status_code` values. The target language (e.g., Python, Java, C++) depends on the chosen compiler invocation.

**Example 2: Dynamically Sized String with Length Prefix**

In some networking protocols I have analyzed, strings were not fixed length; rather they were preceded by a length field.  Below demonstrates how to handle a string that begins with a single-byte length indicator.

```yaml
meta:
  id: length_prefixed_string

seq:
  - id: length
    type: u1
  - id: str_data
    type: str
    size: length
    encoding: utf-8
```

In this case:

*   The `seq` defines two fields. First is `length` of type `u1`, an unsigned 8-bit integer.
*   The second field, `str_data` is of type `str`.
*   The important part here is `size: length`. This instructs Kaitai to read as many bytes as indicated by the `length` field *preceding* the string data.
*   `encoding: utf-8` specifies that the parsed string should be decoded using UTF-8.

This definition automatically handles variable string sizes without the need to explicitly calculate string lengths, an important feature when parsing protocols with varying message lengths.

**Example 3:  A Complex Record with Nested Types and Enums**

My work also led me to parse an older firmware image that contained a rather complex header. The header had a combination of flags, a version number, and configuration blocks. Here's a simplified example demonstrating nested types and enumerations within the header structure:

```yaml
meta:
  id: firmware_header

types:
  configuration_block:
    seq:
      - id: config_id
        type: u1
      - id: config_data
        type: u2

enums:
  status_flags:
    0x01: FLAG_ENABLED
    0x02: FLAG_ERROR
    0x04: FLAG_LOCKED

seq:
  - id: version
    type: u2
  - id: flags
    type: u1
    enum: status_flags
  - id: config_count
    type: u1
  - id: config_blocks
    type: configuration_block
    repeat: expr
    repeat-expr: config_count
```

In this more advanced example:

*   A nested `type` called `configuration_block` is defined, consisting of a 1-byte ID and a 2-byte data field.
*   An enumeration `status_flags` is defined, mapping bit flags to human-readable constants.
*   In the main `seq` the `flags` field is read as a single byte but interpreted using the `status_flags` enumeration. The `config_blocks` entry is a sequence of nested structures which are defined via the `configuration_block` type. The `repeat: expr` and `repeat-expr: config_count` indicates that the config blocks will be repeated as many times as indicated by the previously parsed `config_count` field.

This final example shows the power of Kaitai Struct.  By combining nested types, enumerated values, and repeated structures, we can define complex structures concisely and easily without writing parsing code by hand. This facilitates the analysis of even highly complex binary formats.

When implementing Kaitai Struct definitions, several practices are beneficial:

*   **Start Simple:** Begin with the smallest data unit and gradually expand the complexity. Initially parse the known structures, then progressively enhance the definition as more of the data format becomes known.
*   **Refer to Official Documentation:** Kaitai Struct’s documentation is extensive and readily available. Consult this resource when encountering difficulties or for understanding advanced features.
*   **Test and Validate:** After creating a struct, test it with sample binary files.  This confirms that the struct correctly parses the data.
*   **Use Meaningful Names:**  Assign descriptive names to `id`s,  `types`, and `enums`. This improves the clarity and maintainability of the specification, especially when working with complex formats or collaborating with others.
*   **Iterative Refinement:** It is common to discover nuances in the data format as you proceed through analysis and testing, and thus the definition is subject to iterative updates.

Resources beyond the core Kaitai documentation:

*   **Kaitai Struct Gallery:** This repository provides a multitude of pre-defined Kaitai Struct specifications for many common file formats, providing both reference material and a starting point for new definitions. Studying these examples can offer insights into best practices and common patterns.
*   **Community Forums:** Active discussion groups or communities exist where users share their experiences, pose questions, and offer solutions related to Kaitai Struct. Participation can be beneficial when troubleshooting difficult formats or seeking guidance.
*   **GitHub:** The official Kaitai Struct repository and forks contain code examples, bug fixes and feature requests, allowing for closer inspection of the implementation and understanding the nuances of the tool.

In conclusion, using Kaitai Struct simplifies the analysis of binary data. By creating declarations of the data structure and then relying on its compiler to generate parsing code, we avoid manual bit-twiddling, which enhances accuracy, clarity, and reduces development time when working with binary data. Its capacity to define data structures once and reuse across multiple programming languages and contexts proves to be exceptionally useful.
