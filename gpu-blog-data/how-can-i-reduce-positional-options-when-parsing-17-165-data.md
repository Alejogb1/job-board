---
title: "How can I reduce positional options when parsing 17-165 data?"
date: "2025-01-26"
id: "how-can-i-reduce-positional-options-when-parsing-17-165-data"
---

Data structures often present challenges in parsing, especially when dealing with variable lengths and implicit positional assumptions. Within the realm of 17-165 data, the core issue stems from the reliance on positional arguments, which makes processing brittle and difficult to maintain as data formats evolve. The fundamental strategy to mitigate this involves shifting from positional parsing to a named-field approach, where each data element is explicitly labeled.

In my previous work processing telemetry data streams from a legacy satellite system, we faced similar problems with data packets lacking explicit field identifiers. The initial system employed a strictly positional method; position one would represent the satellite ID, positions 2-4 the payload type, and so on. Modifications to the payload or new telemetry streams caused significant rework, requiring constant updates to parsing logic. Consequently, debugging became time-consuming due to the lack of clearly defined associations between the data and its meaning.

The ideal solution involves a transformation: moving from positional dependency to a key-value mapping. This transition provides increased flexibility, making the system more robust and resistant to minor data schema variations. It also significantly enhances the readability of parsing code. The positional nature of the 17-165 schema means that each 'field' can occupy a variable number of bits or bytes, sometimes even crossing byte boundaries. We can begin by constructing a descriptor that outlines the structure of the incoming data. This descriptor acts as a key, defining field names and their corresponding starting position and lengths.

The first step typically involves converting the incoming bitstring (or byte array) into a more manageable data structure based on this descriptor. Consider an example where the '17' data part represents a series of flags. Let's assume this section has three fields: `status_flag` (2 bits), `error_code` (5 bits), and `sequence_id` (10 bits), and further the subsequent '165' data contains `sensor_reading_1` (16 bits) and `sensor_reading_2` (16 bits).

**Code Example 1: Initial Positional Parsing (Illustrative)**

```python
def parse_positional_data(data):
    status_flag = (data[0] >> 6) & 0x03  # bits 7-6
    error_code = (data[0] >> 1) & 0x1F  # bits 5-1
    sequence_id = ((data[0] & 0x01) << 9) | (data[1] << 1) | ((data[2] >> 7) & 0x01)  # bits 0-7, 8-15, 16

    sensor_reading_1 = (data[2] << 8) | data[3]
    sensor_reading_2 = (data[4] << 8) | data[5]

    return {
        "status_flag": status_flag,
        "error_code": error_code,
        "sequence_id": sequence_id,
        "sensor_reading_1": sensor_reading_1,
        "sensor_reading_2": sensor_reading_2
    }

# Example Data (6 bytes):
example_data = bytearray([0b10101001, 0b11001100, 0b00110011, 0b10101010, 0b01010101, 0b11110000])

parsed_result = parse_positional_data(example_data)
print(parsed_result)
```

This illustrates the manual bit manipulation required when relying on positional assumptions. This code is hard to read, requires constant verification, and modifications to the data structure will force a complete code overhaul. Furthermore, a new team member would need to understand not just the code, but also the implicit positional mappings to make changes.

To improve, we need a descriptor.

**Code Example 2: Descriptor-Based Parsing**

```python
def parse_descriptor_data(data, descriptor):
    parsed_data = {}
    for field_name, params in descriptor.items():
        start_bit = params['start_bit']
        bit_length = params['bit_length']
        value = 0
        current_bit = start_bit
        while current_bit < start_bit + bit_length:
          byte_index = current_bit // 8
          bit_offset = current_bit % 8
          bits_to_read = min(bit_length - (current_bit - start_bit), 8-bit_offset)
          
          mask = ((1 << bits_to_read)-1) << (8 - bits_to_read - bit_offset)
          value = (value << bits_to_read) | ((data[byte_index] & mask) >> (8 - bits_to_read - bit_offset))
          
          current_bit += bits_to_read
        parsed_data[field_name] = value

    return parsed_data

descriptor = {
  "status_flag": {"start_bit": 0, "bit_length": 2},
  "error_code": {"start_bit": 2, "bit_length": 5},
  "sequence_id": {"start_bit": 7, "bit_length": 10},
  "sensor_reading_1": {"start_bit": 17, "bit_length": 16},
  "sensor_reading_2": {"start_bit": 33, "bit_length": 16}
}


# Example Data (6 bytes):
example_data = bytearray([0b10101001, 0b11001100, 0b00110011, 0b10101010, 0b01010101, 0b11110000])


parsed_result = parse_descriptor_data(example_data, descriptor)
print(parsed_result)
```
This version decouples the parsing logic from the hard-coded offsets. The descriptor defines the structure, allowing the parsing function to operate generically. Now, if a new field is added, we need only modify the descriptor. The parsing logic stays intact. This is an improvement, but still requires manual bit manipulation in the extraction.

While the descriptor allows a clean separation between definition and use, directly dealing with individual bits can become tedious and error-prone as schema complexities increase. A better solution involves utilizing a library designed to handle binary structure parsing. Libraries such as Construct or bitstring in Python allow for the definition of data structures that handle bit-level extraction and conversion directly.

**Code Example 3: Construct Library Parsing (Conceptual)**

```python
from construct import Struct, BitsInteger, Byte, Const, Int16ub

#Assuming we are dealing with Big Endian in this case.
#Example with Construct:

data_structure = Struct(
    "status_flag" / BitsInteger(2),
    "error_code" / BitsInteger(5),
    "sequence_id" / BitsInteger(10),
     "sensor_reading_1" / Int16ub,
     "sensor_reading_2" / Int16ub
)

example_data = bytearray([0b10101001, 0b11001100, 0b00110011, 0b10101010, 0b01010101, 0b11110000])


parsed_result = data_structure.parse(bytes(example_data))
print(parsed_result)
```
This last example, while not directly executable without the library, demonstrates the power of a higher-level abstraction. The data structure is defined once and the library handles the bit-level extraction and type conversions. If the data changes, we adjust the definition, instead of modifying bit-twiddling code. This significantly reduces the chance of introducing errors and also improves readability. This is not a perfect solution, as there is still some manual work needed in building the struct, but these structures can be designed to be reused across multiple parsing applications, leading to increased maintainability.

For further study, I'd recommend researching libraries such as Construct (Python), Kaitai Struct, or similar binary parsing tools specific to your language. Examining existing protocols like CAN bus or Modbus TCP also reveals real-world scenarios where descriptor-based parsing excels. Further, focus on methods of code generation using data descriptions, and techniques for handling data of variable size, within the bounds of the known 17-165 format. These techniques, and adopting a library designed for this type of work can greatly reduce the complexity in parsing 17-165 data.
