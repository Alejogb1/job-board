---
title: "Do IEC 61131/61499 PLC function blocks utilize OPC UA for data transport?"
date: "2025-01-30"
id: "do-iec-6113161499-plc-function-blocks-utilize-opc"
---
IEC 61131-3 and IEC 61499 function blocks do not inherently utilize OPC UA for data transport.  My experience implementing distributed control systems over fifteen years, including projects involving both standards, reveals a crucial distinction:  IEC 61131-3 and 61499 define the programming model and application structure, while OPC UA defines a communication protocol.  They are largely orthogonal, though often used together.

**1. Clear Explanation:**

IEC 61131-3 specifies a standardized programming language for programmable logic controllers (PLCs).  Its function blocks encapsulate specific functionalities, and they communicate internally via defined interfaces within the PLC's runtime environment.  This internal communication is typically handled by the PLC's proprietary communication mechanism, often a high-speed, deterministic bus.  There's no mandate for OPC UA in 61131-3.

IEC 61499, a newer standard, builds upon 61131-3 but introduces a distributed architecture. It emphasizes component reusability and allows function blocks to be deployed across multiple devices.  However, the specification itself does not mandate any specific communication protocol between these distributed function blocks.  The inter-device communication could employ various protocols, including OPC UA, but it's not a requirement.  The choice often depends on factors such as existing infrastructure, real-time requirements, and security considerations.

OPC UA (Open Platform Communications Unified Architecture) serves as a general-purpose communication protocol designed for industrial automation. Its strengths lie in interoperability, security, and scalability.  It’s a powerful tool for integrating various systems and devices, including PLCs programmed with 61131-3 or 61499, but its use is not implicitly defined by either PLC programming standard.

Therefore, while OPC UA can be a highly beneficial choice for connecting PLCs programmed according to IEC 61131-3 or 61499 standards to other systems or to enable communication between distributed function blocks in a 61499 architecture,  the standards themselves do not prescribe its use.  The decision to employ OPC UA rests solely on the system design and the integration requirements.


**2. Code Examples with Commentary:**

These examples are illustrative and simplified; real-world implementations would be more complex and would account for error handling and specific PLC vendor APIs.

**Example 1: IEC 61131-3 Function Block (Structured Text)**

```structuredtext
FUNCTION_BLOCK MyFunctionBlock
VAR_INPUT
    InputValue : INT;
END_VAR
VAR_OUTPUT
    OutputValue : INT;
END_VAR

// Simple calculation - no OPC UA involved
OutputValue := InputValue * 2;

END_FUNCTION_BLOCK
```

This example shows a simple function block in Structured Text, a language supported by IEC 61131-3.  The communication within this function block is purely internal to the PLC's runtime environment. No OPC UA calls or interactions are present. The communication with other parts of the system would be handled by the PLC's I/O system.

**Example 2:  IEC 61499 Function Block (Simplified C-like Representation)**

```c
// Simplified representation - actual implementation varies greatly
typedef struct {
  int input;
  int output;
} MyFB;

int MyFB_execute(MyFB *fb) {
  fb->output = fb->input + 10;
  return 0;
}

// Communication handled by external system (e.g. using OPC UA)
// Assume a mechanism exists for passing 'fb' to and from the communication layer
```

This pseudo-code illustrates a basic 61499 function block. The core logic is separated from the communication aspects.  The `MyFB_execute` function performs the core task. The communication with other function blocks or systems (potentially via OPC UA) would be implemented externally to this function, possibly through an integration layer provided by the 61499 runtime environment.  The `fb` structure would be exchanged via the chosen communication protocol, which could be OPC UA.

**Example 3:  OPC UA Client Accessing PLC Data**

```python
# Python OPC UA client (simplified)
import opcua

client = opcua.Client("opc.tcp://myplcserver:4840")
client.connect()

# Access a variable (e.g., from a 61131-3 or 61499 function block)
var = client.get_node("ns=2;i=10")  # Node ID depends on PLC configuration
value = var.get_value()
print(value)

client.disconnect()
```

This Python code demonstrates an OPC UA client connecting to a PLC server (potentially hosting 61131-3 or 61499 function blocks). It retrieves a value from a specific node on the server. The PLC server needs to be configured to expose its data through OPC UA.  This example highlights how OPC UA can be used to access data from a PLC regardless of the underlying programming standard.  However, the data itself *originates* from the function blocks, but the data exchange is managed by the OPC UA communication layer.


**3. Resource Recommendations:**

For deeper understanding of IEC 61131-3, consult the official standard document and related literature on PLC programming.  Similarly, explore the IEC 61499 standard and associated resources for its distributed architecture and function block models.  To master OPC UA, study its specification and numerous available textbooks and training materials on the topic.  Several industrial automation handbooks provide comprehensive coverage of these technologies and their interrelation.  Finally, focusing on case studies of industrial automation projects that utilize these technologies in tandem will provide valuable insight into real-world integration techniques.
