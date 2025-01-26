---
title: "Is this soft core radiation-tolerant?"
date: "2025-01-26"
id: "is-this-soft-core-radiation-tolerant"
---

Given the increasing prevalence of embedded systems operating in harsh radiation environments, evaluating the inherent radiation tolerance of a soft core processor is a critical step in system design. My experience, particularly with custom FPGA-based systems for aerospace applications, has highlighted that the question "Is this soft core radiation-tolerant?" is fundamentally nuanced. The answer is not a simple yes or no, but instead depends heavily on the specific radiation effects being considered and the design implementation choices surrounding the soft core.

A soft core processor, being implemented in programmable logic rather than as a hardened integrated circuit, is susceptible to radiation-induced errors at the logic gate level. These errors manifest primarily as Single Event Effects (SEEs). SEEs, caused by a single ionizing particle striking a sensitive region, can broadly be classified into Single Event Upsets (SEUs) and Single Event Latch-ups (SELs). An SEU is a transient change in the state of a memory element (flip-flop, RAM bit), potentially causing a bit flip and erroneous computation. An SEL is a more severe event, leading to excessive current draw and potentially permanent device damage. SELs are less of a concern in most modern FPGAs due to built-in latch-up mitigation features, but SEUs remain a primary concern for soft core robustness.

Crucially, the "radiation-tolerance" of a soft core is a function of its specific architecture, the FPGA fabric it's instantiated on, and the design choices made by the implementer. A soft core that employs extensive pipelining and a large number of registers, for example, will present a larger area susceptible to SEUs compared to a simpler, non-pipelined design. Similarly, the FPGA's architecture, process technology, and fabrication process will influence the SEU susceptibility at a fundamental level. While all soft cores are intrinsically vulnerable, techniques can be applied to mitigate the effects of these errors.

Let's consider a simple example: a basic 8-bit accumulator built as a soft core. Without any error mitigation, an SEU striking the accumulator register would directly corrupt the accumulator's value.

```c
// Basic 8-bit Accumulator (no error mitigation)
#include <stdint.h>

uint8_t accumulator = 0;

void accumulate(uint8_t data){
  accumulator += data;
}

// Usage:
// accumulate(5);  //accumulator = 5
// accumulate(10); //accumulator = 15
```

In this code, a single bit flip within the `accumulator` variable in memory, caused by an SEU, would lead to incorrect accumulation results. The inherent soft core implementation of this accumulator, whether derived from a hardware description language (HDL) or an FPGA's intellectual property (IP) core, could be compromised.

A common mitigation strategy is to introduce Triple Modular Redundancy (TMR), where three identical modules perform the same operation concurrently and a voting circuit chooses the majority output. This approach is particularly effective at masking single bit flips caused by SEUs in one of the redundant modules.

```c
// 8-bit Accumulator with Triple Modular Redundancy (TMR)

#include <stdint.h>

uint8_t accumulator1 = 0;
uint8_t accumulator2 = 0;
uint8_t accumulator3 = 0;

uint8_t vote(uint8_t a, uint8_t b, uint8_t c) {
    uint8_t majority = 0;
    for (int i = 0; i < 8; i++) {
        uint8_t bit1 = (a >> i) & 0x01;
        uint8_t bit2 = (b >> i) & 0x01;
        uint8_t bit3 = (c >> i) & 0x01;
        if ((bit1 + bit2 + bit3) >= 2) {
            majority |= (1 << i);
        }
    }
    return majority;
}


void accumulate_tmr(uint8_t data){
  accumulator1 += data;
  accumulator2 += data;
  accumulator3 += data;
}

uint8_t get_accumulator(){
   return vote(accumulator1,accumulator2,accumulator3);
}

// Usage:
// accumulate_tmr(5); //accumulator1=5, accumulator2=5, accumulator3=5
// uint8_t acc = get_accumulator(); //acc = 5
```

In this modified example, three accumulators perform the same operation, and a voting function compares their results bit-by-bit, outputting the value held by the majority. If an SEU corrupts the value of one of the accumulators, the voting function will still output the correct result. Note that while the C-code performs the accumulation, the actual implementation in the soft core would need to replicate the accumulator logic three times and implement the voting logic using flip-flops and gates.

Another mitigation strategy involves using Error Correction Codes (ECC), particularly with memory elements. An ECC can detect and sometimes correct errors, allowing data to be recovered from a corrupted memory location. A simple example can be seen in Hamming Code implementation:

```c
// Example of Hamming Code implementation (simplified)
#include <stdint.h>

// Function to generate Hamming code for a 4-bit value
uint8_t generateHammingCode(uint8_t data) {
    uint8_t code = 0;
    uint8_t d1 = (data >> 0) & 0x01;
    uint8_t d2 = (data >> 1) & 0x01;
    uint8_t d3 = (data >> 2) & 0x01;
    uint8_t d4 = (data >> 3) & 0x01;

    uint8_t p1 = d1 ^ d2 ^ d4;
    uint8_t p2 = d1 ^ d3 ^ d4;
    uint8_t p3 = d2 ^ d3 ^ d4;

    code = (p1 << 0) | (p2 << 1) | (d1 << 2) | (p3 << 3) | (d2 << 4) | (d3 << 5) | (d4 << 6);
    return code;
}

// Function to correct a single error in a Hamming code
uint8_t correctHammingCode(uint8_t code) {
    uint8_t p1 = (code >> 0) & 0x01;
    uint8_t p2 = (code >> 1) & 0x01;
    uint8_t d1 = (code >> 2) & 0x01;
    uint8_t p3 = (code >> 3) & 0x01;
    uint8_t d2 = (code >> 4) & 0x01;
    uint8_t d3 = (code >> 5) & 0x01;
    uint8_t d4 = (code >> 6) & 0x01;

    uint8_t syndrome = (p1 ^ d1 ^ d2 ^ d4) | ((p2 ^ d1 ^ d3 ^ d4) << 1) | ((p3 ^ d2 ^ d3 ^ d4) << 2);
    uint8_t correctedCode = code;

    switch (syndrome) {
    case 1: correctedCode ^= (1 << 0); break;
    case 2: correctedCode ^= (1 << 1); break;
    case 3: correctedCode ^= (1 << 2); break;
    case 4: correctedCode ^= (1 << 3); break;
    case 5: correctedCode ^= (1 << 4); break;
    case 6: correctedCode ^= (1 << 5); break;
    case 7: correctedCode ^= (1 << 6); break;
    }

     uint8_t corrected_d1 = (correctedCode >> 2) & 0x01;
     uint8_t corrected_d2 = (correctedCode >> 4) & 0x01;
     uint8_t corrected_d3 = (correctedCode >> 5) & 0x01;
     uint8_t corrected_d4 = (correctedCode >> 6) & 0x01;
     return (corrected_d1 << 0) | (corrected_d2 << 1) | (corrected_d3 << 2) | (corrected_d4 << 3);
}

// Usage
// uint8_t data = 10;
// uint8_t code = generateHammingCode(data); // code contains ECC
// code ^= (1<<1);  // Simulate a bit-flip
// uint8_t correctedData = correctHammingCode(code); //correctedData == 10

```
This simplified example demonstrates how 4-bit data can be encoded with an additional parity bits. This code demonstrates correction capabilities but, in reality, the data and code bits are stored across multiple physical locations on the FPGA, where this code is only an example in principle. In a complete soft core design, this technique might be applied to the data stored in registers and RAM elements. In practice, this code would be integrated as hardware using HDL logic.

From my experience, achieving genuine radiation tolerance for a soft core processor is a multi-faceted effort. Mitigation techniques always come at the expense of increased resource utilization and often performance overhead. A design trade-off needs to be evaluated in light of the operating environment. The use of TMR and ECC, as shown above, are common strategies. Furthermore, techniques like scrubbing are essential in mitigating the accumulation of bit flips over time in the FPGA's configuration memory. This involves periodically re-writing the FPGA’s configuration to correct errors.

For further reading on this subject, I recommend consulting resources from organizations such as the Institute of Electrical and Electronics Engineers (IEEE), focusing on papers detailing radiation effects on microelectronics, particularly in FPGAs. Also relevant are textbooks covering digital design and fault-tolerant computing. Reports from NASA and the European Space Agency (ESA) related to radiation hardening in space electronics also offer valuable insights. Finally, FPGA vendors often publish design guides and application notes discussing techniques to improve soft core radiation tolerance, which are helpful when working with a particular FPGA platform.
