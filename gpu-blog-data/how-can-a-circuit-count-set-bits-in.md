---
title: "How can a circuit count set bits in a 15-bit input?"
date: "2025-01-30"
id: "how-can-a-circuit-count-set-bits-in"
---
Implementing a circuit to count set bits (bits with a value of 1) within a 15-bit input requires a systematic approach, fundamentally relying on combinational logic. Specifically, we need a structure that effectively sums the individual bit values, treating each bit as either 0 or 1. My experience in digital design has led me through various implementations, and the method I find most efficient for this scenario involves a layered approach using full adders.

At its core, a full adder takes three inputs: two bits to be added, and a carry-in bit. It produces two outputs: a sum bit and a carry-out bit. These carry-out bits propagate to the next stage of the adder. This operation is instrumental because it facilitates the addition of multiple bits. In this context, we don't have multi-bit values we are adding, rather individual bit values of either 0 or 1. Nevertheless, the full adder can function as the elemental building block in our design.

To count the set bits in a 15-bit input, we must consider this as a summation problem of individual bit values. Since 15-bit is not a power of 2, a few approaches need to be carefully considered, namely: using full adders to sum 2 bits, then sum the result with the next bit, repeating, or employing a tree-like adder structure. The tree-like structure offers better performance as it reduces the propagation delay through the circuit, and is the one we'll focus on.

The first layer of our tree would sum pairs of input bits. With 15 input bits (let's designate them as `b[0]` to `b[14]`), we can combine `b[0]` and `b[1]` with the first adder, `b[2]` and `b[3]` with the second, and so on. This process results in eight sums (seven using all of the input bits, plus one dummy sum that adds the carry out from the seventh pair and a logic 0 together), and seven carry outputs that also need to be added in subsequent stages. Each full adder, then, becomes a crucial block in this process.

For the next stage, the sums from the first layer and the carry outs from the first layer become the new inputs to another layer of adders. We repeat this process of summing pairs, and their carry outs, until we have a final two numbers that sum the totals across all the layers. The final layer adds these two, the sum and carry-out from the penultimate stage, providing a binary count of set bits. The output of the final adder, which is a four bit number, will be a binary representation of the number of set bits in the original 15-bit input.

The final adder must consider the propagation of carry outs from each addition, requiring careful selection and connection of full adders. Since a maximum of 15 set bits will result in a maximum number of 15 which can be represented in binary as `1111`, we will need 4 output bits from our final adder. This method efficiently utilizes parallel computation, resulting in faster computation of the total number of set bits compared to a simple chain of adders.

Below are three code examples that provide different approaches to implementing this bit counting logic, although the descriptions are written generically because hardware description language like Verilog, VHDL is not available in this environment.

**Example 1: Layered Full Adders in a Tree-Like Structure**

```
// Pseudocode representing the layered full adder structure for a 15-bit input
// Assuming 'b' is a 15-bit input array

// Stage 1: Summing pairs of input bits
sum1_0, carry1_0 = full_adder(b[0], b[1], 0);  // No carry-in for first full adder
sum1_1, carry1_1 = full_adder(b[2], b[3], 0);
sum1_2, carry1_2 = full_adder(b[4], b[5], 0);
sum1_3, carry1_3 = full_adder(b[6], b[7], 0);
sum1_4, carry1_4 = full_adder(b[8], b[9], 0);
sum1_5, carry1_5 = full_adder(b[10], b[11], 0);
sum1_6, carry1_6 = full_adder(b[12], b[13], 0);
sum1_7, carry1_7 = full_adder(b[14], 0, 0);

// Stage 2: Combining sums and carries from Stage 1
sum2_0, carry2_0 = full_adder(sum1_0, sum1_1, carry1_0);
sum2_1, carry2_1 = full_adder(sum1_2, sum1_3, carry1_1);
sum2_2, carry2_2 = full_adder(sum1_4, sum1_5, carry1_2);
sum2_3, carry2_3 = full_adder(sum1_6, sum1_7, carry1_3);
sum2_4, carry2_4 = full_adder(carry1_4, carry1_5, carry1_6);
sum2_5, carry2_5 = full_adder(carry1_7, 0, 0);

//Stage 3
sum3_0, carry3_0 = full_adder(sum2_0, sum2_1, carry2_0);
sum3_1, carry3_1 = full_adder(sum2_2, sum2_3, carry2_1);
sum3_2, carry3_2 = full_adder(sum2_4, sum2_5, carry2_2);

//Stage 4
sum4_0, carry4_0 = full_adder(sum3_0, sum3_1, carry3_0);
sum4_1, carry4_1 = full_adder(sum3_2, 0, carry3_1);

//Final Stage
final_sum, final_carry = full_adder(sum4_0, sum4_1, carry4_0)
// Output is the final sum and final_carry.
// output = [ final_carry, sum4_1, sum4_0, final_sum ];
```

This code demonstrates the general structure for implementing this logic by showing how the full adder units connect together to produce a result. Each line represents an instantiation of a full adder component with the sum and carry outputs used as the inputs to the next layer of full adders, culminating in a four bit final output.

**Example 2: Conceptual Stepwise Approach with Single Full Adder Reuse**

```
// Pseudocode representing iterative full adder usage
// Assuming 'b' is a 15-bit input array

sum_result = 0;
carry_result = 0;

for i from 0 to 14
   sum_temp, carry_temp = full_adder(b[i], sum_result, carry_result);
   sum_result = sum_temp;
   carry_result = carry_temp;

// Output is a combination of sum_result and carry_result, will need more bits to represent correct answer
// Need additional full adders to sum carry_result with sum_result, as carry can accumulate
```

This example conceptually demonstrates how one full adder could be reused repeatedly to achieve bit counting. However, this is not ideal in hardware because it implies sequential operations, which would make the circuit much slower.  The single full adder implementation is not practical for actual hardware because of the propagation delay that will be induced by repeatedly passing a carry output into the next execution of the single full adder. Therefore it is not the method described above for the optimal approach.

**Example 3: Abbreviated Pseudocode Illustrating Multi-Stage Concept**

```
// Pseudocode representing multi-stage summing with abstraction
// Assuming 'b' is a 15-bit input array

stage1_results = sum_pairs(b); // Abstracted function to produce Stage 1 sums and carries
stage2_results = sum_pairs(stage1_results.sums, stage1_results.carries);
stage3_results = sum_pairs(stage2_results.sums, stage2_results.carries);
stage4_results = sum_pairs(stage3_results.sums, stage3_results.carries);
final_results = sum_pairs(stage4_results.sums, stage4_results.carries);
// Output is final_results, four bits

//sum_pairs function takes an array of bits and an array of carries and does the summing, return the result sums and carry outputs
```

This abbreviated example emphasizes the layered concept. The function `sum_pairs` is a placeholder representing multiple full adder units in parallel, performing a similar action to what is described in Example 1, but conceptually. This example can convey the logic without the details of full adder connections, making it clearer to understand the overall multi-layer processing architecture. This high-level approach is useful for understanding the process from a systems-design point of view.

For further study, I recommend focusing on resources that delve into the following topics. First, look for comprehensive textbooks that discuss digital logic design, particularly focusing on combinational logic circuits. Sections on full adders, half adders, and binary adders are crucial. Second, materials on computer architecture, particularly those covering arithmetic logic units (ALUs), will provide a broader context for why bit counting is important. Finally, exploring resources dealing with hardware description languages such as Verilog or VHDL can bridge the gap between theoretical understanding and practical hardware implementation. These resources combined will provide a robust understanding of the problem and its solutions.
