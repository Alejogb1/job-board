---
title: "Why are Verilog outputs showing X and Z instead of 0 and 1?"
date: "2025-01-30"
id: "why-are-verilog-outputs-showing-x-and-z"
---
The appearance of 'X' and 'Z' values on Verilog outputs stems fundamentally from the language's four-valued logic system.  Unlike simpler binary representations, Verilog incorporates 'X' to represent an unknown value and 'Z' to represent a high-impedance state.  Understanding the distinction between these undefined states and their propagation through a design is critical to debugging and ensuring functional correctness. My experience debugging complex FPGA designs across numerous projects has highlighted the frequent misinterpretation of these states, leading to seemingly inexplicable simulation results.


**1. Clear Explanation of X and Z**

Verilog's four-valued logic system extends the typical binary {0, 1} with 'X' and 'Z'.  'X' signifies an unknown logic value.  This uncertainty can arise from several sources:  uninitialized registers, conflicting assignments within a single timestep (e.g., assigning both 0 and 1 to the same variable concurrently), or connections to unconnected ports.  Crucially, 'X' propagates through combinatorial logic.  If an input to an AND gate is 'X', the output will also be 'X', regardless of the other input's value.  This behavior accurately reflects the inherent uncertainty: if one input's state is unknown, the output's state must also be unknown.

'Z' represents a high-impedance state, typically associated with tri-state buffers or buses.  A high-impedance output essentially disconnects from the circuit, neither asserting a 0 nor a 1. This is crucial for shared buses where only one device should drive the bus at a time to prevent conflicts.  Unlike 'X', 'Z' does not necessarily propagate as 'Z' through combinatorial logic.  How 'Z' behaves depends heavily on the surrounding circuitry and the specific logic functions involved.  For instance, a 'Z' connected to an AND gate will typically result in a 0, but a 'Z' connected to an OR gate can result in an 'X' if the other input is unknown.

The simulation tools often highlight these situations through warning messages.  Ignoring these warnings leads to inaccurate simulations and may manifest as unpredictable behavior in synthesized hardware.


**2. Code Examples with Commentary**

**Example 1: Uninitialized Register**

```verilog
module uninit_reg;
  reg [7:0] data;
  initial begin
    $monitor("Time: %t, data: %b", $time, data);
    #10;
  end
endmodule
```

In this example, the register `data` is not initialized.  The simulator will assign it an 'X' value initially.  The `$monitor` system task will display 'X' until a value is explicitly assigned.  Failure to initialize registers can lead to spurious X's propagating through the design.


**Example 2: Conflicting Assignments**

```verilog
module conflicting_assign;
  reg out;
  initial begin
    out = 0;
    #5 out = 1;
    #5 out = 0; //Conflicting assignment within the same time step
    $monitor("Time: %t, out: %b", $time, out);
    #10;
  end
endmodule
```

This example illustrates concurrent assignments within the same time step.  The assignments to `out` at time 5 and 5 are made simultaneously, resulting in an 'X' value.  The simulator will report this conflict, and the output `out` will become 'X'. This highlights the importance of careful timing and avoiding simultaneous conflicting assignments.


**Example 3: Tri-state Buffer**

```verilog
module tristate_example;
  reg enable;
  reg in;
  wire out;

  assign out = enable ? in : 8'bz; // Tri-state buffer

  initial begin
    enable = 1; in = 1; #10;
    enable = 0; in = 0; #10;
    enable = 1; in = 0; #10;
    $monitor("Time: %t, enable: %b, in: %b, out: %b", $time, enable, in, out);
    #10;
  end
endmodule
```

This demonstrates a tri-state buffer. When `enable` is 0, the output `out` is in a high-impedance state ('Z').  The `8'bz` assignment explicitly sets the output to this state.  Observe the output's behavior: when disabled, the output goes to 'Z', and the subsequent logic depends on how this 'Z' is handled by connected components. Note that the simulation will typically present a 'Z' in this case.


**3. Resource Recommendations**

For a deeper understanding, I recommend studying the Verilog Language Reference Manual provided by your synthesis tool vendor.  Pay particular attention to the sections on data types, operators, and simulation behavior.  Additionally, mastering the use of simulation debugging tools within your Integrated Development Environment (IDE) is crucial for pinpointing the source of 'X' and 'Z' values within your design.   Familiarity with the documentation concerning your specific FPGA architecture and its handling of high-impedance states is also valuable.  Finally, reviewing case studies and examples from experienced Verilog developers can provide practical insights into common pitfalls and effective debugging strategies.  These resources, when used diligently, significantly enhance your ability to resolve issues involving unexpected 'X' and 'Z' outputs in your Verilog designs.
