---
title: "Why does my VHDL code, synthesizable and functional in simulation, fail on the FPGA?"
date: "2025-01-30"
id: "why-does-my-vhdl-code-synthesizable-and-functional"
---
The disconnect between functional simulation and successful FPGA implementation in VHDL often stems from subtle differences in how simulation tools and synthesis tools interpret code, particularly concerning timing assumptions and resource utilization. In my experience, several recurring issues contribute to this problem. A common initial error involves overlooking the crucial distinction between zero-delay simulation and real-world hardware behavior.

**Understanding the Discrepancy**

Simulation tools often operate under ideal conditions. They can, for example, execute multiple processes concurrently without any delay, assuming instant propagation of signals. This works for verifying the logic but is entirely unrealistic in the physical world where signal propagation, logic gate delays, and clock skews all play a significant role. These factors are either ignored or simplified during simulation. For instance, simulation typically handles clock signals as perfect, sharp transitions, whereas in reality, clock signals exhibit rise and fall times, jitter, and skew. Furthermore, simulation might allow the creation of infinitely long, non-physical delays (using the `after` keyword without considering hardware timing), which will not translate to a hardware implementation.

Another critical difference arises from the synthesis process. Synthesizers transform RTL code into a physical implementation on the FPGA, mapping logic and memory blocks to actual hardware resources. During synthesis, the tool attempts to optimize for area and performance, and it might drastically change the underlying implementation while maintaining the functionality. However, synthesis may struggle to interpret certain VHDL constructs that lack a direct hardware equivalent or require specific implementation details not initially addressed in your code. Implicit resource assumptions that work in simulation may create resource conflicts during synthesis. Specifically, inference of flip-flops, latches, or block RAM resources can drastically affect the implementation.

Finally, it’s imperative to consider design constraints. Synthesizers use design constraints such as clock periods, input/output delays, and timing exceptions to guide the optimization process. Insufficient or inaccurate constraints can cause the tool to create an implementation that will fail to meet timing requirements, or even worse, cause the design to fail to operate correctly.

**Code Examples and Commentary**

Here are examples of common issues I've encountered, each followed by a correction:

**Example 1: Improper Use of `after` Clause in Combinatorial Logic**

```vhdl
-- Inefficient and potentially non-synthesizable
process(a,b)
begin
  c <= a and b after 5 ns; 
end process;
```

**Commentary:**
While this syntax is valid for simulation, and will produce the AND output 'c' after a 5 ns delay, it’s not translatable to hardware. The delay is an instruction to a simulator, not something directly synthesizable. The synthesis tool will usually ignore this delay, effectively implementing `c <= a and b;` which might lead to discrepancies in expected behavior. Using delays in combinatorial logic is, in general, considered a bad practice.

```vhdl
--Correct implementation for combinational logic:
process (a, b)
begin
  c <= a and b; 
end process;
```

**Commentary:**
Here, the process has been written without specifying the delay. The output 'c' is a direct function of the inputs 'a' and 'b' through a simple AND gate.  The hardware implementation will still exhibit gate delays, but the synthesizer will handle the appropriate timing closure during place-and-route. No explicit delay is needed.

**Example 2: Latches Inferred unintentionally**

```vhdl
process(enable, data_in)
begin
  if (enable = '1') then
    data_out <= data_in;
  end if;
end process;
```

**Commentary:**
This example shows the common issue of unintentional latch creation, which leads to instability. If the 'enable' is '0', what happens to `data_out`? The VHDL code does not specify behavior for that condition. Thus, a latch will be inferred, holding the last known value of the `data_out`. This can create timing problems since the feedback path is implicit, can easily violate setup/hold time constraints, and is difficult to debug. Latches are rarely desirable in FPGAs due to their susceptibility to glitches and timing issues.

```vhdl
process(enable, data_in)
begin
    if (enable = '1') then
        data_out <= data_in;
    else
        data_out <= '0';  --or any other default value that makes sense
    end if;
end process;
```

**Commentary:**
By adding the `else` clause, we explicitly define what should happen to the output when the 'enable' is low, and prevent the formation of a latch. The synthesizer can now infer logic without feedback. This ensures deterministic behavior, and also helps debugging efforts. This design will behave more predictably in hardware.

**Example 3: Uncontrolled Clock Domain Crossing**

```vhdl
--Poor Clock Domain Crossing
process (clk_a)
begin
    if rising_edge(clk_a) then
       signal_b <= some_other_signal;
    end if;
end process;

process(clk_b)
begin
   if rising_edge(clk_b) then
     final_output <= signal_b;
   end if;
end process;
```

**Commentary:**
This example attempts to move the signal 'some_other_signal' from the clock domain `clk_a` to clock domain `clk_b` directly without any synchronization.  This is a textbook example of a clock domain crossing (CDC) violation.  Simulation will likely show this working because the simulator may not accurately model the asynchronicity and the timing relationship between `clk_a` and `clk_b`. However, this is highly prone to metastable conditions in the hardware, resulting in unpredictable behavior and data corruption.

```vhdl
--Correct Clock Domain Crossing (using a two-stage synchronizer)
process (clk_a)
begin
  if rising_edge(clk_a) then
      signal_b_sync1 <= some_other_signal;
  end if;
end process;

process (clk_b)
begin
  if rising_edge(clk_b) then
    signal_b_sync2 <= signal_b_sync1;
    final_output  <= signal_b_sync2;
  end if;
end process;
```

**Commentary:**
This corrected example uses a standard two-stage synchronizer. The signal `some_other_signal` is captured by a flip-flop clocked by `clk_a` and the output of that flip-flop is then used as an input to a second flip-flop clocked by `clk_b`. This reduces the likelihood of metastability at the cost of latency (in this example, one clock cycle in `clk_b` domain). There are other more complex synchronizers depending on the application but this simple solution is appropriate for most situations where data transfer happens between asynchronous clock domains.

**Recommendations and Resources**

To mitigate issues between simulation and synthesis, adopt the following recommendations:

1.  **Follow Synthesis Guidelines:** Refer to the specific FPGA vendor’s coding guidelines or synthesis documentation. Most vendors provide specific information about the VHDL constructs they support and their best implementation practices. This will help ensure that your design is synthesizable and meets all required performance goals.
2.  **Use Timing Constraints Early:** Define timing constraints (clock period, input/output delays) in your design and ensure they reflect the target operating conditions. Early use of timing constraints helps the synthesizer make better decisions and avoid timing violations which often cause the hardware to fail. 
3.  **Thoroughly Simulate and Test:** While simulation is not a perfect replication of the hardware, rigorous testing with multiple vectors can help uncover many problems early. Pay close attention to edge cases and corner conditions. Also, be sure to examine resource utilization reports from the synthesis tools and make sure no resources have been under-utilized or are causing bottlenecks.
4.  **Employ Static Timing Analysis:** Review static timing analysis reports generated by your FPGA vendor's tools. These reports will reveal setup and hold time violations, which may not be apparent in simulation. These reports are critical for proper hardware operation. 
5.  **Refer to General FPGA Design Resources:** Look into textbooks and online guides on FPGA design with VHDL. There are resources that cover topics such as synchronous design, clock domain crossing, and RTL best practices in a more theoretical yet highly practical way.

In summary, discrepancies between simulation and FPGA implementation arise from simplified timing models in simulation tools, resource mapping conflicts during synthesis, and lack of proper constraints. By employing meticulous coding practices, considering hardware implications of VHDL, and paying close attention to design constraints and timing analysis, you can achieve a more robust and successful design implementation in hardware.
