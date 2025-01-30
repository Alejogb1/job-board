---
title: "How do I view changed internal signals in Quartus, Modelsim, and SystemVerilog?"
date: "2025-01-30"
id: "how-do-i-view-changed-internal-signals-in"
---
Observing internal signals within a design flow encompassing Quartus Prime, ModelSim, and SystemVerilog requires a methodical approach leveraging the strengths of each tool.  My experience debugging complex FPGA designs over the last decade has highlighted the crucial role of strategic signal visibility in efficient verification.  The key is to understand that signal access is not a monolithic process, but rather a layered approach involving design coding practices, simulation testbenches, and finally, post-synthesis/implementation debugging within the Quartus environment.

**1.  Clear Explanation of the Methodology**

Effective signal visibility begins at the SystemVerilog source code level.  Proper signal naming conventions and the strategic use of `$display` statements, along with more sophisticated verification methodologies, form the bedrock of effective debugging. During simulation with ModelSim, waveforms are instrumental for visualizing signal behavior. However, these waveforms only reflect the simulated behavior, not necessarily the final, synthesized hardware.  Therefore, post-synthesis or post-implementation debugging within Quartus Prime is often necessary to confirm that the implemented design functions as intended.  This often requires different signal access mechanisms depending on the level of abstraction.

The process can be broken down into these phases:

* **SystemVerilog Coding for Observability:**  Design your SystemVerilog code with observability in mind.  Meaningful signal names are paramount. Avoid generic names;  use descriptive names that directly reflect the signal's function.  For critical signals, consider adding dedicated monitoring mechanisms within the design, potentially using dedicated monitoring processes, or assertions that trigger on specific conditions.  These assertions may log to a file, facilitating later analysis.

* **ModelSim Simulation and Waveform Analysis:**  Employ ModelSim to simulate your design.  Add signals of interest to the waveform window. This allows for dynamic analysis of the signal behavior during simulation. The timing diagram provided by ModelSim is invaluable in understanding signal interactions and identifying potential issues before synthesis. Pay particular attention to signal transitions, delays, and unexpected values.  Testbenches should be comprehensive, exercising various scenarios to ensure complete signal coverage.

* **Quartus Prime Post-Synthesis/Post-Implementation Debugging:** After successful synthesis and implementation in Quartus Prime, the signals you observed in simulation might not directly map to the final hardware implementation.  This is especially true after optimization steps taken by the synthesis and fitter tools.  Quartus Prime offers several mechanisms for probing signals within the implemented design:

    * **SignalTap II Embedded Logic Analyzer:** This powerful tool allows you to embed logic analyzers within your design, capturing signal values during operation. You define signals to be captured, and SignalTap II generates a memory block to store the captured data.  This data can then be viewed post-implementation.  Note that SignalTap II introduces resource overhead and may necessitate design modifications to accommodate it.  This tool is ideal for post-implementation verification when simulation is insufficient.

    * **Timing Analyzer:** Quartus Prime's Timing Analyzer can help uncover timing violations and provide insights into signal propagation delays.  While not directly showing signal values, understanding the timing characteristics is critical in resolving functional issues.

    * **Post-Implementation Simulation:** After implementation, you can generate a post-implementation simulation model in Quartus Prime.  This simulation model reflects the actual implemented hardware, accounting for optimizations. Simulating with this model can validate whether the implemented design behaves as expected.  However, it requires careful consideration of the necessary testbenches.


**2. Code Examples with Commentary**

**Example 1: SystemVerilog with Assertions for Monitoring**

```systemverilog
module my_module (
  input clk,
  input rst,
  input data_in,
  output data_out
);

  logic internal_signal;

  always_ff @(posedge clk) begin
    if (rst) begin
      internal_signal <= 0;
    end else begin
      internal_signal <= data_in;
    end
  end

  assign data_out = internal_signal;

  // Assertion to monitor internal_signal
  assert property (@(posedge clk) disable iff (rst)
                   $past(internal_signal) !== internal_signal |-> $display("Internal signal changed! New value: %b", internal_signal));


endmodule
```

This example shows a simple module with an assertion that triggers and displays a message whenever `internal_signal` changes.  This is a rudimentary form of monitoring; for more complex scenarios, consider using more advanced assertion techniques and logging mechanisms.


**Example 2: ModelSim Waveform Monitoring**

ModelSim's graphical interface enables adding signals to a waveform window for visualization.  This is done through the ModelSim GUI, not directly within the SystemVerilog code.  After simulation, the waveform window provides a visual representation of signal values over time, enabling identification of discrepancies.  No code snippet is required here, as the process is entirely GUI-based.


**Example 3: SignalTap II Configuration (Conceptual)**

SignalTap II configuration involves using the Quartus Prime GUI.  While no code is written directly, a configuration file implicitly defines which signals to capture and the parameters of the embedded logic analyzer.  This file details the trigger conditions, depth of capture, and the output format of the captured data.  Accessing this configuration is done via the Quartus Prime GUI, not via direct code manipulation.  The conceptual representation below depicts the essence of such a configuration:

```
SignalTap II Configuration:
  Signal Name:           internal_signal
  Trigger Condition:      data_in == 1'b1
  Depth:                 1024
  Output Format:         VCD
```

This configuration would capture values of `internal_signal` when `data_in` is high, storing up to 1024 samples in a Value Change Dump (VCD) file for later analysis.


**3. Resource Recommendations**

* Quartus Prime Handbook
* ModelSim User Manual
* SystemVerilog for Verification (book)
* Advanced Digital Design with Verilog HDL (book)


This multifaceted approach, combining careful SystemVerilog design practices, comprehensive ModelSim simulations, and strategic utilization of Quartus Prime's debugging tools, provides a robust methodology for comprehensively monitoring internal signals throughout the entire design flow. Remember that efficient debugging demands proactive planning at each stage, starting with a well-structured and observable design.
