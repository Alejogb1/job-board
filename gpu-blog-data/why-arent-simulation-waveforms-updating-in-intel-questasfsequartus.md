---
title: "Why aren't simulation waveforms updating in Intel Questas_fse/Quartus II?"
date: "2025-01-30"
id: "why-arent-simulation-waveforms-updating-in-intel-questasfsequartus"
---
Simulation waveforms failing to update in Intel's Questa*-fse*/Quartus II environments are frequently a consequence of mismatches between the simulation setup and the underlying hardware description language (HDL) code or the testbench design. My experience across multiple complex ASIC and FPGA verification projects has repeatedly highlighted this issue. This commonly observed behavior isn't usually a bug in the tools themselves, but rather a symptom of inadequately configured simulation parameters or subtle errors in the design and testing methodologies.

The primary cause stems from an incomplete understanding of the compilation, elaboration, and simulation processes within these tools. When a design is compiled, the HDL code is translated into an intermediate representation. Elaboration builds upon this by linking design components and resolving interconnections. Finally, simulation executes the design using stimulus provided by the testbench. Any discrepancy, however slight, between how the code is conceived and how it is being simulated will manifest as a seemingly unresponsive waveform viewer.

Specifically, simulation visibility relies on a meticulous specification of what internal signals and variables are intended for observation. If the testbench does not explicitly instruct the simulator to monitor particular nodes or if it does so improperly, the waveform window will indeed fail to show the expected changes. Similarly, timing inaccuracies in the testbench may preclude signals from settling before the waveform window's sampling rate occurs; the perceived lack of updates might be due to an incorrect timescale or sampling period, rather than the simulation not working as intended. Often, the simulator is correctly executing and propagating signals, but those signals aren't being captured at the proper time or granularity.

Let's examine some concrete scenarios that underscore these challenges. Consider a simple register-based design with a clock signal.

**Example 1: Inadequate Testbench Signal Declaration**

Assume we have the following Verilog module, `simple_register.v`:

```verilog
module simple_register (
    input  wire clk,
    input  wire data_in,
    output reg  data_out
);

always @(posedge clk)
    data_out <= data_in;

endmodule
```

The accompanying testbench, `simple_register_tb.v`, might be erroneously written as follows:

```verilog
module simple_register_tb;

reg clk;
reg data_in;
wire data_out;

simple_register uut (
    .clk(clk),
    .data_in(data_in),
    .data_out(data_out)
);

initial begin
    clk = 0;
    data_in = 0;
    #10;
    data_in = 1;
    #10;
    data_in = 0;
    forever #5 clk = ~clk;
end

endmodule
```

Here, while `data_out` is connected to the instance of `simple_register`, it is not explicitly being monitored by the testbench or simulator. Thus, if we open a waveform viewer, we would see `clk` and `data_in` changing but `data_out` will likely be 'X' or indeterminate. This is because we haven't instructed the simulator to expose it to the viewing tool. To correct this, we must modify the testbench, for instance by adding a `dumpvars` statement to enable waveform capture for `data_out`:

```verilog
module simple_register_tb;

reg clk;
reg data_in;
wire data_out;

simple_register uut (
    .clk(clk),
    .data_in(data_in),
    .data_out(data_out)
);

initial begin
    $dumpfile("wave.vcd");
    $dumpvars(0, simple_register_tb); //Include all signals in the module.
    clk = 0;
    data_in = 0;
    #10;
    data_in = 1;
    #10;
    data_in = 0;
    forever #5 clk = ~clk;
end

endmodule
```
The `$dumpvars` function, when used in the `initial` block, directs the simulator to record all changes within the specified scope into a Value Change Dump (VCD) file. The waveform viewer subsequently uses this file to display simulation results.

**Example 2: Incorrect Simulation Time Scale**

Suppose we have a module that uses a small delay:

```verilog
module delay_module (
    input  wire in,
    output reg out
);

always @*
  #1 out = in;

endmodule
```
And a testbench:

```verilog
module delay_module_tb;

reg in;
wire out;

delay_module dut (
  .in(in),
  .out(out)
);

initial begin
  $dumpfile("wave.vcd");
  $dumpvars(0, delay_module_tb);

  in = 0;
  #5 in = 1;
  #5 in = 0;
  #100 $finish;
end

endmodule
```
The problem might not be immediately visible in the waveform unless a critical detail is considered. The default timescale of 1ns is assumed by most simulators unless told otherwise. If we have a simulation with this time scale, a 1-time unit delay in the hardware corresponds to 1 ns. However, if we accidentally have our viewer zoomed out, a 1 ns delay may not be visible. Thus, the waveform appears not to update when it truly is, simply at a scale where the change is imperceptible. The resolution of the viewer may need to be adjusted. Alternatively, a timescale directive like ``` `timescale 1ps/1ps``` can be added in the module to specify a finer time scale. This would have the effect of displaying a 1 unit time delay as 1 ps, which may be easier to view at the default zoom level. This is often overlooked but can be the cause of a perceived unresponsive simulation environment.

**Example 3: Overly Complex Signal Naming in an Elaborated Design**
In more complex designs which instantiate other modules, a problem can emerge. Let's consider the following, slightly more advanced scenario:

```verilog
module sub_module (input wire a, output reg b);
    always @* b = ~a;
endmodule

module top_module (input wire c, output wire d);
  sub_module sub(.a(c), .b(d));
endmodule

module top_module_tb;
  reg c;
  wire d;
  top_module top(.c(c), .d(d));
    initial begin
      $dumpfile("wave.vcd");
      $dumpvars(0, top_module_tb);
      c = 0;
      #10 c = 1;
      #10 c = 0;
      #100 $finish;
    end
endmodule
```

If you simply declare a module to `dumpvars` as is, it can be difficult to pinpoint exactly what signals are where in the elaborated design. For instance, the internal signal 'b' within `sub_module` will not be visible on the waveforms. Furthermore, `d`, an output of `top_module` is actually the internal signal 'b' within the instance of `sub_module`. We need to specifically trace to that module's signal in order to see those changes. To observe `sub_module.b`, we would have to tell the dumpvars function specifically where that signal lives using a hierarchical path: `$dumpvars(0, top_module_tb.top.sub.b);`. Furthermore, adding multiple nested modules only exacerbates this problem. Thus, correctly using path-based naming during debugging is essential. This might not manifest as 'no changes' on a waveform, but 'unexpected changes' can be just as confusing. Understanding hierarchy becomes necessary to debug these cases.

To effectively troubleshoot waveform update issues, several resources are instrumental. First, the Questa-fse user manual thoroughly details the simulation process, including waveform viewing, timescale specification, and testbench integration; its section on VCD file creation is particularly relevant. The Intel Quartus II handbook contains valuable guidance on setting up simulation projects and troubleshooting common issues, particularly in the context of FPGA-based designs. Many advanced simulation debugging resources are available through general academic textbooks as well. Often the problem isn't that the simulation *isn't working* but that we aren't viewing its results *correctly*.

In my experience, meticulous attention to detail in testbench design, deliberate signal visibility configuration, and a firm grasp of hierarchical design in advanced HDL design are the foundational prerequisites for successful waveform analysis in environments such as Questa-fse and Quartus II. These are not mere "nice to haves," but essential practices for any verification effort.
