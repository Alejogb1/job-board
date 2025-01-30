---
title: "How to exclude specific paths from set_false_path?"
date: "2025-01-30"
id: "how-to-exclude-specific-paths-from-setfalsepath"
---
The `set_false_path` constraint in digital circuit synthesis tools, such as those used in FPGA and ASIC design, dictates that a specified timing path should not be analyzed for timing violations. This is a critical tool when certain paths are inherently not relevant for timing closure due to design-specific logic or architectural constraints. However, the need often arises to exclude specific sub-paths within a broad `set_false_path` declaration, refining the constraint application.

I have personally encountered scenarios, particularly in complex System-on-Chip (SoC) designs involving asynchronous clock domain crossings (CDCs), where applying a blanket `set_false_path` between two clock domains was insufficient. The issue emerged when certain paths, although logically spanning the domains, required timing analysis due to specific signal usage within a common module. Ignoring these specific paths could lead to unexpected race conditions and functional errors. Therefore, a mechanism to selectively exempt paths from an existing `set_false_path` constraint became vital.

To understand this need more deeply, consider a scenario involving a bus interface between two clock domains: `clk_A` and `clk_B`.  We initially define `set_false_path` from all paths in the `clk_A` domain to all paths in the `clk_B` domain using their clock names. However, imagine that a control signal, derived from `clk_A`, goes through a synchronization stage and then is used in `clk_B` to control a critical operation on data received from `clk_B`. The path involving this control signal needs timing analysis to ensure proper operation. The broad `set_false_path` now incorrectly includes this specific, critical path.

The exclusion of specific paths within a `set_false_path` constraint cannot be directly achieved using a subtractive approach in the common synthesis languages like Synopsys Design Constraints (SDC). Instead, one must utilize the inverse – setting explicit *timing requirements* on the paths that *should* be analyzed. The `set_max_delay` or `set_min_delay` constraints, when applied to specific path segments, effectively override any prior `set_false_path` that may overlap those paths, forcing the timing analyzer to consider them. The synthesis tool prioritizes these specific timing requirements over the general false path specification.

The principle is to define the broadest false path initially and then use `set_max_delay` (or `set_min_delay`) to carve out exceptions for paths that do require timing analysis. This hierarchical approach is fundamental for achieving accurate timing analysis while also correctly ignoring intentionally asynchronous path segments.

Here are three code examples illustrating this approach:

**Example 1: Basic False Path Exclusion**

```sdc
# Initially set all paths between clk_A and clk_B as false
set_false_path -from [get_clocks clk_A] -to [get_clocks clk_B]

# Define the critical path that requires timing analysis using set_max_delay.
# Assuming the signal named 'control_sync_out' has the timing path that matters.
set_max_delay -from [get_pins sync_stage/control_sync_out] -to [get_pins critical_logic/control_in] 1.5
```

**Commentary:**

In this example, the first line sets a false path between all endpoints clocked by `clk_A` and all endpoints clocked by `clk_B`. The second line then sets a maximum delay of 1.5 time units (the unit is determined by the time scale) between the output of a synchronization stage, which outputs `control_sync_out`, and the input pin, `control_in`, of the critical logic block where this signal is received within the `clk_B` domain. Note that this implicitly tells the tool to analyze the timing of that specific path, effectively ignoring the earlier `set_false_path` constraint in this instance. The signal `control_sync_out` is assumed to be an output of a register, not a pure combinational logic output, to avoid creating additional timing path definitions.

**Example 2: Specifying Path Through Hierarchical Elements**

```sdc
# Broad false path definition remains the same
set_false_path -from [get_clocks clk_A] -to [get_clocks clk_B]

# Exclude specific path through an instance called 'bus_arbiter'
set_max_delay -from [get_pins bus_arbiter/arb_req_out] -to [get_pins sync_logic/arb_req_in] 2.0
```

**Commentary:**

This example is very similar to the previous one but specifies the path exclusion at a higher level within the design hierarchy. It sets a `set_max_delay` constraint between a specific output pin, `arb_req_out` of a module `bus_arbiter` and the input pin `arb_req_in` of another module, named `sync_logic`.  The `get_pins` command is essential here for specifying the physical path of signals that traverse through hierarchy. Similar to before, `arb_req_out` is assumed to be a register output pin for proper timing analysis.

**Example 3: Exclusion involving multiple sequential elements within the critical path**

```sdc
# Broad false path definition remains
set_false_path -from [get_clocks clk_A] -to [get_clocks clk_B]

# Exclude a multi-stage pipeline by using multiple set_max_delay commands
set_max_delay -from [get_pins stage1_reg/data_out] -to [get_pins stage2_reg/data_in] 1.0
set_max_delay -from [get_pins stage2_reg/data_out] -to [get_pins stage3_reg/data_in] 1.2
set_max_delay -from [get_pins stage3_reg/data_out] -to [get_pins final_logic/data_in] 0.8
```

**Commentary:**

In this example, a timing path that is part of a multi-stage pipeline is excluded from the false path. Instead of only specifying the first and last stages as is often seen, this example breaks the critical path between clock domains into several timing segments. This granularity of control is useful, if not necessary, to accurately reflect the functional aspects of a data flow spanning clock domains. Note again, that the outputs of the registers, `stage1_reg/data_out`, `stage2_reg/data_out`, and `stage3_reg/data_out` are used as starting points for the `set_max_delay` constraints.

When using `set_max_delay` to override a `set_false_path`, ensure that the time values specified are reasonable, based on the physical characteristics and expected operation of the targeted hardware. Incorrectly specified `set_max_delay` values can create timing violations if the synthesis tool attempts to meet those requirements without appropriate logic optimization. Further,  it is critical to verify these timing specifications with formal static timing analysis tools following synthesis to ensure that all timing constraints have been interpreted correctly by the synthesis tool.

For further understanding of the concepts in depth, I recommend exploring the documentation specific to your chosen synthesis tool; usually vendor-specific.  Consult resources which have in-depth explanations of Static Timing Analysis and Timing Constraints, which are typically published by the leading EDA (Electronic Design Automation) tool providers. Furthermore, consider texts or publications that discuss advanced digital logic design or FPGA design that commonly touch on clock domain crossing challenges and how timing constraints are utilized to address them. These provide the theoretical and practical bases for the strategies explained here.
