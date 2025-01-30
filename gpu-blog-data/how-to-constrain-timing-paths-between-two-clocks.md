---
title: "How to constrain timing paths between two clocks or force hold conditions?"
date: "2025-01-30"
id: "how-to-constrain-timing-paths-between-two-clocks"
---
In high-speed digital design, inadequate control over timing path relationships between asynchronous or closely related clock domains frequently leads to metastability, functional errors, or increased power consumption. Consequently, explicitly constraining these paths and, in some cases, intentionally forcing hold violations, becomes critical for reliable system operation. This detailed response explores various techniques for achieving this control, drawing from experience spanning several ASIC and FPGA projects.

Fundamentally, timing constraints inform the synthesis and place-and-route tools about the temporal relationships required for the design to function correctly. Without constraints, the tools attempt to optimize for overall performance without regard for specific inter-clock domain requirements, which can result in race conditions or hold time violations. There are two primary scenarios where controlling these paths is essential: handling asynchronous crossings and addressing closely related clock domains. For asynchronous crossings, the goal is often to slow down data transfer or introduce metastability mitigation techniques. For closely related clocks (e.g., generated clock domains, phase-shifted clocks), the goal might be to allow adequate time for signals to propagate or to intentionally violate hold times for specific purposes, like high-speed data capture.

The most common and robust method to constrain timing paths involves using *set_max_delay* and *set_min_delay* constraints within the synthesis tool’s constraint file. These constraints define the upper and lower bounds, respectively, on the allowed delay between two points in the design, which are usually launch and capture registers. Specifically, to constrain paths crossing between clock domains, a common approach is to use *set_max_delay* to impose a reasonable latency for the data to propagate, which will increase the chance of resolving metastability, and to prevent an excessively fast data transfer that might violate setup times. Conversely, *set_min_delay* is often employed to handle paths where we might want to intentionally violate hold constraints to create a single cycle path. For paths within a single clock domain, these constraints can be used to ensure that clock skew does not result in setup or hold violations on very fast logic paths.

Let's examine several practical examples using standard constraint syntax, applicable across many synthesis and place-and-route tools. Assume we are working with a design containing two clock domains, clk_a and clk_b, where data transfers from clk_a to clk_b through a synchronizer, as well as internal paths within clk_a that are exhibiting very fast logic paths.

**Example 1: Constraining an Asynchronous Crossing Path**

This example demonstrates how to constrain the path transferring data from clock domain clk_a to clock domain clk_b, assuming a simple two-stage synchronizer is implemented. This forces the tool to account for potential metastability by not optimizing the path too aggressively.

```
# Define clock periods (assuming 10ns and 8ns period respectively)
create_clock -period 10.0 -name clk_a [get_ports clk_a]
create_clock -period 8.0 -name clk_b [get_ports clk_b]

# Identify the start and end points of the data path.
set start_point [get_pins data_register_a/Q]
set end_point [get_pins synchronizer_stage2/D]

# Set the maximum delay for the path. A delay of 4ns gives some margin
# This should be lower than the clock cycle of the destination (8ns).
set_max_delay 4.0 -from $start_point -to $end_point

# Explicitly ignore the hold time requirements since its a synchroniser
set_false_path -from $start_point -to $end_point -hold
```

In this example, *create_clock* statements define the clock frequencies for clk_a and clk_b. The *set start\_point* and *set end\_point* commands identify the critical endpoints for the constrained path, in this case the output of the data register and the input of the second synchronizer stage respectively. Critically, *set_max_delay* defines a maximum allowable delay of 4ns for this crossing. Finally *set_false_path* forces the tool to ignore any hold requirements, acknowledging that this path is not a critical one in terms of hold timing constraints, as the metastability resolution happens over multiple clock cycles. This technique prevents the tool from attempting to optimize this path to an unrealistic delay, thereby indirectly encouraging the creation of a more robust design. I have used this technique on countless high speed asynchronous data transfer links, achieving great results.

**Example 2: Forcing a Hold Violation**

There are instances where a slight hold violation is intentional, especially when very high speed capture is involved. The following example deliberately uses *set_min_delay* to create a single clock cycle path. This specific use case is uncommon, so use it carefully.

```
# Assuming a single clock named clk_fast which operates at a period of 3ns
create_clock -period 3.0 -name clk_fast [get_ports clk_fast]

# Identify the start and end points. Data moves from source_register to capture_register
set start_point [get_pins source_register/Q]
set end_point [get_pins capture_register/D]

# Force a small negative delay between source_register and capture_register, creating a hold violation
set_min_delay -0.5 -from $start_point -to $end_point
```

Here, the constraint *set_min_delay -0.5* creates a requirement where the data arrives at the capture register earlier than the clock edge by 0.5ns, forcing a hold violation. Although seemingly counter intuitive, this has proven useful in custom high speed capture circuits that require this condition for correct data sampling at high speeds. It is extremely important to meticulously analyze the specific situation that requires this sort of timing and verify it experimentally on an evaluation board before deploying to production silicon.

**Example 3: Constraining Fast Internal Logic Paths**

This example demonstrates constraining internal paths within a clock domain that have extremely fast logic and could lead to potential hold violations.

```
# Assuming a single clock named clk_core which operates at a period of 5ns
create_clock -period 5.0 -name clk_core [get_ports clk_core]

# Identify the start and end points within the clk_core domain.
set start_point [get_pins fast_register_a/Q]
set end_point [get_pins fast_register_b/D]

# Set the min delay, which means the path will have a minimum logic and routing delay
# This also means if the logic path is too fast, the place and route tool must
# introduce extra delay
set_min_delay 0.8 -from $start_point -to $end_point
```

Here, the *set_min_delay* forces a minimum delay of 0.8ns between two registers within the same clock domain. This can be necessary when synthesis optimizations result in extremely short logic paths, increasing the likelihood of hold violations due to variations in gate delays or clock skew. It prevents the synthesis tool from making the path too short and thereby forces the tool to insert buffer logic on the path and avoid hold violations, which would not be otherwise captured with setup time constraints.

In addition to explicit *set_max_delay* and *set_min_delay* constraints, using *set_clock_groups* to define logical clock groups is important. This declaration tells the tool which clocks are logically independent from one another and prevents paths across clock domain being considered for the normal timing optimization. Without defining clock groups, the tool might attempt to over-constrain crossing paths by trying to meet timing requirements as if they are paths within the same clock domain. In some cases, clock groups can be configured to be asynchronous with each other, indicating paths from a clock group to another do not require special optimization and can simply be ignored.

Finally, for practical implementation, designers should consult their synthesis and place-and-route tool documentation for the specific constraint syntax. Timing reports, specifically *hold* and *setup* reports should be carefully inspected to verify the tool has adhered to the timing specifications.

For further information and examples, I recommend reviewing textbooks on ASIC and FPGA design methodologies, as well as online resource websites by EDA vendors. Books focusing on timing analysis and clock domain crossing are especially useful for a comprehensive understanding of this topic. Additionally, consulting with senior colleagues or participating in design forums can provide invaluable insights and tips for practical applications of timing constraints.
