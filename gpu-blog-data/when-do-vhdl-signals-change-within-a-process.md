---
title: "When do VHDL signals change within a process?"
date: "2025-01-30"
id: "when-do-vhdl-signals-change-within-a-process"
---
Signal updates in VHDL processes are governed by a precise set of rules tied to the process's sensitivity list and the execution model of VHDL.  My experience debugging complex FPGA designs, particularly those involving intricate state machines and asynchronous interfaces, has underscored the crucial role of understanding this mechanism.  A failure to fully grasp the timing semantics can lead to subtle, difficult-to-detect race conditions and ultimately, non-functional hardware.  The fundamental insight is this: signal updates within a process do *not* occur instantaneously; instead, they are scheduled for update at the end of the process's execution.

**1.  Explanation of Signal Update in VHDL Processes**

VHDL employs a delta-cycle-based simulation model.  Each simulation cycle is divided into a series of delta cycles.  Within a single simulation cycle, multiple delta cycles might occur.  A delta cycle signifies a moment of signal value propagation.

Consider a process with a sensitivity list.  When a signal in the sensitivity list changes its value, the process is triggered. The process executes sequentially, reading the current values of signals.  However, any assignments made to signals within the process are *not* immediately reflected.  These assignments are instead scheduled for update at the end of the *current* delta cycle, after the process finishes executing.  Critically, these updated values only become visible in the *next* delta cycle. This is why VHDL is often described as having a "zero-delay" model for signal updates within a process.

If multiple processes are triggered within the same delta cycle and update the same signal, the order of execution of those processes determines the final value of the signal at the end of that delta cycle. This is where potential for race conditions exist.  Sequential statements within a process also execute strictly sequentially, preventing parallel updates within a single process.


**2. Code Examples with Commentary**

Let's illustrate this with examples.  I've encountered situations similar to these during my work developing high-speed data acquisition systems.

**Example 1: Simple Signal Assignment**

```vhdl
library ieee;
use ieee.std_logic_1164.all;

entity simple_process is
  port (
    clk : in std_logic;
    reset : in std_logic;
    a : in std_logic;
    b : out std_logic
  );
end entity;

architecture behavioral of simple_process is
  signal internal_b : std_logic;
begin
  process (clk, reset)
  begin
    if reset = '1' then
      internal_b <= '0';
    elsif rising_edge(clk) then
      internal_b <= a;
    end if;
  end process;

  b <= internal_b;
end architecture;
```

In this example, the internal signal `internal_b` is updated at the end of the delta cycle in which the `rising_edge(clk)` condition is met. The output signal `b` is assigned the value of `internal_b` concurrently; therefore, `b` will reflect the updated value of `internal_b` in the next delta cycle. The assignment to `internal_b` within the process does *not* affect the value of `b` during the same delta cycle.

**Example 2: Multiple Processes and Race Condition**

```vhdl
library ieee;
use ieee.std_logic_1164.all;

entity race_condition is
  port (
    clk : in std_logic;
    reset : in std_logic;
    a : in std_logic;
    b : out std_logic
  );
end entity;

architecture behavioral of race_condition is
  signal shared_signal : std_logic;
begin
  process (clk, reset)
  begin
    if reset = '1' then
      shared_signal <= '0';
    elsif rising_edge(clk) then
      shared_signal <= a;
    end if;
  end process;

  process (clk, reset, a)
  begin
    if reset = '1' then
      shared_signal <= '1';  --Potential Race condition
    elsif rising_edge(clk) then
      if a = '1' then
        shared_signal <= '0';
      end if;
    end if;
  end process;

  b <= shared_signal;
end architecture;
```

This example demonstrates a potential race condition.  Both processes update `shared_signal`. Depending on the simulator's process scheduling, either '0' or '1' might be assigned to `shared_signal`. The outcome is non-deterministic, making this design unreliable. Proper synchronization mechanisms are necessary to resolve such issues.

**Example 3:  Illustrating Delta Cycles with WAIT**

```vhdl
library ieee;
use ieee.std_logic_1164.all;

entity delta_cycles is
  port (
    start : in std_logic;
    result : out std_logic_vector(1 downto 0)
  );
end entity;

architecture behavioral of delta_cycles is
  signal a : std_logic;
  signal b : std_logic;
begin
  process (start)
  begin
    if start = '1' then
      a <= '1';
      b <= a; --Assignment within same delta cycle
      wait until false; --Process suspends until next delta cycle
      result <= a & b; --Reflects update
    end if;
  end process;
end architecture;
```

This example employs `wait until false` to explicitly pause process execution.  The assignment to `b` is within the same delta cycle as the assignment to `a`.  However, the `result` signal will only reflect the updated values after this delta cycle completes. The `wait` statement ensures that the update of `a` and `b` are observed in the following delta cycle. This highlights the fact that updates within a process are not immediate.


**3. Resource Recommendations**

For a deeper understanding, I recommend consulting the VHDL Language Reference Manual, focusing particularly on the sections detailing process execution, signal assignments, and the delta cycle model.  A thorough study of simulation and timing diagrams generated from VHDL simulations is indispensable for grasping the subtle timing intricacies.  Further, practicing with various scenarios, including those with multiple processes and asynchronous events, will solidify your understanding.  Finally, understanding the specific capabilities and constraints of your target FPGA architecture is crucial in avoiding potential problems arising from the interplay between VHDL code and hardware implementation.  This includes understanding clock speeds and the impact of signal propagation delays.
