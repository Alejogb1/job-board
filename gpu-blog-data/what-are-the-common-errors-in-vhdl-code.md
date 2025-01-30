---
title: "What are the common errors in VHDL code?"
date: "2025-01-30"
id: "what-are-the-common-errors-in-vhdl-code"
---
Over my years working on FPGA projects, I've consistently observed a recurring set of errors in VHDL code, typically stemming from a misunderstanding of the language's concurrent nature or a failure to accurately model hardware behavior. These errors are not unique to beginners; even experienced engineers can fall into these traps under pressure. Understanding these common pitfalls is crucial for producing robust and predictable digital designs.

One frequent error I've encountered involves incomplete sensitivity lists in `process` blocks. In VHDL, a `process` statement executes whenever an event occurs on a signal specified in its sensitivity list. If a signal that affects the process's output is *not* included in the list, the behavior will be incorrect and lead to simulation-synthesis mismatches. This is particularly problematic with combinational logic implemented inside a `process`. The simulator might execute the process correctly based on a full sensitivity, whereas the synthesized hardware will only update its output when one of the specified signals in the sensitivity list changes. This leads to incorrect outputs and a non-functional design.

Consider a simple example of an AND gate:

```vhdl
-- Incorrect VHDL code with incomplete sensitivity list
entity and_gate is
  Port ( a : in  STD_LOGIC;
         b : in  STD_LOGIC;
         c : out STD_LOGIC);
end and_gate;

architecture Behavioral of and_gate is
begin
    process (a)
    begin
        c <= a and b; -- b is missing from the sensitivity list
    end process;
end Behavioral;
```

In this code, the `process` is only sensitive to signal `a`. If `b` changes, the output `c` will *not* be updated, despite the result of `a and b` changing. This creates a critical discrepancy between simulation and synthesis. The simulator, which often monitors all signals, might show the expected behavior, while the synthesized hardware will fail. The correct implementation should include both `a` and `b` in the sensitivity list:

```vhdl
-- Correct VHDL code with complete sensitivity list
entity and_gate is
  Port ( a : in  STD_LOGIC;
         b : in  STD_LOGIC;
         c : out STD_LOGIC);
end and_gate;

architecture Behavioral of and_gate is
begin
    process (a, b)
    begin
        c <= a and b;
    end process;
end Behavioral;
```

Another prevalent issue arises from inadvertently creating latches. Latches, while sometimes useful in specific memory structures, are typically undesirable in most combinational logic designs because they introduce timing hazards and complicate analysis. A latch is inferred when a signal is assigned a value inside a `process` under certain conditions and then isn't assigned a value in *all* other conditions that might occur.  This can arise due to incomplete conditional assignment structures within a `process`.

For instance, consider this implementation of a multiplexer:

```vhdl
-- Incorrect VHDL code creating latches
entity multiplexer is
    Port ( sel : in  STD_LOGIC;
           in1 : in  STD_LOGIC;
           in2 : in  STD_LOGIC;
           out1 : out STD_LOGIC);
end multiplexer;

architecture Behavioral of multiplexer is
begin
    process (sel, in1, in2)
    begin
        if sel = '1' then
            out1 <= in1; -- Only assigned under this condition
        end if;
		-- No else clause for when sel = '0'
    end process;
end Behavioral;
```

In this instance, if `sel` is '0', no assignment is made to `out1`. In order to "hold" a value when `sel` is '0', the synthesizer infers a latch because the logic needs to maintain a past state. The corrected version, to implement a multiplexer without generating a latch, is:

```vhdl
-- Correct VHDL code avoiding latches
entity multiplexer is
    Port ( sel : in  STD_LOGIC;
           in1 : in  STD_LOGIC;
           in2 : in  STD_LOGIC;
           out1 : out STD_LOGIC);
end multiplexer;

architecture Behavioral of multiplexer is
begin
    process (sel, in1, in2)
    begin
        if sel = '1' then
            out1 <= in1;
		else
			out1 <= in2;
        end if;
    end process;
end Behavioral;
```

By including an `else` clause, we ensure that `out1` is assigned a value under all circumstances, preventing latch inference.

A final, frequent error relates to incorrectly utilizing synchronous design principles when implementing sequential logic.  Specifically, not adhering to a single clock edge for state updates is a significant source of issues, often resulting in metastable behavior and unpredictable results. Attempting to update registers on both the rising and falling edges of a clock or using multiple clock domains within a single process will lead to timing violations and non-deterministic operations. A well-structured sequential design should only use a single clock edge and update all registers and flip-flops based on this single timing reference.

Consider the following problematic code example:

```vhdl
-- Incorrect VHDL code using both clock edges
entity counter is
    Port ( clk : in  STD_LOGIC;
           rst : in  STD_LOGIC;
           count : out INTEGER range 0 to 15);
end counter;

architecture Behavioral of counter is
signal count_s : INTEGER range 0 to 15 := 0;
begin
    process (clk, rst)
    begin
        if rst = '1' then
            count_s <= 0;
        elsif rising_edge(clk) then
          count_s <= count_s + 1;
        elsif falling_edge(clk) then
            count_s <= count_s + 1;
        end if;
    end process;
  count <= count_s;
end Behavioral;
```
The process attempts to increment the counter on both edges of the clock signal. This practice creates a problem because many hardware elements can only be triggered by one specific clock edge. This code should be corrected by only updating the counter register on a single, defined clock edge:

```vhdl
-- Correct VHDL code using only one clock edge
entity counter is
    Port ( clk : in  STD_LOGIC;
           rst : in  STD_LOGIC;
           count : out INTEGER range 0 to 15);
end counter;

architecture Behavioral of counter is
signal count_s : INTEGER range 0 to 15 := 0;
begin
    process (clk, rst)
    begin
        if rst = '1' then
            count_s <= 0;
        elsif rising_edge(clk) then
          count_s <= count_s + 1;
        end if;
    end process;
  count <= count_s;
end Behavioral;
```

The corrected version only increments the counter on the rising edge of the clock, which is the standard practice for many synchronous digital circuits.

In summary, VHDL's concurrency and hardware-centric nature demand careful attention to detail to avoid these common errors. Incomplete sensitivity lists, accidental latch creation, and deviations from synchronous clocking principles are all sources of significant problems that I’ve encountered in past designs. To bolster your VHDL skills, I would highly recommend referring to resources like "Digital Design Principles and Practices" by John F. Wakerly. Books focused on synthesis of digital systems provide a deeper understanding of hardware implications behind VHDL coding. Additionally, exploring vendor-specific documentation for your target FPGA or ASIC technology can be immensely helpful in understanding the nuances of how the VHDL code is translated into actual hardware. Thoroughly understanding these common pitfalls and continually reviewing your design methodologies are essential in creating robust, predictable digital systems.
