---
title: "Why does the upper digit of the two-digit BCD counter not increment beyond '0000' in VHDL?"
date: "2025-01-30"
id: "why-does-the-upper-digit-of-the-two-digit"
---
A common pitfall in designing Binary-Coded Decimal (BCD) counters arises from incomplete carry handling, particularly when cascading counters for multi-digit representations. I've seen this numerous times in my work, where novice designers often assume that a simple binary increment will suffice for BCD, leading to the problem you're experiencing: the upper digit stalls at '0000'. The core issue lies in the fact that BCD encoding represents decimal digits using a subset of the available binary values, specifically 0-9 (0000-1001). A standard binary counter increments up to 15 (1111) before wrapping around. This means that a BCD counter must actively detect the transition beyond '9' (1001) and reset to '0' while generating a carry signal for the next digit. If that carry is not correctly processed in the higher digit, it will indeed remain at '0000'.

Let’s examine why this happens. A naive approach, employing a standard binary counter, will increment past 9, say to 10 (1010) and beyond. A two-digit BCD counter needs to increment the lower digit from 0-9, then reset to zero and propagate a carry to the upper digit. The upper digit also needs to correctly handle that carry. If the lower digit performs the reset but the upper digit does not have the logic in place to increment only on receiving the carry, it will remain static. In your specific situation, where the upper digit is not incrementing, the critical logic of the higher digit correctly observing and responding to the lower digit's carry has likely not been implemented, or perhaps is not functioning correctly due to a logical or timing issue within the design.

Let's illustrate with code examples. These are simplified examples; a full design would require considerations like clock enabling, asynchronous reset, etc. I am basing the VHDL used on IEEE standard library versions.

**Example 1: Basic Lower Digit BCD Counter (Incorrect for Multi-Digit)**

This first example shows an incomplete approach which would work for a single digit, but would not function correctly for multiple digits. This does demonstrate how a single digit would increment and reset.

```vhdl
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity bcd_counter_lower is
    Port ( clk    : in  std_logic;
           reset  : in  std_logic;
           count_out : out std_logic_vector (3 downto 0);
           carry_out : out std_logic
            );
end entity bcd_counter_lower;

architecture Behavioral of bcd_counter_lower is
    signal count : unsigned (3 downto 0) := "0000";
begin
    process(clk, reset)
    begin
        if reset = '1' then
            count <= "0000";
            carry_out <= '0';
        elsif rising_edge(clk) then
            if count = 9 then
                count <= "0000";
                carry_out <= '1';
             else
                count <= count + 1;
                carry_out <= '0';
            end if;
        end if;
    end process;
    count_out <= std_logic_vector(count);
end Behavioral;
```

*Commentary:* This code implements a single-digit BCD counter.  When the `count` reaches 9, it resets to 0 and asserts the `carry_out` signal for one clock cycle. However, the carry-out signal is not persistent - it’s only a single pulse. If the upper digit is based on similar logic, it will not increment correctly. In addition, this counter assumes a rising edge on the clock. The reset logic, which sets the count and carry to '0', is simple but important.

**Example 2: Naive Upper Digit (Incorrect)**

This second example is a flawed example of an upper digit using the carry_out from the first example. It only demonstrates the problem.

```vhdl
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity bcd_counter_upper_bad is
    Port ( clk    : in  std_logic;
           reset  : in  std_logic;
           carry_in  : in std_logic;
           count_out : out std_logic_vector (3 downto 0)
            );
end entity bcd_counter_upper_bad;

architecture Behavioral of bcd_counter_upper_bad is
    signal count : unsigned (3 downto 0) := "0000";
begin
    process(clk, reset)
    begin
        if reset = '1' then
            count <= "0000";
        elsif rising_edge(clk) then
            if carry_in = '1' then
                count <= count + 1;
            end if;
        end if;
    end process;
    count_out <= std_logic_vector(count);
end Behavioral;
```

*Commentary:* This is where the issue arises. The upper digit in this example, `bcd_counter_upper_bad`, is only observing the *instantaneous* value of `carry_in`. If it's a short pulse it will likely miss it. Also, the carry signal is only one clock cycle; therefore, this implementation would not be able to increment the upper digit properly if the carry does not align perfectly with its clock edge or is too short. Even if it did catch the carry it would not act correctly. It would increment each time carry in was a 1. This illustrates the issue. Note also the synchronous reset.

**Example 3: Correct Upper Digit with Synchronous Carry**

This final example is a more correct version of the upper digit, which demonstrates how the carry signal should be handled correctly:

```vhdl
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity bcd_counter_upper_correct is
    Port ( clk    : in  std_logic;
           reset  : in  std_logic;
           carry_in  : in std_logic;
           count_out : out std_logic_vector (3 downto 0)
            );
end entity bcd_counter_upper_correct;

architecture Behavioral of bcd_counter_upper_correct is
    signal count : unsigned (3 downto 0) := "0000";
    signal carry_in_d  : std_logic := '0';
begin
    process(clk, reset)
    begin
        if reset = '1' then
            count <= "0000";
            carry_in_d <= '0';
        elsif rising_edge(clk) then
              carry_in_d <= carry_in;
            if carry_in_d = '1' then
               if count = 9 then
                     count <= "0000";
                  else
                     count <= count + 1;
               end if;
            end if;
        end if;
    end process;
    count_out <= std_logic_vector(count);
end Behavioral;
```

*Commentary:* In this improved version, the `carry_in` is registered using `carry_in_d`, a signal which stores the previous carry value. This ensures that a carry occurring even momentarily will correctly result in an increment to the higher digit. Note that the upper digit here also now has the BCD logic implemented, preventing it from going above 9.  The `carry_in` signal from the previous example is used to update the current value stored in `carry_in_d`, and that value is then observed. This allows for reliable capture of the lower digit's carry-out signal. This is crucial. The code also shows the BCD implementation for the upper digit, which now prevents it from going over 9. If this logic is not implemented in the upper digits, an increment above 9 will occur.

To resolve your issue, carefully review your higher digit counter logic. Ensure that: 1) the lower digit correctly generates a carry signal when it rolls over from '9' to '0'. 2) the higher digit *registers* that carry (as shown in example 3) and does not rely solely on the carry signal being high at the exact moment the higher digit increments.  3) the higher digit also correctly implements BCD logic to prevent it from incrementing beyond 9. The examples illustrate this issue with the incorrect and correct implementations. Without careful handling of the carry signal, particularly for multiple digits, your design will exhibit the symptom you have observed: the upper digit remaining at '0000'.

For further study, I would recommend focusing on texts covering digital logic design with VHDL and specifically pay attention to sections dealing with synchronous design, counters, and BCD encoding. Look for examples that illustrate the need for carry registers. Furthermore, reviewing any documentation or application notes from the FPGA/ASIC vendor you are targeting can provide specific examples and guidance related to your platform. Look for descriptions of timing-safe synchronous design practices. Reading through the IEEE VHDL standards document for a deeper understanding of the language is also useful.
