---
title: "How to create a delay after a signal's falling edge in VHDL?"
date: "2025-01-30"
id: "how-to-create-a-delay-after-a-signals"
---
Within digital logic design, precise timing control is often paramount. Implementing a delay following a signal’s negative transition (falling edge) requires a careful consideration of VHDL constructs and their implications in hardware synthesis. I've encountered this scenario numerous times, specifically during asynchronous interface design and debouncing circuits, where precise pulse widths are critical. My experiences have shown that while seemingly straightforward, naive implementations can introduce unintended timing hazards.

The core challenge lies in the fact that VHDL, particularly when targeting FPGA or ASIC synthesis, doesn’t directly translate to sequential “wait” statements in the same way a software programming language might. Instead, we leverage the inherent nature of synchronous digital circuits, using registers to create a time-delayed version of the input signal. The key is to understand that in hardware description, we’re not ‘waiting’ in time; rather, we’re clocking the signal through a series of registers, each stage adding a delay of one clock cycle. This clocked approach ensures robust, predictable behavior in a real-world implementation.

A fundamental method to generate a post-falling edge delay involves capturing the edge and then propagating that captured state through a chain of D flip-flops. Each flip-flop stage, driven by the system clock, contributes a delay equivalent to one clock period. To detect the falling edge, I’ll use an edge detector which, when it registers a high to low transition, latches a pulse. The number of subsequent stages in our delay chain determines the total delay period.

**Code Example 1: Basic Falling Edge Delay**

```vhdl
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity falling_edge_delay is
    Port ( clk : in STD_LOGIC;
           input_sig : in STD_LOGIC;
           delayed_sig : out STD_LOGIC);
end entity falling_edge_delay;

architecture behavioral of falling_edge_delay is
    signal input_sig_reg : STD_LOGIC := '0';
    signal falling_edge : STD_LOGIC := '0';
	 signal delay_1 : STD_LOGIC := '0';
	 signal delay_2 : STD_LOGIC := '0';

begin
    process(clk)
    begin
        if rising_edge(clk) then
            input_sig_reg <= input_sig; -- Capture the current value of input signal
				if input_sig_reg = '1' and input_sig = '0' then -- Detect falling edge 
					falling_edge <= '1'; -- Pulse is set high on detection
				else
					falling_edge <= '0'; -- Pulse returns to low after edge detected
				end if;
				delay_1 <= falling_edge;
				delay_2 <= delay_1;
				
			end if;
    end process;
	
	delayed_sig <= delay_2;

end architecture behavioral;
```

In this first example, `input_sig_reg` retains the previous state of `input_sig` allowing a comparison to detect falling edges. The `falling_edge` signal becomes high on the falling edge, and stays high for one clock cycle. The signal `delay_1` captures this pulse and transfers it to `delay_2` on the next clock cycle. Therefore, the signal `delay_2` will output a pulse delayed by two clock cycles from the initial falling edge. This represents a straightforward two clock cycle delay but can be extended by chaining more registers. The crucial aspect is that the process is clocked, making it synchronous and predictable.

**Code Example 2: Parameterized Delay**

```vhdl
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity parameterized_delay is
    Generic ( DELAY_CYCLES : integer := 3 );
    Port ( clk : in STD_LOGIC;
           input_sig : in STD_LOGIC;
           delayed_sig : out STD_LOGIC);
end entity parameterized_delay;

architecture behavioral of parameterized_delay is
    signal input_sig_reg : STD_LOGIC := '0';
    signal falling_edge : STD_LOGIC := '0';
    type delay_array is array (0 to DELAY_CYCLES -1) of STD_LOGIC;
    signal delay_line : delay_array := (others => '0');

begin
	
    process(clk)
    begin
        if rising_edge(clk) then
            input_sig_reg <= input_sig;
				if input_sig_reg = '1' and input_sig = '0' then
					falling_edge <= '1';
				else
					falling_edge <= '0';
				end if;

				delay_line(0) <= falling_edge;
				for i in 1 to DELAY_CYCLES-1 loop
					delay_line(i) <= delay_line(i-1);
				end loop;
				
        end if;
    end process;
    
	delayed_sig <= delay_line(DELAY_CYCLES-1);

end architecture behavioral;
```

This version introduces a generic parameter, `DELAY_CYCLES`, which controls the length of the delay. A signal array, `delay_line`, is used to store the delayed states. The for loop dynamically creates the series of flip-flops. By changing the `DELAY_CYCLES` parameter during instantiation, the desired post falling edge delay is achieved. This parameterized approach offers greater flexibility and reusability. Note that the first element of `delay_line` is assigned to the falling edge and then the signal is propagated through the rest of the array, each stage delaying the signal by one clock cycle.

**Code Example 3: Delay with Asynchronous Reset**

```vhdl
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity reset_delay is
    Generic ( DELAY_CYCLES : integer := 4 );
    Port ( clk : in STD_LOGIC;
           reset : in STD_LOGIC;
           input_sig : in STD_LOGIC;
           delayed_sig : out STD_LOGIC);
end entity reset_delay;

architecture behavioral of reset_delay is
    signal input_sig_reg : STD_LOGIC := '0';
    signal falling_edge : STD_LOGIC := '0';
    type delay_array is array (0 to DELAY_CYCLES -1) of STD_LOGIC;
    signal delay_line : delay_array := (others => '0');


begin
    process(clk, reset)
    begin
        if reset = '1' then
			delay_line <= (others => '0');
			input_sig_reg <= '0';
			falling_edge <= '0';
        elsif rising_edge(clk) then
				input_sig_reg <= input_sig;
				if input_sig_reg = '1' and input_sig = '0' then
					falling_edge <= '1';
				else
					falling_edge <= '0';
				end if;
				delay_line(0) <= falling_edge;
				for i in 1 to DELAY_CYCLES-1 loop
					delay_line(i) <= delay_line(i-1);
				end loop;
        end if;
    end process;

    delayed_sig <= delay_line(DELAY_CYCLES-1);

end architecture behavioral;
```

The inclusion of an asynchronous reset ( `reset` ) in this design is crucial for handling power-up states or unexpected circuit behavior. The reset overrides the clocked process, forcing all registers in the delay line to zero. This ensures a known initial state, preventing potential glitches or unpredictable behavior in the downstream logic. The `reset` signal, when asserted high, immediately sets all signals to their initial values. This addition improves the robustness of the design in a real-world context.

When designing such circuits, I always consider the following: The frequency of the clock `clk`, the required delay duration, and the target technology. If a shorter delay than one clock cycle is needed, one must consider techniques such as clock manipulation, specific delay elements within the target FPGA architecture or specific ASIC library cells. These methods are technology dependent. The number of stages added impacts resource utilization and should be balanced against performance requirements.

For further learning and practical application, I suggest studying: Advanced Digital Design with the VHDL and Digital Design Principles and Practices, with a focus on sequential circuit design and timing analysis. These resources offer thorough coverage on digital circuit design principles, VHDL programming for synthesis, and timing analysis, all crucial for creating reliable and predictable hardware designs. Additionally, I recommend consulting vendor documentation for specific FPGA or ASIC families to understand the capabilities of their synthesis tools. This deep dive allows for an optimized implementation of the described delay mechanisms.
