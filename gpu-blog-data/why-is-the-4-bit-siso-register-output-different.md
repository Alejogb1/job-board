---
title: "Why is the 4-bit SISO register output different from expected in VHDL?"
date: "2025-01-30"
id: "why-is-the-4-bit-siso-register-output-different"
---
The discrepancy between expected and observed outputs in a 4-bit Serial-In, Serial-Out (SISO) register implemented in VHDL often arises from subtle interactions between timing, clocking, and the inherent sequential nature of the register's operation, particularly during simulation or hardware implementation. I have encountered this issue numerous times during my work on digital signal processing hardware, often requiring meticulous analysis of both the VHDL code and the intended operational behaviour. Specifically, the problem tends to manifest when the simulator's clocking model does not perfectly mirror the real-world hardware behaviour, or when the user misinterprets the timing implications of the synchronous register implementation.

A SISO register, fundamentally, is a cascade of D-type flip-flops. Each flip-flop stores one bit of the register's data, and on each rising (or falling, depending on implementation) clock edge, the value at the D input is latched and presented to the Q output. When connecting these flip-flops serially, the output of one becomes the input of the next, forming a shift register. The serial input feeds the first flip-flop, and the final flip-flop’s output becomes the register’s serial output. The expected output, therefore, is a delayed version of the serial input, shifted by as many clock cycles as there are flip-flops in the chain.

The most common deviations from this expectation stem from the initial conditions, the clocking methodology, and the interpretation of simulation results. When not explicitly initialized, flip-flops often start in an 'unknown' state (represented by 'U' in most VHDL simulators). If the serial input begins immediately after the start of simulation, the initial 'unknown' state will propagate through the register, creating confusion about when the actual input data appears at the output. Furthermore, incorrect clocking configurations, especially asynchronous clocking of the register in a system which expects synchronous operation or incorrect handling of the clock enable signal, can result in seemingly random shifts and erroneous data being captured. Furthermore, subtle race conditions in the input data relative to the clock edge can lead to unpredictable data values being captured in the flip-flops, especially if the clock period is shorter than the setup time requirements.

Let's examine three VHDL code snippets, along with commentary, to illustrate potential sources of the problem:

**Example 1: Uninitialized Register and Immediate Input**

```vhdl
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity siso_register is
  port (
    clk : in  std_logic;
    serial_in : in  std_logic;
    serial_out : out std_logic
  );
end entity siso_register;

architecture Behavioral of siso_register is
  signal q0, q1, q2, q3 : std_logic;
begin
  process (clk)
  begin
    if rising_edge(clk) then
      q0 <= serial_in;
      q1 <= q0;
      q2 <= q1;
      q3 <= q2;
    end if;
  end process;
  serial_out <= q3;
end architecture Behavioral;

```

*   **Commentary:** This code is a straightforward implementation of a 4-bit SISO register. Critically, it does not initialize the flip-flop signals q0, q1, q2 and q3. In simulation, these will initially take an unknown state 'U'. Consequently, when serial_in starts toggling, the first few outputs will be unpredictable due to the propagation of the initial 'U' states through the register stages before valid data occupies the entire shift chain. This leads to an incorrect output if the simulator is relied upon for the first four clock cycles. Additionally, this code demonstrates the fundamental structure of the shift register and highlights the need to explicitly initialize registers for predictable simulation behaviour. The real hardware may exhibit similar but undefined behaviour at power up.

**Example 2: Register with Initialization and Delayed Input**

```vhdl
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity siso_register_init is
  port (
    clk : in  std_logic;
    serial_in : in  std_logic;
    serial_out : out std_logic
  );
end entity siso_register_init;

architecture Behavioral of siso_register_init is
  signal q0, q1, q2, q3 : std_logic := '0'; -- Initialized to '0'
begin
  process (clk)
  begin
    if rising_edge(clk) then
      q0 <= serial_in;
      q1 <= q0;
      q2 <= q1;
      q3 <= q2;
    end if;
  end process;
  serial_out <= q3;
end architecture Behavioral;
```

*   **Commentary:** This example is very similar to the first but introduces a key difference: it initializes the internal flip-flop signals `q0`, `q1`, `q2`, and `q3` to '0'. This will resolve the problem with an unknown initial state being propagated through the register, ensuring that the output remains at '0' for the first four clock cycles after which the incoming data appears shifted by four cycles, at the output `serial_out`. This ensures a more predictable and easily verified simulation behaviour. It is crucial in any shift register implementation to have well defined initial states at the beginning of operations and reset sequences can play that role in hardware implementations. The delay between a valid serial input and the valid serial output is clearly visible here.

**Example 3: Register with Clock Enable**

```vhdl
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity siso_register_enable is
  port (
    clk : in  std_logic;
    enable : in std_logic;
    serial_in : in  std_logic;
    serial_out : out std_logic
  );
end entity siso_register_enable;

architecture Behavioral of siso_register_enable is
  signal q0, q1, q2, q3 : std_logic := '0';
begin
  process (clk)
  begin
    if rising_edge(clk) then
      if enable = '1' then
        q0 <= serial_in;
        q1 <= q0;
        q2 <= q1;
        q3 <= q2;
      end if;
    end if;
  end process;
  serial_out <= q3;
end architecture Behavioral;
```

*   **Commentary:** This example demonstrates the addition of a clock enable input. If the `enable` signal is '0' during a rising clock edge, no data shifting occurs, and the current data is maintained. If `enable` is '1', data shifts as in the previous examples. Errors with this implementation can arise if the user expects the register to shift data continuously while the enable line is inactive. Simulation may show that data is captured correctly with a rising edge of enable in sync with clock edges, yet unexpected behaviour may result if the enable and clock edges are not synchronized in real hardware implementations. Moreover, the behaviour during the simulation and real-hardware implementation may vary significantly unless the enable signal is synchronous with the clock, and its setup and hold times with respect to clock transitions have been taken into account. This case shows that control logic of the register, not just data capturing, also needs to be properly analyzed to ensure expected behaviour.

To further understand VHDL simulation nuances, one should explore resources that focus on synchronous design methodology, VHDL best practices, and timing analysis for digital circuits. Several excellent textbooks on digital design principles discuss the theory behind sequential logic, flip-flop characteristics, and clocking techniques. Application notes from various FPGA vendors usually provide specific guidelines for implementing shift registers and related circuitry, including how to avoid pitfalls related to the initial states, enable signals and clocking considerations. Finally, exploring specific documentation of your simulation tools can reveal their particular interpretations of timing behaviours that may need to be taken into account during both simulation and hardware implementations. These resources, if studied well, can significantly improve one’s understanding of and ability to correctly implement the logic of a SISO register.
