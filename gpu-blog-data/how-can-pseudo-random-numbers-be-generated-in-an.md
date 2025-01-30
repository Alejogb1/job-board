---
title: "How can pseudo-random numbers be generated in an FPGA?"
date: "2025-01-30"
id: "how-can-pseudo-random-numbers-be-generated-in-an"
---
The core challenge in generating pseudo-random numbers (PRNs) within an FPGA stems from the inherently deterministic nature of digital hardware. Unlike general-purpose CPUs, where operating systems and runtime environments often provide library functions for randomness, FPGAs demand a more fundamental approach. The foundation for PRN generation on an FPGA lies in leveraging linear feedback shift registers (LFSRs), a practical method for creating sequences that exhibit properties resembling randomness.

An LFSR is essentially a shift register where the input to the leftmost bit is derived from a linear function of the existing register state, typically involving XOR operations. The selection of the 'tap' positions – the bits that are XORed – dictates the length and randomness characteristics of the generated sequence. A crucial element in the design is a 'primitive polynomial', a mathematical construct that guarantees the generated sequence will cycle through all possible states (excluding the zero state) before repeating itself. The length of the sequence is then given by 2^n - 1, where 'n' is the register's width.

Implementing an LFSR in an FPGA requires understanding the basic hardware primitives and coding with hardware description languages (HDLs) such as Verilog or VHDL. In terms of the hardware itself, registers are constructed using flip-flops and the XOR operation is realized using an XOR gate. The tap selection corresponds to the wiring configuration that connects the output of certain flip-flops to the inputs of the XOR gates feeding the feedback.

**Example 1: A 4-bit LFSR in Verilog**

```verilog
module lfsr_4bit (
  input clk,
  input reset,
  output reg [3:0] prn
);

  always @(posedge clk or posedge reset) begin
    if (reset) begin
      prn <= 4'b0001; // Initialize to a non-zero value
    end else begin
      prn <= {prn[2:0], prn[3] ^ prn[0]};
    end
  end
endmodule
```

This Verilog code defines a module `lfsr_4bit` implementing a 4-bit LFSR. The `prn` register stores the current state, representing the generated pseudo-random number. The initialization to `4'b0001` avoids the zero state, which would cause the sequence to stall. In the non-reset condition, the shift operation occurs, moving the current register state to the left, with the new LSB calculated by XORing the output of the third and first bits. The polynomial implemented here is x^4 + x + 1.

This example generates a sequence with a period of 15 (2^4 - 1). Each clock cycle shifts and generates a new value. The `reset` input allows for setting an initial value to the register, an important consideration for deterministic testing of the design.

**Example 2: An 8-bit LFSR with Parameterizable Tap Positions in VHDL**

```vhdl
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity lfsr_8bit is
    Port ( clk : in  std_logic;
           reset : in  std_logic;
           prn : out  std_logic_vector(7 downto 0));
end entity lfsr_8bit;

architecture behavioral of lfsr_8bit is
  signal shift_reg : std_logic_vector(7 downto 0);
  constant TAP_POSITIONS : std_logic_vector(7 downto 0) := "01010001"; -- x^8 + x^6 + x^4 + x + 1
begin
  process (clk, reset)
  begin
      if reset = '1' then
          shift_reg <= "00000001";
      elsif rising_edge(clk) then
          shift_reg <= shift_reg(6 downto 0) & (shift_reg(7) xor shift_reg(6) xor shift_reg(4) xor shift_reg(0));
      end if;
  end process;
    prn <= shift_reg;
end architecture behavioral;
```

This VHDL example showcases an 8-bit LFSR with a parameterized tap position via the `TAP_POSITIONS` constant. This allows for easy modification of the polynomial used to generate the PRN sequence. The implementation still uses a shift register and a combination of XOR operations. The tap selection `TAP_POSITIONS` effectively specifies x^8 + x^6 + x^4 + x + 1. The result is an 8-bit pseudo-random sequence, cycling through 255 (2^8 - 1) states. The shift and XOR logic remains similar to the Verilog case, but the syntax differs. The parameterization makes it easier to experiment with different polynomials.

**Example 3: An LFSR with an additional XOR stage to enhance randomness in Verilog**

```verilog
module lfsr_advanced (
  input clk,
  input reset,
  output reg [7:0] prn
);

  reg [7:0] shift_reg;

  always @(posedge clk or posedge reset) begin
    if (reset) begin
      shift_reg <= 8'hAA;
    end else begin
      shift_reg <= {shift_reg[6:0], shift_reg[7] ^ shift_reg[4] ^ shift_reg[3] ^ shift_reg[1]};
    end
  end

  always @(posedge clk) begin
      prn <= shift_reg ^ (shift_reg << 1);
  end
endmodule
```

In this example, we maintain an internal 8-bit `shift_reg` driven by an LFSR with the polynomial x^8 + x^4 + x^3 + x + 1. However, the output `prn` is derived by XORing the internal register with its own left-shifted version. This additional XOR stage aims to improve the statistical properties of the pseudo-random sequence and reduce potential short-term correlations within the output, which is a common concern in simpler LFSR implementations. The choice of the shift amount `<< 1` can also be considered as a parameter for additional experimentation. While not drastically altering the sequence's period, it enhances distribution properties.

Key considerations when generating PRNs on FPGAs include:

*   **Initialization:** Avoiding the zero state is critical for LFSRs, so an initialization mechanism must be included.
*   **Period Length:** Choosing an appropriate primitive polynomial ensures a maximal-length sequence. Lookup tables or precomputed polynomials for commonly used lengths are available.
*   **Seed Values:** While the initialization value can affect the starting point of the PRN sequence, it doesn’t impact its randomness, assuming the generator is maximal-length. Different seed values merely shift where within the long cycle you start from.
*   **Output Bias:** In some cases, the least significant bits of an LFSR output can exhibit some bias. It is advisable to use the high-order bits or additional logic (like the XORing in Example 3) to minimize such biases.
*   **Testing:** Validate PRN outputs using statistical tests of randomness, ensuring they meet the requirements of the application.
*   **Resource Utilization:** Simple LFSRs are highly efficient on FPGAs in terms of logic usage and clock cycles. However, more complex PRNGs might be required for specific applications at the cost of resource utilization.

For further understanding and design implementation, the following resources are valuable:

*   **Textbooks on Digital Design:** These cover the fundamentals of flip-flops, registers, and combinational logic, which are the building blocks of LFSRs.
*   **FPGA vendor documentation:** Guides from Xilinx, Intel, Lattice, and other FPGA manufacturers often provide examples and best practices for implementing various logic functions, including PRN generation.
*   **Published scientific papers on random number generation:** Deeper dives into the mathematics of pseudo-random number generation and associated testing methods can be found in research publications.
*   **Online communities for FPGA design:** Forums and communities offer practical solutions, troubleshooting advice, and various discussions relating to LFSR implementation.

Successfully implementing pseudo-random number generators in an FPGA demands an understanding of the underlying principles of LFSRs and hardware description language implementations. By considering initialization, period length, bias, and statistical properties, one can generate sequences suitable for a variety of applications.
