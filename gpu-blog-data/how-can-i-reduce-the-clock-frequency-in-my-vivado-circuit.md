---
title: "How can I reduce the clock frequency in my Vivado circuit?"
date: "2025-01-26"
id: "how-can-i-reduce-the-clock-frequency-in-my-vivado-circuit"
---

The primary method to reduce clock frequency within a Vivado design involves leveraging clock management primitives, specifically those that allow for clock division. Direct manipulation of the system clock source frequency at the hardware level is often not within the purview of standard HDL design practices; instead, we focus on deriving slower clocks from a faster, established source.

Clock division is essentially the process of creating a new clock signal with a frequency that is an integer fraction of an original clock. This is commonly achieved using dedicated clock management resources such as MMCMs (Mixed-Mode Clock Managers) or PLLs (Phase-Locked Loops) within the Xilinx FPGA architecture. These primitives provide highly configurable clock generation capabilities, enabling us to derive multiple clocks with differing frequencies and phase relationships. My experience across various embedded systems projects has consistently pointed towards this approach as the most reliable and manageable for controlling clock speeds within a design.

Implementing clock division using MMCMs or PLLs involves instantiating these primitives within our HDL code. I typically prefer using the Vivado IP catalog to generate these cores as it provides a more visual interface and eases the configuration process. However, direct instantiation of the underlying primitives is also possible and gives greater control at the cost of verbosity. The fundamental concept, however, remains consistent regardless of how the primitive is instantiated. We specify the input clock frequency, the desired output clock frequency, and any other relevant parameters such as jitter requirements and allowable phase shift. The MMCM/PLL then handles the internal logic to achieve this.

The most critical considerations when dividing clock frequencies revolve around the potential introduction of clock skew, increased latency, and the need for proper reset synchronization. Clock skew, differences in arrival times of a clock signal at different points in the circuit, can become more prominent with divided clocks, particularly with long routing paths or non-balanced topologies. Proper clock domain crossing (CDC) techniques must be employed when interfacing between the different clock domains formed by the original and divided clock signals to prevent metastability issues. Furthermore, the latency introduced by clock division primitives themselves will contribute to overall system latency and must be accounted for when planning timing constraints.

Here are three code examples illustrating different aspects of clock frequency reduction, leveraging the `MMCME2_ADV` primitive:

**Example 1: Basic Clock Division**

```vhdl
library ieee;
use ieee.std_logic_1164.all;

entity clock_divider_basic is
    Port ( clk_in   : in  std_logic;
           clk_out  : out std_logic;
           reset : in std_logic
           );
end clock_divider_basic;

architecture Behavioral of clock_divider_basic is

    signal clk_fb : std_logic;

    component MMCME2_ADV
    generic (
        BANDWIDTH           : string := "OPTIMIZED";
        CLKFBOUT_MULT_F     : real   := 5.0;
        CLKFBOUT_PHASE      : real   := 0.0;
        CLKFBOUT_USE_FINE_PS : boolean := FALSE;
        CLKIN1_PERIOD       : real   := 10.0;  -- input clock period 10ns = 100MHz
        CLKOUT0_DIVIDE_F    : real   := 10.0;
        CLKOUT0_DUTY_CYCLE  : real   := 0.5;
        CLKOUT0_PHASE       : real   := 0.0;
        CLKOUT1_DIVIDE      : integer := 1;
        CLKOUT1_DUTY_CYCLE  : real   := 0.5;
        CLKOUT1_PHASE       : real   := 0.0;
        CLKOUT2_DIVIDE      : integer := 1;
        CLKOUT2_DUTY_CYCLE  : real   := 0.5;
        CLKOUT2_PHASE       : real   := 0.0;
        CLKOUT3_DIVIDE      : integer := 1;
        CLKOUT3_DUTY_CYCLE  : real   := 0.5;
        CLKOUT3_PHASE       : real   := 0.0;
        CLKOUT4_DIVIDE      : integer := 1;
        CLKOUT4_DUTY_CYCLE  : real   := 0.5;
        CLKOUT4_PHASE       : real   := 0.0;
        CLKOUT5_DIVIDE      : integer := 1;
        CLKOUT5_DUTY_CYCLE  : real   := 0.5;
        CLKOUT5_PHASE       : real   := 0.0;
        CLKOUT6_DIVIDE      : integer := 1;
        CLKOUT6_DUTY_CYCLE  : real   := 0.5;
        CLKOUT6_PHASE       : real   := 0.0;
        CLKOUT_USE_FINE_PS : boolean := FALSE;
        DIVCLK_DIVIDE      : integer := 1;
        REF_JITTER1         : real   := 0.0
     )
    port (
    CLKFBOUT   : out  std_logic;
    CLKOUT0    : out  std_logic;
    CLKOUT1    : out  std_logic;
    CLKOUT2    : out  std_logic;
    CLKOUT3    : out  std_logic;
    CLKOUT4    : out  std_logic;
    CLKOUT5    : out  std_logic;
    CLKOUT6    : out  std_logic;
    CLKIN1     : in  std_logic;
    PWRDWN     : in  std_logic;
    RESET      : in  std_logic;
    LOCKED    : out std_logic;
    CLKFBIN   : in std_logic
    );
    end component;

    signal locked_sig : std_logic;

begin

    mmcm_inst : MMCME2_ADV
    generic map (
        CLKIN1_PERIOD       => 10.0,  -- 100 MHz input
        CLKOUT0_DIVIDE_F    => 10.0   -- Output clock is 10MHz
    )
    port map (
        CLKFBOUT   => clk_fb,
        CLKOUT0    => clk_out,
        CLKIN1     => clk_in,
        PWRDWN     => '0',
        RESET      => reset,
        LOCKED     => locked_sig,
        CLKFBIN    => clk_fb
    );
end Behavioral;

```

*   **Commentary:** This code defines an entity called `clock_divider_basic` which accepts an input clock signal, `clk_in`, and provides a slower clock output `clk_out`. It instantiates the `MMCME2_ADV` primitive, and configures it to divide the input clock frequency by a factor of 10.  The input clock is set to 100MHz by the `CLKIN1_PERIOD` generic.  The output clock will be 10MHz, configured via `CLKOUT0_DIVIDE_F`. Note the reset and power down inputs to the MMCM are set to '0' except for the `reset`, which allows for reseting the clock divider. Note that the feedback clock path is also wired to the output.  The `LOCKED` signal indicates a stable operation from the MMCM.

**Example 2: Multiple Output Clocks from a single MMCM**

```vhdl
library ieee;
use ieee.std_logic_1164.all;

entity clock_divider_multiple is
    Port ( clk_in    : in  std_logic;
           clk_out1  : out std_logic;
           clk_out2  : out std_logic;
           reset     : in std_logic
           );
end clock_divider_multiple;

architecture Behavioral of clock_divider_multiple is

    signal clk_fb : std_logic;

    component MMCME2_ADV
    generic (
    BANDWIDTH           : string := "OPTIMIZED";
        CLKFBOUT_MULT_F     : real   := 5.0;
        CLKFBOUT_PHASE      : real   := 0.0;
        CLKFBOUT_USE_FINE_PS : boolean := FALSE;
        CLKIN1_PERIOD       : real   := 10.0;  -- input clock period 10ns = 100MHz
        CLKOUT0_DIVIDE_F    : real   := 10.0;
        CLKOUT0_DUTY_CYCLE  : real   := 0.5;
        CLKOUT0_PHASE       : real   := 0.0;
        CLKOUT1_DIVIDE      : integer := 20;
        CLKOUT1_DUTY_CYCLE  : real   := 0.5;
        CLKOUT1_PHASE       : real   := 0.0;
        CLKOUT2_DIVIDE      : integer := 1;
        CLKOUT2_DUTY_CYCLE  : real   := 0.5;
        CLKOUT2_PHASE       : real   := 0.0;
        CLKOUT3_DIVIDE      : integer := 1;
        CLKOUT3_DUTY_CYCLE  : real   := 0.5;
        CLKOUT3_PHASE       : real   := 0.0;
        CLKOUT4_DIVIDE      : integer := 1;
        CLKOUT4_DUTY_CYCLE  : real   := 0.5;
        CLKOUT4_PHASE       : real   := 0.0;
        CLKOUT5_DIVIDE      : integer := 1;
        CLKOUT5_DUTY_CYCLE  : real   := 0.5;
        CLKOUT5_PHASE       : real   := 0.0;
        CLKOUT6_DIVIDE      : integer := 1;
        CLKOUT6_DUTY_CYCLE  : real   := 0.5;
        CLKOUT6_PHASE       : real   := 0.0;
        CLKOUT_USE_FINE_PS : boolean := FALSE;
        DIVCLK_DIVIDE      : integer := 1;
        REF_JITTER1         : real   := 0.0
    )
    port (
        CLKFBOUT    : out  std_logic;
        CLKOUT0     : out  std_logic;
        CLKOUT1     : out  std_logic;
        CLKOUT2     : out  std_logic;
        CLKOUT3     : out  std_logic;
        CLKOUT4     : out  std_logic;
        CLKOUT5     : out  std_logic;
        CLKOUT6     : out  std_logic;
        CLKIN1      : in  std_logic;
        PWRDWN      : in  std_logic;
        RESET       : in  std_logic;
        LOCKED      : out std_logic;
        CLKFBIN    : in std_logic
    );
    end component;

    signal locked_sig : std_logic;

begin
   mmcm_inst : MMCME2_ADV
    generic map (
        CLKIN1_PERIOD       => 10.0,  -- 100 MHz input
        CLKOUT0_DIVIDE_F    => 10.0,   -- Output 1 : 10 MHz
        CLKOUT1_DIVIDE      => 20    -- Output 2: 5 Mhz
    )
    port map (
        CLKFBOUT   => clk_fb,
        CLKOUT0    => clk_out1,
        CLKOUT1    => clk_out2,
        CLKIN1     => clk_in,
        PWRDWN     => '0',
        RESET      => reset,
        LOCKED     => locked_sig,
        CLKFBIN     => clk_fb
    );
end Behavioral;
```

*   **Commentary:**  This example illustrates how multiple clock frequencies can be generated from a single `MMCME2_ADV` instance. Here, `clk_out1` is derived by dividing the input clock by 10, resulting in a 10 MHz output, while `clk_out2` is derived by dividing the input clock by 20, producing 5MHz clock.  The ability to generate multiple clocks simplifies design, reducing resource overhead.

**Example 3:  Clock division using integer values**

```vhdl
library ieee;
use ieee.std_logic_1164.all;

entity clock_divider_integer is
    Port ( clk_in    : in  std_logic;
           clk_out  : out std_logic;
           reset  : in std_logic
           );
end clock_divider_integer;

architecture Behavioral of clock_divider_integer is

    signal clk_fb : std_logic;

    component MMCME2_ADV
    generic (
        BANDWIDTH           : string := "OPTIMIZED";
        CLKFBOUT_MULT_F     : real   := 5.0;
        CLKFBOUT_PHASE      : real   := 0.0;
        CLKFBOUT_USE_FINE_PS : boolean := FALSE;
        CLKIN1_PERIOD       : real   := 10.0;  -- input clock period 10ns = 100MHz
        CLKOUT0_DIVIDE_F    : real   := 1.0;
        CLKOUT0_DUTY_CYCLE  : real   := 0.5;
        CLKOUT0_PHASE       : real   := 0.0;
        CLKOUT1_DIVIDE      : integer := 5;
        CLKOUT1_DUTY_CYCLE  : real   := 0.5;
        CLKOUT1_PHASE       : real   := 0.0;
        CLKOUT2_DIVIDE      : integer := 1;
        CLKOUT2_DUTY_CYCLE  : real   := 0.5;
        CLKOUT2_PHASE       : real   := 0.0;
        CLKOUT3_DIVIDE      : integer := 1;
        CLKOUT3_DUTY_CYCLE  : real   := 0.5;
        CLKOUT3_PHASE       : real   := 0.0;
        CLKOUT4_DIVIDE      : integer := 1;
        CLKOUT4_DUTY_CYCLE  : real   := 0.5;
        CLKOUT4_PHASE       : real   := 0.0;
        CLKOUT5_DIVIDE      : integer := 1;
        CLKOUT5_DUTY_CYCLE  : real   := 0.5;
        CLKOUT5_PHASE       : real   := 0.0;
        CLKOUT6_DIVIDE      : integer := 1;
        CLKOUT6_DUTY_CYCLE  : real   := 0.5;
        CLKOUT6_PHASE       : real   := 0.0;
        CLKOUT_USE_FINE_PS : boolean := FALSE;
        DIVCLK_DIVIDE      : integer := 1;
        REF_JITTER1         : real   := 0.0
    )
    port (
        CLKFBOUT    : out  std_logic;
        CLKOUT0     : out  std_logic;
        CLKOUT1     : out  std_logic;
        CLKOUT2     : out  std_logic;
        CLKOUT3     : out  std_logic;
        CLKOUT4     : out  std_logic;
        CLKOUT5     : out  std_logic;
        CLKOUT6     : out  std_logic;
        CLKIN1      : in  std_logic;
        PWRDWN      : in  std_logic;
        RESET       : in  std_logic;
        LOCKED       : out std_logic;
        CLKFBIN     : in std_logic
    );
    end component;

    signal locked_sig : std_logic;

begin
    mmcm_inst : MMCME2_ADV
    generic map (
        CLKIN1_PERIOD  => 10.0,  -- 100 MHz input
        CLKOUT0_DIVIDE_F => 1.0,
        CLKOUT1_DIVIDE    => 5    -- Output: 20MHz
    )
    port map (
        CLKFBOUT   => clk_fb,
        CLKOUT0    => open,
        CLKOUT1     => clk_out,
        CLKIN1     => clk_in,
        PWRDWN     => '0',
        RESET       => reset,
        LOCKED       => locked_sig,
        CLKFBIN     => clk_fb
    );
end Behavioral;

```

*   **Commentary:** This final example shows the use of integer clock division factors, via the `CLKOUT1_DIVIDE` parameter. The initial output `CLKOUT0` is set to 1.0 (no division), effectively disabling it by using 'open'.  The second output, `CLKOUT1` is set to divide the input clock by 5. Using integer division factors is essential when fractional frequencies are not needed. This example provides a clock output of 20MHz.

For further exploration of this topic, the Xilinx documentation provides detailed information on the MMCME2_ADV and PLL primitives. The Vivado Design Suite's user guides, specifically those pertaining to clocking, are also invaluable resources. Additionally, several books and online courses on FPGA design cover clock management in detail. These resources will provide a deeper understanding of the underlying principles, the finer control available via the primitives, and best practices for clock distribution networks within your design. Understanding these concepts will enable robust designs with precise clock management.
