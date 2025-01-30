---
title: "How can an FPGA be used to generate PWM signals?"
date: "2025-01-30"
id: "how-can-an-fpga-be-used-to-generate"
---
Pulse Width Modulation (PWM) signal generation is a fundamental capability required in a wide array of embedded systems, from motor control to dimming LEDs. FPGAs, with their inherent parallelism and configurability, provide an exceptionally versatile and efficient platform for implementing precise PWM generation logic. Unlike microcontroller-based solutions that often rely on hardware timers and peripherals, FPGAs allow for customized PWM generation tailored to specific application needs, enabling higher frequencies, improved resolution, and independent control of multiple channels without resource contention. My experience in developing high-speed data acquisition systems and custom hardware for robotic platforms has repeatedly underscored the advantages of using FPGAs for this purpose.

The core principle behind generating a PWM signal using an FPGA involves comparing a continuously incrementing counter value against a predetermined duty cycle value. When the counter is less than the duty cycle, the output signal is driven high; conversely, when the counter exceeds the duty cycle, the output signal is driven low. This process is repeatedly cycled based on a defined clock frequency, thereby creating a periodic waveform where the pulse width is proportional to the configured duty cycle. In essence, the duty cycle dictates the percentage of time the output signal is high within each period.

The frequency of the PWM signal is directly related to the clock driving the counter and the maximum value of the counter, known as the period or counter limit. A higher clock frequency allows for higher PWM frequencies but also impacts the maximum attainable resolution, which is limited by the number of bits in the counter and duty cycle registers.

Let's explore how this concept translates into practical FPGA implementation using VHDL, a common hardware description language.

**Example 1: Basic Single-Channel PWM Implementation**

This example demonstrates the fundamental building blocks for a single-channel PWM generator with a fixed frequency and a configurable duty cycle.

```vhdl
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity pwm_generator is
    Port ( clk       : in  std_logic;
           duty_cycle : in  unsigned(7 downto 0);  -- 8-bit duty cycle input
           pwm_out   : out std_logic
         );
end entity pwm_generator;

architecture behavioral of pwm_generator is
    signal counter   : unsigned(7 downto 0) := (others => '0');
    signal period    : unsigned(7 downto 0) := to_unsigned(255,8); -- Fixed Period
begin
    process (clk)
    begin
        if rising_edge(clk) then
            if counter < period then
                counter <= counter + 1;
            else
                counter <= (others => '0');
            end if;
        end if;
    end process;

    pwm_out <= '1' when counter < duty_cycle else '0';
end architecture behavioral;
```

**Commentary:**

This VHDL code defines an entity `pwm_generator` with a clock input, an 8-bit duty cycle input, and a single PWM output. Inside the architecture, a counter (`counter`) is incremented on each clock rising edge until it reaches the defined period, which is 255 in this case, causing the counter to roll over to zero. The PWM output (`pwm_out`) is driven high when the counter value is less than the provided `duty_cycle` value and is driven low otherwise. This effectively generates a PWM signal whose duty cycle is determined by the input `duty_cycle`. A fixed period is implemented, resulting in a consistent frequency.  The choice of 8-bit values for both the counter and duty cycle allows for 256 discrete duty cycle values, ranging from 0% to 100%.

**Example 2: PWM Generator with Configurable Period**

This example extends the basic implementation to allow for runtime configuration of the PWM period, thereby enabling frequency adjustment.

```vhdl
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity pwm_generator_configurable is
    Port ( clk        : in  std_logic;
           duty_cycle  : in  unsigned(7 downto 0);
           period      : in  unsigned(7 downto 0);  -- Configurable period
           pwm_out    : out std_logic
         );
end entity pwm_generator_configurable;

architecture behavioral of pwm_generator_configurable is
    signal counter   : unsigned(7 downto 0) := (others => '0');
begin
    process (clk)
    begin
        if rising_edge(clk) then
            if counter < period then
                counter <= counter + 1;
            else
                counter <= (others => '0');
            end if;
        end if;
    end process;

    pwm_out <= '1' when counter < duty_cycle else '0';
end architecture behavioral;
```

**Commentary:**

This revised entity `pwm_generator_configurable` introduces an additional input, `period`, allowing an external source to change the counter limit during runtime. This modification allows for dynamic control over the PWM signal frequency. The functionality of the counter and duty cycle comparison remains unchanged; only the period value is now externally configurable, offering a higher level of flexibility compared to the previous example. The resolution is still 8-bit, limiting it to 256 distinct duty cycles, but the period is no longer hardcoded.

**Example 3: Multi-Channel PWM Generation with a Shared Counter**

The following example illustrates how to efficiently generate multiple PWM signals using a single shared counter. This is advantageous in scenarios where multiple outputs need to be controlled, such as in multi-motor drive systems or display backlight control.

```vhdl
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity multi_channel_pwm is
    Port ( clk           : in  std_logic;
           duty_cycle_0 : in  unsigned(7 downto 0);
           duty_cycle_1 : in  unsigned(7 downto 0);
           duty_cycle_2 : in  unsigned(7 downto 0);
           period       : in  unsigned(7 downto 0);
           pwm_out_0    : out std_logic;
           pwm_out_1    : out std_logic;
           pwm_out_2    : out std_logic
         );
end entity multi_channel_pwm;

architecture behavioral of multi_channel_pwm is
    signal counter    : unsigned(7 downto 0) := (others => '0');
begin
    process (clk)
    begin
       if rising_edge(clk) then
            if counter < period then
                counter <= counter + 1;
            else
                counter <= (others => '0');
            end if;
        end if;
    end process;


    pwm_out_0 <= '1' when counter < duty_cycle_0 else '0';
    pwm_out_1 <= '1' when counter < duty_cycle_1 else '0';
    pwm_out_2 <= '1' when counter < duty_cycle_2 else '0';


end architecture behavioral;
```

**Commentary:**

In this example, the entity `multi_channel_pwm` takes multiple duty cycle inputs (`duty_cycle_0`, `duty_cycle_1`, `duty_cycle_2`) and outputs three independent PWM signals (`pwm_out_0`, `pwm_out_1`, `pwm_out_2`). A single shared counter is utilized, where its value is compared independently against each of the different duty cycles. This method conserves FPGA resources by avoiding the need for redundant counter logic. Each output effectively generates a PWM waveform with a duty cycle that is independently controlled.  This approach allows for synchronized PWM waveforms, sharing the same period, but each with unique pulse width modulation.

For further exploration of FPGA-based digital design and PWM techniques, I recommend consulting texts such as "Digital Design Using VHDL" by William J. Dally and R. Curtis Harting, as well as "FPGA Prototyping by VHDL Examples" by Pong P. Chu. Specific vendor documentation for your chosen FPGA platform is also essential for understanding available hardware resources and optimization techniques. Numerous academic papers and conference proceedings offer advanced discussions on specific PWM applications and design trade-offs. Understanding core VHDL concepts and digital design principles will be crucial to build increasingly sophisticated PWM controllers.
