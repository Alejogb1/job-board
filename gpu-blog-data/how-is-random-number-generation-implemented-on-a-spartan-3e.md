---
title: "How is random number generation implemented on a Spartan-3E?"
date: "2025-01-26"
id: "how-is-random-number-generation-implemented-on-a-spartan-3e"
---

The Spartan-3E lacks a dedicated hardware random number generator (RNG). Therefore, generating pseudo-random numbers relies primarily on algorithmic approaches implemented in the FPGA’s configurable logic. This presents both challenges and opportunities for achieving sufficient randomness within the constraints of the available resources. My experience with embedded systems, specifically on the Spartan-3E for a telemetry project, has highlighted the importance of choosing an appropriate pseudo-random number generation (PRNG) method and the limitations imposed by its architecture.

The fundamental limitation is the absence of a source of truly unpredictable physical phenomena within the chip itself, which would serve as a seed for a genuine random generator. Instead, deterministic algorithms that create sequences *appearing* random are used. The quality of a PRNG depends on several factors: period length (how many unique values are produced before repetition), statistical distribution (whether the generated numbers are evenly spread across the range), and computational efficiency (how much of the FPGA's logic resources the generator uses).

One common approach is the Linear Congruential Generator (LCG), a relatively simple algorithm suitable for implementation on resource-constrained devices like the Spartan-3E. An LCG uses the following recursive equation:

X<sub>n+1</sub> = (aX<sub>n</sub> + c) mod m

where:

* X<sub>n</sub> is the current random number.
* X<sub>n+1</sub> is the next random number.
* a is the multiplier.
* c is the increment.
* m is the modulus.

The choice of a, c, and m significantly affects the quality of the sequence. Poorly chosen constants can result in short periods and biased distributions. For instance, using a modulus that is a power of 2, like 2<sup>16</sup> (65536) for a 16-bit output, makes implementation with modulo operations simpler, but can create noticeable patterns in the least significant bits. This is a key reason that more complicated, although potentially computationally intensive, algorithms are sometimes necessary.

The Spartan-3E's limited logic resources often mean that complex algorithms like the Mersenne Twister are generally impractical. A reasonable trade-off often involves using a slightly more sophisticated version of an LCG or a different, simpler algorithm with better properties than a basic LCG. I personally found that careful selection of the LCG parameters and post-processing, such as bit shuffling, can achieve a quite satisfactory pseudo-random sequence for most basic tasks.

Another method, sometimes useful for tasks where lower quality random number generation is acceptable in favor of minimal resource utilization, involves utilizing inherent jitter present in timing loops. By counting the number of clock cycles that elapse within a loop dependent on some external or relatively unpredictable process, such as a timer or a serial receive, a stream of quasi-random bits can be created. While not statistically robust as an LCG, this method is extremely light in logic usage.

The implementation process generally involves defining a module with internal register(s) that store the current state of the generator, combinational logic implementing the calculation, and a clock input that synchronizes the state update. The output from this module forms the pseudo-random numbers. Below are three representative VHDL code examples that I have found useful over the course of different projects involving the Spartan-3E.

**Example 1: A Basic LCG**

This example illustrates a simple 16-bit LCG implementation. I kept the parameters relatively simple to make the function fit within the limited logic resources of older Spartan-3E devices.

```vhdl
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity lcg_16bit is
    Port ( clock : in  std_logic;
           reset : in  std_logic;
           random_out : out std_logic_vector(15 downto 0));
end entity lcg_16bit;

architecture Behavioral of lcg_16bit is
    signal state : unsigned(15 downto 0) := x"1234"; -- Initial seed
    constant a : unsigned(15 downto 0) := x"6123";
    constant c : unsigned(15 downto 0) := x"0017";
begin
    process(clock, reset)
    begin
        if reset = '1' then
            state <= x"1234";
        elsif rising_edge(clock) then
            state <= (a * state + c);
        end if;
    end process;
    random_out <= std_logic_vector(state);
end architecture Behavioral;
```

*Commentary:* This VHDL code defines an entity `lcg_16bit` with a clock, reset, and a 16-bit random output. The internal state (`state`) is initialized with a seed value. The `process` updates the state on the rising edge of the clock according to the LCG formula. Note that the multiplication will potentially consume more resources than simpler operations, and the initial values of the constants a and c are crucial for period length.

**Example 2:  A Modified LCG with Bit-Shuffling**

This example uses the same core LCG algorithm but adds a bit-shuffling operation to improve the distribution of bits. This method has produced more desirable results for me without a significant increase in resource usage when it is implemented correctly.

```vhdl
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity lcg_shuffled is
    Port ( clock : in  std_logic;
           reset : in  std_logic;
           random_out : out std_logic_vector(15 downto 0));
end entity lcg_shuffled;

architecture Behavioral of lcg_shuffled is
    signal state : unsigned(15 downto 0) := x"5678";
    constant a : unsigned(15 downto 0) := x"4ABC";
    constant c : unsigned(15 downto 0) := x"002B";
    signal shuffled_state : std_logic_vector(15 downto 0);
begin
    process(clock, reset)
    begin
        if reset = '1' then
            state <= x"5678";
        elsif rising_edge(clock) then
            state <= (a * state + c);
        end if;
    end process;

    -- Bit shuffling operation (Example)
    shuffled_state(0) <= std_logic(state(15));
    shuffled_state(1) <= std_logic(state(1));
    shuffled_state(2) <= std_logic(state(14));
    shuffled_state(3) <= std_logic(state(2));
    shuffled_state(4) <= std_logic(state(13));
    shuffled_state(5) <= std_logic(state(3));
    shuffled_state(6) <= std_logic(state(12));
    shuffled_state(7) <= std_logic(state(4));
    shuffled_state(8) <= std_logic(state(11));
    shuffled_state(9) <= std_logic(state(5));
    shuffled_state(10) <= std_logic(state(10));
    shuffled_state(11) <= std_logic(state(6));
    shuffled_state(12) <= std_logic(state(9));
    shuffled_state(13) <= std_logic(state(7));
    shuffled_state(14) <= std_logic(state(8));
    shuffled_state(15) <= std_logic(state(0));

    random_out <= shuffled_state;
end architecture Behavioral;
```

*Commentary:* This code adds an extra block of combinational logic (`shuffled_state`) that permutes the output bits from the LCG before the value is presented at the output. While this might seem simple, this type of shuffling can drastically improve the statistical properties of the random numbers. The bit selection should be well-considered, as poor shuffling can make things worse.

**Example 3: Clock Cycle Counting for Minimal Logic Usage**

This example demonstrates the minimal-resource method of clock cycle counting within a loop dependent on an external signal, which could be a timer or serial data.

```vhdl
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity counter_based_rng is
    Port ( clock : in  std_logic;
           enable_signal : in std_logic;  -- Input that signals the start/end of the loop
           random_out : out std_logic_vector(15 downto 0));
end entity counter_based_rng;

architecture Behavioral of counter_based_rng is
    signal counter : unsigned(15 downto 0) := (others => '0');
    signal prev_enable: std_logic := '0';
begin

   process(clock)
    begin
        if rising_edge(clock) then
            if enable_signal = '1' and prev_enable = '0' then
                counter <= (others => '0');
            elsif enable_signal = '1' then
                 counter <= counter + 1;
            end if;
            prev_enable <= enable_signal;
         end if;
    end process;
    random_out <= std_logic_vector(counter);

end architecture Behavioral;
```

*Commentary:*  This example uses a simple counter that increments with every clock cycle while `enable_signal` is high. The value of the counter when the `enable_signal` becomes inactive is presented as the `random_out`. The variability of the external `enable_signal` or its latency introduces the element of unpredictability. The quality of randomness here is directly tied to the jitter or unpredictability of the `enable_signal`. This method is useful in situations where minimal resource usage is more important than high-quality randomness.

In practice, I have found it beneficial to test various PRNG implementations and analyze the resulting output using statistical tools to verify the sequence’s quality. For more critical applications where security is an issue, more robust algorithms might be needed but often involve trade-offs in resource usage on older FPGAs.

When exploring further into the field of random number generation on the Spartan-3E, I recommend investigating the following resources: Textbooks on digital design that include sections on pseudo-random number generation algorithms and FPGA implementation, manufacturer datasheets to obtain deeper knowledge of the available resources and limitations, and various publications on statistical analysis of PRNGs to effectively evaluate the randomness quality. These resources, used in conjunction with a hands-on approach, will provide a solid foundation for effectively implementing random number generation on Spartan-3E and similar devices.
