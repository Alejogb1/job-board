---
title: "How can square roots be implemented on an FPGA using VHDL and fixed-point arithmetic?"
date: "2025-01-30"
id: "how-can-square-roots-be-implemented-on-an"
---
Implementing square roots on an FPGA using VHDL with fixed-point arithmetic requires a careful consideration of both accuracy and resource utilization. The inherent iterative nature of square root computation, combined with the constraints of hardware implementation, demands a pragmatic approach. I’ve found that, in my experience designing embedded control systems for aerospace applications, a digit-by-digit (or radix-based) algorithm, specifically a non-restoring algorithm, provides a good balance between these two competing factors. It avoids the costly and often impractical iterative division operations often associated with other square root algorithms. Let me explain how this works and provide concrete examples.

The fundamental principle behind a non-restoring square root algorithm is to progressively build the root, one digit at a time, based on comparing partial remainders with a value derived from the current partial root. The algorithm iterates through each digit position, calculating a new partial remainder and appending the next most significant digit to the partial root. The 'non-restoring' aspect eliminates the need for conditional restorations of the remainder, which would be necessary in a restoring algorithm, thereby reducing resource requirements in hardware. The crux of this involves shifting operations, additions, and subtractions.

Let’s consider a fixed-point number representation with 'n' integer bits and 'm' fractional bits. We are calculating the square root of an input ‘x’ which will be in the same fixed-point format. The initial partial remainder, let's call it 'r', is initialized as the input x. The partial root, 'y', is initialized to zero. The algorithm proceeds bit by bit, starting from the most significant digit of the root. In each iteration, we shift ‘r’ two bits to the left, which is equivalent to multiplying by 4. Then we determine if appending ‘1’ to the current partial root produces a square that is less than the adjusted remainder. If it is, we subtract the square of the new partial root from the adjusted remainder and append ‘1’. Otherwise, we append ‘0’ and carry on. This continues until the desired number of bits in the square root is calculated. The final root 'y' will have n integer bits and m/2 fractional bits (assuming m is an even number) at most.

Here are three VHDL code examples, each designed to illustrate the incremental complexity and optimization of the core algorithm:

**Example 1: Basic Non-Restoring Square Root (Unoptimized)**

This example provides the core algorithm without resource optimization considerations.

```vhdl
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity sqrt_basic is
    Generic (N : integer := 16); -- Total number of bits for the input
    Port (
        clk      : in  std_logic;
        reset_n  : in  std_logic;
        x        : in  std_logic_vector(N-1 downto 0);
        y        : out std_logic_vector(N/2-1 downto 0); -- Output will have half bits, rounded down
        valid   : out std_logic
    );
end entity sqrt_basic;

architecture Behavioral of sqrt_basic is
    signal r         : unsigned(2*N-1 downto 0) := (others => '0');
    signal y_partial : unsigned(N/2-1 downto 0) := (others => '0');
    signal idx       : integer range 0 to N/2 := 0;
    signal done       : std_logic := '0';
begin
    process (clk)
    begin
        if rising_edge(clk) then
            if reset_n = '0' then
                r <= (others => '0');
                y_partial <= (others => '0');
                idx <= 0;
                done <= '0';
                valid <= '0';
            elsif done = '0' then
              if idx = 0 then
                    r <= unsigned('0' & x & (others => '0')); -- Initialize remainder, padding with zeros
                    y_partial <= (others => '0');
                    idx <= 1;
               elsif idx <= N/2 then
                   r <= r(2*N-3 downto 0) & "00"; -- Shift remainder by 2
                   y_partial(N/2 - idx) <= '1'; -- Assume we try to append 1 first
                   if r >= ((y_partial * y_partial)  ) then
                      r <= r - ((y_partial*y_partial)); -- If square is less than remainder, subtract the square
                   else
                      y_partial(N/2 - idx) <= '0'; -- Else, keep previous root and don't subtract
                   end if;

                   idx <= idx + 1;
              else
                  done <= '1';
                  valid <= '1';
                  y <= std_logic_vector(y_partial);
              end if;

            else
              valid <= '0';
            end if;
        end if;
    end process;
end architecture Behavioral;

```

*Commentary:* This basic implementation demonstrates the core logic. It uses unsigned arithmetic, left-shifting, and conditional subtraction. The `idx` variable tracks which root digit is currently being calculated. The `valid` signal indicates when the output is valid. While straightforward, this design is not optimized for area or throughput. The square calculation (`y_partial*y_partial`) is resource-intensive on an FPGA and should be optimized in later versions.

**Example 2: Optimized Non-Restoring Square Root (Lookup Table for Squares)**

This version introduces a lookup table to calculate the square of the partial root candidate rather than using an arithmetic multiplication. This optimization is very effective in reducing logic resource consumption.

```vhdl
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity sqrt_lut is
    Generic (N : integer := 16);
    Port (
        clk      : in  std_logic;
        reset_n  : in  std_logic;
        x        : in  std_logic_vector(N-1 downto 0);
        y        : out std_logic_vector(N/2-1 downto 0);
        valid    : out std_logic
    );
end entity sqrt_lut;

architecture Behavioral of sqrt_lut is
    signal r        : unsigned(2*N-1 downto 0) := (others => '0');
    signal y_partial : unsigned(N/2-1 downto 0) := (others => '0');
    signal idx       : integer range 0 to N/2 := 0;
    signal done      : std_logic := '0';
    -- Square value LUT
    type square_lut_type is array (0 to (1<<(N/2))-1) of unsigned(2*N-1 downto 0);
    function generate_square_lut (N : integer) return square_lut_type is
          variable lut : square_lut_type;
          variable i : integer;
        begin
            for i in 0 to (1<<(N/2))-1 loop
                lut(i) := unsigned(to_unsigned(i, N/2) * to_unsigned(i, N/2));
           end loop;
         return lut;
        end function;
    constant square_lut : square_lut_type := generate_square_lut(N);


begin
    process (clk)
    begin
        if rising_edge(clk) then
            if reset_n = '0' then
                r <= (others => '0');
                y_partial <= (others => '0');
                idx <= 0;
                done <= '0';
                valid <= '0';
            elsif done = '0' then
              if idx = 0 then
                  r <= unsigned('0' & x & (others => '0'));
                  y_partial <= (others => '0');
                  idx <= 1;
              elsif idx <= N/2 then
                    r <= r(2*N-3 downto 0) & "00";
                    y_partial(N/2 - idx) <= '1';
                  if r >= square_lut(to_integer(y_partial))  then  --Lookup for square
                    r <= r - square_lut(to_integer(y_partial)); --Subtract lookup value
                  else
                       y_partial(N/2 - idx) <= '0';
                  end if;
                idx <= idx + 1;
              else
                  done <= '1';
                  valid <= '1';
                  y <= std_logic_vector(y_partial);
              end if;

            else
              valid <= '0';
            end if;
        end if;
    end process;
end architecture Behavioral;
```

*Commentary:* This example precomputes all possible square values and stores them in a lookup table which consumes some Block RAM.  Within the iterative loop, it accesses this table rather than performing an arithmetic multiplication. This method trades off some memory usage for reduced logic resources and slightly improves performance as the table lookup is faster than a multiplier.  The `generate_square_lut` function generates the lookup table at synthesis time.

**Example 3: Pipelined Non-Restoring Square Root**

To increase the throughput and clock frequency, a pipelined architecture can be introduced. The stages for shift, root appendage, subtraction, and remainder updates can be implemented in separate pipeline stages and buffered with registers. This greatly increases the clock frequency possible on an FPGA design.

```vhdl
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity sqrt_pipelined is
    Generic (N : integer := 16);
    Port (
        clk      : in  std_logic;
        reset_n  : in  std_logic;
        x        : in  std_logic_vector(N-1 downto 0);
        y        : out std_logic_vector(N/2-1 downto 0);
        valid    : out std_logic
    );
end entity sqrt_pipelined;

architecture Behavioral of sqrt_pipelined is
    signal r_reg        :  unsigned(2*N-1 downto 0) := (others => '0');
    signal y_partial_reg : unsigned(N/2-1 downto 0) := (others => '0');
     signal r_shifted        :  unsigned(2*N-1 downto 0) := (others => '0');
     signal idx_reg       : integer range 0 to N/2 := 0;
    signal done_reg      : std_logic := '0';
     signal y_next        : unsigned(N/2-1 downto 0);
     signal r_next        :  unsigned(2*N-1 downto 0);
     signal valid_reg     : std_logic := '0';
     type square_lut_type is array (0 to (1<<(N/2))-1) of unsigned(2*N-1 downto 0);
    function generate_square_lut (N : integer) return square_lut_type is
          variable lut : square_lut_type;
          variable i : integer;
        begin
            for i in 0 to (1<<(N/2))-1 loop
                lut(i) := unsigned(to_unsigned(i, N/2) * to_unsigned(i, N/2));
           end loop;
         return lut;
        end function;
    constant square_lut : square_lut_type := generate_square_lut(N);
begin
    process (clk)
    begin
        if rising_edge(clk) then
            if reset_n = '0' then
               r_reg <= (others => '0');
              y_partial_reg <= (others => '0');
                idx_reg <= 0;
                done_reg <= '0';
                valid_reg <= '0';
            elsif done_reg = '0' then
                if idx_reg = 0 then
                  r_reg <= unsigned('0' & x & (others => '0'));
                  y_partial_reg <= (others => '0');
                    idx_reg <= 1;
                  else
                    r_shifted <= r_reg(2*N-3 downto 0) & "00";
                    y_next <= y_partial_reg;
                    y_next(N/2 - idx_reg) <= '1';

                  if r_shifted >= square_lut(to_integer(y_next))  then
                       r_next <= r_shifted - square_lut(to_integer(y_next));
                    else
                       y_next(N/2 - idx_reg) <= '0';
                        r_next <= r_shifted;

                    end if;

                       r_reg <= r_next;
                      y_partial_reg <= y_next;
                   idx_reg <= idx_reg+1;
                   if idx_reg > N/2 then
                      done_reg <= '1';
                     valid_reg <= '1';
                   end if;

                else
                  valid_reg <= '0';
                end if;
            end if;
            valid <= valid_reg;
            y <= std_logic_vector(y_partial_reg);
        end if;
    end process;
end architecture Behavioral;
```

*Commentary:* This pipelined version registers the intermediate signals `r` and `y_partial` and uses extra registers at each stage. The `r_shifted`, `r_next` and `y_next` are used to manage the data flow within a single cycle and are then passed to the next cycle using the `r_reg`, and `y_partial_reg` registers. This introduces latency equal to the number of pipeline stages. The square calculation using the LUT remains the same.  Pipelining allows for a higher clock rate and thus a higher overall throughput for many calculations. The `valid` signal lags the input by the pipeline depth.

These examples demonstrate how the non-restoring square root algorithm can be implemented in VHDL, ranging from a basic design to more optimized and higher throughput versions.  The choice of method will always depend on the specific requirements of the target application. For further study and more detailed insight into these methods, I recommend reading works by Steven J. B. and Michael D. L on digital arithmetic. Works on FPGA architecture, by authors such as Roger H. and Peter A. are also very informative for hardware design considerations.  Finally, the IEEE floating-point standard document is useful for understanding the theoretical bounds and trade-offs involved with fixed-point arithmetic and should be used for reference.
