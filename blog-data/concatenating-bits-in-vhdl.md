---
title: "concatenating bits in vhdl?"
date: "2024-12-13"
id: "concatenating-bits-in-vhdl"
---

Alright so you're asking about concatenating bits in VHDL right Been there done that a thousand times It's like the bread and butter of hardware design but I get it especially if you're just starting it can be a bit confusing So let's dive in shall we

First off concatenation in VHDL is pretty straightforward but you gotta understand how the language treats bit vectors and arrays You see VHDL isn't just like your typical programming language where you're just throwing around data It’s hardware description language so everything you write translates to hardware connections Think of it like you're building circuits not just writing instructions

When you're talking about bits you're usually working with signals or variables that are defined as std_logic_vector or sometimes just bit_vector If you're new to this I highly recommend picking up "Digital Design Principles and Practices" by John F Wakerly it’s a solid foundation I used it when I was getting my start back in the old days

Now the actual concatenation uses the `&` operator It's pretty simple you just put the vectors or bits you want to join together with `&` between them Like if you have two vectors `a` and `b` you just use `a & b` and that makes a new vector that's the length of `a` plus the length of `b`

Let me give you an example of a scenario where I ran into this You know I was designing this custom serial protocol decoder and needed to combine a bunch of control bits with the actual data bits before sending it out This was way back using a Xilinx Spartan 3 FPGA those were the days you had to be so careful about resource usage because the FPGAs weren't as beefy as what we have today

So my code looked something like this in the end something like this:

```vhdl
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity serial_encoder is
  Port (
    clk : in  std_logic;
    data_in : in  std_logic_vector(7 downto 0);
    control_in : in std_logic_vector(2 downto 0);
    serial_out : out std_logic
  );
end entity serial_encoder;

architecture behavioral of serial_encoder is
    signal shift_reg : std_logic_vector(10 downto 0); -- combined control + data
    signal data_counter : integer range 0 to 10;
begin
  process(clk)
  begin
    if rising_edge(clk) then
        if data_counter = 0 then
            shift_reg <= control_in & data_in & '1'; -- add stop bit and combine
            data_counter <= 10;
        elsif data_counter > 0 then
            shift_reg <= shift_reg(9 downto 0) & '0'; -- shift and make place for next bit
            data_counter <= data_counter -1;
        end if;
        serial_out <= shift_reg(10); -- output the MSB of shift reg
    end if;
  end process;
end architecture behavioral;
```

See how I combined `control_in` and `data_in` with that stop bit I was talking about to create the actual `shift_reg` signal That is concatenating a 3-bit vector with an 8-bit vector and then with 1 bit to create the complete shift register with 11 bits. This is a pretty common use case where you might need to assemble different parts of a data packet

One thing to watch out for is that the order matters when concatenating `a & b` is not the same as `b & a` The resulting vector is created in the order you put the terms In my early days I once flipped the order of these signals and spent half a day debugging why the receiver was getting corrupted data Turns out the control bits and the data bits were swapped I learned my lesson the hard way always double check your concatenation order

Another thing that’s important to know is how to concatenate constants or literals you can also just use the `&` with simple values like '1' or '0' like I did in the example above when I added that stop bit to the shift register or you can also concatenate like this `"00"` or `"10"` this is how you concatenate string literals in VHDL as bit vectors So you might want to create something like `"00" & data_in` This creates a vector with two zeros on the MSB then the bits of the signal data_in following that.

I also want to show you how it might look when using a mux This is another real world example when I was building a configurable data path where different control signals selected which piece of the data was going to outputted You would do it like this:

```vhdl
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity data_mux is
    Port (
        sel     : in  std_logic_vector (1 downto 0);
        data_a : in std_logic_vector (7 downto 0);
        data_b : in std_logic_vector (3 downto 0);
        data_c : in std_logic_vector (1 downto 0);
        data_out: out std_logic_vector (10 downto 0)
    );
end data_mux;


architecture arch_data_mux of data_mux is
begin
    process(sel, data_a, data_b, data_c)
    begin
        case sel is
            when "00" => data_out <= "000" & data_a & "0"; -- select A padded with 0s
            when "01" => data_out <= "00" & data_b & "0000000"; -- select B padded with 0s
            when "10" => data_out <= "00000000" & data_c & "00"; -- select C padded with 0s
            when others => data_out <= (others => '0'); -- default all 0s
        end case;
    end process;
end arch_data_mux;
```
Here you see I am concatenating some literal values like "000" or "00" with the data signals. This is a common way to make different data widths match the output width. I know some guys like to use functions but sometimes a simple mux is easier to read and faster to implement if you are not using it in a loop or some other place where reuse is important.

Ok here is one last use case that is useful to show you The thing about hardware is you are usually dealing with more complex data paths and structures so lets say you need to deal with data and metadata which in some of my past projects were necessary. This example also shows you how to use vector slices

```vhdl
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity data_metadata_combiner is
    Port (
        data_in: in std_logic_vector(31 downto 0);
        metadata_in: in std_logic_vector(7 downto 0);
        combined_out : out std_logic_vector (39 downto 0)
    );
end data_metadata_combiner;

architecture arch_data_metadata_combiner of data_metadata_combiner is
    signal temp_data : std_logic_vector (31 downto 0);
    signal temp_metadata : std_logic_vector (7 downto 0);
    signal temp_combined : std_logic_vector (39 downto 0);
begin
    temp_data <= data_in;
    temp_metadata <= metadata_in;
    temp_combined <= temp_metadata & temp_data; -- metadata first data after

    combined_out <= temp_combined;

    -- example of slice operation as a comment to show that
    -- combined_out(39 downto 32) is equal to temp_metadata
    -- combined_out(31 downto 0) is equal to temp_data

end architecture arch_data_metadata_combiner;
```

This example simply shows that you can concatenate two variables that are of type `std_logic_vector` And it also shows that `combined_out(39 downto 32)` is a slice of the combined vector that is equivalent to `temp_metadata` as a comment to remind you of what the signal is made of. This is important when debugging to keep track of the parts

I hope these examples were useful and that you can apply them to your projects also check out the book "VHDL Primer" by J Bhasker is also a very good read and helped me when I first started.

Oh and one more thing I found this joke somewhere on some forum once I'm not sure who said it but: Why did the VHDL programmer quit his job? Because he didn't get arrays! Hehe. But yeah in all seriousness VHDL concatenation is a powerful and necessary part of writing hardware descriptions so make sure you nail it down correctly.

Remember to be meticulous about your bit widths and the order of your concatenation operations because those are easy mistakes to make. Good luck with your VHDL projects and don't hesitate to ask if you hit any more snags!
