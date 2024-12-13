---
title: "case when vhdl syntax?"
date: "2024-12-13"
id: "case-when-vhdl-syntax"
---

Okay so you're asking about `CASE` statements in VHDL huh Been there done that tons of times Let's break it down in a way that makes sense

Basically you're dealing with a construct that lets you pick different actions based on the value of some expression It's like a switch statement in C or Java but in the context of hardware design with all its nuances

I've wrestled with this thing countless times My first FPGA project back in college was a disaster of tangled `if` and `else` statements Then I found `CASE` and life became way simpler Seriously though it's crucial for creating clean and readable hardware description language

So how does it look in practice Here are some basic structures and common pitfalls to avoid I’ll throw in some code snippets cause who doesn't love a good code example

**Basic `CASE` Structure**

The standard `CASE` uses a selection variable to evaluate against different values Each value is a possible 'case' It's like checking if a variable is equal to one value or the other

```vhdl
process(sel_signal)
begin
    case sel_signal is
        when "00" =>
            output_signal <= "1010";
        when "01" =>
            output_signal <= "0101";
        when "10" =>
            output_signal <= "1100";
        when "11" =>
            output_signal <= "0011";
        when others =>
            output_signal <= "0000"; -- Default for anything else
    end case;
end process;
```

Here `sel_signal` is the selection input It dictates which output is assigned to `output_signal` It's important to include the `others` case to handle unexpected inputs It prevents unintended latches in synthesised hardware I had a project once where I forgot this and it nearly drove me nuts debugging what felt like a ghost in the machine

**`CASE` with Enumerated Types**

VHDL shines with enumerated types They really make your code self-documenting Imagine defining a set of states for a state machine And `CASE` is perfect for state transitions with those

```vhdl
type state_type is (IDLE, LOAD, PROCESS, DONE);
signal current_state : state_type;

process(clk, reset)
begin
    if reset = '1' then
        current_state <= IDLE;
    elsif rising_edge(clk) then
        case current_state is
            when IDLE =>
                if start_signal = '1' then
                    current_state <= LOAD;
                end if;
            when LOAD =>
                -- Do loading stuff
                current_state <= PROCESS;
            when PROCESS =>
                 -- Do process stuff
                current_state <= DONE;
            when DONE =>
                -- Finish all and go back
                 current_state <= IDLE;
        end case;
    end if;
end process;
```

Here `state_type` is a custom enum. This is the kind of thing I really needed to get better at because it made my designs more understandable and manageable debugging a bunch of states without names is basically the equivalent of trying to navigate without a map and I’ve experienced that pain more than once Believe me

**`CASE` with Range**

Sometimes you need to check if the variable falls within a range and that also is very much possible this is useful when handling data processing or control signals

```vhdl
process(input_val)
begin
    case input_val is
        when 0 to 9 =>
           output_code <= "0000";
        when 10 to 19 =>
           output_code <= "0001";
        when 20 to 29 =>
           output_code <= "0010";
        when others =>
            output_code <= "1111";
    end case;
end process;
```
In this instance I have used ranges to generate specific codes based on values this is particularly great when you have to do some calculations and those calculations could provide values that are close but not exact and you do need to handle those cases in a specific manner

**Things to Keep in Mind**

*   **Completeness:** Like I mentioned before you must cover all possibilities either explicitly or with an `others` clause Otherwise the synthesis tools might infer a latch that you did not want which can cause weird behaviour I had a design that worked fine in simulation and failed mysteriously on hardware because I forgot this rule It is very important to respect that
*   **Selection Variable Type:** The type of the selection signal has to be compatible with the `CASE` statement For example using range with a std\_logic\_vector is possible but not as intuitive as using integer variables or enumerated types
*   **Hardware Implications:** Remember that VHDL describes hardware so every statement you write translates into gates and flip-flops A badly implemented `CASE` can lead to inefficient hardware Remember that everything you write will be turned into real physical gates and wires and logic components

**How it works under the hood**

On the synthesis side `CASE` statements are generally mapped to multiplexer circuits You see a `CASE` statement and that gets turned into a mux with multiple inputs one for each `when` clause and a select input that depends on your input variable. This can become complex quickly especially with many cases so clarity becomes imperative for readability

I have also experimented with designs where the tools optimized the CASE statement to something completely different than just a multiplexer that is usually the case if your `when` expressions are more complicated then a straight forward equal operation The synthesis tools are really good at finding the best structure in many cases

**Debugging Tips**

Simulation is key here I would always create testbenches to check how the `CASE` works with different inputs Use waveforms to visualize what's happening and see if it matches your expectations A simple logic analyzer on the FPGA is another lifesaver it allows you to have real data to look at

**Further Reading**

For more in-depth knowledge I would highly suggest "VHDL Primer" by J. Bhasker it is a solid reference book that explains all that you need regarding VHDL syntax with many examples and also "Digital Design Principles and Practices" by John F. Wakerly this is a very good book to understand the relation between Hardware description languages and the hardware that it represents

Anyways I hope this clarifies the mystery of the `CASE` statement in VHDL Trust me it's a powerful tool once you get a handle on it Keep practicing and you will get better at it it’s all about working in practice with that
Oh and one last joke why was the VHDL programmer bad at poker because they always declared the output as 'Z' for high impedance
