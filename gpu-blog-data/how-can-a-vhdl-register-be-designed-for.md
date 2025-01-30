---
title: "How can a VHDL register be designed for a push button?"
date: "2025-01-30"
id: "how-can-a-vhdl-register-be-designed-for"
---
A push button interface in a VHDL design requires robust debouncing and synchronization to prevent spurious signal transitions from affecting downstream logic. I’ve frequently encountered this issue during my tenure working with FPGA-based embedded systems. The core problem arises from the mechanical nature of a push button – when pressed or released, the contacts do not make or break cleanly. This produces a series of rapid open-and-close events, referred to as ‘contact bounce,’ that can appear as multiple logical changes instead of a single intended transition. Proper handling necessitates a multi-stage approach.

First, a simple VHDL register directly mirroring the button’s raw input is insufficient. The inherent bounce can cause unpredictable behavior. The key is to incorporate both a debouncing circuit and a synchronizer. The debouncer filters out these fast transitions, allowing only stable states to register. Simultaneously, the synchronizer, typically a two-stage flip-flop, mitigates any metastability issues arising from the asynchronous nature of the button's signal with respect to the system clock.

The debouncing stage operates on the principle of temporal filtering. We need to define a minimum period – typically several milliseconds – during which the button signal must remain stable for it to be considered a valid state change. This is accomplished using a counter and a comparison. If the signal remains constant for a period exceeding this threshold defined by the counter limit, the debounced output will change state; otherwise, the state remains unchanged.

The synchronization stage involves passing the debounced output through two D-type flip-flops clocked by the system clock. This aligns the asynchronous debounced button signal to our synchronous system domain. This two-stage synchronization helps minimize the probability of metastability, a condition where a flip-flop's output is indeterminate. I've found this two-stage approach adequate for most typical button input applications.

Here are three code examples illustrating increasingly complex implementation:

**Example 1: Simple Debouncer**

```vhdl
entity simple_debouncer is
    Port ( clk        : in  STD_LOGIC;
           button_in  : in  STD_LOGIC;
           button_out : out STD_LOGIC );
end entity simple_debouncer;

architecture Behavioral of simple_debouncer is
    constant DEBOUNCE_LIMIT : integer := 10000; -- Example debounce limit
    signal   counter        : integer range 0 to DEBOUNCE_LIMIT := 0;
    signal   button_reg     : STD_LOGIC := '0';
begin
    process(clk)
    begin
        if rising_edge(clk) then
            if button_in = button_reg then
                if counter < DEBOUNCE_LIMIT then
                    counter <= counter + 1;
                end if;
            else
                counter <= 0;
                button_reg <= button_in;
            end if;

            if counter = DEBOUNCE_LIMIT then
                button_out <= button_in;
            end if;
        end if;
    end process;
end architecture Behavioral;
```

*Commentary:* This first example implements a basic debouncer. The `DEBOUNCE_LIMIT` constant determines the debounce duration. The `counter` increments when the button input is stable. If the input changes, the counter resets. Only when the counter reaches the `DEBOUNCE_LIMIT`, is `button_out` updated. The key weakness here is the direct connection of the asynchronous `button_in` to a clocked process that can cause metastability issues. It also doesn’t handle synchronisation so should only be used where the signal being debounced is synchronous or the risks are understood and acceptable.

**Example 2: Debouncer with Basic Synchronisation**

```vhdl
entity debouncer_with_sync is
    Port ( clk        : in  STD_LOGIC;
           button_in  : in  STD_LOGIC;
           button_out : out STD_LOGIC );
end entity debouncer_with_sync;

architecture Behavioral of debouncer_with_sync is
    constant DEBOUNCE_LIMIT : integer := 10000; -- Example debounce limit
    signal   counter        : integer range 0 to DEBOUNCE_LIMIT := 0;
    signal   button_reg     : STD_LOGIC := '0';
    signal   debounced_sig  : STD_LOGIC;
    signal   sync_reg_1   : STD_LOGIC := '0';
    signal   sync_reg_2   : STD_LOGIC := '0';
begin
    --Debounce logic
    process(clk)
    begin
        if rising_edge(clk) then
            if button_in = button_reg then
                if counter < DEBOUNCE_LIMIT then
                    counter <= counter + 1;
                end if;
            else
                counter <= 0;
                button_reg <= button_in;
            end if;
            if counter = DEBOUNCE_LIMIT then
                debounced_sig <= button_in;
            end if;
        end if;
    end process;

    --Synchronization logic
    process (clk)
    begin
        if rising_edge(clk) then
            sync_reg_1 <= debounced_sig;
            sync_reg_2 <= sync_reg_1;
        end if;
    end process;
    button_out <= sync_reg_2;
end architecture Behavioral;
```

*Commentary:* This improves upon the first example by adding a two-stage synchronization to the debounced signal.  `debounced_sig` represents the output from the debouncer and this is passed into two cascaded flip-flops (`sync_reg_1` and `sync_reg_2`). The output of the second stage is the synchronized output, `button_out`. This approach minimizes metastability. This design is more robust than Example 1 and is suitable for many applications, but might be considered redundant by more experienced VHDL engineers for the simple problem of a single debounced and synchronised signal.

**Example 3: Modular Debouncer with Synchronisation and Edge Detection**

```vhdl
entity modular_button_interface is
    Port ( clk         : in  STD_LOGIC;
           button_in   : in  STD_LOGIC;
           button_out_p : out STD_LOGIC;
           button_out_r : out STD_LOGIC;
           button_out   : out STD_LOGIC );
end entity modular_button_interface;

architecture Behavioral of modular_button_interface is
    constant DEBOUNCE_LIMIT : integer := 10000; -- Example debounce limit
    signal   counter        : integer range 0 to DEBOUNCE_LIMIT := 0;
    signal   button_reg     : STD_LOGIC := '0';
    signal   debounced_sig  : STD_LOGIC;
    signal   sync_reg_1   : STD_LOGIC := '0';
    signal   sync_reg_2   : STD_LOGIC := '0';
    signal   previous_sync_state : STD_LOGIC := '0';
begin
    --Debounce logic
    process(clk)
    begin
        if rising_edge(clk) then
            if button_in = button_reg then
                if counter < DEBOUNCE_LIMIT then
                    counter <= counter + 1;
                end if;
            else
                counter <= 0;
                button_reg <= button_in;
            end if;
            if counter = DEBOUNCE_LIMIT then
                debounced_sig <= button_in;
            end if;
        end if;
    end process;

    --Synchronization logic
    process (clk)
    begin
        if rising_edge(clk) then
            sync_reg_1 <= debounced_sig;
            sync_reg_2 <= sync_reg_1;
        end if;
    end process;

    -- Output the stable output
    button_out <= sync_reg_2;

    -- Edge detection logic
    process(clk)
    begin
        if rising_edge(clk) then
            button_out_r <= '0'; -- reset each cycle
            button_out_p <= '0'; -- reset each cycle

            if(sync_reg_2 = '1' and previous_sync_state = '0') then
                button_out_p <= '1';
            end if;
            if (sync_reg_2 = '0' and previous_sync_state = '1') then
                button_out_r <= '1';
            end if;
             previous_sync_state <= sync_reg_2;
        end if;
    end process;

end architecture Behavioral;
```

*Commentary:*  This version introduces modularity and adds edge detection functionality. It provides not just the debounced and synchronized signal (`button_out`), but also separate outputs for rising edge (`button_out_p`) and falling edge (`button_out_r`). These edge-triggered signals are more useful for triggering specific actions in downstream logic. The `previous_sync_state` register is used to identify the transitions. This version encapsulates much more functionality required for a simple button interface making the component highly re-usable in other designs.

For deeper understanding and further techniques related to VHDL register design and digital systems, I would recommend consulting texts focusing on digital design using VHDL. Resources covering synchronous and asynchronous design techniques as well as texts exploring the practicalities of FPGA design, will give a more fundamental understanding of the trade-offs involved when implementing such interfaces. Examining online design documents associated with specific FPGA architectures will also reveal more specific device implementations that are highly optimised. Furthermore, design guides associated with many commercially available FPGA development boards will also contain examples for how to design these interfaces.

Through practical application and study, I've learned that a well-designed push-button register is critical for reliable system performance, especially when dealing with the complexities of real-world hardware interfaces.
