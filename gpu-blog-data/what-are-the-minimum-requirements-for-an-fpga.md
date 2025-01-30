---
title: "What are the minimum requirements for an FPGA PLB slave module?"
date: "2025-01-30"
id: "what-are-the-minimum-requirements-for-an-fpga"
---
The core requirement for a functioning FPGA PLB (Processor Local Bus) slave module hinges on adhering to the PLB protocol's defined handshake mechanism.  My experience implementing numerous PLB-based systems across various Xilinx FPGA families underscores the importance of precise signal timing and proper register mapping.  Deviation from the specified protocol invariably leads to unpredictable behavior, data corruption, or complete system failure. This response will detail the minimum requirements, illustrated with code examples leveraging VHDL.

**1.  Clear Explanation:**

A PLB slave module, fundamentally, is a collection of registers accessible by a PLB master (typically a processor core).  The PLB protocol dictates a specific transaction sequence involving a series of control and data signals.  These signals must be correctly implemented to ensure reliable communication. At a minimum, a PLB slave needs:

* **PLB Interface Signals:** These are the signals that connect the slave module to the PLB bus.  They include, but aren't limited to:
    * `PADDR`:  Address input from the master indicating which register is being accessed.
    * `PWRITE`:  Write enable signal from the master.
    * `PWDATA`:  Write data from the master.
    * `PREADY`:  Slave readiness signal, indicating that it's ready to accept a transaction.
    * `PENABLE`:  Enable signal from the master initiating a transaction.
    * `PRESET`:  Reset signal for the slave.
    * `PREAD`: Read enable signal from the master.
    * `PRDATA`: Read data output to the master.
    * `PSLVERR`: Slave error signal.
    * `PPROT`: Protection signal indicating access permission levels.

* **Register Map:** This defines the addresses and widths of the accessible registers within the slave module.  A well-defined register map is crucial for predictability and maintainability. Each register must be assigned a unique address within the slave's address space.

* **Register Logic:** This comprises the combinational and sequential logic responsible for reading and writing to the registers, handling address decoding, and managing the PLB handshake. This logic processes the PLB interface signals, performs the requested operation (read or write), and asserts the appropriate response signals.

* **Address Decoding Logic:** This is responsible for determining whether the current address presented on the PLB bus corresponds to a register within the slave module's address space. If the address matches, the slave responds accordingly; otherwise, it remains inactive.  Efficient address decoding is crucial for preventing unintended access to registers.

Failure to properly implement any of these components will prevent the slave from functioning correctly within the PLB system.


**2. Code Examples with Commentary:**

**Example 1:  Simplified PLB Slave (VHDL)**

This example shows a minimal slave with a single 32-bit register.  It omits error handling and protection signals for brevity.

```vhdl
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity simple_plb_slave is
    port (
        clk : in std_logic;
        rst : in std_logic;
        plb_addr : in std_logic_vector(31 downto 0);
        plb_write : in std_logic;
        plb_writedata : in std_logic_vector(31 downto 0);
        plb_read : in std_logic;
        plb_enable : in std_logic;
        plb_ready : out std_logic;
        plb_readdata : out std_logic_vector(31 downto 0);
        my_register : inout std_logic_vector(31 downto 0)
    );
end entity;

architecture behavioral of simple_plb_slave is
    signal reg_value : std_logic_vector(31 downto 0) := x"00000000";
begin
    process (clk, rst)
    begin
        if rst = '1' then
            reg_value <= x"00000000";
            plb_ready <= '0';
        elsif rising_edge(clk) then
            if plb_enable = '1' then
                if plb_write = '1' then
                    reg_value <= plb_writedata;
                end if;
                plb_ready <= '1';
            else
                plb_ready <= '0';
            end if;
            if plb_read = '1' then
                plb_readdata <= reg_value;
            end if;

        end if;
    end process;
    my_register <= reg_value;
end architecture;
```

This code demonstrates the basic read and write operations.  `reg_value` holds the register's data.  `plb_ready` signals readiness.  Address decoding is implicitly assumed to be handled externally.


**Example 2: Incorporating Address Decoding**

This example adds address decoding for a single 32-bit register at address 0x1000.

```vhdl
-- ... (previous declarations) ...

architecture behavioral of plb_slave_with_decode is
    signal reg_value : std_logic_vector(31 downto 0) := x"00000000";
begin
    process (clk, rst)
    begin
        if rst = '1' then
            reg_value <= x"00000000";
            plb_ready <= '0';
        elsif rising_edge(clk) then
            if plb_enable = '1' and plb_addr = x"00001000" then --Address decode
                if plb_write = '1' then
                    reg_value <= plb_writedata;
                end if;
                plb_ready <= '1';
            else
                plb_ready <= '0';
            end if;
            if plb_read = '1' and plb_addr = x"00001000" then --Address decode
                plb_readdata <= reg_value;
            end if;
        end if;
    end process;
    my_register <= reg_value;
end architecture;
```

This enhances the previous example by only responding to transactions targeting address 0x1000.


**Example 3: Multi-Register Slave with Error Handling**

This example demonstrates a slave with multiple registers and basic error handling.

```vhdl
-- ... (previous declarations, expanded for multiple registers) ...

architecture behavioral of multi_reg_slave is
    type reg_array is array (0 to 3) of std_logic_vector(31 downto 0);
    signal registers : reg_array := (others => x"00000000");
    signal plb_err : std_logic := '0';
begin
    process (clk, rst)
    begin
        if rst = '1' then
            registers <= (others => x"00000000");
            plb_ready <= '0';
            plb_err <= '0';
        elsif rising_edge(clk) then
            plb_ready <= '0';
            plb_err <= '0';
            if plb_enable = '1' then
                case plb_addr(11 downto 2) is -- Assuming 10-bit address space within slave
                    when x"000" =>
                        if plb_write = '1' then
                            registers(0) <= plb_writedata;
                        end if;
                        plb_ready <= '1';
                    when x"001" =>
                        if plb_write = '1' then
                            registers(1) <= plb_writedata;
                        end if;
                        plb_ready <= '1';
                    when others =>
                        plb_err <= '1'; --Address out of range
                end case;
            end if;
            if plb_read = '1' then
                case plb_addr(11 downto 2) is
                    when x"000" => plb_readdata <= registers(0);
                    when x"001" => plb_readdata <= registers(1);
                    when others => plb_readdata <= x"00000000";
                end case;
            end if;
        end if;
    end process;
end architecture;

```

This example shows a more realistic implementation with multiple registers and error handling for out-of-range addresses.  Note that a more robust error handling mechanism would be needed in a production environment.


**3. Resource Recommendations:**

For a comprehensive understanding of the PLB protocol, consult the relevant documentation provided by the FPGA vendor (e.g., Xilinx's UGxxx documents on the PLB).  Study examples of PLB slave implementations within the vendor's example designs and reference designs.  Familiarize yourself with advanced VHDL concepts, specifically those related to concurrent signal assignments and finite state machines, crucial for implementing complex PLB slaves.  Finally, utilize a robust simulation environment (e.g., ModelSim, Vivado Simulator) to thoroughly test the design before implementation on the FPGA.
