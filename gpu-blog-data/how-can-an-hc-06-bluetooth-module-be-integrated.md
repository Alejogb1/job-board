---
title: "How can an HC-06 Bluetooth module be integrated with an FPGA?"
date: "2025-01-30"
id: "how-can-an-hc-06-bluetooth-module-be-integrated"
---
Integrating an HC-06 Bluetooth module with a Field-Programmable Gate Array (FPGA) necessitates a careful consideration of the differences in their communication protocols and data handling capabilities. The HC-06, a serial communication module, primarily uses UART, while FPGAs, at their core, operate on logic levels and require a defined interface for handling incoming and outgoing data. My experience with embedded systems, including several projects involving custom communication protocols on FPGAs, has underscored the importance of addressing timing, synchronization, and data representation when bridging such divergent architectures.

The core challenge is translating the asynchronous serial data from the HC-06 into a format the FPGA can reliably process, and vice versa. This requires constructing a custom UART receiver and transmitter logic within the FPGA’s programmable fabric. The HC-06 typically operates at TTL voltage levels (0-3.3V or 0-5V), making it mostly compatible with standard FPGA I/O banks. However, electrical considerations, such as pull-up or pull-down resistors, should be reviewed based on the FPGA development board’s specifications and the HC-06’s datasheet. The interaction revolves around three primary signal lines: RX (receive), TX (transmit), and GND (ground). The VCC line for power is assumed to be provided separately and is not a focus of this discussion.

For the FPGA to receive data from the HC-06, we must implement a UART receiver module. This involves detecting the start bit (a logic low level), sampling the incoming data line at the correct baud rate (typically 9600, 38400, or 115200), and reassembling the data bits into a byte. A common approach involves an oversampling technique to reduce the chance of sampling in the middle of signal transitions. The following VHDL snippet demonstrates a basic UART receiver:

```vhdl
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity uart_rx is
    generic (
        CLOCK_FREQUENCY : integer := 50000000;  -- System clock frequency
        BAUD_RATE       : integer := 115200   -- Baud rate of HC-06
    );
    port (
        clk     : in  std_logic;
        rx_in   : in  std_logic; -- Input from HC-06 RX line
        data_out : out std_logic_vector(7 downto 0); -- Received byte
        data_valid : out std_logic -- Indicates new data is available
    );
end entity uart_rx;

architecture behavioral of uart_rx is
    constant OVERSAMPLE_RATE : integer := 16; -- Oversampling factor
    constant TICKS_PER_BIT  : integer := CLOCK_FREQUENCY / (BAUD_RATE * OVERSAMPLE_RATE);
    type state_type is (IDLE, START_BIT, DATA_BITS, STOP_BIT);
    signal current_state : state_type := IDLE;
    signal bit_counter : integer range 0 to 10 := 0; -- 1 start, 8 data, 1 stop
    signal sample_counter : integer range 0 to TICKS_PER_BIT - 1 := 0;
    signal rx_data_reg : std_logic_vector(7 downto 0);
    signal data_valid_int : std_logic := '0';

begin
    process(clk)
    begin
        if rising_edge(clk) then
            case current_state is
                when IDLE =>
                    data_valid_int <= '0';
                    if rx_in = '0' then -- Start bit detected
                        current_state <= START_BIT;
                        sample_counter <= 0;
                        bit_counter <= 0;
                    end if;

                when START_BIT =>
                    sample_counter <= sample_counter + 1;
                    if sample_counter = TICKS_PER_BIT/2 then
                      if rx_in = '0' then
                        current_state <= DATA_BITS;
                        sample_counter <= 0;
                      else
                       current_state <= IDLE;
                      end if;
                    elsif sample_counter = TICKS_PER_BIT - 1 then
                      sample_counter <= 0;
                    end if;

                when DATA_BITS =>
                    sample_counter <= sample_counter + 1;
                    if sample_counter = TICKS_PER_BIT - 1 then
                      sample_counter <= 0;
                      rx_data_reg(bit_counter) <= rx_in;
                      bit_counter <= bit_counter + 1;
                        if bit_counter = 8 then
                          current_state <= STOP_BIT;
                        end if;
                     end if;

                when STOP_BIT =>
                    sample_counter <= sample_counter + 1;
                    if sample_counter = TICKS_PER_BIT/2 then
                       if rx_in = '1' then
                        current_state <= IDLE;
                        data_valid_int <= '1';
                       else
                        current_state <= IDLE;
                        data_valid_int <= '0';
                       end if;
                    elsif sample_counter = TICKS_PER_BIT - 1 then
                     sample_counter <= 0;
                    end if;

            end case;
            data_out <= rx_data_reg;
            data_valid <= data_valid_int;
         end if;
     end process;

end architecture behavioral;
```

This VHDL code implements a finite state machine to capture each bit of the incoming serial stream. The `OVERSAMPLE_RATE` parameter increases sampling frequency to better mitigate noise.  After receiving the stop bit, the `data_valid` signal is asserted, indicating that a complete byte has been received and stored in `data_out`.

Conversely, sending data from the FPGA to the HC-06 requires a UART transmitter module. This process involves taking a byte of data, serializing it into a stream of bits, and outputting it on the FPGA's TX line. The following VHDL snippet illustrates a basic UART transmitter:

```vhdl
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity uart_tx is
    generic (
        CLOCK_FREQUENCY : integer := 50000000;  -- System clock frequency
        BAUD_RATE       : integer := 115200   -- Baud rate of HC-06
    );
    port (
        clk      : in  std_logic;
        tx_out   : out std_logic; -- Output to HC-06 TX line
        data_in  : in  std_logic_vector(7 downto 0); -- Byte to be transmitted
        data_ready : in std_logic; -- Data available flag
        tx_busy : out std_logic -- Indicates if module is busy transmitting data
    );
end entity uart_tx;

architecture behavioral of uart_tx is
  constant TICKS_PER_BIT  : integer := CLOCK_FREQUENCY / BAUD_RATE;
  type state_type is (IDLE, START_BIT, DATA_BITS, STOP_BIT);
  signal current_state : state_type := IDLE;
  signal bit_counter : integer range 0 to 7 := 0;
  signal sample_counter : integer range 0 to TICKS_PER_BIT - 1 := 0;
  signal tx_data_reg : std_logic_vector(7 downto 0);
  signal tx_busy_int : std_logic := '0';

begin
    process(clk)
    begin
        if rising_edge(clk) then
            case current_state is
                when IDLE =>
                    tx_out <= '1'; -- Idle state (high)
                    tx_busy_int <= '0';
                    if data_ready = '1' then
                      tx_data_reg <= data_in;
                       current_state <= START_BIT;
                       sample_counter <= 0;
                       bit_counter <= 0;
                       tx_busy_int <= '1';
                    end if;

                when START_BIT =>
                    sample_counter <= sample_counter + 1;
                     tx_out <= '0';
                    if sample_counter = TICKS_PER_BIT - 1 then
                       sample_counter <= 0;
                        current_state <= DATA_BITS;
                     end if;

                when DATA_BITS =>
                     sample_counter <= sample_counter + 1;
                     tx_out <= tx_data_reg(bit_counter);
                     if sample_counter = TICKS_PER_BIT - 1 then
                         sample_counter <= 0;
                         bit_counter <= bit_counter + 1;
                         if bit_counter = 8 then
                             current_state <= STOP_BIT;
                          end if;
                     end if;

                when STOP_BIT =>
                     sample_counter <= sample_counter + 1;
                     tx_out <= '1';
                     if sample_counter = TICKS_PER_BIT - 1 then
                        sample_counter <= 0;
                        current_state <= IDLE;
                        tx_busy_int <= '0';
                     end if;
            end case;
           tx_busy <= tx_busy_int;
        end if;
     end process;

end architecture behavioral;
```

This transmitter takes input `data_in` when `data_ready` is high and transmits it serially on the `tx_out` pin. The `tx_busy` signal is asserted to indicate when the transmitter is in operation. Both receiver and transmitter modules utilize the same `CLOCK_FREQUENCY` and `BAUD_RATE` generics for timing consistency.

Finally, consider the integration of these two modules within an FPGA design. A simplified example using VHDL illustrates this:

```vhdl
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity top_level is
    port (
        clk     : in  std_logic;
        rx_in   : in  std_logic; -- Input from HC-06 RX
        tx_out  : out std_logic; -- Output to HC-06 TX
        led_out : out std_logic_vector(7 downto 0) -- Example output LEDs
    );
end entity top_level;

architecture behavioral of top_level is
  signal rx_data : std_logic_vector(7 downto 0);
  signal rx_valid : std_logic;
  signal tx_data : std_logic_vector(7 downto 0);
  signal tx_ready : std_logic := '0';
  signal tx_busy : std_logic;

begin
    -- Instantiate UART Receiver
   rx_module : entity work.uart_rx
        generic map(
           CLOCK_FREQUENCY => 50000000,
           BAUD_RATE => 115200
        )
        port map(
            clk => clk,
            rx_in => rx_in,
            data_out => rx_data,
            data_valid => rx_valid
        );

    -- Instantiate UART Transmitter
    tx_module : entity work.uart_tx
        generic map(
           CLOCK_FREQUENCY => 50000000,
           BAUD_RATE => 115200
        )
        port map(
           clk => clk,
           tx_out => tx_out,
           data_in => tx_data,
           data_ready => tx_ready,
           tx_busy => tx_busy
        );

   process (clk)
   begin
      if rising_edge(clk) then
         if rx_valid = '1' then
            led_out <= rx_data; -- Update LEDs with received data
            tx_data <= rx_data; -- Echo the received data
            tx_ready <= '1';
          elsif tx_busy = '0' then
           tx_ready <= '0';
         end if;
      end if;
   end process;

end architecture behavioral;
```

In this top-level design, the `uart_rx` and `uart_tx` modules are instantiated. Upon reception of valid data (`rx_valid = '1'`), the received byte is displayed on LEDs and transmitted back to the HC-06 (echo functionality). This demonstrates a basic bidirectional communication setup.  Care should be taken when integrating multiple modules as this implementation is highly simplified for demonstration purposes.

For further study, I recommend focusing on resources that detail digital design principles, such as those found in standard textbooks on digital logic and computer architecture. Additionally,  manufacturer documentation specific to your FPGA family (Xilinx, Intel, Lattice, etc.) and associated development boards provides valuable information on I/O characteristics and clocking resources. Exploring detailed UART tutorials will also improve your grasp of its intricacies.  Finally, working through open-source FPGA projects that include UART communication provides practical context and a wealth of reusable code.
