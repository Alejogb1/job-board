---
title: "How can VHDL code be used to read parallel ADC inputs?"
date: "2025-01-30"
id: "how-can-vhdl-code-be-used-to-read"
---
Synchronizing and capturing data from multiple parallel analog-to-digital converter (ADC) outputs in VHDL requires careful consideration of timing, data validity, and potential skew. I've encountered this issue multiple times in projects involving high-speed data acquisition systems. A successful implementation hinges on understanding the ADC's timing characteristics and effectively using VHDL's capabilities to create a reliable capture mechanism.

Fundamentally, parallel ADCs output multiple bits simultaneously representing the converted analog voltage. These bits change on a clock edge provided either by the ADC or externally. My experience has shown that the critical aspects to address in VHDL are: firstly, ensuring the capture clock used to latch the ADC data is synchronized with the ADC's data valid signal (if one exists); secondly, properly handling the bit width of the parallel output; and thirdly, if multiple ADCs are used, creating a structured approach for accessing each of their data streams. The goal in VHDL is to create a register that stores the parallel output data when it's valid and accessible for downstream processing.

Consider a simplified scenario: a single 8-bit ADC whose output is sampled on the rising edge of a clock signal, ‘clk’. Furthermore, the ADC provides a ‘data_valid’ signal that indicates when the output is stable and ready to be read. This is a crucial aspect for reliable data capture. The following code illustrates this basic implementation.

```vhdl
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity adc_interface is
    Port ( clk       : in  std_logic;
           adc_data  : in  std_logic_vector(7 downto 0);
           data_valid : in std_logic;
           captured_data : out std_logic_vector(7 downto 0)
          );
end entity adc_interface;

architecture behavioral of adc_interface is
    signal internal_captured_data : std_logic_vector(7 downto 0);
begin
    process(clk)
    begin
        if rising_edge(clk) then
           if data_valid = '1' then
              internal_captured_data <= adc_data;
           end if;
        end if;
    end process;

    captured_data <= internal_captured_data;

end architecture behavioral;
```

In this code, a synchronous process, sensitive to the rising edge of the `clk` signal, captures the `adc_data` into an internal register, `internal_captured_data`, only when the `data_valid` signal is high. This is essential because relying solely on the clock could result in incorrect data readings if the ADC output is still changing when the clock edge occurs. The captured data is then made available at the output `captured_data`. This pattern of synchronization via a data valid signal is common in hardware interfacing and one I consistently use in high-speed acquisition systems. This approach prioritizes data integrity over absolute timing precision, accepting a one-clock cycle latency to achieve reliable readings.

Next, let's examine a scenario involving two 12-bit ADCs. This time, assume no explicit data valid signal is provided; instead, the data is valid one clock cycle after a conversion start signal, ‘conv_start’, from the ADC. The system requires a single capture clock, and hence a delay to match the ADC data settling time.

```vhdl
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity dual_adc_interface is
    Port ( clk          : in  std_logic;
           adc1_data    : in  std_logic_vector(11 downto 0);
           adc2_data    : in  std_logic_vector(11 downto 0);
           conv_start   : in  std_logic;
           captured_data_adc1 : out std_logic_vector(11 downto 0);
           captured_data_adc2 : out std_logic_vector(11 downto 0)
          );
end entity dual_adc_interface;

architecture behavioral of dual_adc_interface is
    signal internal_captured_data_adc1 : std_logic_vector(11 downto 0);
    signal internal_captured_data_adc2 : std_logic_vector(11 downto 0);
    signal data_ready: std_logic;
begin

   process(clk)
    begin
        if rising_edge(clk) then
            data_ready <= conv_start;
          if (data_ready = '1') then
              internal_captured_data_adc1 <= adc1_data;
              internal_captured_data_adc2 <= adc2_data;
          end if;
        end if;
    end process;


    captured_data_adc1 <= internal_captured_data_adc1;
    captured_data_adc2 <= internal_captured_data_adc2;


end architecture behavioral;
```

Here, the `conv_start` signal is internally delayed by one clock cycle using a flip-flop implemented through `data_ready`. On the subsequent clock cycle, when `data_ready` is high, the ADC outputs are captured. The key here is the synchronisation achieved by using a sequential element. This approach relies on the ADC’s timing specifications, as it's crucial that the data is valid one clock cycle after the conversion start, and this must be carefully verified against the manufacturer’s data sheet, a step I've learned to be meticulous about. Separate internal registers, `internal_captured_data_adc1` and `internal_captured_data_adc2` are used to hold the captured data for each ADC. It exemplifies the approach used for handling multiple parallel inputs.

Finally, consider a more complex scenario with a data valid signal for each ADC. The code structure in this case would involve multiple parallel processes, each dealing with one ADC.

```vhdl
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity multi_adc_interface is
    generic ( NUM_ADCS : positive := 4;
              ADC_WIDTH : positive := 10);
    Port ( clk            : in  std_logic;
           adc_data     : in  std_logic_vector(NUM_ADCS*ADC_WIDTH -1 downto 0);
           data_valid : in std_logic_vector(NUM_ADCS -1 downto 0);
           captured_data : out std_logic_vector(NUM_ADCS*ADC_WIDTH -1 downto 0)
          );
end entity multi_adc_interface;

architecture behavioral of multi_adc_interface is
    signal internal_captured_data : std_logic_vector(NUM_ADCS*ADC_WIDTH -1 downto 0);

begin

    gen_adc : for i in 0 to NUM_ADCS -1 generate
        signal internal_data: std_logic_vector(ADC_WIDTH-1 downto 0);
     begin
        process(clk)
        begin
            if rising_edge(clk) then
                if data_valid(i) = '1' then
                   internal_data <= adc_data((i+1)*ADC_WIDTH -1 downto (i*ADC_WIDTH));
                    internal_captured_data((i+1)*ADC_WIDTH -1 downto (i*ADC_WIDTH)) <= internal_data;
                end if;
            end if;
        end process;
        end generate;

    captured_data <= internal_captured_data;

end architecture behavioral;
```

This example leverages the VHDL generate statement, allowing the code to be parameterized based on the number of ADCs and their bit width. The use of indexed signals makes the code more scalable and easier to maintain as it avoids repetition in the code. Each process captures its associated ADC data when the corresponding ‘data_valid’ signal is asserted and stores it into a section of the `internal_captured_data`. This structured approach is beneficial for large-scale systems, which often contain numerous ADCs.

Implementing parallel ADC interfaces in VHDL requires a strong understanding of the ADC's timing, as detailed in its datasheet. The design must address potential issues such as setup and hold times of the ADC, clock domain crossings if the ADC operates at a different frequency, and skew between data signals. For further study, I would recommend examining literature on digital logic design, specifically focusing on synchronous design techniques, and also VHDL design methodologies used for high-speed data acquisition systems. Texts covering signal integrity considerations are also valuable to ensure that the captured data reflects the true analog input. Finally, researching vendor-specific documentation for ADC chips is critical to understand particular timing considerations.
