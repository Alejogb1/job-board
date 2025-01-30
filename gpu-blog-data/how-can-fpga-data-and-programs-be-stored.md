---
title: "How can FPGA data and programs be stored persistently?"
date: "2025-01-30"
id: "how-can-fpga-data-and-programs-be-stored"
---
An FPGA's inherent reconfigurability, while powerful, presents a challenge for persistent storage of its configuration data and the programs that execute within it. Unlike processors that load software from non-volatile memory upon power-up, an FPGA's logic fabric is effectively blank until a configuration bitstream is loaded. The bitstream, containing the detailed instructions for interconnecting the configurable logic blocks (CLBs), lookup tables (LUTs), and other resources, must be stored externally and reloaded after each power cycle or reconfiguration. Over my years developing embedded systems incorporating FPGAs, I've encountered various strategies to address this need.

The most common method for persistent storage relies on external non-volatile memory (NVM) connected to the FPGA. Typically, this involves using flash memory, such as NOR flash or SPI flash, due to their relatively high density, low cost, and widespread availability. The FPGA boot process is designed to access this flash memory on power-up, read the appropriate bitstream, and program itself. The flash memory effectively acts as a persistent repository for the FPGA's personality. Additionally, I’ve observed that some FPGAs incorporate internal NVM, albeit usually limited in size. This is often used for storing very small configuration files or acting as a second-stage bootloader to minimize external dependencies.

Beyond the simple bitstream, application-specific data generated or used by the FPGA often needs persistent storage. This is handled differently depending on the volume and required access speeds. Low-volume, infrequently updated parameters might reside in the same NVM as the bitstream, or a dedicated section within it. Larger datasets, especially those used actively during FPGA operation, are better stored on separate memory devices. In my experience, I've found that NAND flash, or even external RAM with battery backup, are suitable options, depending on the particular system's performance needs. Careful planning regarding data storage mapping within these devices is important as is selecting a device with sufficient endurance and suitable access time characteristics for the application.

The interaction between the FPGA and the external NVM is mediated by a configuration interface, frequently adhering to a standard protocol. For SPI flash, the Serial Peripheral Interface (SPI) is common, while parallel NOR flash devices require a wider bus. An interface controller is needed on the FPGA fabric, or built into the FPGA’s silicon directly, responsible for initiating reads from the NVM and programming the configuration cells of the FPGA. This interface is frequently implemented within the FPGA design and is usually part of an initialization sequence automatically executed upon power-up or by explicitly initiated through the FPGA system. The selection of the NVM type has a major impact on this interface implementation.

Here are several code examples, showcasing various approaches to persistent storage. For brevity these examples are conceptual. The specific implementation details will always be dependent on the FPGA platform and connected peripherals.

**Example 1: Basic Bitstream Loading from SPI Flash (Conceptual VHDL)**

This example illustrates the conceptual behavior of an FPGA interface designed to load a configuration bitstream from SPI flash memory. Specific implementation may vary based on the SPI flash device specifics.

```vhdl
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity spi_flash_loader is
    port (
        clk         : in  std_logic; -- System clock
        reset_n     : in  std_logic; -- Active low reset
        spi_sclk    : out std_logic; -- SPI clock
        spi_mosi    : out std_logic; -- SPI master out slave in
        spi_miso    : in  std_logic; -- SPI master in slave out
        spi_cs_n    : out std_logic; -- SPI chip select (active low)
        load_done : out std_logic  -- Indicates bitstream load completion
    );
end entity spi_flash_loader;

architecture behavioral of spi_flash_loader is

    -- Internal states and constants.

    type state_type is (IDLE, SEND_CMD, READ_DATA, CHECK_DONE);
    signal current_state : state_type := IDLE;

    constant READ_CMD  : std_logic_vector(7 downto 0) := x"03"; -- SPI read command
    constant START_ADDR : std_logic_vector(23 downto 0) := x"000000";-- Start Address

    signal spi_data_out : std_logic_vector(7 downto 0);
    signal spi_data_in  : std_logic_vector(7 downto 0);
    signal bitstream_counter : integer := 0;
    signal data_valid : std_logic := '0';

begin
    process(clk)
    begin
        if rising_edge(clk) then
            if reset_n = '0' then
                current_state <= IDLE;
                spi_sclk <= '0';
                spi_cs_n <= '1';
                load_done <= '0';
                 bitstream_counter <= 0;
            else
                case current_state is
                    when IDLE =>
                       spi_cs_n <= '0';
                        spi_data_out <= READ_CMD;
                        current_state <= SEND_CMD;

                    when SEND_CMD =>
                        -- Send read command, address etc... over SPI
                        if (bitstream_counter < 4) then -- command and address
                             spi_sclk <= not spi_sclk;
                            if spi_sclk = '1' then
                               if bitstream_counter = 0 then
                                  spi_data_out <=  READ_CMD;
                                elsif bitstream_counter = 1 then
                                  spi_data_out <= START_ADDR(23 downto 16);
                                elsif bitstream_counter = 2 then
                                  spi_data_out <= START_ADDR(15 downto 8);
                                elsif bitstream_counter = 3 then
                                   spi_data_out <= START_ADDR(7 downto 0);
                                end if;
                              bitstream_counter <= bitstream_counter + 1;
                            end if;
                        else
                            bitstream_counter <= 0;
                            current_state <= READ_DATA;
                            spi_sclk <= '0';
                        end if;


                    when READ_DATA =>
                        -- Read data bytes from SPI flash and pass to the FPGA configuration module
                         spi_sclk <= not spi_sclk;
                         if spi_sclk = '1' then
                             spi_data_in <= spi_miso; -- Sample the data
                              -- Simulate that the configuration process uses the received data, not directly write into memory
                            data_valid <= '1';
                           if bitstream_counter > 10000 then
                                current_state <= CHECK_DONE;
                           end if;
                           bitstream_counter <= bitstream_counter + 1;
                         end if;

                    when CHECK_DONE =>
                        spi_cs_n <= '1';
                        load_done <= '1';
                        current_state <= IDLE;

                    when others =>
                         current_state <= IDLE;
                end case;
                if data_valid = '1' then
                   -- configuration process of FPGA.
                   data_valid <= '0';
               end if;
                spi_mosi <= spi_data_out(7);
            end if;
        end if;
    end process;
end architecture behavioral;

```

This example presents a simplified state machine implementation. It begins in IDLE, transitions to SEND_CMD to transmit the SPI read command and start address, then enters the READ_DATA state to continuously receive data from the SPI flash and passes data for FPGA configuration. Finally, the CHECK_DONE state de-asserts the chip select and flags load completion. The actual configuration of the FPGA hardware itself is not included for brevity.

**Example 2: Storing FPGA Application Data in External RAM (Conceptual C/C++ on Host)**

This second example showcases conceptual code running on a host processor, that communicates with the FPGA, to read and write data from an external RAM using an interface over PCIe.

```c++
#include <iostream>
#include <vector>
#include <unistd.h> //for usleep
// Assume a library with helper function is available to access PCIe and handle transactions
#include "fpga_pcie_lib.h"

const uint64_t RAM_BASE_ADDR = 0x00000000; // Start address of external RAM as seen by FPGA
const uint64_t DATA_SIZE     = 1024; // 1 KB data

int main() {
    FPGADevice fpgaDevice; // Assume this is already initialised by the library

    std::vector<uint8_t> data_to_write(DATA_SIZE);
     std::vector<uint8_t> data_read_back(DATA_SIZE);

    // Initialize some data to be written
    for (size_t i = 0; i < DATA_SIZE; ++i) {
       data_to_write[i] = (uint8_t)i;
    }

    // Write data to external RAM
    std::cout << "Writing data to FPGA external RAM..." << std::endl;
    if (fpgaDevice.writeData(RAM_BASE_ADDR, data_to_write.data(), DATA_SIZE) != 0) {
        std::cerr << "Error writing data to RAM." << std::endl;
        return 1;
    }
     usleep(100000);

    // Read data back from external RAM
    std::cout << "Reading data from FPGA external RAM..." << std::endl;
     if (fpgaDevice.readData(RAM_BASE_ADDR, data_read_back.data(), DATA_SIZE) != 0){
       std::cerr << "Error reading data from RAM." << std::endl;
       return 1;
    }
     usleep(100000);

    // Verify the data
    bool data_match = true;
     for (size_t i = 0; i < DATA_SIZE; i++) {
        if (data_to_write[i] != data_read_back[i]) {
            data_match = false;
           break;
        }
     }

    if (data_match) {
        std::cout << "Data verification successful." << std::endl;
    } else {
       std::cout << "Data verification failed." << std::endl;
    }

    return 0;
}
```
This example highlights data transfer between a host processor and the FPGA. The host sends data to the FPGA which is written to an external RAM. A subsequent read is then performed to verify the data. The `FPGADevice` class is conceptual and encapsulates the interaction with the FPGA over PCIe. The actual interaction is dependent on the specific FPGA and interfaces.

**Example 3: Storing FPGA Parameters in Configuration Flash (Conceptual C)**

This example demonstrates how firmware code running on the FPGA’s embedded processor (e.g., a soft core processor such as MicroBlaze or NIOS) might read configuration parameters from flash memory after boot.

```c
#include <stdint.h>
#include <stdio.h>

// Assume there are some flash access functions available.
extern int flash_read(uint32_t address, uint8_t *data, uint32_t size);

#define CONFIG_OFFSET 0x100000 // Starting address for configuration parameters
#define CONFIG_PARAM1_OFFSET (CONFIG_OFFSET + 0)
#define CONFIG_PARAM2_OFFSET (CONFIG_OFFSET + 4)

typedef struct {
    uint32_t parameter1;
    uint32_t parameter2;
} FPGA_config_t;

FPGA_config_t config_data;

void load_parameters_from_flash(){
    if (flash_read(CONFIG_PARAM1_OFFSET, (uint8_t *)&config_data.parameter1, sizeof(uint32_t)) != 0) {
      printf("Error loading parameter 1 from Flash.\n");
      return;
    }
   if (flash_read(CONFIG_PARAM2_OFFSET, (uint8_t *)&config_data.parameter2, sizeof(uint32_t)) != 0) {
      printf("Error loading parameter 2 from Flash.\n");
      return;
    }
}

int main(){

    load_parameters_from_flash();

    printf("FPGA Configuration: \n");
    printf("Param 1 = %lu \n",config_data.parameter1);
    printf("Param 2 = %lu \n",config_data.parameter2);

    //Use parameters to configure FPGA system.

    return 0;
}
```
This example outlines how parameters stored in flash can be loaded by the FPGA’s embedded processor during the program’s start up. The code uses an assumed `flash_read` function to read the configuration parameters from defined memory locations. These parameters are then utilized to initialize the FPGA system.

For learning more about FPGA persistent storage, I'd recommend consulting resources related to specific FPGA vendor's documentation like Xilinx or Intel. Further, delving into embedded system design books is advisable, which cover various memory technology options in addition to protocols like SPI and PCIe. Understanding the intricacies of board design guides and configuration memory datasheets for your chosen components are also crucial. Finally, research into common embedded software development practices, covering aspects such as file system storage on memory devices can also prove beneficial. The appropriate selection depends on the needs of the application.
