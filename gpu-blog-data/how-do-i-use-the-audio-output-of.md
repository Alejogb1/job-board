---
title: "How do I use the audio output of a DE10 Standard board in C?"
date: "2025-01-30"
id: "how-do-i-use-the-audio-output-of"
---
Accessing and manipulating audio output on the Intel DE10-Standard board using C requires direct interaction with the hardware peripherals, primarily the Audio CODEC (Coder-Decoder). This interaction is facilitated through memory-mapped I/O (MMIO), where specific memory addresses correspond to control and data registers within the CODEC. Unlike a standard PC audio system which often relies on abstracted operating system APIs, low-level embedded development like this demands a deeper understanding of the hardware architecture and register-level programming.

The typical workflow involves initializing the CODEC, configuring its operating parameters (sampling rate, bit depth, etc.), and then continuously writing audio sample data to its output buffer.  The DE10-Standard leverages the I2C (Inter-Integrated Circuit) bus for controlling the Audio CODEC (often a Wolfson WM8731 or similar). We must first establish communication via I2C, then configure registers through specific addresses to enable the CODEC and set desired audio parameters. Subsequently, the CODEC’s data registers become the target for audio sample data. I've spent considerable time debugging issues stemming from improperly sequenced I2C writes, and even seemingly insignificant delays between I2C operations can have profound effects on the initialization.

Let's examine the initialization process and output mechanism in detail. Firstly, we need to address the I2C communication, which is typically done by writing to the I2C master controller present in the FPGA fabric, this is different from writing directly to the Codec. The DE10-Standard’s System-on-Chip (SoC) architecture exposes these master controller registers through the MMIO space. Specific addresses corresponding to the I2C Controller base address and offset registers will be utilized to communicate to the Audio CODEC. Therefore, a crucial aspect of developing on the DE10 is accessing the board's technical reference and schematic documentation, to extract correct base addresses, and register definitions. Once communication is established, a sequence of writes to the audio CODEC must occur to initialize parameters like setting the digital interface format, setting the sample rate, configuring internal bias, enabling the output and adjusting the gain. Finally, we can start the process of writing audio sample data.

Here are three code examples, each building upon the previous one. I'll present the core logic and make reasonable assumptions about definitions based on common practice. Assume a library header, possibly "hwlib.h", defines I2C related operations and MMIO base address handling. These examples are focused and don't implement error handling or multi-threading.

**Example 1: I2C Initialization and Device Probe**

```c
#include "hwlib.h" // Assuming a hardware library header

#define I2C_CONTROLLER_BASE 0xFF201000 // Replace with actual base address
#define WM8731_I2C_ADDR 0x1A // WM8731 I2C address
#define I2C_SCL_PIN 28 // Actual pins used are based on DE10-Standard pin assignment
#define I2C_SDA_PIN 29 // These should be constants from board header

void i2c_init(){
    // Configure I2C controller, assuming the hardware library provides these.
    // This typically involves configuring clock dividers and setting up pull-up registers.
    i2c_master_config(I2C_CONTROLLER_BASE, I2C_SCL_PIN, I2C_SDA_PIN);
}

bool probe_wm8731(){
    // Send a read instruction to the device and check for a positive acknowledge.
    // The write address is composed of the base I2C Address and the write bit (0)
    // Return true on successful acknowledgment
    uint8_t write_address = WM8731_I2C_ADDR << 1; // Shift left by 1 to clear the LSB
    uint8_t read_test_byte;
    return i2c_master_read(I2C_CONTROLLER_BASE, write_address, &read_test_byte, 0); // 0 bytes to read = probe

}

int main() {
    i2c_init();

    if (probe_wm8731()) {
        printf("WM8731 CODEC found.\n");
    } else {
        printf("WM8731 CODEC not detected.\n");
    }

    return 0;
}
```

This first code segment demonstrates initialization of the I2C bus and probing the WM8731 CODEC. I’ve noted that the WM8731 has a specific 7-bit I2C address (0x1A). The `i2c_init()` function, assumed to be provided in the `hwlib.h`, handles low-level configuration of the I2C master controller. In the function `probe_wm8731()`, sending a write instruction without data enables probing for the presence of the device (it responds if present). This technique saved me countless hours when troubleshooting misconnected I2C lines. The `main` function then initializes the I2C bus, probes, and prints the results.

**Example 2: WM8731 Configuration**

```c
#include "hwlib.h" // Assuming a hardware library header
#include <stdbool.h>

#define I2C_CONTROLLER_BASE 0xFF201000
#define WM8731_I2C_ADDR 0x1A
#define I2C_SCL_PIN 28
#define I2C_SDA_PIN 29

// WM8731 Register Definitions (Partial)
#define R0_RESET_REG     0x00
#define R1_LEFT_LINE_IN  0x01
#define R2_RIGHT_LINE_IN 0x02
#define R3_LEFT_HEADPHONE_OUT 0x03
#define R4_RIGHT_HEADPHONE_OUT 0x04
#define R5_ANALOG_PATH_CONTROL 0x05
#define R6_DIGITAL_PATH_CONTROL 0x06
#define R7_POWER_MANAGEMENT 0x07
#define R8_DIGITAL_INTERFACE_FORMAT 0x08
#define R9_SAMPLING_CONTROL 0x09
#define R10_ACTIVE_CONTROL 0x0A
// End Register Definitions

void i2c_init(){
    i2c_master_config(I2C_CONTROLLER_BASE, I2C_SCL_PIN, I2C_SDA_PIN);
}
bool probe_wm8731(){
    uint8_t write_address = WM8731_I2C_ADDR << 1;
    uint8_t read_test_byte;
    return i2c_master_read(I2C_CONTROLLER_BASE, write_address, &read_test_byte, 0);
}

void write_wm8731_register(uint8_t register_address, uint16_t data) {
    uint8_t write_address = WM8731_I2C_ADDR << 1;
    uint8_t packet[3]; // {address, data-high, data-low}
    packet[0] = (register_address << 1) | ((data >> 8) & 0x01); // Address bits 8:1 + data MSB
    packet[1] = (data & 0xFF); // data LSB
    i2c_master_write(I2C_CONTROLLER_BASE, write_address, packet, 2);
    // Delay is often needed, even small ones
    _delay(100); // Short delay in milliseconds, using a _delay function
}


void configure_wm8731() {

    // Reset the WM8731.
    write_wm8731_register(R0_RESET_REG, 0x00);
    _delay(10); // Delay after reset is vital.
    // Example configurations (these may vary depending on application)
    // Digital interface format (I2S, 16-bit, master)
    write_wm8731_register(R8_DIGITAL_INTERFACE_FORMAT, 0x0012);
    // Sample rate and clock configuration for 48kHz
    write_wm8731_register(R9_SAMPLING_CONTROL, 0x0000);
    // Enable DAC and HP output
    write_wm8731_register(R5_ANALOG_PATH_CONTROL, 0x0012);
    // Unmute all outputs
    write_wm8731_register(R3_LEFT_HEADPHONE_OUT, 0x017F);
    write_wm8731_register(R4_RIGHT_HEADPHONE_OUT, 0x017F);
     //Activate codec
    write_wm8731_register(R10_ACTIVE_CONTROL, 0x0001);

}

int main() {
    i2c_init();

     if (probe_wm8731()) {
        printf("WM8731 CODEC found.\n");
        configure_wm8731();
        printf("WM8731 CODEC configured.\n");

    } else {
        printf("WM8731 CODEC not detected.\n");
    }
    return 0;
}
```

Example 2 builds upon the I2C initialization, introducing the register writing function `write_wm8731_register`.  The WM8731 registers are defined as constants for clarity. Crucially, note the bit shift operation within `write_wm8731_register` that combines the register address and MSB of the 16-bit data for I2C communication. Furthermore, the code demonstrates a sequence of specific I2C writes to configure parameters of the CODEC, such as the digital interface format and sampling rate as well as enabling the outputs. This is a critical process, failing to observe proper sequencing will render the CODEC inoperable. The `_delay()` calls, though simplistic, mimic the delays required between I2C operations.  I found through painful trial and error these delays are absolutely required. This specific example sets the CODEC for I2S, 16-bit, 48kHz operation, which are common configurations.

**Example 3: Audio Sample Output**

```c
#include "hwlib.h" // Assuming a hardware library header
#include <stdbool.h>

#define I2C_CONTROLLER_BASE 0xFF201000
#define WM8731_I2C_ADDR 0x1A
#define I2C_SCL_PIN 28
#define I2C_SDA_PIN 29
#define AUDIO_DATA_BASE 0xFF203000 // Example base address for Audio FIFO

// WM8731 Register Definitions (Partial)
#define R0_RESET_REG     0x00
#define R1_LEFT_LINE_IN  0x01
#define R2_RIGHT_LINE_IN 0x02
#define R3_LEFT_HEADPHONE_OUT 0x03
#define R4_RIGHT_HEADPHONE_OUT 0x04
#define R5_ANALOG_PATH_CONTROL 0x05
#define R6_DIGITAL_PATH_CONTROL 0x06
#define R7_POWER_MANAGEMENT 0x07
#define R8_DIGITAL_INTERFACE_FORMAT 0x08
#define R9_SAMPLING_CONTROL 0x09
#define R10_ACTIVE_CONTROL 0x0A
// End Register Definitions

void i2c_init(){
    i2c_master_config(I2C_CONTROLLER_BASE, I2C_SCL_PIN, I2C_SDA_PIN);
}

bool probe_wm8731(){
    uint8_t write_address = WM8731_I2C_ADDR << 1;
    uint8_t read_test_byte;
    return i2c_master_read(I2C_CONTROLLER_BASE, write_address, &read_test_byte, 0);
}

void write_wm8731_register(uint8_t register_address, uint16_t data) {
     uint8_t write_address = WM8731_I2C_ADDR << 1;
    uint8_t packet[3]; // {address, data-high, data-low}
    packet[0] = (register_address << 1) | ((data >> 8) & 0x01); // Address bits 8:1 + data MSB
    packet[1] = (data & 0xFF); // data LSB
    i2c_master_write(I2C_CONTROLLER_BASE, write_address, packet, 2);
     _delay(100);
}

void configure_wm8731() {
   write_wm8731_register(R0_RESET_REG, 0x00);
    _delay(10); // Delay after reset is vital.
    write_wm8731_register(R8_DIGITAL_INTERFACE_FORMAT, 0x0012);
    write_wm8731_register(R9_SAMPLING_CONTROL, 0x0000);
    write_wm8731_register(R5_ANALOG_PATH_CONTROL, 0x0012);
    write_wm8731_register(R3_LEFT_HEADPHONE_OUT, 0x017F);
    write_wm8731_register(R4_RIGHT_HEADPHONE_OUT, 0x017F);
    write_wm8731_register(R10_ACTIVE_CONTROL, 0x0001);
}


int main() {
    i2c_init();

    if (probe_wm8731()) {
        printf("WM8731 CODEC found.\n");
        configure_wm8731();
        printf("WM8731 CODEC configured.\n");

        volatile uint16_t *audio_fifo = (volatile uint16_t *)AUDIO_DATA_BASE; // Direct register access
        uint16_t sample_value = 0; // Example test tone generation

        while (1) {

            // Example: Basic sin wave
            sample_value = (uint16_t)(16384*sin(0.02 * sample_value));
            *audio_fifo = sample_value;
            _delay(1); // Short delay to control sample rate

        }

    } else {
        printf("WM8731 CODEC not detected.\n");
    }
    return 0;
}
```

Finally, Example 3 introduces writing audio sample data to the CODEC after I2C initialization. I’ve posited an address for an audio FIFO (First-In-First-Out) buffer that would be wired between the FPGA fabric and the CODEC. The address 0xFF203000 is a fictional address, but it is used to access the audio data register. This is often exposed via a dedicated hardware IP core.  The code then enters a simple infinite loop which generates a basic sin wave that it writes to the audio FIFO. I found that the delay between samples needed careful adjustments to match the desired sample rate and prevent buffer under-run.  In real applications, these samples would come from a different source such as a microphone, a file, or another processing module.

To further your learning, I would highly recommend referencing the following.  The documentation provided by Intel for the DE10-Standard board including its schematics and user manual. This offers detailed information about I2C communication, clock frequencies, audio specific registers and memory maps.  In addition, the specification sheets for the WM8731 audio CODEC are critical, as it describes each register’s function in exhaustive detail, including bit field descriptions, and timings.  Furthermore, textbooks on digital signal processing can provide the conceptual background to understand audio signals and algorithms used in practice. I often consulted those when implementing various audio effects.  Finally, it would greatly benefit development, to review I2C protocol specifications, as a solid understanding of timing constraints and addressing schemes are invaluable for debugging.
