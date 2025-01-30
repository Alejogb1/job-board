---
title: "How can I implement an I2C master to interface with a TMP007 sensor module?"
date: "2025-01-30"
id: "how-can-i-implement-an-i2c-master-to"
---
I’ve encountered the challenge of interfacing with the Texas Instruments TMP007 infrared thermopile sensor via I2C numerous times, specifically in embedded systems projects requiring precise non-contact temperature measurement. The core issue lies in orchestrating the I2C communication protocol to correctly configure the sensor and extract the raw temperature data. The TMP007, like many I2C peripherals, requires a specific sequence of register writes and reads to achieve this, which necessitates a carefully crafted driver implementation. I'll detail my approach, providing concrete code examples and contextual explanations.

The I2C protocol, at its essence, involves a master device (in our case, a microcontroller) sending addresses and commands over a two-wire serial bus (SDA and SCL) to one or more slave devices, in this instance, the TMP007.  The process always starts with a start condition, followed by the slave address (7 bits), then a read/write bit (0 for write, 1 for read). If writing, the master sends the register address followed by the data.  If reading, the master sends the register address, a repeated start condition, the slave address with read bit, and the slave returns the requested data. Crucially, I2C communication requires strict timing adherence and acknowledgement bits following each byte sent or received.  A failure in these can lead to communication failures.

The TMP007’s operational sequence begins with configuring the desired conversion rate and enabling the sensor. The configuration occurs via register address 0x02. The datasheet details the specific bit fields and their meanings. For instance, a value of `0x01` in the most significant bit enables the sensor and `0x00`, `0x01`, `0x02`, and `0x03` in bits 4-3 set the conversion rate to 16, 4, 1, and 1/4 Hz, respectively. After configuration, the raw temperature is available in registers 0x00 and 0x01. The reading from the two registers needs to be interpreted as a 16-bit twos-complement value, and further scaled as documented in the datasheet (multiplied by a factor of 0.0078125 to arrive at the final temperature in °C.) Failure to adhere to this sequence will result in incorrect or nonsensical temperature readings. Let’s now examine some concrete implementation examples to illustrate this process. I will use C, a language typically employed in embedded systems.

**Code Example 1: I2C Initialization and Helper Functions**

This first example demonstrates the fundamental setup for I2C communication. It does not interact with the TMP007 directly, instead establishing the underlying communication capability and providing helper functions to send and receive bytes over I2C.

```c
#include <stdint.h>
#include "my_i2c_library.h"  // Assume this contains the hardware-specific I2C functions

#define TMP007_ADDR 0x28  // 7-bit I2C address of the TMP007 sensor


void i2c_init() {
  // Hardware-specific I2C initialization (clock, pins, etc.)
  // Implementation depends on the target microcontroller.
  i2c_hardware_init(); // This would be implemented in my_i2c_library.h/.c
}

uint8_t i2c_send_byte(uint8_t address, uint8_t data) {
   //Send a start condition. Implementation details vary.
   i2c_start();
   if (i2c_write_byte((address << 1) | 0) == 0) // Address + write bit
    {
       return 0; //Nack
    }
    if (i2c_write_byte(data) == 0)
    {
      return 0; //Nack
    }
   i2c_stop();
    return 1; //Ack

}

uint8_t i2c_receive_byte(uint8_t address, uint8_t *data) {
  i2c_start();
  if (i2c_write_byte((address << 1) | 1) == 0)
  {
    return 0; //Nack
  }
    *data = i2c_read_byte();
    i2c_stop();
    return 1; //Ack
}

uint8_t i2c_send_register_address(uint8_t address, uint8_t reg_addr)
{
    // Send start condition
    i2c_start();
     if (i2c_write_byte((address << 1) | 0) == 0) {
         return 0; //Nack
     }
    // Send the register address to read from
    if (i2c_write_byte(reg_addr) == 0) {
          return 0; //Nack
    }
    return 1; //Ack
}
```

*Explanation:* This code initializes the I2C hardware and provides low-level byte-oriented send and receive functions. Note that actual hardware interaction is abstracted within `my_i2c_library.h/.c` (implementation details depend on the particular microcontroller used.) `i2c_send_byte` sends data to the specified device address, prepended with the I2C write bit. `i2c_receive_byte` issues a read request to the I2C device and returns the received byte. The `i2c_send_register_address` prepares the I2C device for a register read/write operation, by writing to slave's address and the register we want to interact with.

**Code Example 2: TMP007 Configuration**

The next code snippet shows how I enable the TMP007, configure its conversion rate, and demonstrate that using the provided functions, interaction with the TMP007 sensor is simplified.

```c
#define TMP007_CONFIG_REG 0x02 //Configuration register address

uint8_t tmp007_configure() {

  uint8_t config_data = 0x01;  // Set CR=0b01, Enable Conversion

  //Configure using 1Hz conversion rate.
  // config_data = config_data | (0x02<<3);

  if (i2c_send_register_address(TMP007_ADDR, TMP007_CONFIG_REG)==0) {
    return 0; //Nack
  }
  if (i2c_send_byte(0, config_data)==0){
     return 0;
   }
  return 1; //Ack
}

```

*Explanation:* This function sets the config register at address `TMP007_CONFIG_REG`, enabling the device with a conversion rate of 16Hz. To modify the configuration, bits 4 and 3 of the data can be set to `00`, `01`, `10`, or `11` to correspond to conversion rates of 16, 4, 1, or 1/4Hz. Crucially, I first send the address of the device and address register using the helper function, then the byte of configuration data.

**Code Example 3: Reading Raw Temperature Data**

This last code excerpt demonstrates how I read the raw temperature data from the TMP007.

```c
#define TMP007_TEMP_REG_MSB 0x00  // Temperature register MSB address
#define TMP007_TEMP_REG_LSB 0x01  // Temperature register LSB address

int16_t tmp007_read_temperature() {
  uint8_t msb, lsb;

   if (i2c_send_register_address(TMP007_ADDR, TMP007_TEMP_REG_MSB)==0) {
    return -1; //Nack
  }

  if( i2c_receive_byte(TMP007_ADDR, &msb) == 0){
    return -1; //Nack
  }

   if (i2c_send_register_address(TMP007_ADDR, TMP007_TEMP_REG_LSB)==0) {
     return -1; //Nack
   }

  if (i2c_receive_byte(TMP007_ADDR, &lsb) == 0){
    return -1; //Nack
  }


  int16_t raw_temp = (msb << 8) | lsb;

  return raw_temp;
}

float tmp007_convert_temperature(int16_t raw_temp) {
    float temp_c = raw_temp * 0.0078125;
    return temp_c;
}
```

*Explanation:*  The function `tmp007_read_temperature` retrieves the most significant byte (MSB) and least significant byte (LSB) of the raw temperature data, forming a 16-bit value and returning it as a signed integer. The conversion to a floating point value in Celsius is performed in `tmp007_convert_temperature`. Note that I first have to read register 0x00, then 0x01 separately because the data is split across two registers. I send the address of the register before receiving each byte of temperature data. Error handling returns -1 if there was a Nack. The final step in utilizing the data from the sensor is a conversion from the raw digital reading to temperature in degrees Celsius using the conversion factor provided in the sensor's datasheet. This scaling is performed via the `tmp007_convert_temperature` function.

For further study, I would recommend exploring Texas Instruments’ documentation for the TMP007 sensor itself. The datasheet is crucial for understanding the specific register maps, configuration options, and conversion factors. Additionally, studying the I2C bus specification (available from NXP) can enhance your understanding of the communication protocol itself. The book "Embedded Systems: Real-Time Interfacing to Arm Cortex-M Microcontrollers" by Jonathan Valvano is also highly recommended for those seeking in-depth knowledge of embedded systems and microcontroller interfaces. Finally, microcontroller manufacturers usually offer example code that will be useful as a starting point, although be aware that sometimes these examples can be unnecessarily convoluted.
