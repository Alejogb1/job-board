---
title: "How can I dim all but one torch?"
date: "2025-01-30"
id: "how-can-i-dim-all-but-one-torch"
---
The core challenge in selectively dimming multiple light sources, modeled here as "torches," lies in the independent control of each source's intensity.  A naive approach might involve manipulating global brightness settings, but this lacks the granularity required for precise individual control. My experience designing embedded systems for theatrical lighting control informs my approach to this problem.  The solution necessitates a system capable of addressing each torch individually, rather than treating them as a monolithic unit.

**1. Clear Explanation**

The fundamental requirement is a system with dedicated control channels for each torch. This can be achieved through several methods, each with varying levels of complexity and scalability.  The simplest approach involves using a separate control signal for each torch, typically through a parallel output interface. More sophisticated systems might leverage serial communication protocols like I2C or SPI, allowing for control of many torches with fewer physical lines.  However, even with serial communication, each torch still requires a unique address to distinguish it from others within the network.

Regardless of the communication method, a crucial aspect is the signal used to control brightness.  Analog control, using pulse-width modulation (PWM), offers smooth dimming capabilities.  PWM generates a square wave whose duty cycle (the proportion of time the signal is high) dictates the average power delivered to the torch, thus controlling its brightness.  A higher duty cycle translates to higher brightness, while a lower duty cycle results in dimmer light.  Digital control, while simpler to implement, offers less precise dimming and is generally less preferred for this application.

The algorithm controlling dimming can be straightforward. It involves setting the duty cycle of the PWM signal for each torch independently.  The selected torch will maintain a high duty cycle (e.g., 100%), while all others will have their duty cycle reduced to a desired low value (e.g., 0% for complete dimming, or a small percentage for a low-level glow).

**2. Code Examples with Commentary**

These examples illustrate different approaches to controlling multiple torches, focusing on PWM control.  Assume each torch is connected to a specific pin on a microcontroller.  For simplicity, I've omitted error handling and hardware initialization. These are crucial in production code but would obfuscate the core dimming logic.

**Example 1: Direct Port Manipulation (Microcontroller-Specific)**

This example uses direct port manipulation, a method common in low-level microcontroller programming. It is highly microcontroller-specific and lacks portability.  I've used a fictional microcontroller architecture for illustrative purposes.

```c
#include <regdef.h> // Fictional microcontroller header

// Define torch control pins
#define TORCH1 PORTB, 0
#define TORCH2 PORTB, 1
#define TORCH3 PORTB, 2

// Function to set PWM duty cycle for a specific torch
void setTorchBrightness(unsigned char torch, unsigned char brightness) {
    // Assume 8-bit PWM resolution (0-255)
    if (torch == 0) {
        PORTB |= (1 << 0); //Set the pin high
        // Fictional PWM register manipulation. Replace with actual code for your microcontroller.
        PWM_REG[0] = brightness;
    } else if (torch == 1) {
        PORTB |= (1 << 1);
        PWM_REG[1] = brightness;
    } else if (torch == 2) {
        PORTB |= (1 << 2);
        PWM_REG[2] = brightness;
    }
}


int main() {
    // Dim all torches except torch 1
    setTorchBrightness(0, 255); // Torch 1 at full brightness
    setTorchBrightness(1, 0);   // Torch 2 completely off
    setTorchBrightness(2, 0);   // Torch 3 completely off

    // ... rest of the main loop ...
    return 0;
}
```

**Example 2: Using a Timer and Interrupt (More Efficient)**

This approach leverages a microcontroller's timer and interrupt capabilities for more efficient PWM generation, allowing for simultaneous control of multiple torches without blocking the main program.  This code is more portable than direct port manipulation, though still requires adaptation for the specific microcontroller.


```c
#include <stdint.h> // Standard integer types
// Fictional timer and interrupt header
#include <timer_interrupt.h>

// Array to store brightness levels for each torch
uint8_t torchBrightness[3];

// Timer interrupt service routine (ISR)
void timerISR() {
    static uint8_t torchIndex = 0;

    // Check if the current torch needs updating
    if(torchBrightness[torchIndex] > 0){
        // Fictional PWM output toggling
        TOGGLE_PWM_OUTPUT(torchIndex);
    }

    torchIndex = (torchIndex + 1) % 3;
}

int main() {
  // Initialize timer and interrupt (Microcontroller specific)
  initTimerInterrupt();

  // Set brightness levels
  torchBrightness[0] = 255; // Torch 1 full brightness
  torchBrightness[1] = 0;   // Torch 2 off
  torchBrightness[2] = 0;   // Torch 3 off

  while (1) {
      // ... other program logic ...
  }
  return 0;
}

```

**Example 3:  Higher-Level Library Approach**

This approach uses a fictional higher-level library that abstracts away the low-level hardware details, making the code more portable and easier to read.

```c
#include "pwm_library.h" // Fictional PWM library

int main() {
  // Initialize PWM library (with microcontroller-specific parameters)
  pwm_init();

  // Set up 3 PWM channels
  pwm_setup(0, PWM_CHANNEL_1);
  pwm_setup(1, PWM_CHANNEL_2);
  pwm_setup(2, PWM_CHANNEL_3);


  // Set brightness levels.  Assume values 0-100%
  pwm_setDutyCycle(0, 100); // Torch 1 full brightness
  pwm_setDutyCycle(1, 0);   // Torch 2 off
  pwm_setDutyCycle(2, 0);   // Torch 3 off

  while (1) {
      // ... other program logic ...
  }
  return 0;
}
```


**3. Resource Recommendations**

For deeper understanding of microcontroller programming, consult introductory texts on embedded systems design and specific documentation for your chosen microcontroller.  Advanced topics such as real-time operating systems (RTOS) can greatly enhance the responsiveness and scalability of your lighting control system.  Relevant literature on PWM signal generation and its application in power control will be valuable.  Explore resources on various serial communication protocols, particularly I2C and SPI, for more complex multi-torch systems.  A thorough understanding of digital logic and electronic circuitry will also be beneficial.
