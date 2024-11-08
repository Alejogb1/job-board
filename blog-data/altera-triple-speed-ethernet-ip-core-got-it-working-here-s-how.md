---
title: "Altera Triple Speed Ethernet IP Core:  Got it working, here's how!"
date: '2024-11-08'
id: 'altera-triple-speed-ethernet-ip-core-got-it-working-here-s-how'
---

```vhdl
--  Check the clock and reset signals using spare pins and an oscilloscope.
--  Verify the clock frequency and duty cycle are correct.
--  Ensure the reset signal is active low and its timing is correct.

--  Double-check the power supply for the Ethernet core and the daughter card.
--  Verify that the power supply meets the requirements for the core and the transceiver.

--  Review the pin assignments for the Ethernet core and the daughter card.
--  Ensure that all pins are correctly connected and that the pin standards are properly configured.

--  Verify the TSE core's configuration settings, especially in System Console.
--  Ensure the core's registers are initialized correctly, including the MAC address, speed and duplex settings.

--  Test the packet generator module independently.
--  Verify that the module can generate and transmit valid Ethernet packets.

--  Use a logic analyzer to capture the signals on the MII/GMII interfaces.
--  Analyze the signals to determine if any errors or inconsistencies exist.

--  If using a reconfig block, ensure it is properly configured and functioning.

--  If necessary, use a debug tool such as SignalTap to monitor the internal signals of the Ethernet core.
--  This can help identify any issues that are not visible on the external signals.
```
