---
title: "Why can't iCEprog find the FTDI USB device on the Alchitry CU?"
date: "2025-01-30"
id: "why-cant-iceprog-find-the-ftdi-usb-device"
---
The inability of iCEprog to detect the FTDI USB device on an Alchitry CU board typically stems from a combination of driver conflicts, incorrect device configurations, and, occasionally, hardware-related issues. Having encountered this problem numerous times across various FPGA development boards during my tenure in embedded systems, I've found that a methodical approach to troubleshooting is crucial. The FTDI chip, specifically the FT2232H or similar variant often utilized in these contexts, acts as a critical bridge for communication between the host computer and the FPGA. When this bridge malfunctions, programming and debugging become impossible.

Firstly, it's imperative to understand the driver interaction. The FTDI chip needs an appropriate driver installed on the host computer to expose the device as a serial port (or multiple virtual ports). This driver is typically provided by FTDI directly or integrated into the operating system. Conflicts often arise when multiple versions of FTDI drivers exist on the system, or when a different serial port driver (such as a driver from a virtual serial port application) incorrectly claims the device. These conflicts can prevent iCEprog from establishing a connection. The operating system, upon connecting a new USB device, attempts to assign a driver. This assignment can fail if the driver isn't present or if there are competing drivers. The result is that the USB device may enumerate, appear in Device Manager, but without being correctly associated with a functional serial port.

Another factor is the configuration of the FTDI chip itself. While uncommon on Alchitry CU boards due to their relatively standardized configuration, misconfiguration is a possible cause. This configuration is usually set by the manufacturer and allows the FTDI chip to operate in different modes (e.g., UART, FIFO, MPSSE). If the firmware on the chip becomes corrupted or is accidentally changed, communication problems will occur. Less frequently, the FTDI chip itself may be faulty. This would be a hardware-level issue and require a different type of investigation, such as an examination of voltage levels, or using a different computer to diagnose.

Furthermore, the Alchitry CU has a USB micro-B connector that is used for both power and communication. A damaged cable, a faulty USB port on the computer, or a power issue can interfere with the ability of the computer to enumerate the FTDI device. A poor connection at the board could create intermittent communication, which would result in the iCEprog program failing to detect the device on repeated connection attempts. The FTDI chip has a clock signal. If this signal is not present or unstable, the chip will not function properly.

To address this, a systematic approach is necessary:

**1. Verify Driver Installation and Compatibility:**

The first step is to ensure the correct FTDI drivers are installed. This should involve identifying the exact model of FTDI chip used on the Alchitry CU and downloading the matching drivers from the official FTDI website. After installation, examine the device manager (in Windows) or equivalent system tools on your operating system. The FTDI device should enumerate under 'Ports (COM & LPT)' as two serial ports if the driver is installed correctly and the chip is configured correctly. If it appears as a device with a yellow exclamation mark, that indicates that the driver is either not installed correctly or is incompatible.

**Code Example 1 (Python, using pyftdi library, for verification):**

```python
from pyftdi.ftdi import Ftdi
from pyftdi.usbtools import UsbTools

try:
    devices = UsbTools.find_all(all_devices=True)
    for device in devices:
      if "FTDI" in device.description:
        print("Found FTDI device:", device.description, "at ", device.location)
    ftdi = Ftdi()
    ftdi.open(device.location)
    print("FTDI device successfully opened.")
    ftdi.close()
except Exception as e:
    print("Error finding or opening FTDI device:", e)
```

**Commentary:**
This Python snippet using the `pyftdi` library attempts to enumerate all USB devices and then specifically search for an FTDI device. If found, the code then attempts to open and close that device. If there is an issue, this gives an indication if the problem is on the driver or device level, or simply that a device is not found. The code requires that `pyftdi` is installed and assumes a default configuration is used by the FTDI. It does not identify which port the FTDI is connected to. This example helps determine if the driver is generally functional.

**2. Check for Driver Conflicts:**

If an FTDI device is detected, further checks should be done to check for conflicts. This could mean using the device manager in Windows or using command line tools on macOS or Linux to examine the drivers loaded for the FTDI device. Sometimes multiple entries for FTDI devices may be present due to different driver versions. If this occurs, the user should uninstall the old drivers. The most accurate test would be to boot into a known clean OS or Virtual Machine and see if the FTDI chip functions. If it does not, the issue is likely at the hardware level.

**3. Test and Validate the USB connection:**

Verify the USB cable and connection to the PC. Try a different cable known to be functional, and try a different USB port on the PC. If other USB devices are also not functioning on a specific port, the problem could be on the system hardware, and not the Alchitry CU.

**Code Example 2 (Command-line using `dmesg` on Linux or `system_profiler SPUSBDataType` on MacOS):**

```bash
# Linux
dmesg | grep FTDI

# MacOS
system_profiler SPUSBDataType | grep FTDI
```
**Commentary:**

These commands will print messages relating to FTDI devices, as registered by the kernel, showing potential driver issues, the correct enumerated COM port. The output may show the device being disconnected or having intermittent issues. This allows for real time monitoring of device connections. On MacOS, this outputs a more verbose view of the USB devices which may also help indicate which ports are allocated for the chip. This helps identify low-level issues in the OS's communication with the USB device, not visible in standard applications.

**4. Reinstallation:**
Once hardware is verified and a driver identified, reinstalling the latest FTDI drivers may solve the issue. Prior to doing this, be sure to uninstall the existing FTDI drivers and ensure they are removed from the computer using the control panel (windows) or via the command line on Linux or macOS.

**5. Hardware Checks:**
If the software and driver approaches do not resolve the issue, the next step is to inspect the hardware. Inspect the board for physical damage, such as broken components or shorts. If another Alchitry CU is available, test it to rule out a faulty board. It may also be worth checking if the power on the board is stable using a multimeter.

**Code Example 3 (Python, using pyftdi to test communication - requires further hardware setup):**

```python
from pyftdi.ftdi import Ftdi
import time

try:
    ftdi = Ftdi()
    ftdi.open(device.location)  # Assuming 'device.location' was identified earlier
    print("FTDI device opened, testing MPSSE")
    ftdi.set_bitmode(0xFF, 0xFF) # Setting it to an output configuration
    time.sleep(0.1)
    ftdi.write(b'\xFF') # Try to write some data.
    time.sleep(0.1)
    value = ftdi.read()
    print("Bytes read: ", value)

    ftdi.close()
    print("FTDI device closed.")
except Exception as e:
    print("Error testing FTDI device:", e)

```

**Commentary:**
This Python script, still using pyftdi, attempts to configure the FTDI in MPSSE (Multi-Protocol Synchronous Serial Engine) mode, specifically for testing its input/output capabilities. It sends a byte of data (0xFF) and then attempts to read it back from a device connected to the output pins. This will only work if something is connected to the output pins. This step would require the user to have additional testing equipment, but will show if the communication path to the chip is intact. If communication fails at this point, a hardware issue is likely. The code assumes the location of the device was identified in prior steps. This is a basic test and needs hardware connected to verify a hardware path is open.

**Resource Recommendations:**

*   **Manufacturer Documentation:** The Alchitry website and documentation for their CU board should be the first resource consulted. Often, specific driver installation instructions and troubleshooting guides are available.
*   **FTDI Website:** This is the primary source for official FTDI drivers. Download the latest versions directly from their site. Refer to their application notes for device configuration details.
*   **FPGA Community Forums:** Online forums dedicated to FPGAs and embedded development can provide valuable support from other users who may have experienced similar issues.
*   **Operating System Support Pages:** These pages usually provide detailed instructions on how to diagnose and resolve USB driver issues.

In summary, diagnosing communication issues between iCEprog and an FTDI device on the Alchitry CU involves verifying driver functionality, resolving conflicts, validating USB connection integrity, and, if needed, conducting hardware checks. The methodology outlined here is applicable beyond this specific use case. A careful, iterative approach will help to isolate the root cause of the problem and ultimately restore proper communication to the board.
