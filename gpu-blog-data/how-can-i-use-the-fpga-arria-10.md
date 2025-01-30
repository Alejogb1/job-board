---
title: "How can I use the FPGA Arria 10 passive serial driver in Linux?"
date: "2025-01-30"
id: "how-can-i-use-the-fpga-arria-10"
---
The Arria 10 FPGA utilizes a passive serial (PS) configuration mode, allowing a host processor, such as a Linux system, to directly load the FPGA bitstream. This mode contrasts with active serial, where the FPGA itself manages the configuration process from an external memory. Understanding the nuances of the PS driver within the Linux kernel is crucial for successful FPGA integration. My experience with integrating custom logic onto embedded systems using Altera, now Intel, FPGAs, has highlighted the challenges and solutions surrounding this interface. Specifically, I've encountered scenarios where the provided vendor tools were insufficient, necessitating a deeper dive into driver mechanics.

The core functionality of the PS driver resides in enabling the transfer of configuration data to the FPGA's dedicated programming pins. Unlike active modes which rely on onboard flash memory or external devices, in PS mode, the host CPU must manage the entire bitstream loading sequence. This typically involves manipulating specific GPIO pins connected to the FPGA’s configuration inputs. These pins are usually CLK, DATA, and nCONFIG, alongside others like nSTATUS and CONF_DONE for handshake signals. The specific behavior of these pins during configuration is rigorously defined in Intel’s documentation for the Arria 10 series. The driver’s primary responsibility is to sequence through the prescribed bitstream transfer process. The bitstream itself, typically a compiled '.rbf' file, is a binary representation of the logic design intended for the FPGA.

The Linux kernel exposes the PS configuration interface through a character device. This allows user-space programs to interact with the underlying hardware using standard file operations – read, write, and ioctl. Crucially, direct memory access (DMA) can be employed for significantly faster transfers of the bitstream. The device driver, often a part of the board support package (BSP) or a custom module developed by the user, initializes the necessary hardware resources, including allocating memory for the bitstream and setting up the GPIO lines. A primary concern in this operation is the proper handling of timing constraints and handshake signals. Incorrect sequencing or timing can result in failed configuration attempts and require careful debugging, often with the aid of logic analyzers.

Furthermore, the driver needs to handle different power states of the FPGA. This might include putting the FPGA into reset prior to configuration or verifying that the FPGA has been reset, and confirming the ‘DONE’ status after a successful transfer. Error handling is another crucial aspect. The driver must gracefully recover from potential failures, such as a corrupted bitstream, and report the error condition to the user. Finally, the driver typically exposes a series of device attributes for querying status and triggering different operations, like reset or configuration.

Here are three code examples illustrating crucial facets of working with an Arria 10 PS driver in Linux:

**Example 1: Basic Bitstream Transfer**

This example focuses on the core operation of transferring the bitstream from memory to the FPGA.

```c
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>

// Assumes /dev/fpga_config exists and is the device node for the PS interface

#define CONFIG_DATA_SIZE 1024*1024 // Example bitstream size, adjust as needed

int main() {
    int fd;
    unsigned char *config_data;
    ssize_t bytes_written;

    // Load the bitstream
    config_data = malloc(CONFIG_DATA_SIZE);
    if(config_data == NULL) {
        perror("Failed to allocate memory for config data");
        return -1;
    }
	
    FILE* fp = fopen("/path/to/my_bitstream.rbf", "rb");
	if(fp == NULL) {
        perror("Failed to open the bitstream file");
        free(config_data);
        return -1;
	}
    fread(config_data, 1, CONFIG_DATA_SIZE, fp);
	fclose(fp);
	
    fd = open("/dev/fpga_config", O_WRONLY);
    if (fd == -1) {
        perror("Failed to open device");
        free(config_data);
        return -1;
    }
    
    bytes_written = write(fd, config_data, CONFIG_DATA_SIZE);
    if (bytes_written != CONFIG_DATA_SIZE) {
        perror("Failed to write configuration data");
        close(fd);
        free(config_data);
        return -1;
    }

    printf("Configuration data written successfully.\n");

    close(fd);
    free(config_data);
    return 0;
}
```

*Commentary:* This code snippet illustrates a simple file write operation to the device node. In a practical driver, the `write()` call would trigger a series of hardware operations, including the pin toggling and timing related to sending bitstream data through the PS interface. The `/dev/fpga_config` is a placeholder; the actual device path depends on your kernel configuration. Error handling is included, but more robust mechanisms would be required for production use cases. I’ve found the process of allocating the buffer, loading the RBF file, and calling the driver's write method to be the most straightforward method, especially when initially testing the kernel module.

**Example 2: Resetting the FPGA**

This code highlights the reset functionality, which is often implemented as an ioctl operation.

```c
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>

#define FPGA_RESET _IO('F', 0x01)

int main() {
    int fd;

    fd = open("/dev/fpga_config", O_RDWR);
    if (fd == -1) {
        perror("Failed to open device");
        return -1;
    }

    if (ioctl(fd, FPGA_RESET) == -1) {
        perror("Failed to reset FPGA");
        close(fd);
        return -1;
    }
	
    printf("FPGA reset initiated.\n");

    close(fd);
    return 0;
}
```

*Commentary:* Here, `FPGA_RESET` is a user-defined ioctl command. This value will depend entirely on how the specific driver was implemented. The driver, upon receiving this ioctl call, would manipulate the FPGA's nCONFIG pin to trigger a reset sequence. Based on my development experiences, I often implemented a dedicated ioctl command for this feature for the sake of debugging. Proper timing and debouncing is crucial in the actual driver to properly initiate reset operations, and incorrect implementation can cause the FPGA to not load properly.

**Example 3: Polling the FPGA for Completion**

This example demonstrates how to check the configuration process has successfully concluded, often done through reading a CONF_DONE signal through the kernel device.

```c
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <stdint.h>
#include <stdbool.h>

#define FPGA_GET_STATUS _IOR('F', 0x02, uint32_t)
#define CONFIG_DONE_BIT 0x01

int main() {
    int fd;
	uint32_t status;
    bool config_done;


    fd = open("/dev/fpga_config", O_RDWR);
    if (fd == -1) {
        perror("Failed to open device");
        return -1;
    }
	
	config_done = false;
    
    for(int i=0; i<10; i++)
	{
        if(ioctl(fd, FPGA_GET_STATUS, &status) == -1) {
            perror("Failed to get FPGA status");
            close(fd);
            return -1;
        }
		if(status & CONFIG_DONE_BIT)
		{
			config_done = true;
			break;
		}
		sleep(1); // wait a bit, assuming driver can update at this rate
	}
	
    if(config_done)
        printf("FPGA Configuration is complete.\n");
    else
		printf("FPGA Configuration has timed out.\n");
	
    close(fd);
    return 0;
}
```

*Commentary:* This example uses another custom ioctl command, `FPGA_GET_STATUS`, to retrieve the configuration status of the FPGA. This status information would be a bit mask which has a bit dedicated to confirming whether the FPGA has been configured correctly. In the kernel module this ioctl would actually query the CONF_DONE pin of the FPGA and translate it into this mask. The driver must ensure the pin’s value is correctly read, and the correct status value is presented to the user space. The poll loop with a small sleep time is a common practice when polling a hardware status signal. I've frequently used loops similar to this when debugging hardware issues.

Resource recommendations for further understanding the Arria 10 PS driver include the Intel Arria 10 documentation, specifically the device handbook, and the Linux kernel source itself, particularly the device driver examples present in the `drivers/` directory. Exploring the generic GPIO subsystem documentation is also useful. Additionally, embedded systems development books, specifically those covering device driver development on Linux, can provide a broader understanding of the concepts involved. Vendor provided example drivers are also valuable to review even when customization is required.
