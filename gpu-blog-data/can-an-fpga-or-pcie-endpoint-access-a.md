---
title: "Can an FPGA or PCIe endpoint access a SOM's on-chip RAM?"
date: "2025-01-30"
id: "can-an-fpga-or-pcie-endpoint-access-a"
---
The crucial factor determining whether an FPGA or PCIe endpoint can access a System-on-Module's (SOM) on-chip RAM hinges on the SOM's architecture and the available interfaces.  Direct access is rarely provided for security and design reasons.  In my experience working on embedded systems for over a decade, including projects integrating Xilinx FPGAs with various ARM-based SOMs, I've observed that such access typically requires an intermediary layer or a carefully designed memory-mapped interface.

**1. Explanation:**

On-chip RAM within an SOM is typically part of the system's private memory space, managed by the SOM's processor.  Direct access from an external device like an FPGA or a PCIe endpoint would require a dedicated interface specifically exposed by the SOM vendor.  This is uncommon for several reasons:

* **Security Concerns:** Providing direct access compromises the integrity of the system.  Unauthorized access to the on-chip RAM could lead to data breaches or system instability.

* **Design Complexity:**  Implementing and maintaining a robust, secure interface for direct memory access (DMA) is complex, potentially affecting the SOM's power consumption and performance.

* **Vendor Dependence:** The specifics of any such interface are entirely dependent on the SOM's architecture and the manufacturer's design choices. There’s no standardized approach.

Instead of direct access, several architectural solutions are commonly employed to facilitate data transfer:

* **Shared Memory through Inter-processor Communication (IPC):** The SOM's processor can act as a mediator. The FPGA or PCIe endpoint communicates with the SOM's processor (e.g., via a UART, SPI, or a higher-bandwidth interface like AXI) and requests data from the on-chip RAM.  The processor then retrieves the data and sends it back. This adds latency but ensures controlled access.

* **Memory-Mapped I/O:**  The SOM might expose specific memory addresses that the FPGA or PCIe endpoint can write to or read from. This usually involves utilizing a dedicated peripheral on the SOM that interfaces with external devices. This requires careful configuration and understanding of the SOM's memory map.  The on-chip RAM is not directly accessed, but its contents are indirectly reflected through the peripheral.

* **DMA Controller with Dedicated Memory Region:** A more sophisticated approach involves the use of a DMA controller on the SOM.  The SOM's processor can configure the DMA controller to transfer data between the on-chip RAM and a dedicated memory region accessible by the FPGA or PCIe endpoint.  This offers higher throughput compared to IPC but requires significant configuration and careful management to avoid conflicts.


**2. Code Examples:**

The following examples are illustrative and rely on hypothetical interfaces.  The actual implementation depends heavily on the specific hardware and software involved.

**Example 1:  Shared Memory via UART (Conceptual):**

```c++
// On SOM (running on the SOM's processor)
void process_request(uint8_t *data, int size) {
  // Read data from on-chip RAM based on the request
  // ...
  // Send data back via UART
  uart_transmit(data, size);
}

// On FPGA (VHDL - Simplified)
process (clk) begin
  if rising_edge(clk) then
    if (request_ready) then
      uart_transmit(request);
      wait until response_ready;
      process_response;
    end if;
  end if;
end process;
```

This demonstrates the basic principle of using UART for communication. The FPGA sends a request, and the SOM's processor responds after accessing the RAM.


**Example 2: Memory-Mapped I/O (Conceptual C++):**

```c++
// On PCIe Endpoint (C++)
#include <iostream>

// Assume base address is defined elsewhere
const uint32_t base_address = 0x40000000;

int main() {
  volatile uint32_t *memory_mapped_register = (uint32_t *)base_address;
  *memory_mapped_register = 0x1234; // Write to a register reflecting on-chip RAM content
  uint32_t value = *memory_mapped_register; // Read from the register
  std::cout << "Value read: 0x" << std::hex << value << std::endl;
  return 0;
}
```

This example showcases the idea of memory-mapped I/O; however, the underlying mechanism for mapping the SOM's peripheral to the PCIe address space is not shown and is highly system-specific.


**Example 3: DMA Transfer (Conceptual Python - High-Level):**

```python
# On FPGA (Python controlling DMA – simplified)
import struct

# Assume DMA functions are provided via a library
# ...

dma_address = 0x10000000  # Address of the SOM's dedicated DMA region
data = [10, 20, 30, 40]

# Write data to DMA region
dma_write(dma_address, struct.pack('<IIII', *data))

# Trigger DMA transfer configured by the SOM’s processor
# ...

# Read data back from DMA region
received_data = dma_read(dma_address, 4*4) # Read 4 integers.
```


This example provides a very high-level conceptual view of utilizing a DMA controller to transfer data. The crucial element is the interaction with a dedicated memory region which is indirectly mapped to the SOM’s on-chip RAM through the processor’s DMA controller configuration.


**3. Resource Recommendations:**

For in-depth understanding, I suggest consulting the datasheets and application notes provided by the SOM vendor, specifically focusing on documentation detailing interfaces, memory maps, and DMA controller configuration.  Also, review relevant documentation on FPGA programming (using HDL languages like VHDL or Verilog) and the chosen interface protocol (e.g., AXI, PCIe specifications).  Finally, consult textbooks on embedded system design and computer architecture.  These resources will provide the necessary level of detail needed to tackle the intricacies involved in such systems.
