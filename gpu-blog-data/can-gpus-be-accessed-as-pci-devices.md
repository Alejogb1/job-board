---
title: "Can GPUs be accessed as PCI devices?"
date: "2025-01-30"
id: "can-gpus-be-accessed-as-pci-devices"
---
Modern graphics processing units (GPUs), despite their specialized architecture for parallel computation, are indeed fundamentally accessible as Peripheral Component Interconnect (PCI) devices within a computer system. This isn't a theoretical possibility, but a core aspect of how they integrate with the motherboard and the overall system architecture. I've worked extensively with low-level system programming, including custom device drivers, which has afforded me direct experience in this interaction.

The PCI interface serves as the primary communication channel between the CPU, chipset, and numerous peripheral devices, and the GPU is no exception. The system BIOS (or UEFI firmware) initializes and enumerates all detected PCI devices at startup, assigning each a unique address within the PCI address space. The operating system then utilizes these addresses to interact with each device, including GPUs. This interaction involves memory-mapped I/O (MMIO), where certain regions of the PCI address space are mapped to memory addresses, allowing the CPU to send commands to the GPU and read responses. These commands are specific to the GPU architecture and include operations like loading shaders, launching computation kernels, and transferring data to and from the GPU's dedicated memory.

This access is not solely limited to the CPU. Direct Memory Access (DMA) capabilities of the PCI bus allows the GPU to directly access system memory and other PCI devices without involving the CPU as an intermediary. This is crucial for performance because it reduces latency and offloads data transfer tasks from the CPU, freeing it up for other computations. This is particularly evident when handling complex graphical scenes or large datasets for scientific computation. Furthermore, the GPU’s dedicated memory itself resides in the PCI address space, allowing for low-level interactions that bypass higher-level drivers, such as during debugging or custom hardware interface development.

Let's look at three practical examples illustrating the access mechanisms involved. These examples simplify some implementation complexities to focus on the essential PCI interface interaction.

**Example 1: Detecting a GPU on the PCI Bus (using a conceptual pseudocode)**

```pseudocode
function find_gpu_on_pci() {
    for each pci_device in discovered_pci_devices {
        // PCI devices have vendor and device IDs
        if (pci_device.vendor_id == NVIDIA_VENDOR_ID || pci_device.vendor_id == AMD_VENDOR_ID) { 
            // These are common identifiers for known GPU vendors
             if (pci_device.device_class == GRAPHICS_CLASS) { 
                 // This confirms that the PCI device is a graphics card
                 print("GPU found at address: ", pci_device.address)
                 print("Vendor ID: ", pci_device.vendor_id)
                 print("Device ID: ", pci_device.device_id)
                 return pci_device // Device found
             }
        }
    }
     print ("No GPU found.")
    return null; // No compatible device found
}

```

This pseudocode depicts a simple search algorithm that would iterate through the enumerated PCI devices. The vendor and device identifiers (NVIDIA_VENDOR_ID, AMD_VENDOR_ID, etc.) are unique numerical values assigned to every PCI component by its manufacturer. In reality, vendor and device IDs can be retrieved from the PCI configuration space, a specific memory range associated with each PCI device. Additionally, the device class (GRAPHICS_CLASS) is a PCI standard identifying different types of devices; graphics cards consistently report this class code. This fundamental process demonstrates how the system software can identify and distinguish GPUs from other PCI peripherals.

**Example 2: Accessing GPU Configuration Space using a simplified C structure**

```c
#define PCI_CONFIG_SPACE_SIZE 256  //Standard PCI config space size

typedef struct {
    uint16_t vendor_id;     // offset 0x0
    uint16_t device_id;     // offset 0x2
    uint16_t command;       // offset 0x4
    uint16_t status;        // offset 0x6
    uint8_t  revision_id;   // offset 0x8
    uint8_t  class_code[3]; // offset 0x9-0xB
    uint8_t  cache_line_size; // offset 0xC
    uint8_t  latency_timer;  // offset 0xD
    uint8_t  header_type;   // offset 0xE
    uint8_t  bist;           // offset 0xF
    uint32_t base_address_registers[6]; // offset 0x10 - 0x24,  base address for MMIO regions
    // .... other fields (not shown for clarity)
} pci_config_space_t;


// Conceptual function (actual MMIO access is platform specific)
uint32_t read_pci_config_space_field(uint32_t pci_address, uint32_t offset, uint32_t size) {

  volatile uint8_t *config_space_ptr = (volatile uint8_t*)(pci_address + offset);
  uint32_t value = 0;

  if(size == 1) value = *config_space_ptr;
  if(size == 2) value = *((uint16_t*)config_space_ptr);
  if(size == 4) value = *((uint32_t*)config_space_ptr);
  return value;
}

// Example usage:
void print_gpu_info(uint32_t pci_gpu_address) {
    pci_config_space_t config;

    // Hypothetically populate the config structure. The method to actually fetch this varies
    // Platform specific details are abstracted away here.
    config.vendor_id = read_pci_config_space_field(pci_gpu_address,0x0, 2);
    config.device_id = read_pci_config_space_field(pci_gpu_address,0x2, 2);
    config.class_code[0] = read_pci_config_space_field(pci_gpu_address,0x9, 1);
    config.class_code[1] = read_pci_config_space_field(pci_gpu_address,0xA, 1);
     config.class_code[2] = read_pci_config_space_field(pci_gpu_address,0xB, 1);
    uint32_t mmio_base_address = read_pci_config_space_field(pci_gpu_address, 0x10, 4);


    printf("GPU Vendor ID: 0x%x\n", config.vendor_id);
    printf("GPU Device ID: 0x%x\n", config.device_id);
    printf("GPU Class Code: 0x%x %x %x \n", config.class_code[0], config.class_code[1], config.class_code[2]);
    printf("MMIO Base address: 0x%x\n", mmio_base_address);
}

```

This code illustrates a simplified representation of how we might access and interpret the PCI configuration space of a GPU. A `pci_config_space_t` structure is declared to hold various vital information, including vendor and device IDs, the class code, and base address registers. Note that accessing the physical memory address of the PCI config space (represented by *pci_address* here) is system-dependent and needs specific operating system or platform calls; here it’s abstracted via the *read_pci_config_space_field* method. The example shows how we could potentially retrieve the MMIO region’s base address; this region is where registers for direct hardware access reside. It highlights the low-level structures involved in communication with a GPU at the PCI level. In a real driver, there would be explicit code to translate between PCI bus addresses and physical or virtual address mappings.

**Example 3: Triggering a simple GPU operation via memory-mapped I/O (Conceptual pseudocode)**

```pseudocode
// Assuming mmio_base_address was discovered earlier. It should map a physical MMIO region
// into the virtual address space

function write_to_gpu_mmio_register(gpu_mmio_base_address, register_offset, value) {
   volatile uint32_t *register_ptr = (volatile uint32_t*)(gpu_mmio_base_address + register_offset);
   *register_ptr = value;  // write value to the register
}

function trigger_simple_gpu_calculation(mmio_base_address) {
  const uint32_t  START_CALCULATION_REG = 0x100;  // Hypothetical address of a register
  const uint32_t START_VALUE = 0x1; // Value to write to that register to start a calculation

   // This writes to a specific register within the GPU's MMIO address space
    write_to_gpu_mmio_register(mmio_base_address, START_CALCULATION_REG, START_VALUE);

  //  Polling would be done for completion (not shown)
   print("Calculation started");

}

```

This pseudocode shows a simplified concept of how a CPU would send a command to the GPU using MMIO.  The `write_to_gpu_mmio_register` function simulates how the CPU can write to specific addresses within the GPU's MMIO region.  In reality, many operations are not single-register writes but instead, multi-step command sequences involving DMA transfers of code and data.  This is a highly abstracted example, and the exact registers, their offsets, and values are vendor-specific. The purpose here is to showcase the principle of interacting with the GPU by writing to memory-mapped hardware registers.

To further explore this subject, I recommend consulting the following resources: “PCI System Architecture” by Don Anderson and Tom Shanley; "Understanding the Linux Kernel" by Daniel P. Bovet and Marco Cesati, which covers low-level driver interactions; and finally, vendor documentation for the specific GPU models. These will reveal the intricacies and variations involved in GPU hardware interaction at the PCI level. While modern operating systems provide higher-level APIs and libraries (like CUDA, OpenCL, Direct3D), understanding the fundamentals of the underlying PCI device access remains invaluable, especially for those involved in driver development or custom hardware integration.
