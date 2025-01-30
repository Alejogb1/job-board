---
title: "How can AXI4 signals be separated in HLS?"
date: "2025-01-30"
id: "how-can-axi4-signals-be-separated-in-hls"
---
High-Level Synthesis (HLS) tools often present challenges when dealing with the complex, parallel nature of AXI4 streams.  My experience optimizing data transfer in large-scale FPGA designs highlighted the crucial need for disciplined signal separation within the HLS environment, especially when working with multiple AXI4 masters and slaves.  The key lies in understanding the inherent structure of the AXI4 protocol and leveraging HLS pragmas to manage data flow effectively.  Improper handling can lead to resource contention and ultimately, suboptimal performance.

**1. AXI4 Signal Separation Strategies in HLS**

The AXI4 interface, with its read and write channels, comprises multiple signals operating concurrently.  Directly interfacing with all signals simultaneously within a single HLS function is generally undesirable.  This approach significantly complicates code readability, verification, and synthesis, often resulting in inefficient resource utilization.  Instead, a strategic approach involving signal partitioning and dedicated interface functions proves significantly more effective.

The primary strategy revolves around decomposing the AXI4 interface into smaller, manageable components. This can be achieved through several techniques:

* **Dedicated Read/Write Functions:**  Create separate HLS functions for AXI4 read and write operations. Each function will manage a subset of the AXI4 signals specific to its operation.  This modularity simplifies code design, improves maintainability, and allows for parallel processing if appropriate.

* **Data Structuring:**  Define HLS data structures to represent the relevant data fields from the AXI4 bus. This facilitates cleaner access to specific data elements within the HLS design, avoiding direct manipulation of individual signals. This approach enhances code clarity and allows the HLS compiler to optimize data movement more effectively.

* **Pragma-Based Control:**  Leverage HLS pragmas such as `#pragma HLS INTERFACE` to specify the interface type for each signal group.  This enables explicit control over signal mapping to AXI4 channels, which is essential for managing multiple streams.  Careful use of `s_axilite` (for control signals) and `m_axi` (for data streams) pragmas is paramount.

* **Data Buffering:**  Strategic use of internal buffers within the HLS function can decouple data flow from the AXI4 interface. This is particularly important for mitigating timing constraints and preventing resource contention when dealing with high-throughput data streams.  FIFO interfaces can be particularly useful in this context.


**2. Code Examples and Commentary**

The following examples illustrate different approaches to separating AXI4 signals in HLS, using a simplified scenario of reading and writing data to an AXI4-Lite slave and an AXI4-Stream master, respectively.  Note that these examples are simplified for illustrative purposes and might need adjustments based on the specific HLS tool and target platform.

**Example 1: Separating Read and Write Operations for AXI4-Lite**

```c++
#include <ap_int.h>

// Define data structure for AXI4-Lite access
typedef struct {
  ap_uint<32> data;
  ap_uint<1>  write;
  ap_uint<1>  read;
} axi4_lite_data;

void axi4_lite_access(axi4_lite_data *axi_data) {
  #pragma HLS INTERFACE axis port=axi_data
  // ... Read and write operations separated by conditional logic ...
  if (axi_data->write == 1) {
      //Write operation to AXI4-Lite using axi_data->data
  } else if (axi_data->read == 1) {
      //Read operation from AXI4-Lite into axi_data->data
  }
}
```

This example uses a structure to encapsulate the AXI4-Lite signals, creating a single interface, but separating read and write operations internally through conditional statements.  This approach is suitable for less complex scenarios.


**Example 2: Dedicated Functions for AXI4-Stream Read and Write**

```c++
#include <ap_axi_sdata.h>

typedef ap_axiu<32,1,1,1> axi_stream_data;

void axi_stream_read(axi_stream_data *in, ap_uint<32> *data_buffer, int size) {
  #pragma HLS INTERFACE axis port=in
  #pragma HLS INTERFACE m_axi port=data_buffer offset=slave bundle=data_bus
  #pragma HLS INTERFACE s_axilite port=size bundle=control
  // ... Read data from AXI4-Stream 'in' and store into 'data_buffer' ...
}

void axi_stream_write(ap_uint<32> *data_buffer, axi_stream_data *out, int size) {
  #pragma HLS INTERFACE m_axi port=data_buffer offset=slave bundle=data_bus
  #pragma HLS INTERFACE axis port=out
  #pragma HLS INTERFACE s_axilite port=size bundle=control
  // ... Read data from 'data_buffer' and write to AXI4-Stream 'out' ...
}
```

Here,  separate functions manage AXI4-Stream reads and writes.  This improves modularity and enables better optimization by the HLS compiler for each operation. The use of `m_axi` and `axis` pragmas explicitly defines the interfaces.


**Example 3:  Using FIFOs for decoupling**

```c++
#include <ap_axi_sdata.h>
#include <hls_stream.h>

typedef ap_axiu<32,1,1,1> axi_stream_data;
typedef hls::stream<ap_uint<32>> data_fifo;

void process_stream(axi_stream_data *in, axi_stream_data *out) {
  #pragma HLS INTERFACE axis port=in
  #pragma HLS INTERFACE axis port=out
  data_fifo buffer;
  #pragma HLS STREAM variable=buffer depth=1024
  //Read from input stream into FIFO
  //...
  //Process data from FIFO
  //...
  //Write from FIFO to output stream
  //...
}

```

This demonstrates the use of an HLS stream (`hls::stream`) as a FIFO buffer to decouple the input and output AXI4-Streams.  This is particularly useful for handling bursty or asynchronous data flows, significantly improving performance and mitigating resource conflicts. The `depth` pragma controls the FIFO size.

**3. Resource Recommendations**

For a deeper understanding of AXI4 and its interaction with HLS, I would recommend consulting the documentation for your specific HLS tool.  Thoroughly examine the documentation on pragmas and interface declarations, paying particular attention to memory mapping and data transfer optimization techniques.  Furthermore, studying the design examples provided within the HLS tool's resources will prove immensely valuable.  Finally, consider acquiring and reviewing advanced FPGA design texts focusing on high-performance computing and memory management.  These resources will provide a more comprehensive theoretical foundation to supplement practical experience.
