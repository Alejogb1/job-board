---
title: "How to design a Vivado HLS module to read from a FIFO?"
date: "2025-01-26"
id: "how-to-design-a-vivado-hls-module-to-read-from-a-fifo"
---

High-throughput data processing often demands efficient interaction with First-In, First-Out (FIFO) buffers in hardware. When targeting an FPGA using Vivado High-Level Synthesis (HLS), correctly designing modules to read from these FIFOs is crucial for optimal performance. I've personally encountered numerous bottlenecks arising from improperly handled FIFO interfaces, reinforcing the need for a clear, systematic approach.

The core challenge lies in the inherently sequential nature of FIFO operations. Unlike memory access, reading from a FIFO isn't as straightforward as providing an address. Instead, we need to manage a 'valid' signal indicating the presence of data and a 'ready' signal indicating the downstream module's capacity to receive it. Failure to adhere to these signaling protocols can result in deadlocks or data loss. Furthermore, efficiently pipelining these operations within the HLS framework requires careful consideration.

The basic structure of a Vivado HLS module interacting with a FIFO involves the following: a FIFO input interface, an internal processing loop, and logic to conditionally read from the FIFO. Let's delve into specifics. We’ll utilize the `hls::stream` data type, which is Vivado HLS’s abstraction for FIFOs. It allows us to focus on the logic without manually crafting low-level FIFO interfaces. Assume we want to implement a simple module that reads 8-bit data from a FIFO, and process it (in this example, just adding 1).

**Code Example 1: Basic FIFO Read**

```c++
#include "ap_int.h"
#include "hls_stream.h"

void fifo_read_basic(hls::stream<ap_uint<8> > &input_fifo, hls::stream<ap_uint<8> > &output_fifo) {
#pragma HLS INTERFACE axis port=input_fifo
#pragma HLS INTERFACE axis port=output_fifo

    ap_uint<8> data_in;

    while (true) {
        input_fifo.read(data_in);
        output_fifo.write(data_in + 1);
    }
}
```

This example showcases the simplest method. We define `input_fifo` as a stream of 8-bit unsigned integers (`ap_uint<8>`). The `#pragma HLS INTERFACE axis port=input_fifo` directive tells the HLS tool that this port will be a stream with AXI Stream handshaking. Inside the `while(true)` loop, `input_fifo.read(data_in)` attempts to read data. Note that `read()` is a *blocking* operation. The function pauses until data becomes available in the FIFO. Similarly, `output_fifo.write` transmits the processed data. While functional, this implementation is inefficient for high-throughput applications because the pipeline cannot advance if data is not readily available. Each iteration of the loop is dependent on the completion of the read operation from the FIFO.

**Code Example 2: Pipelined FIFO Read with Loop Exit Condition**

```c++
#include "ap_int.h"
#include "hls_stream.h"

void fifo_read_pipelined(hls::stream<ap_uint<8> > &input_fifo, hls::stream<ap_uint<8> > &output_fifo, int num_transfers) {
#pragma HLS INTERFACE axis port=input_fifo
#pragma HLS INTERFACE axis port=output_fifo
#pragma HLS INTERFACE s_axilite port=num_transfers bundle=control

    ap_uint<8> data_in;

    for (int i = 0; i < num_transfers; i++) {
#pragma HLS PIPELINE
       input_fifo.read(data_in);
       output_fifo.write(data_in + 1);
    }
}
```

This example introduces several refinements.  Firstly, `num_transfers` is included as an input controlling the number of elements to process. We declare it as an AXI Lite slave port (using `#pragma HLS INTERFACE s_axilite`) for control from a processor or other external sources. Then, the processing now occurs inside a standard `for` loop controlled by `num_transfers`. Crucially, the `#pragma HLS PIPELINE` directive tells HLS to pipeline the loop. This enables HLS to overlap iterations and operate on multiple elements in parallel by inserting pipeline stages. In essence, the `read` operation is now performed with more independent stages.  It’s important to note that while the pipeline is initiated, it is still necessary that data are available in the FIFO and that the downstream module is ready to receive them. If there is backpressure from either direction, the pipeline will stall.

**Code Example 3: Read with Polling and Back Pressure Handling**

```c++
#include "ap_int.h"
#include "hls_stream.h"

void fifo_read_polling(hls::stream<ap_uint<8> > &input_fifo, hls::stream<ap_uint<8> > &output_fifo, int num_transfers) {
#pragma HLS INTERFACE axis port=input_fifo
#pragma HLS INTERFACE axis port=output_fifo
#pragma HLS INTERFACE s_axilite port=num_transfers bundle=control

    ap_uint<8> data_in;

    for (int i = 0; i < num_transfers; i++) {
#pragma HLS PIPELINE
        while (input_fifo.empty()) {
          // Spin while the fifo is empty
        }
       input_fifo.read(data_in);

       while(output_fifo.full())
       {
         // Spin while the fifo is full
       }
        output_fifo.write(data_in + 1);
    }
}
```

This example enhances the previous one by introducing polling to handle situations where the FIFO may be empty or full. Before attempting a `read` or a `write`,  we check whether the `input_fifo` is `empty()` and the `output_fifo` is `full()` respectively. These `empty()` and `full()` member functions provide access to the current state of the stream. If either condition is true, we enter a spin-wait loop until the condition becomes false.  This prevents the pipeline from stalling indefinitely and improves robustness in scenarios where FIFO data availability is not guaranteed. The polling does introduce a degree of inefficiency.  The code executes extra cycles while spinning on the empty/full conditions. Careful analysis of the system’s data rates is often critical to ensure that this is not a large overhead.  In scenarios with predictable data flow, simple pipelining may be preferable.

**Resource Recommendations**

When designing Vivado HLS modules interacting with FIFOs, several resources will be immensely helpful.  Firstly, carefully review the official Xilinx documentation for Vivado HLS, particularly the sections relating to streaming interfaces (`hls::stream`) and AXI Stream (AXIS).   These sections provide detailed descriptions of the available directives and optimization techniques. Look closely at sections pertaining to pipeline scheduling and the performance implications. Further, the Xilinx forums are a good resource for specific technical questions, particularly when encountering errors or design challenges. Many experienced users discuss their approaches and solutions to problems related to FIFOs and data movement.  Finally, look for practical examples of HLS designs targeting AXI interfaces.  These often demonstrate advanced optimization techniques such as burst transfers and different clock domain crossing approaches. The concepts of backpressure and handshaking will also need more specific examination.

In conclusion, designing a Vivado HLS module to read from a FIFO involves a blend of understanding FIFO mechanics, careful coding, and strategic use of HLS directives. We have looked at a basic read implementation, a pipelined approach, and a more robust method that checks for availability. Correctly applying these techniques is paramount to achieving efficient, high-throughput hardware accelerators. As experience is the best teacher, practical projects experimenting with variations of these approaches will undoubtedly provide more understanding.
