---
title: "Should FPGA design be included in a computer science curriculum?"
date: "2025-01-30"
id: "should-fpga-design-be-included-in-a-computer"
---
FPGA design, often perceived as an electrical engineering domain, offers a profoundly beneficial perspective when integrated into a computer science curriculum. My experience designing custom hardware accelerators for image processing pipelines has consistently underscored how a foundational understanding of FPGAs drastically improves a computer scientist’s comprehension of computational limits and optimization techniques. This transcends typical software development, exposing students to the physical underpinnings of computation and enabling them to design truly parallel and performant systems.

The core benefit of including FPGA design stems from its ability to illustrate the direct connection between logical functions and their hardware implementation. Unlike abstract software, where layers of abstraction hide the underlying machine, FPGAs provide a highly tangible platform. Students learn to translate algorithms directly into configurable hardware, witnessing the physical manifestations of Boolean logic and parallel processing. This offers an essential counterpoint to the traditional von Neumann architecture focus of most computer science programs. The ability to manipulate logic at this level fosters a deeper understanding of how computational tasks are ultimately executed and what constraints they face. This is a critical skill for any computer scientist seeking to create efficient, low-latency systems.

Furthermore, FPGA design compels a move away from purely sequential programming paradigms. Students are forced to consider resource utilization (logic elements, memory blocks), timing constraints, and parallelism from the ground up. This exposure to hardware-level parallelism, which is inherent in FPGA fabrics, is invaluable in the modern computing landscape where multi-core and GPU architectures demand highly parallelized code to achieve high performance. Working with hardware description languages (HDLs) like Verilog or VHDL provides a drastically different approach to problem-solving compared to high-level languages, enhancing critical thinking and system architecture skills. The student learns to think not just algorithmically but also architecturally, designing entire systems from a set of configurable logic gates and registers. This shift in perspective facilitates the design of algorithms optimized for real-world hardware, a competency increasingly demanded by industry.

Let’s examine several code examples to illustrate these points:

**Example 1: A Simple Adder in Verilog**

```verilog
module adder (
  input  wire [7:0] a,
  input  wire [7:0] b,
  output wire [7:0] sum
);

  assign sum = a + b;

endmodule
```

This extremely basic Verilog module shows the fundamental structure of FPGA design. We define a module named `adder`, specifying the inputs `a` and `b`, and output `sum`, each as 8-bit wide bit vectors (wires). The `assign` statement demonstrates that the output `sum` is continuously computed as the sum of inputs `a` and `b`. While seemingly trivial, this code directly translates into logic gates (specifically, an 8-bit adder circuit) within the FPGA fabric. A computer science student gains a concrete understanding of how addition is physically implemented, a stark contrast to the abstract addition operations within a C++ program, for example. This example, when synthesized, shows the very low-level implementation of an arithmetic operation.

**Example 2: A FIFO (First-In, First-Out) Buffer in VHDL**

```vhdl
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity fifo is
    generic (DEPTH : positive := 16; DATA_WIDTH : positive := 8);
    port (
        clk     : in  std_logic;
        reset   : in  std_logic;
        wr_en   : in  std_logic;
        wr_data : in  std_logic_vector(DATA_WIDTH-1 downto 0);
        rd_en   : in  std_logic;
        rd_data : out std_logic_vector(DATA_WIDTH-1 downto 0);
        full    : out std_logic;
        empty   : out std_logic
    );
end entity fifo;

architecture behavioral of fifo is

    type mem_type is array (0 to DEPTH-1) of std_logic_vector(DATA_WIDTH-1 downto 0);
    signal mem      : mem_type := (others => (others => '0'));
    signal wr_ptr   : integer range 0 to DEPTH-1 := 0;
    signal rd_ptr   : integer range 0 to DEPTH-1 := 0;
    signal count    : integer range 0 to DEPTH := 0;


begin
    process(clk, reset)
    begin
        if reset = '1' then
            mem <= (others => (others => '0'));
            wr_ptr <= 0;
            rd_ptr <= 0;
            count <= 0;

        elsif rising_edge(clk) then
            if (wr_en = '1') and (count < DEPTH) then
                mem(wr_ptr) <= wr_data;
                wr_ptr <= (wr_ptr + 1) mod DEPTH;
                count <= count + 1;
            end if;

            if (rd_en = '1') and (count > 0) then
                rd_data <= mem(rd_ptr);
                rd_ptr <= (rd_ptr + 1) mod DEPTH;
                count <= count - 1;
            end if;

        end if;
    end process;

    full <= '1' when count = DEPTH else '0';
    empty <= '1' when count = 0 else '0';
end architecture behavioral;

```
This example uses VHDL, another common HDL, to design a First-In, First-Out buffer (FIFO). A FIFO is a fundamental building block in many digital systems. The VHDL code defines the entity with generic parameters for depth and data width, along with the necessary input and output ports. Within the architecture, we see the explicit declaration of a memory array, read/write pointers, and a counter. The process block describes the sequential logic, updating memory, pointers, and the count based on clock edges and control signals. This directly maps to memory blocks, registers, and control logic within the FPGA. Students can grasp how data buffers are implemented at the hardware level, how timing and synchronization matter. The concept of concurrent processes becomes a tangible aspect of the implementation.

**Example 3: A Parallel Convolution Filter in Verilog (Simplified)**
```verilog
module convolution_filter (
    input wire clk,
    input wire [7:0] pixel_in,
    input wire valid_in,
    output reg [7:0] pixel_out,
    output reg valid_out
);

  reg [7:0] line_buffer [0:2];
  reg [7:0] filtered_pixel;
  reg [1:0] counter;

  always @(posedge clk) begin
    if (valid_in) begin
      line_buffer[0] <= line_buffer[1];
      line_buffer[1] <= line_buffer[2];
      line_buffer[2] <= pixel_in;

      if(counter == 2'b11) begin // Simple 3x1 kernel; can be made generic
            filtered_pixel <= (line_buffer[0] + line_buffer[1] + line_buffer[2])/3 ;
            pixel_out <= filtered_pixel;
            valid_out <= 1'b1;
      end else begin
          counter <= counter + 1;
          valid_out <= 1'b0;
      end
    end
    else begin
        valid_out <= 1'b0;
        counter <= 2'b00;
    end
  end
endmodule
```
This simplified Verilog code illustrates a basic 1D convolution filter, commonly used in image processing. It introduces concepts of pipelining and data flow. Using a register array (`line_buffer`), the module accumulates three input pixels before performing the filter operation. This parallel processing ability is directly observable in the design. The `if-else` block with `valid_in` and `valid_out` showcase a typical handshaking control scheme found in many hardware accelerators. A computer science student with FPGA experience would see beyond the sequential-looking code, recognizing the inherent parallelism within the logic which is fundamentally different compared to a similar software implementation running on a processor with serial execution.

To augment a computer science curriculum, specific resources are necessary. It is crucial to have access to a textbook focused on digital logic design. Books covering digital design with Verilog and VHDL, which offer good synthesis coverage, are invaluable. Furthermore, texts focusing on computer architecture with an emphasis on hardware design complement the practical hands-on aspects. These resources, combined with hands-on lab work using FPGA development boards, provide a complete picture. Software tools from FPGA vendors like Xilinx and Intel (formerly Altera) provide the necessary environments for development, simulation, and synthesis. These should be accessible to students for practical learning, especially for simulation, which makes sure the hardware is working as expected.

In conclusion, integrating FPGA design into a computer science curriculum is not merely adding an extra topic; it is about broadening the fundamental understanding of computation. It empowers students to become not just proficient programmers but also system architects who can design software and hardware in concert, addressing real-world performance bottlenecks. The ability to see computation as a physical process through the lens of FPGA design is an invaluable asset in the modern landscape, and is one that every computer science graduate should have.
