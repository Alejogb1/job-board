---
title: "How can migen or chisel HDL be used on PYNQ FPGA boards?"
date: "2025-01-30"
id: "how-can-migen-or-chisel-hdl-be-used"
---
Using Migen or Chisel on PYNQ FPGA boards requires a multi-faceted approach, primarily focused on bridging the gap between these high-level hardware description languages and the Vivado toolchain that PYNQ relies upon. The crux of the matter is that neither Migen nor Chisel directly generate the Xilinx `.xpr` project files or `.bit` files expected by PYNQ. Instead, they generate intermediate forms, namely Verilog or VHDL, which then must be incorporated into a Vivado project for synthesis, implementation, and bitstream generation. I have firsthand experience with this during the development of custom peripherals for sensor data processing on a Zynq-7000 series PYNQ board.

The process involves three essential stages: HDL generation, Vivado project integration, and PYNQ driver development. First, Migen or Chisel is utilized to create the desired hardware logic, exporting it as synthesizable Verilog or VHDL. Second, this generated HDL is added as a user-defined IP core within a Vivado project, alongside the PYNQ base overlay's IP. Finally, the necessary Python drivers, exploiting the PYNQ framework, are crafted to interact with the created hardware.

Migen, written in Python, benefits from Python's powerful abstractions and concise syntax. Its procedural nature makes it suitable for complex control logic and data path specification. For example, if I were creating a simple shift register in Migen for my PYNQ project, the Python code would resemble the following:

```python
from migen import *
from migen.genlib.cdc import MultiReg

class ShiftRegister(Module):
    def __init__(self, width, depth):
        self.i = Signal(width)
        self.o = Signal(width)
        self.clk = ClockSignal()
        self.depth = depth

        ###

        shift_reg = Signal(width * depth)
        
        self.sync += shift_reg.eq(Cat(self.i, shift_reg[:(width*(depth-1))]))
        self.sync += self.o.eq(shift_reg[(width*(depth-1)):])
        

        ###
    
if __name__ == "__main__":
    
    from migen.fhdl.verilog import convert
    
    shifter = ShiftRegister(8, 16)
    v = convert(shifter, ios=[shifter.i, shifter.o, shifter.clk], name="shifter_migen")
    print(v.verilog)
    with open("shifter_migen.v", "w") as f:
        f.write(v.verilog)
```

This Migen code defines a `ShiftRegister` module with an input `i`, output `o`, clock `clk` , shift register width, and shift register depth as input parameters. The core logic uses Migen’s synchronous assignment `self.sync +=` to create the shift register itself.  The last synchronous assignment, `self.sync += self.o.eq(shift_reg[(width*(depth-1)):])`, takes the output of the shift register. Importantly, the `convert` function generates the Verilog output; in actual PYNQ deployment, this would not be printed to the standard out but instead written to a file that would then be integrated into the Vivado IP project. The  `ios` parameter specifies the external signals, crucial for proper port definition in Vivado. The resulting `shifter_migen.v`  is the file you will need to add to Vivado.

Chisel, in contrast, uses Scala as its host language and adopts a functional paradigm, often resulting in concise and highly parameterizable designs. Consider a similar shift register designed using Chisel. I would typically employ something like this:

```scala
import chisel3._

class ShiftRegister(width: Int, depth: Int) extends Module {
  val io = IO(new Bundle {
    val i = Input(UInt(width.W))
    val o = Output(UInt(width.W))
  })

  val shiftReg = Reg(Vec(depth, UInt(width.W)))

    shiftReg(0) := io.i
  for (i <- 1 until depth) {
      shiftReg(i) := shiftReg(i-1)
  }
  io.o := shiftReg(depth-1)

}
 

object ShiftRegisterGen extends App {
  println(chisel3.Driver.emitVerilog(new ShiftRegister(8, 16)))
}
```

This Chisel code constructs a `ShiftRegister` module utilizing Scala's `for` loops to concisely describe the shift register logic. The `Reg(Vec(depth, UInt(width.W)))` line declares a register array of specified width and depth, which will be used for the shift register. The first shift register location, `shiftReg(0)` takes the module input and each of the other values in the register takes the previous register value. The `io.o := shiftReg(depth-1)` sets the module output to the final value of the register. Finally, the `Driver.emitVerilog` method within the `ShiftRegisterGen` object produces the Verilog output.  Again, in practical PYNQ usage, you would direct this to a file rather than the standard output for inclusion in your Vivado project.

Once the Verilog from Migen or Chisel is generated, the next crucial step is integrating it into a Vivado project targeting the specific PYNQ board. This is done by creating a new custom IP core and adding the generated HDL files. After the logic is in the block design, the appropriate connections are made, typically using the AXI interconnect to the Zynq processing system. The design then undergoes synthesis, implementation, and bitstream generation. For my projects, I’ve often found that debugging during the Vivado implementation step tends to be the most challenging due to the interactions of timing constraints and placement of resources.

The final piece involves the development of Python drivers using PYNQ’s abstraction layers.  These drivers, leveraging PYNQ's capabilities, provide high-level access to your custom hardware. For example, if I wanted to write to a register, I would use the `xlnk.cma_allocate` memory allocation function to create a memory object at the base address of my module (defined in Vivado address map).  Then I could write the data I needed to the desired register offset. The following illustrates the python logic I would use to write the input for a Migen or Chisel-generated module:

```python
from pynq import Overlay
from pynq import MMIO
import numpy as np

# Load the PYNQ overlay
overlay = Overlay("your_overlay.bit") #replace this with your overlay

# Get the address of your custom IP
base_address = overlay.ip_dict["shifter_migen_0"]["addr_range"][0] # replace shifter_migen_0 with your IP name

# Create a memory-mapped I/O object
mmio = MMIO(base_address, 256) #256 is a size of the total MMIO range. Change as needed.

# Example data to shift in
input_data = 0xAA

# Write the data to the input register
mmio.write(0x0, input_data) #assumes input is at offset 0. If not, change this.

#read the output register after enough clock cycles for data to propagate. If the output is offset 0x4, use
# output_data = mmio.read(0x4)
# print (output_data)

```

The Python code first loads the bitstream and then retrieves the base address of the created module from the loaded overlay using the `overlay.ip_dict` method. Then the `MMIO` method initializes the memory map of the peripheral using the base address and the total address range. Then you can use the `write` function to write data to register offsets in your custom module's address space. The example also gives an example on how you could read the output by using the `read` function at the appropriate offset. Note that in this context, we have omitted any control signals, however, control signals can be accessed the same way through the appropriate base address offset.

In my experience, careful attention to address maps in Vivado is crucial for successful driver development. Mismatched addresses between the Python code and the hardware can result in subtle and perplexing issues. Also, timing issues, if unaddressed during the Vivado implementation stage, can cause instability or prevent your custom logic from functioning correctly.

For further exploration, I recommend focusing on resources concerning the specifics of using Migen or Chisel for FPGA design. Additionally, Xilinx documentation regarding the Vivado IP Integrator and AXI interfaces will prove invaluable for navigating the hardware integration process. Finally, the official PYNQ documentation itself will detail the appropriate driver development methodologies and API calls necessary for controlling your custom hardware. These materials, taken as a whole, should enable successful use of Migen and Chisel on PYNQ platforms.
