---
title: "how to declare and use 1d and 2d byte arrays in verilog?"
date: "2024-12-13"
id: "how-to-declare-and-use-1d-and-2d-byte-arrays-in-verilog"
---

Okay so you wanna juggle byte arrays in Verilog right I've been there trust me It's not exactly Python with its easy lists but it's doable and yeah you gotta be careful about how you set things up especially when you're pushing them into hardware

Alright first let's tackle the 1D byte array a simple case I've used this countless times for things like FIFO buffers or just storing small data packets Basically you declare a reg type with a width equal to your byte and then you define an array of them this is the usual way to simulate a byte array

```verilog
module one_dimensional_byte_array (
  input clk,
  input [7:0] data_in,
  input wr_en,
  input rd_en,
  output reg [7:0] data_out
);

  parameter ARRAY_SIZE = 16;
  reg [7:0] byte_array [0:ARRAY_SIZE-1]; // Declare the array of 8-bit registers
  reg [3:0] wr_ptr;
  reg [3:0] rd_ptr;

  always @(posedge clk) begin
    if (wr_en) begin
       byte_array[wr_ptr] <= data_in;
      wr_ptr <= wr_ptr + 1;
    end
    if (rd_en) begin
      data_out <= byte_array[rd_ptr];
      rd_ptr <= rd_ptr + 1;
    end
  end
endmodule
```

See that's the typical 1D array declaration `reg [7:0] byte_array [0:ARRAY_SIZE-1];` It says declare a register named `byte_array` of 8 bits for each element and allocate `ARRAY_SIZE` which is parameterized for some flexibility if you wanna use this multiple times this way the array elements are addressed from index 0 to `ARRAY_SIZE-1`.

Now when you need to read or write you do it via indexed access like `byte_array[wr_ptr]` or `byte_array[rd_ptr]` the example is a simplified version of FIFO but the main idea of array declaration and access is there. Pay attention the `wr_ptr` and `rd_ptr` should not exceed the boundaries of your array you probably will have to implement some overflow control logic that’s beyond the scope here but just a heads up. Also notice that arrays in Verilog are static meaning their size is fixed at compile time you can not change their size on the fly.

Okay moving on to 2D byte arrays now things get a little more interesting this isn’t like C where you can simply declare `int arr[rows][cols]` in Verilog we are still dealing with arrays of registers but we think of them conceptually as a 2D matrix. I mostly used these for image processing applications where each pixel is represented by a byte or for memory structures.

```verilog
module two_dimensional_byte_array (
  input clk,
  input [7:0] data_in,
  input [3:0] row_idx,
  input [3:0] col_idx,
  input wr_en,
  input rd_en,
  output reg [7:0] data_out
);

  parameter ROWS = 8;
  parameter COLS = 8;
  reg [7:0] byte_matrix [0:ROWS-1][0:COLS-1]; // 2D byte array
    
    always @(posedge clk) begin
        if (wr_en) begin
          byte_matrix[row_idx][col_idx] <= data_in;
        end
        if (rd_en) begin
          data_out <= byte_matrix[row_idx][col_idx];
        end
    end
endmodule
```
In this example, I declared `reg [7:0] byte_matrix [0:ROWS-1][0:COLS-1];` which can store `ROWS * COLS` bytes. It is a bit like an Excel sheet but everything is stored sequentially in the memory. The indices access should go as `byte_matrix[row_idx][col_idx]`. `row_idx` and `col_idx` are just the access indices you need to manage them correctly otherwise you will read and write to wrong memory locations that can lead to very hard to debug issues.
Here’s a thing I wish I had known when I was starting. In Verilog, arrays are not passed as a pointer or reference like in other software languages so always remember this when instancing modules. When you instance the previous examples you do not pass byte arrays directly you use inputs and outputs to read and write to those byte arrays.

Finally a more complex one where we are using the array to store some kind of image data in a more efficient way. Usually I would use memory modules like SRAM modules but this can simulate one in behavioral verilog:

```verilog
module byte_matrix_with_memory_access (
  input clk,
  input [15:0] addr,
  input [7:0] data_in,
  input wr_en,
  input rd_en,
  output reg [7:0] data_out
);

  parameter ROWS = 16;
  parameter COLS = 16;
  reg [7:0] byte_matrix [0:(ROWS*COLS)-1]; // 1D but used as 2D with calculations
  
  always @(posedge clk) begin
      if (wr_en) begin
        byte_matrix[addr] <= data_in;
      end
      if (rd_en) begin
        data_out <= byte_matrix[addr];
      end
  end
endmodule
```
This module is taking a 16-bit address input `addr` that we use to access the byte array `byte_matrix` which is declared as a 1D array. However we treat this array as a 2D matrix by converting the row and column address into the correct 1D index. Let’s say your `ROWS` and `COLS` parameters are set to 16. If you wanna access the pixel (5,3) then you need to convert (5,3) to a 1D address using the following formula `addr = 5*COLS + 3` which is `5*16 + 3` which is `83`. So the address `83` will give you the access to that particular byte. This way is less verbose than declaring a 2D array but needs a little math calculation for access but it's much closer to how hardware would access memory. This implementation is a bit faster in simulation and in hardware compared to the previous one because accessing a single level array is faster than accessing a 2 level array from a hardware point of view. (This is also less verbose in terms of Verilog code and easier to parameterize).

Some things you should remember here:
*  **Address decoding:** When you use an address like in my last example you need to make sure that your address is correctly mapped to your desired row and column location, otherwise, you are going to be reading and writing on wrong memory locations.
*   **Timing considerations:**  Always keep timing in mind when you are working on hardware your clock cycles are precious also consider pipelining when necessary that is beyond the scope here.
*   **Tools and Simulation:**   You really need to use the correct simulator to test your logic and testcases. I always simulate these byte arrays before synthesizing it into hardware to avoid headaches and debugging on hardware. There are a lot of simulators you could use. I started using Modelsim but now I am using Vivado's internal simulator.
* **Memory Organization**: How you map your data affects performance. If you access your memory row by row it would be faster to have the data organized in that manner rather than column by column in real hardware.

If you want to get deeper into it I'd recommend some books rather than links. I found that “Digital Design and Computer Architecture” by Harris and Harris goes into good detail about memory structures and how you would implement them in hardware. Also “Computer Architecture A Quantitative Approach” by Hennessy and Patterson is a very good book to understand memory organization and performance but it focuses more on the general system aspect rather than low-level Verilog implementation.

And hey one last thing when you're trying to read from a byte array and nothing's coming out don't panic it's probably just that the wires are shy.

Hope this helps and happy coding.
