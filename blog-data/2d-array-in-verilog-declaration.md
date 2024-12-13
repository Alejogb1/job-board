---
title: "2d array in verilog declaration?"
date: "2024-12-13"
id: "2d-array-in-verilog-declaration"
---

Alright so you're wrestling with declaring 2D arrays in Verilog huh Been there done that multiple times It's not as straightforward as some other languages which is why I figure you're here I've been elbow deep in hardware design for what feels like forever and Verilog's quirky nature on multidimensional arrays has bitten me more than once Let me walk you through it with some examples and maybe save you the headache I went through

First off Verilog isn’t exactly thrilled with the term “2D array” or “multidimensional array” It prefers to think of them as arrays of vectors or memory depending on what you’re actually doing This distinction is crucial for understanding how to declare and use them So when you're thinking 2D array Verilog thinks "array of bit vectors"

Let’s start with a basic example say you need a 2D array that’s 4 rows deep and each row is a 8 bit wide vector This would represent a 4x8 matrix of bits Here’s how you declare it

```verilog
module example_array_declaration;

  reg [7:0] my_2d_array [0:3]; // 4 rows each 8 bits wide

  initial begin
    $display("Initial Values of the Array");
    for (integer i = 0; i < 4; i = i + 1) begin
        $display("Row %0d: %b", i, my_2d_array[i]);
    end
	my_2d_array[0] = 8'b10101010;
	my_2d_array[1] = 8'b01010101;
    $display("After assign value to row 0 and 1 of the Array");
     for (integer i = 0; i < 4; i = i + 1) begin
        $display("Row %0d: %b", i, my_2d_array[i]);
    end

  end
endmodule

```

Okay let's break it down `reg [7:0]` this declares a bit vector of 8 bits from bit 7 down to bit 0 The next part `my_2d_array` is just your array's name `[0:3]` that specifies the array size or the number of elements in the outer dimension so you have 4 rows or the depth of your 2D structure I’ve also included a small testbench snippet to initialize and display the array to check for you This pattern `reg [bit_width-1:0] array_name [array_depth]` is the most basic way to declare the 2d-ish structures in verilog and it should cover most use cases

Now you can access a specific row using its index directly so `my_2d_array[0]` accesses the first 8 bit vector `my_2d_array[1]` accesses the second one and so on If you want to access a specific bit within a specific row you would do it like `my_2d_array[0][2]` which access the third bit from the first row

Now things get a bit more interesting when you start thinking about memory like for RAM or ROM implementations You’re still dealing with the idea of an “array” but the Verilog constructs you would use can be subtly different

For something like ROM for example you might have a large “2D” structure where each row is actually a word or data and the row index is its address You would declare it much like our last example the difference lies in the usage here's a snippet to show you this difference:

```verilog
module rom_example;

  parameter ROM_DEPTH = 256;
  parameter WORD_WIDTH = 16;
  reg [WORD_WIDTH-1:0] rom_data [0:ROM_DEPTH-1]; // ROM storage

  initial begin
    // Simulating ROM initialization
    for (integer i = 0; i < ROM_DEPTH; i = i + 1) begin
      rom_data[i] = i * 16'h10; // Fill the ROM with some data
    end

      // Displaying some values
      $display("ROM Value at address 0: %h", rom_data[0]);
      $display("ROM Value at address 1: %h", rom_data[1]);
      $display("ROM Value at address 10: %h", rom_data[10]);


  end

endmodule
```

In this example `rom_data` is like a table where each row `[0:ROM_DEPTH-1]` holds a 16-bit `[WORD_WIDTH-1:0]` word so we have a memory structure of 256 x 16 The code shows how to initialize such ROM with some data and display it Now when you’re thinking of a real ROM you’d load actual data into it not some sequential numbers you probably read that from a memory file of some sort This structure is super common for implementing actual hardware ROM

It's important to remember that in hardware everything needs to be implemented using physical elements so it can be a bit different from software perspective where you can declare large multidimensional arrays with ease You must have a deep understanding of the underlying hardware that’s why it is common to use arrays like these to implement memories that are actually mapped in the hardware

Now if you’re dealing with RAM it gets even more interesting because the RAM itself needs to handle read and write operations but that doesn’t change the declaration logic It is still the same idea of an array of bit vectors Consider the next example where I'm creating an example of a simplified RAM:

```verilog
module ram_example;

    parameter RAM_DEPTH = 256;
    parameter DATA_WIDTH = 32;

    reg [DATA_WIDTH-1:0] ram [0:RAM_DEPTH-1]; // RAM storage
    reg [7:0] address; // Address line
    reg [DATA_WIDTH-1:0] write_data; // data to be write to RAM
    reg we; // write enable signal

    always @(posedge we) begin
        ram[address] <= write_data; // write data when WE is high
    end

    initial begin
        // Initial data set
        ram[1] = 32'hFFFFFFFF;
        ram[2] = 32'h00000000;
        ram[3] = 32'h11111111;

        // simulate write to address 10
        address = 10;
        write_data = 32'hAABBCCDD;
        we = 1;
        #1;
        we = 0; // reset the write signal

        // Displaying some values from the memory
        $display("RAM Value at address 0: %h", ram[0]);
        $display("RAM Value at address 1: %h", ram[1]);
        $display("RAM Value at address 2: %h", ram[2]);
        $display("RAM Value at address 3: %h", ram[3]);
        $display("RAM Value at address 10: %h", ram[10]);

    end
endmodule
```
In this example the structure of ram is an array that holds 32 bit of data with a depth of 256 elements so we have a 256x32 memory The code shows a basic approach how the memory would work in hardware the actual implementation can be different You can see that I'm using an always block to write the data to the memory and simulating a single cycle write to address 10 after some initial values assigned

Now let me share a bit from my past This actually happened a while back I remember once designing a video processing IP where I needed to store a frame buffer and I was trying to declare it like `reg [pixel_width-1:0] frame_buffer [height][width]` and oh boy did it not work out So then I looked at the syntax and remembered what I explained earlier I had the width and height mixed up and my dimensions were completely messed up it took me like 2 hours to figure that out It turned out I needed to think of that as a single large array not nested arrays inside each other because I had to manage the addressing of each pixel individually like they were stored in one big continuous chunk of memory or RAM I still get shivers when I think about it but hey that's how you learn the hard way.

One little gotcha to watch out for is that you can’t just go wild with very large arrays especially in FPGA designs or ASICs Synthesis tools do have their limits and you need to optimize how you use these structures and you must consider the hardware impact of it all too Large RAMs or ROMS could end up using substantial amount of hardware resources or be quite slow So it is essential to be realistic with your designs and hardware resources you are targeting

So what do you need to consider when you need to implement these structures? Firstly think carefully about if you actually need a multidimensional structure or if it can be achieved with a single dimensional array that can be mapped to a physical memory structure like in the example I just showed. Secondly you should think of how the addresses are going to be implemented inside your module For that you can use address decoders or even counters depending on your needs. Finally make sure you are using the correct data types so you do not overflow the size you think you want

Also don’t forget to check your resource usage when simulating If you're using some kind of FPGA vendor simulator it is not enough to just simulate and make sure your design is working properly you need to check resource usage and timing. Remember you’re implementing this in hardware not just software

Lastly the thing I didn't mention is that there is a way to declare multidimensional arrays with another form of syntax using packed and unpacked arrays However I do not recommend that for beginners and for most use cases because it is usually confusing and can make you spend too much time figuring out what is going on So stick to arrays of vectors it is more understandable and more straightforward to maintain and it covers almost any use cases

As for resources for learning more I’d suggest looking into textbooks like “Digital Design and Computer Architecture” by Harris and Harris or “FPGA Prototyping Using Verilog Examples” by Pong P Chu those books cover in details how hardware architectures work and how verilog is used to implement the designs. There are also lots of excellent whitepapers and technical documentation on FPGA vendor sites these can be extremely useful in understanding the more nuances of implementing these structures. Trust me going through these resources will save you countless hours of head scratching debugging.

So in summary you are declaring arrays of vectors in Verilog and that is how you must think about it Keep in mind the hardware implications and physical hardware behind your declarations That should keep you out of trouble and make your designs work smoothly Oh and one more thing never underestimate the power of a good old reset signal in your design… it can be a lifesaver and you might end up using it on the project you are working on or maybe not?
