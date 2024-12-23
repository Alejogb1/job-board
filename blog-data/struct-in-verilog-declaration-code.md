---
title: "struct in verilog declaration code?"
date: "2024-12-13"
id: "struct-in-verilog-declaration-code"
---

 so you're asking about declaring structs in Verilog eh Been there done that seen the t-shirt and probably designed the circuit that prints the t-shirt yeah I've had my fair share of run-ins with Verilog and its quirks especially when it comes to organizing data into something more structured than a bunch of loose wires. You see in the real world we ain't dealing with just single bits or integers we're handling packets data streams and complex control signals you know the stuff that actually makes things tick. So yeah structs are pretty crucial.

let's talk shop. In Verilog unlike in high level languages like C or Python you don't have a dedicated `struct` keyword per se. Instead you achieve a similar effect using `packed` arrays. Yes I know sounds weird right first time I encountered it I did a double take and wondered if I had accidentally downloaded a different language manual but no it's just how Verilog rolls.

The concept here is to create a multi-bit signal where each part of the signal represents a member of your structure. You're essentially concatenating smaller bit fields to create a larger composite signal. The beauty or the curse of it depends on your mood is that you control the bits directly. No automatic padding or memory management here its all on you mate. That means you gotta be really careful with your bit widths and make sure you’re not stepping on any toes. Now you might be wondering why is it like that simple answer hardware control my man hardware control.

Let's dive into some code examples. I'll walk you through it like I'm explaining it to my past self circa 2008 when I was probably pulling all-nighters for my digital design courses good times I guess.

**Example 1: A Simple Packet Structure**

Suppose we're working with a packet that has a header a payload and a checksum typical eh Well here's how we could model it in Verilog using packed arrays

```verilog
module packet_struct;
  parameter HEADER_WIDTH = 8;
  parameter PAYLOAD_WIDTH = 16;
  parameter CHECKSUM_WIDTH = 4;

  // Define the packed array representing the packet
  typedef struct packed {
    logic [HEADER_WIDTH-1:0] header;
    logic [PAYLOAD_WIDTH-1:0] payload;
    logic [CHECKSUM_WIDTH-1:0] checksum;
  } packet_t;

  packet_t my_packet;

  initial begin
    my_packet.header = 8'hAA;
    my_packet.payload = 16'h1234;
    my_packet.checksum = 4'h5;

    $display("Header: %h", my_packet.header);
    $display("Payload: %h", my_packet.payload);
    $display("Checksum: %h", my_packet.checksum);

    $display("Entire Packet: %h", my_packet);
  end

endmodule
```

 let's break this down. We start by defining some `parameter`s for the width of each field header payload checksum. This is good practice cause it makes your code more readable and easier to modify later. Then we create a `typedef` which is the crucial bit here with the name `packet_t` The `struct packed` syntax is used to group these fields together into a single datatype. And we use `logic` for our signals. Finally we create an instance of our packet struct called `my_packet` and assign values to each member using the dot operator. And well I'm using the display command to show that everything works and is accessible.

**Example 2: Nested Structs**

Now things can get a bit more complicated. Let's say we need to have nested structures where one struct contains another one. This happens all the time in complex hardware designs. So for example let's pretend our header from the previous example contains source and destination address.

```verilog
module nested_struct;
  parameter ADDR_WIDTH = 4;

  typedef struct packed {
    logic [ADDR_WIDTH-1:0] source;
    logic [ADDR_WIDTH-1:0] destination;
  } header_t;

  parameter PAYLOAD_WIDTH = 16;
  parameter CHECKSUM_WIDTH = 4;

    typedef struct packed {
    header_t header;
    logic [PAYLOAD_WIDTH-1:0] payload;
    logic [CHECKSUM_WIDTH-1:0] checksum;
  } packet_t;


  packet_t my_packet;

  initial begin
    my_packet.header.source = 4'h1;
    my_packet.header.destination = 4'h2;
    my_packet.payload = 16'hABCD;
    my_packet.checksum = 4'h9;

    $display("Source: %h", my_packet.header.source);
    $display("Destination: %h", my_packet.header.destination);
    $display("Payload: %h", my_packet.payload);
    $display("Checksum: %h", my_packet.checksum);
     $display("Entire Packet: %h", my_packet);
  end
endmodule
```

So here we first define a `header_t` struct which contains source and destination addresses both 4 bit wide. Then we use the `header_t` struct as a part of the `packet_t` struct. Accessing the members of this nested structure requires chaining dot operators like `my_packet.header.source`. This is pretty much as straightforward as it gets and if you get lost here maybe you are in the wrong profession just kidding.

**Example 3: Structs in Modules**

Now let’s get to the real stuff. Usually you won’t just define and display a struct in the `initial` block that would be boring. You’d use them in your modules. So here we create a module that takes the packet we defined in the first example.

```verilog
module packet_processor #(
    parameter HEADER_WIDTH = 8,
    parameter PAYLOAD_WIDTH = 16,
    parameter CHECKSUM_WIDTH = 4
    )
    (input logic [HEADER_WIDTH+PAYLOAD_WIDTH+CHECKSUM_WIDTH-1:0] packet_in,
    output logic [PAYLOAD_WIDTH-1:0] payload_out);

  // Define the packed array representing the packet
  typedef struct packed {
    logic [HEADER_WIDTH-1:0] header;
    logic [PAYLOAD_WIDTH-1:0] payload;
    logic [CHECKSUM_WIDTH-1:0] checksum;
  } packet_t;

  packet_t input_packet;
    assign input_packet = packet_in;
    assign payload_out = input_packet.payload;

endmodule

module testbench;
    parameter HEADER_WIDTH = 8;
    parameter PAYLOAD_WIDTH = 16;
    parameter CHECKSUM_WIDTH = 4;
    logic [HEADER_WIDTH+PAYLOAD_WIDTH+CHECKSUM_WIDTH-1:0] test_packet;
    logic [PAYLOAD_WIDTH-1:0] processed_payload;

    initial begin
        test_packet = {8'hAB, 16'h1234, 4'h2};
        #10;
        $display("Test Packet is %h",test_packet);
        $display("Processed Payload is %h",processed_payload);
        #10;
    end
    packet_processor #(
        .HEADER_WIDTH(HEADER_WIDTH),
        .PAYLOAD_WIDTH(PAYLOAD_WIDTH),
        .CHECKSUM_WIDTH(CHECKSUM_WIDTH)
    ) dut (
        .packet_in(test_packet),
        .payload_out(processed_payload)
    );
endmodule
```

So in this example we have a `packet_processor` module that takes a packet as input and outputs the payload only. To demonstrate this we create a `testbench` module where we define a test packet and then we instantiate our `packet_processor` and then display the result. This is very very very simplified version of what real hardware looks like but you can see that structs make passing signals around much more convenient. You see how the `input` is just a single flat bus that is later decomposed into different parts using struct. This is very powerful way to handle different kind of signals in hardware. And since we are passing the parameter as well this can be a generic module that can handle different kinds of packets.

Now a few words of wisdom from someone who has spent way too many hours debugging Verilog code.

**Things to watch out for**

*   **Bit width errors**: Make sure the total bit width of your packed array matches the total bit width of the signal you're assigning it to. Mismatches can lead to all sorts of weird bugs and head scratching moments trust me on this. And while Verilog might not complain to you about mismatches the hardware will.
*   **Endianness**: Verilog is endianness agnostic meaning that its on you to make sure that the order of the bits in the struct is how the rest of your system expects it. This has been the source of many many many debugging sessions.
*   **Debugging**: Debugging packed arrays can be tricky because you don’t see the individual fields in many waveform viewers as clear as you would see them in a high level programming language. I recommend spending time mastering your simulation tool and learning all its bells and whistles.
*   **Readability**: As your structures get complex and deeply nested it can get difficult to read the code. So proper documentation and consistent formatting is very important.
*   **Reusability**: You can make use of `typedef` to create reusable structs across your design. This is very important because otherwise you'll be defining structs everywhere in your project which is a big no-no.

**Further Learning**

If you want to get deeper I wouldn't really recommend going online and searching for random posts. If you want to seriously understand the topic then you should look for formal resources such as:

*   **"SystemVerilog for Design Second Edition"** by Stuart Sutherland and David Doman it has many good examples and good explanations of the topic.
*   **"Digital Design Principles and Practices"** by John F Wakerly. This is not specifically for Verilog but if you understand hardware you understand Verilog.

Yeah so that’s pretty much it for structs in Verilog. It's not exactly the most elegant solution but it works and that's what matters in the end. As I always say when I am working on hardware "it just needs to work" and it doesn't have to be pretty. So go out there and create some cool circuits. Oh and one last thing never ever underestimate the power of a good cup of coffee and a clear head debugging hardware can be pain but it is very rewarding in the end.
