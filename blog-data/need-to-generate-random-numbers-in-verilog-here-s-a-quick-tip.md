---
title: "Need to Generate Random Numbers in Verilog? Here's a Quick Tip!"
date: '2024-11-08'
id: 'need-to-generate-random-numbers-in-verilog-here-s-a-quick-tip'
---

```verilog
module test;
   initial begin
      reg[15:0]a;
      reg [15:0] b;

      integer    i,j;
      for (i=0; i<6; i=i+1)
        begin
           a=$urandom%10; 
           #100;
           b=$urandom%20;
           $display("A %d, B: %d",a,b);    
        end 
      $finish;
   end
endmodule
```
