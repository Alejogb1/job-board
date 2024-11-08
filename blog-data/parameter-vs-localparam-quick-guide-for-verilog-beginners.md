---
title: '"Parameter vs Localparam: Quick Guide for Verilog Beginners"'
date: '2024-11-08'
id: 'parameter-vs-localparam-quick-guide-for-verilog-beginners'
---

```verilog
module myram #(
    parameter WORD_SIZE = 1,
    parameter ADDR_SIZE = 1
) (
    input wire [ADDR_SIZE-1:0] addr,
    inout wire [WORD_SIZE-1:0] data,
    // ...
);
    localparam MEM_SIZE = WORD_SIZE * (1 << ADDR_SIZE);
    // Use MEM_SIZE several times in block.
...
endmodule

module myram_tb;
    myram #(
        .ADDR_SIZE(2),
        .WORD_SIZE(2)
    ) top (
        /* wires */
    );
endmodule
```
