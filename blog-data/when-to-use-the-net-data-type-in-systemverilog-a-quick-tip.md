---
title: "When to use the 'net' data type in SystemVerilog: A quick tip!"
date: '2024-11-08'
id: 'when-to-use-the-net-data-type-in-systemverilog-a-quick-tip'
---

```systemverilog
// Example of using 'net' for bidirectional ports and multiple drivers
module bidirectional_example (
  input wire clk,
  inout wire [7:0] data,
  output wire read,
  output wire write
);

  // 'wire' is used for bidirectional port 'data'
  assign data = read ? 8'h00 : 8'hFF;
  assign read = 1'b0;

  // 'wire' is used for multiple drivers on 'data'
  assign data = write ? data_in : data_out;
  assign write = 1'b0;

  // 'logic' can be used for internal signals like 'data_in' and 'data_out'
  logic [7:0] data_in, data_out;

endmodule
```
