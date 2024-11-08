---
title: "Unpacking Streams: The Quick and Easy Way"
date: '2024-11-08'
id: 'unpacking-streams-the-quick-and-easy-way'
---

```verilog
  wire unpacked_wire[3:0];
  wire [3:0] packed_wire;

  assign {>>{unpacked_wire}} = packed_wire; 
```
