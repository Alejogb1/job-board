---
title: "Stuck on this SystemVerilog `typedef enum`? Let me show you the light! 💡"
date: '2024-11-08'
id: 'stuck-on-this-systemverilog-typedef-enum-let-me-show-you-the-light'
---

```systemverilog
typedef enum logic [1:0] {S0, S1, S2} statetype;

module tb;

  statetype s;

  initial begin
    s = S0;
    $display("n=%s,s=%0d,", s.name(), s);
    s = 3; // This will cause a warning/error 
    $display("n=%s,s=%0d,", s.name(), s);
  end

endmodule
```
