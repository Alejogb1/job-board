---
title: "Confused by this SystemVerilog typedef enum? Let me explain!"
date: '2024-11-08'
id: 'confused-by-this-systemverilog-typedef-enum-let-me-explain'
---

```systemverilog
typedef enum logic [1:0] {S0, S1, S2} statetype;

module tb;

  statetype s;

  initial begin
    s = S0;
    $display("n=%s,s=%0d,", s.name(), s);
    s = 3; // This assignment will cause a warning/error
    $display("n=%s,s=%0d,", s.name(), s); 
  end

endmodule
```
