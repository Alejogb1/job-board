---
title: "Got SystemVerilog Assertions Acting Up? Here's How to Schedule Them Right!"
date: '2024-11-08'
id: 'got-systemverilog-assertions-acting-up-here-s-how-to-schedule-them-right'
---

```systemverilog
module top;
  bit a,b;
  
  initial begin
    #1 a = 1;
    #1 b = 1;
    #1 a = 0;
    #1 b = 0;
    #1 $finish;
  end
  
  let p_sig_detect = (a == 1) -> (b == 1);

  always_comb
    a_sig_detect : assert (p_sig_detect()) $info("pass"); else $error("fail");
endmodule
```
