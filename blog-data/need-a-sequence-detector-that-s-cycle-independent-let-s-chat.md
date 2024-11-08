---
title: "Need a sequence detector that's cycle-independent? Let's chat!"
date: "2024-11-08"
id: "need-a-sequence-detector-that-s-cycle-independent-let-s-chat"
---

```systemverilog
module seq_detector (
  input clk,
  input reset,
  input data_in,
  output logic detect
);

  typedef enum logic [2:0] {S0, S1, S2, S3, S4} state_t;
  state_t current_state, next_state;

  always_ff @(posedge clk or posedge reset) begin
    if (reset) begin
      current_state <= S0;
    end else begin
      current_state <= next_state;
    end
  end

  always_comb begin
    case (current_state)
      S0: begin
        if (data_in == 1'b1) begin
          next_state = S1;
        end else begin
          next_state = S0;
        end
      end
      S1: begin
        if (data_in == 1'b0) begin
          next_state = S2;
        end else begin
          next_state = S1;
        end
      end
      S2: begin
        if (data_in == 1'b1) begin
          next_state = S3;
        end else begin
          next_state = S2;
        end
      end
      S3: begin
        if (data_in == 1'b0) begin
          next_state = S4;
        end else begin
          next_state = S3;
        end
      end
      S4: begin
        if (data_in == 1'b1) begin
          next_state = S1;
        end else begin
          next_state = S4;
        end
      end
      default: next_state = S0;
    endcase
  end

  assign detect = (current_state == S4);

endmodule
```
