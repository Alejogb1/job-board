---
title: "How can I retain button input values across multiple clock cycles in Verilog for an FPGA?"
date: "2025-01-26"
id: "how-can-i-retain-button-input-values-across-multiple-clock-cycles-in-verilog-for-an-fpga"
---

A common challenge in FPGA design is reliably capturing and maintaining button press states, particularly when the button's physical duration is shorter than a clock cycle. The inherent asynchronous nature of button inputs relative to the FPGA's clock domain necessitates careful handling to avoid metastability and ensure predictable behavior. My experience debugging similar issues across several industrial controllers has highlighted the critical role of debouncing, synchronization, and state-holding mechanisms.

The primary issue lies in the fact that a physical button, when pressed or released, doesn't transition cleanly between logic levels. It exhibits what's known as contact bounce – a rapid oscillation of high and low states – before settling into a stable value. If an FPGA samples this bouncing signal directly, it may interpret multiple rapid transitions as a series of button presses instead of one sustained action. Furthermore, if the button state changes during the setup or hold window of a flip-flop (the storage element typically used), the output can become metastable, resulting in an unpredictable voltage level that can persist for a duration before resolving to either logic high or low. This metastability issue is exacerbated by the asynchronous nature of the button input, as its transitions are not coordinated with the FPGA's clock.

Therefore, the design needs a multi-pronged approach: first, debouncing to eliminate the rapid oscillations; second, synchronizing the debounced signal to the FPGA’s clock domain; and third, holding the synchronized signal using a storage mechanism, typically a flip-flop or a register.

Let's consider a straightforward implementation utilizing a counter-based debounce approach, followed by a synchronizer and a storage register:

```verilog
module button_input_controller (
  input  wire        clk,          // FPGA Clock
  input  wire        button_in,     // Raw button input
  output reg         button_out,    // Synchronized & debounced output
  output reg        button_pressed,  // Indication button is pressed
  output reg        button_released // Indication button is released
);

parameter DEBOUNCE_COUNT_MAX = 2000; // Clock cycles for debounce
reg [11:0] debounce_counter;
reg        debounced_sig;
reg        sync1_sig, sync2_sig;
reg        prev_button_state; // Store previous state

// Debounce Logic: Count up while button is stable
always @(posedge clk) begin
  if (button_in == debounced_sig) begin //button state match
    if (debounce_counter < DEBOUNCE_COUNT_MAX) begin
        debounce_counter <= debounce_counter + 1;
      end
  else
    begin
        debounce_counter <= DEBOUNCE_COUNT_MAX;
    end
  end else begin //button state does not match
        debounce_counter <= 0;
    end
  if (debounce_counter == DEBOUNCE_COUNT_MAX)
        debounced_sig <= button_in;
end

// Synchronizer: Two-stage flip-flop for metastability mitigation
always @(posedge clk) begin
  sync1_sig <= debounced_sig;
  sync2_sig <= sync1_sig;
end

// State Holding and Edge Detection Logic
always @(posedge clk) begin
  button_out <= sync2_sig;

  //detect button presses
  if (sync2_sig == 1'b1 && prev_button_state == 1'b0)
    button_pressed <= 1'b1;
  else
    button_pressed <= 1'b0;
    
    //detect button releases
    if (sync2_sig == 1'b0 && prev_button_state == 1'b1)
        button_released <= 1'b1;
     else
      button_released <= 1'b0;

  prev_button_state <= sync2_sig; // Update previous state for next clock
end

endmodule
```

In this first example, the `button_input_controller` module encapsulates the entire process. The `DEBOUNCE_COUNT_MAX` parameter controls the duration of the debounce. The debouncing section increments a counter whenever the raw button input matches the currently registered debounced signal value. If the counter reaches `DEBOUNCE_COUNT_MAX`, the debounced signal is updated. This ensures a consistent high or low level after the bounce has settled. The two-stage synchronizer, using `sync1_sig` and `sync2_sig`, significantly reduces the probability of metastability propagating into subsequent logic. Finally, a register, `button_out`, holds the synchronized debounced value. Additionally, the logic detects edges, allowing the module to provide single clock cycle pulse outputs `button_pressed` and `button_released`. This module avoids the common practice of delaying or stretching signals, which can complicate debugging and timing analysis.

Now, let's explore a scenario where the button press should be stored until a specific action is performed or a flag is cleared, requiring a latching behavior.

```verilog
module button_latch (
  input wire clk,          // FPGA Clock
  input wire button_in,     // Raw button input
  input wire reset,         // Asynchronous reset
  input wire clear_latch,   // Signal to clear the latch
  output reg button_latched // Latched button press
);

parameter DEBOUNCE_COUNT_MAX = 2000;
reg [11:0] debounce_counter;
reg        debounced_sig;
reg        sync1_sig, sync2_sig;
reg        temp_button_latched;

// Debounce Logic (same as in the first example)
always @(posedge clk) begin
 if (button_in == debounced_sig) begin //button state match
    if (debounce_counter < DEBOUNCE_COUNT_MAX) begin
        debounce_counter <= debounce_counter + 1;
      end
  else
    begin
        debounce_counter <= DEBOUNCE_COUNT_MAX;
    end
  end else begin //button state does not match
        debounce_counter <= 0;
    end
  if (debounce_counter == DEBOUNCE_COUNT_MAX)
        debounced_sig <= button_in;
end
// Synchronizer (same as in the first example)
always @(posedge clk) begin
  sync1_sig <= debounced_sig;
  sync2_sig <= sync1_sig;
end

// Latching Logic
always @(posedge clk or posedge reset) begin
  if (reset) begin
    temp_button_latched <= 1'b0; // Initialize latch to off when reset is asserted
    button_latched      <= 1'b0;
  end else begin
    if (sync2_sig == 1'b1) // If the debounced and synchronized signal goes high
      temp_button_latched <= 1'b1; //set the temp latch.
    if(clear_latch)
    temp_button_latched <= 1'b0;
    button_latched <= temp_button_latched; // Latch the button press
  end
end

endmodule
```

In this `button_latch` module, the debounce and synchronization sections are identical to the first example. However, the core logic involves a latch, `temp_button_latched`, that is set to '1' when a button is pressed. `temp_button_latched` will remain high until the asynchronous `reset` signal or `clear_latch` signal goes high. The `button_latched` output mirrors the value of `temp_button_latched` and represents the persistent button state. This allows the system to react to a button press even if it's very brief, until the `clear_latch` signal or `reset` occurs. This technique of using a secondary latch variable (`temp_button_latched`) and copying the result to the final output (`button_latched`) avoids combinatorial feedback loops.

Finally, let's consider a more robust approach to debounce utilizing a shift register which can improve noise immunity over the counter method:

```verilog
module button_input_shift_reg (
    input wire        clk,
    input wire        button_in,
    output reg        button_out,
    output reg       button_pressed,
    output reg       button_released
);

parameter DEBOUNCE_DEPTH = 10;
reg [DEBOUNCE_DEPTH-1:0] shift_reg;
reg  sync1_sig, sync2_sig;
reg   prev_button_state;

// Synchronizer
always @(posedge clk) begin
    sync1_sig <= button_in;
    sync2_sig <= sync1_sig;
end

//Debounce
always @(posedge clk) begin
   shift_reg <= {shift_reg[DEBOUNCE_DEPTH-2:0],sync2_sig};
end

//State Hold and Edge Detection
always @(posedge clk) begin
    if(&shift_reg)  //if all shift registers are 1 we have a debounced 1
       button_out <= 1'b1;
    else
        button_out <= 1'b0;

 //detect button presses
  if (button_out == 1'b1 && prev_button_state == 1'b0)
    button_pressed <= 1'b1;
  else
    button_pressed <= 1'b0;
    
    //detect button releases
    if (button_out == 1'b0 && prev_button_state == 1'b1)
        button_released <= 1'b1;
     else
      button_released <= 1'b0;

    prev_button_state <= button_out;
end

endmodule
```

In this `button_input_shift_reg` module, the input `button_in` is directly synchronized through a two stage synchronizer. Instead of a counter, a shift register `shift_reg` is used. The depth of the register is controlled by `DEBOUNCE_DEPTH`. The debounced signal is determined by performing a bitwise AND across all registers within `shift_reg`. If all bits within the register are 1, we can safely determine the signal has been stable for the amount of clock cycles specified by `DEBOUNCE_DEPTH`. Similar to the first example, this module also features `button_pressed` and `button_released` outputs which are pulse signals indicating a button transition edge.

In summary, retaining button input values across clock cycles requires a combination of debouncing, synchronization, and storage. The choice between the counter-based, shift-register based, or latched approach depends on the specific application requirements. For additional study, I suggest reviewing literature on digital logic design focusing on asynchronous input handling and metastability mitigation techniques. A deeper understanding of FPGA timing analysis can also provide valuable insights into designing robust and reliable button input systems. Furthermore, examining application notes and datasheets from FPGA vendors regarding best practices for input handling is highly beneficial.
