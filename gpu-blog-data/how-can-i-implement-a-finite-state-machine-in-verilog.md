---
title: "How can I implement a finite state machine in Verilog?"
date: "2025-01-26"
id: "how-can-i-implement-a-finite-state-machine-in-verilog"
---

Implementing a finite state machine (FSM) in Verilog is a fundamental skill for hardware design, particularly when creating controllers and sequential logic. My initial forays into digital design, specifically a custom memory interface for a now-defunct microcontroller project, hammered home the importance of a robust FSM implementation; its correctness directly affects system stability and predictability. In essence, an FSM transitions between defined states based on input conditions, generating outputs specific to each state. In Verilog, this translates into a combination of a synchronous state register and combinational logic for next-state and output calculations.

The core components of a Verilog FSM implementation are:

1.  **State Register:** This is a flip-flop or set of flip-flops that stores the current state. The number of bits required to represent the state depends on the number of distinct states.
2.  **Next-State Logic:** This combinational logic block determines the next state based on the current state and the input conditions. It forms the decision-making mechanism of the FSM.
3.  **Output Logic:** This combinational logic block generates outputs based on the current state. Outputs may be registered or unregistered based on specific design requirements.

The state transition is triggered by the clock edge; typically, this is the positive edge. This synchronous behavior ensures that all state transitions happen simultaneously and deterministically, maintaining the system's stability. The overall structure is best represented using a `case` statement within an `always` block triggered by the clock, with separate combinatorial blocks for next-state and output calculation.

**Code Example 1: A Simple 2-State FSM**

Let's consider a simple FSM with two states: `IDLE` and `ACTIVE`. The transition from `IDLE` to `ACTIVE` occurs when an input signal `start` is asserted, and from `ACTIVE` back to `IDLE` when an input signal `done` is asserted.

```verilog
module simple_fsm (
  input  wire clk,
  input  wire start,
  input  wire done,
  output reg  active_out
);

  // Define State Encoding
  localparam IDLE    = 1'b0;
  localparam ACTIVE  = 1'b1;

  reg current_state;

  // Next State Logic (Combinational)
  reg next_state;
  always @(*) begin
    case (current_state)
      IDLE:  if (start) next_state = ACTIVE; else next_state = IDLE;
      ACTIVE: if (done)  next_state = IDLE;   else next_state = ACTIVE;
      default: next_state = IDLE; // Default case for safety
    endcase
  end

  // State Register (Sequential)
  always @(posedge clk) begin
    current_state <= next_state;
  end

  // Output Logic (Combinational)
  always @(*) begin
    if (current_state == ACTIVE)
       active_out = 1'b1;
    else
       active_out = 1'b0;
  end

endmodule
```

*   **Commentary:** This example introduces basic concepts. `localparam` defines symbolic names for the states, improving code readability. The `next_state` block is purely combinational, avoiding unintended latching. The state register is clocked, implementing the synchronous behavior. The output `active_out` directly reflects the active state of the FSM. Using an `always @(*)` block ensures that the logic updates anytime an input or a current state changes; this is essential for robust hardware designs. The default case ensures the FSM does not end in an unknown state upon power-up.

**Code Example 2: A Traffic Light Controller FSM**

A more complex example is a simplified traffic light controller with states `GREEN`, `YELLOW`, and `RED`. The transitions between states are sequential, with a timer enabling each transition. For simplicity, I'll represent the timer using a clock counter, rather than a full-fledged hardware timer.

```verilog
module traffic_light_fsm (
  input  wire clk,
  input  wire reset, // Sync Reset Signal
  output reg  red_light,
  output reg  yellow_light,
  output reg  green_light
);

  // Define State Encoding
  localparam GREEN  = 2'b00;
  localparam YELLOW = 2'b01;
  localparam RED    = 2'b10;

  reg [1:0] current_state;
  reg [1:0] next_state;
  reg [3:0] timer;
  localparam MAX_COUNT = 4'd10;

  // Next State Logic (Combinational)
  always @(*) begin
    case (current_state)
      GREEN:  if (timer == MAX_COUNT) next_state = YELLOW; else next_state = GREEN;
      YELLOW: if (timer == MAX_COUNT) next_state = RED;    else next_state = YELLOW;
      RED:    if (timer == MAX_COUNT) next_state = GREEN;  else next_state = RED;
      default: next_state = RED; // Default case to prevent unknown state
    endcase
  end

  // State Register (Sequential)
  always @(posedge clk) begin
      if(reset)
        current_state <= RED;
      else
        current_state <= next_state;
  end


    // Timer Logic (Sequential)
  always @(posedge clk) begin
      if(reset)
        timer <= 4'd0;
      else if (timer == MAX_COUNT)
        timer <= 4'd0;
    else
        timer <= timer + 1'b1;
  end

  // Output Logic (Combinational)
  always @(*) begin
      red_light   = (current_state == RED);
      yellow_light= (current_state == YELLOW);
      green_light = (current_state == GREEN);
  end

endmodule
```

*   **Commentary:** This example demonstrates a more complex state transition mechanism using a clock counter to simulate a timer. Notice the synchronous reset implemented in both the state and timer registers. A `default` case is present in the `next_state` block. State-based output enables generating lights based on the current state. I’ve used a simple integer timer for illustrative purposes; a more robust design would use a dedicated timer module for better timekeeping precision. This version also directly demonstrates one hot encoding within the `output logic` block.

**Code Example 3: FSM with an Input Request and Acknowledge**

Consider an FSM managing data transfer with input request (`req`) and acknowledge (`ack`). The FSM is in `IDLE` until a request, then transitions through a `TRANSFER` state while an acknowledge isn't asserted, and returns to idle.

```verilog
module data_transfer_fsm (
  input  wire clk,
  input  wire req,
  input  wire ack,
  output reg  transfer_active
);

  // Define State Encoding
  localparam IDLE     = 2'b00;
  localparam TRANSFER = 2'b01;

  reg [1:0] current_state;
  reg [1:0] next_state;


  // Next State Logic (Combinational)
    always @(*) begin
      case (current_state)
          IDLE:     if(req) next_state = TRANSFER; else next_state = IDLE;
          TRANSFER: if(ack) next_state = IDLE; else next_state = TRANSFER;
          default: next_state = IDLE;
      endcase
    end

  // State Register (Sequential)
    always @(posedge clk) begin
        current_state <= next_state;
    end

  // Output Logic (Combinational)
    always @(*) begin
        transfer_active = (current_state == TRANSFER);
    end


endmodule
```

*   **Commentary:** This example showcases the use of input signals to directly control state transitions. When the `req` signal is asserted, the state transitions to `TRANSFER`. It then remains in `TRANSFER` until `ack` is asserted, highlighting the state waiting for acknowledgement. This represents a common handshake protocol that has been encountered on multiple occasions during custom bus designs. The output `transfer_active` acts as an enable signal driven by the state.

When designing FSMs in Verilog, several considerations are important. Careful state encoding should be done to achieve optimal performance and resource utilization. Using symbolic names for states significantly improves the readability and maintainability of the design. Always include a default case in `case` statements, this ensures that the FSM doesn’t enter into a non-defined state during the initial power up or during error conditions. All FSM logic should be synchronous, using flip-flops, which enables predictable behavior. Avoid using latches by ensuring all conditions are covered by `if-else` blocks within the combinational sections. Thorough testing is essential to verify state transitions and output logic. When dealing with complex FSMs it's good practice to model the state diagram on paper first and use that model during design.

For further study, I recommend delving into literature on digital design that covers finite state machines in the context of Verilog. There are several excellent books, like “Digital Design and Computer Architecture” by Harris & Harris, and “Verilog HDL” by Samir Palnitkar, which cover fundamental principles as well as more complex FSM implementations. Reviewing examples from real-world designs, such as open-source hardware repositories, provides valuable insights. Online communities and forums are useful resources for problem-solving and design discussions. Additionally, tutorials and workshops offered by EDA tool vendors are beneficial. These resources, combined with practical experience, will greatly improve one's ability to effectively implement and manage finite state machines in Verilog.
