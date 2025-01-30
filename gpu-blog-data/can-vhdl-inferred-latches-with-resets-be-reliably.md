---
title: "Can VHDL inferred latches with resets be reliably used in finite state machines?"
date: "2025-01-30"
id: "can-vhdl-inferred-latches-with-resets-be-reliably"
---
VHDL's inherent flexibility, while empowering, often leads to unintended latch inference, particularly in state machine designs.  My experience working on high-speed communication protocols highlighted the criticality of rigorously avoiding such unintended latches, especially when incorporating asynchronous resets. Relying on inferred latches with resets in state machines introduces unpredictable behavior and jeopardizes design stability.  This stems from the unpredictable initialization state of latches in simulation and the potential for metastability issues in hardware implementation.


**1. Clear Explanation of the Problem:**

A finite state machine (FSM) fundamentally relies on predictable state transitions driven by clearly defined inputs and outputs.  A well-designed FSM explicitly defines its state register, its next-state logic, and its output logic.  VHDL synthesizers are tasked with translating this high-level description into a physical implementation.  However, incomplete or ambiguous signal assignments can inadvertently lead to latch inference.  A latch is essentially a memory element that retains its value unless actively updated. Unlike a flip-flop, which necessitates a clock signal for state change, a latch simply holds its previous value until the next assignment.

The problem with inferred latches in FSMs, especially those with asynchronous resets, arises from several factors:

* **Unpredictable Initialization:**  Asynchronous resets are designed to override the current state immediately, independent of the clock.  However, an inferred latch might retain its previous value even after an asynchronous reset, leading to an unpredictable initial state. This is particularly troublesome during simulation and debugging.  The initial state might vary depending on the simulation tool or even the order of signal assignments within the code.

* **Metastability Issues:**  Latches are more susceptible to metastability problems than flip-flops. Metastability arises when the data input to a latch changes near the time when the latch's enable signal is activated.  The latch might enter an undefined state that can propagate through the design, causing intermittent failures. This is exacerbated by asynchronous resets which can change the state arbitrarily close to the latch's enable.

* **Synthesis Tool Dependency:** Different synthesis tools handle latch inference differently.  What might synthesize cleanly with one tool could result in unexpected behavior with another.  This lack of portability undermines the design's reliability across different implementation platforms.

* **Verification Challenges:** Simulating and verifying a design with inferred latches becomes significantly more challenging.  The lack of explicit control over the latch’s behavior obscures the expected state transitions, making debugging and identifying root causes considerably more difficult.


**2. Code Examples with Commentary:**

The following examples demonstrate how unintended latch inference can occur and how to avoid it.  These examples are simplified for clarity.

**Example 1: Unintended Latch Inference**

```vhdl
library ieee;
use ieee.std_logic_1164.all;

entity fsm_bad is
  port (
    clk : in std_logic;
    rst : in std_logic;
    x : in std_logic;
    y : out std_logic
  );
end entity;

architecture behavioral of fsm_bad is
  signal state : std_logic;
begin
  process (clk, rst)
  begin
    if rst = '1' then
      state <= '0';
    elsif rising_edge(clk) then
      case state is
        when '0' =>
          if x = '1' then
            state <= '1';
            y <= '1';
          end if;
        when '1' =>
          state <= '0';
          y <= '0';
        when others =>
          null;
      end case;
    end if;
  end process;
end architecture;
```

**Commentary:** This code might appear correct at first glance, but it’s problematic.  The output `y` is only assigned conditionally within the case statement when `state` is '0'. In other states, `y` is not assigned. This lack of explicit assignment for `y` in all states leads to latch inference for the `y` signal.  A synthesizer may infer a latch to store the last value of `y` until the next assignment.


**Example 2: Correct Implementation using Flip-Flops**

```vhdl
library ieee;
use ieee.std_logic_1164.all;

entity fsm_good is
  port (
    clk : in std_logic;
    rst : in std_logic;
    x : in std_logic;
    y : out std_logic
  );
end entity;

architecture behavioral of fsm_good is
  signal next_state : std_logic;
  signal current_state : std_logic;
begin
  process (clk, rst)
  begin
    if rst = '1' then
      current_state <= '0';
    elsif rising_edge(clk) then
      current_state <= next_state;
    end if;
  end process;

  process (current_state, x)
  begin
    case current_state is
      when '0' =>
        if x = '1' then
          next_state <= '1';
          y <= '1';
        else
          next_state <= '0';
          y <= '0';
        end if;
      when '1' =>
        next_state <= '0';
        y <= '0';
      when others =>
        null;
    end case;
  end process;
end architecture;
```

**Commentary:** This improved version explicitly uses a flip-flop (`current_state`) to store the current state.  The `next_state` signal computes the next state based on the current state and inputs.  Crucially, the output `y` is assigned a value in every state, eliminating the potential for latch inference.


**Example 3:  State Encoding for Robustness**

```vhdl
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity fsm_robust is
  port (
    clk : in std_logic;
    rst : in std_logic;
    x : in std_logic;
    y : out std_logic_vector(1 downto 0)
  );
end entity;

architecture behavioral of fsm_robust is
  type state_type is (IDLE, PROCESSING, DONE);
  signal current_state : state_type;
  signal next_state : state_type;
begin
  process (clk, rst)
  begin
    if rst = '1' then
      current_state <= IDLE;
    elsif rising_edge(clk) then
      current_state <= next_state;
    end if;
  end process;

  process (current_state, x)
  begin
    case current_state is
      when IDLE =>
        if x = '1' then
          next_state <= PROCESSING;
          y <= "01";
        else
          next_state <= IDLE;
          y <= "00";
        end if;
      when PROCESSING =>
          next_state <= DONE;
          y <= "10";
      when DONE =>
          next_state <= IDLE;
          y <= "00";
      when others =>
          null;
    end case;
  end process;
end architecture;
```

**Commentary:** This example uses an enumerated type for state representation, which enhances code readability and maintainability.  This approach is generally preferred for larger, more complex FSMs. The explicit assignment of `y` for each state further minimizes the risk of latch inference.  Using a type ensures clarity and prevents accidental omission of state transitions.


**3. Resource Recommendations:**

For deeper understanding of VHDL synthesis and FSM design, I recommend consulting the VHDL language reference manual, a comprehensive digital design textbook, and a synthesis tool's user manual specific to the tool you intend to utilize.  Pay close attention to sections concerning synthesis optimization and latch avoidance strategies.  Reviewing examples of well-designed FSMs in existing projects is also valuable.  Finally, thoroughly understanding the specifics of your chosen synthesis tool's reporting capabilities is essential for identifying and resolving potential latch inference issues.
