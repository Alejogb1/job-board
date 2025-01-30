---
title: "Why does concurrent array assignment in VHDL wipe the memory structure?"
date: "2025-01-30"
id: "why-does-concurrent-array-assignment-in-vhdl-wipe"
---
Concurrent assignment in VHDL, particularly when targeting memory structures, often results in the seemingly counterintuitive behavior of overwriting the entire memory rather than the single intended address. This arises from a fundamental difference in how VHDL handles concurrent assignments versus the sequential nature implied by many imperative programming languages. My experience designing several custom memory controllers using VHDL clarifies why this happens.

The core issue stems from VHDL's declarative and dataflow-driven approach. A concurrent assignment statement, such as `memory <= new_memory_value when condition else memory;`, doesn't operate like a procedural update. Instead, the statement is continuously evaluated. At a high level, think of it as a constantly re-evaluated expression determining the final value of the signal `memory`. The signal `memory` represents the entire memory structure; in most hardware implementations, this is a register array. The right side of the concurrent assignment must produce the entire new array for the memory.

When you attempt an assignment intending to modify a single location, like `memory(address) <= data`, you are actually creating a new memory array where only that single element differs, while the other elements remain undefined during an update due to implicit updates to the entire memory in concurrent VHDL. This leads to a situation where either an entirely new, partially initialized memory array is assigned (which, in synthesis, may be filled with an initial default value, often zero) or the previous value of the memory is used, but only if the conditional assignment is set up in such a way. This overwrite is the result of the fact that there is no inherent memory mechanism in a bare concurrent assignment. The assignment must be updated for the entire array, not just one element within it. Therefore the statement: `memory(address) <= data;` is syntactically invalid within a process or at the top-level of a design outside a process, but it is often a source of confusion. The syntax would be acceptable within a process if a memory assignment to the entirety of memory is implemented.

Here's a breakdown using three code examples to highlight common mistakes and effective solutions.

**Example 1: The Incorrect Approach - Concurrent Direct Assignment**

```vhdl
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity memory_incorrect is
    Port ( address : in  std_logic_vector(7 downto 0);
           data    : in  std_logic_vector(7 downto 0);
           write_en: in std_logic;
           clk     : in  std_logic;
           memory  : out std_logic_vector(7 downto 0)  array(0 to 255)
          );
end entity memory_incorrect;

architecture Behavioral of memory_incorrect is
    signal mem : std_logic_vector(7 downto 0)  array(0 to 255);
begin
    memory <= mem; -- Connect signal mem to memory output
    mem(to_integer(unsigned(address))) <= data when write_en = '1'; --Incorrect assignment

end Behavioral;
```

This example attempts to write data to the memory array `mem` at the index specified by `address`, concurrently. Critically, this is not how memory is implemented in hardware. It appears as a single statement that is always running. It creates a new memory with a modified element, but implicitly zeros all other elements on a concurrent update. This assignment is a common mistake I've observed, and the intended behavior of writing to a specific address is lost. The resulting memory will not behave as expected.

**Example 2: A Correct Approach - Sequential Access Within a Process**

```vhdl
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity memory_correct is
    Port ( address : in  std_logic_vector(7 downto 0);
           data    : in  std_logic_vector(7 downto 0);
           write_en: in std_logic;
           clk     : in  std_logic;
           memory  : out std_logic_vector(7 downto 0)  array(0 to 255)
          );
end entity memory_correct;

architecture Behavioral of memory_correct is
    signal mem : std_logic_vector(7 downto 0)  array(0 to 255);

begin
    memory <= mem; -- Connect signal mem to memory output
    process(clk)
    begin
        if rising_edge(clk) then
             if write_en = '1' then
                mem(to_integer(unsigned(address))) <= data;
            end if;
        end if;
    end process;
end Behavioral;
```

In this revised example, I've moved the memory update within a clocked process. The key difference here is that we are now implementing memory access on a single clock edge (or a condition). Within the process, sequential statements are executed in order, as one would expect from typical procedural programming. The assignment to `mem(to_integer(unsigned(address)))` now correctly updates only that specific memory location. The `mem` signal retains its value from previous clock cycles unless it is explicitly changed in the current cycle and thus behaves as a register array as intended. This approach accurately captures the intended behavior of a synchronous memory. This is also the typical method used to implement memory in hardware and how I would typically write it.

**Example 3: Another Correct Approach - Using a Function to Generate a New Memory State**

```vhdl
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity memory_correct_functional is
    Port ( address : in  std_logic_vector(7 downto 0);
           data    : in  std_logic_vector(7 downto 0);
           write_en: in std_logic;
           clk     : in  std_logic;
           memory  : out std_logic_vector(7 downto 0)  array(0 to 255)
          );
end entity memory_correct_functional;

architecture Behavioral of memory_correct_functional is
    signal mem : std_logic_vector(7 downto 0)  array(0 to 255);

    function update_memory (current_mem : std_logic_vector(7 downto 0) array(0 to 255);
                            address_in : std_logic_vector(7 downto 0);
                            data_in : std_logic_vector(7 downto 0)
                           ) return std_logic_vector(7 downto 0) array(0 to 255) is
        variable temp_mem : std_logic_vector(7 downto 0) array(0 to 255) := current_mem;
    begin
        temp_mem(to_integer(unsigned(address_in))) := data_in;
        return temp_mem;
    end function;


begin
    memory <= mem; -- Connect signal mem to memory output
    process(clk)
    begin
        if rising_edge(clk) then
            if write_en = '1' then
                mem <= update_memory(mem, address, data);
            end if;
        end if;
    end process;
end Behavioral;
```

This example demonstrates an alternative correct method using a function.  The function `update_memory` takes the current state of the memory array, the address, and the new data and returns the modified memory array. This function implements memory functionality in the same way as example two. This method can make the code easier to read in some circumstances, particularly if memory access becomes more complex (e.g., handling data masks).  The important takeaway is that the concurrent assignment to `mem` is now taking an *entire* new array as its right side and updating it on a per-clock cycle. This example is more verbose, but could offer an alternative for more complex designs.

**Key Takeaways and Recommendations:**

-   **Concurrent assignments operate on entire signals:** In VHDL, concurrent assignments update the entire target signal.  Attempting to modify individual elements directly with such assignments will overwrite the whole structure.

-   **Use clocked processes for sequential behavior:** For memory updates, employ clocked processes to ensure sequential access, and update the memory signal via sequential assignment. This is the proper method to implement a register array.

-   **Functions can aid readability for complex memory operations:**  Consider using functions to encapsulate the logic of generating new memory states, especially when dealing with complex memory access scenarios.

-   **Understand the dataflow paradigm:** Embrace VHDL's dataflow paradigm, which emphasizes continuous signal assignments and the propagation of data. Don’t expect VHDL's signals and processes to behave in the same way as imperative programming variables and statements.

-   **Study memory controller architectures:** Examine existing memory controller implementations (e.g. RAM interfaces) to grasp the best practices for coding memory structures in VHDL.

-   **Consult VHDL textbooks:** Deepen your knowledge with comprehensive VHDL textbooks which explain the language's nuances with greater depth than online tutorials, specifically the sections on concurrent and sequential assignment.

- **Practice through design examples:**  Build simple memory structures using the correct methods as demonstrated, such as single-port and dual-port RAMs.  This hands-on approach significantly improves understanding.

By understanding these fundamental principles, you can avoid the common pitfall of inadvertently wiping out your memory structures through concurrent assignment and confidently design reliable hardware.
