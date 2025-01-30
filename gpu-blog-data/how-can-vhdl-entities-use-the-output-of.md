---
title: "How can VHDL entities use the output of one as input to another?"
date: "2025-01-30"
id: "how-can-vhdl-entities-use-the-output-of"
---
In VHDL, connecting the output of one entity to the input of another, a fundamental aspect of digital system design, relies on the concept of signal declarations and the instantiation of component instances. Signals act as the conductors, transmitting data between different architectural blocks. This process of interconnecting entities is crucial for creating hierarchical designs, thereby enabling the construction of complex digital systems from simpler, manageable units. My practical experience developing FPGA-based communication systems has consistently emphasized the critical nature of accurate signal typing and proper instantiation within a larger architecture.

The core principle revolves around defining signals in an architecture that then serve as the connecting medium between entity instances. When instantiating a component, the port map clause is employed, specifying the correspondence between the ports of the instantiated entity and the signals defined within the architecture. Correctly mapping ports to signals with compatible data types is paramount; any mismatch will lead to compilation errors or, worse, unpredictable runtime behavior. Without understanding this interplay, it’s easy to end up with disconnected or incorrectly wired sub-modules, regardless of the individual components being flawless.

The declaration and use of signals occur at the architecture level. A signal is declared using the keyword `signal` followed by its name, data type, and optionally an initial value. For instance, `signal my_signal : std_logic;` declares a signal named `my_signal` of type `std_logic`. When an entity needs to utilize another’s output, it is vital to declare a signal within the architecture encompassing both entities that has a type that matches the output port. The instance port map clause ensures this newly declared signal connects to the desired output and input ports. Signal declarations are local to a given architecture. This provides encapsulation, as signals are not directly accessible outside of the architecture in which they are defined, avoiding naming conflicts when working in large teams. The architecture encapsulates the component and their connections, creating a modular approach to development.

Let us consider an example with two basic entities: a simple inverter and a buffer. The inverter negates its input and outputs it, while the buffer passes its input unchanged.

**Example 1: Inverter and Buffer Connection**

```vhdl
entity inverter is
  port (
    input_signal : in  std_logic;
    output_signal : out std_logic
  );
end entity inverter;

architecture behavioral of inverter is
begin
  output_signal <= not input_signal;
end architecture behavioral;

entity buffer is
  port (
    input_signal : in  std_logic;
    output_signal : out std_logic
  );
end entity buffer;

architecture behavioral of buffer is
begin
  output_signal <= input_signal;
end architecture behavioral;

entity top_level is
  port (
    input_data  : in  std_logic;
    output_data : out std_logic
  );
end entity top_level;

architecture structural of top_level is
  signal intermediate_signal : std_logic;
  component inverter
    port (
      input_signal : in  std_logic;
      output_signal : out std_logic
    );
  end component;
  component buffer
    port (
      input_signal : in  std_logic;
      output_signal : out std_logic
    );
  end component;

begin
  inv_inst : inverter port map (
    input_signal => input_data,
    output_signal => intermediate_signal
  );
  buf_inst : buffer port map (
    input_signal => intermediate_signal,
    output_signal => output_data
  );
end architecture structural;
```

In this example, the `top_level` entity interconnects the `inverter` and `buffer` components. A signal named `intermediate_signal` of type `std_logic` is declared within the `structural` architecture. The `inverter` instance, `inv_inst`, takes the input signal `input_data` and generates output `intermediate_signal`.  Subsequently,  `intermediate_signal` is used as the input signal to the `buffer` instance, `buf_inst`, with `output_data` getting the final output. This demonstrates the typical signal declaration and port mapping strategy for connecting two entities. The component declaration section is not strictly required if we instantiate entities in the same project (it’s considered implicitly declared), but it's good practice when considering more complex designs or code organization.

Consider a slightly more complex scenario where we introduce a delay element, implemented with a process. The signal we use to connect this component will likely have a different timing characteristic than the input of the buffer. This introduces subtleties in how we manage timing using signals.

**Example 2: Delay Element and Buffer Connection**

```vhdl
entity delay_element is
  port (
    input_signal  : in  std_logic;
    output_signal : out std_logic
  );
end entity delay_element;

architecture behavioral of delay_element is
begin
  process(input_signal)
  begin
   -- Simplified model, single clock domain.  This process isn't synthesizable in practice
    wait for 10 ns;
    output_signal <= input_signal;
  end process;
end architecture behavioral;


entity top_level_delay is
  port (
    input_data  : in  std_logic;
    output_data : out std_logic
  );
end entity top_level_delay;

architecture structural of top_level_delay is
  signal delayed_signal : std_logic;
  component delay_element
    port (
      input_signal  : in  std_logic;
      output_signal : out std_logic
    );
  end component;
  component buffer
    port (
      input_signal : in  std_logic;
      output_signal : out std_logic
    );
  end component;

begin
  delay_inst : delay_element port map (
    input_signal => input_data,
    output_signal => delayed_signal
  );

  buf_inst : buffer port map (
    input_signal => delayed_signal,
    output_signal => output_data
  );
end architecture structural;
```

Here, `top_level_delay` connects a `delay_element` to a `buffer`. The intermediate signal, `delayed_signal`, carries the output of the `delay_element`, which is then fed to the buffer. The important element to note is that `delayed_signal`, even though it has the same logical type as the input signal `input_data`, will now have a delay associated with it. This shows the importance of signal connections, not just the types they carry. When designing with clock edges it’s critical to understand timing budgets between components that are established via the signals connecting them.

Finally, a multi-bit example should clarify the versatility of this approach. Using VHDL vectors of `std_logic`, one can effectively handle buses.

**Example 3: 8-bit Inverter and 8-bit Buffer Connection**

```vhdl
entity inverter8 is
  port (
    input_signal : in  std_logic_vector(7 downto 0);
    output_signal : out std_logic_vector(7 downto 0)
  );
end entity inverter8;

architecture behavioral of inverter8 is
begin
  output_signal <= not input_signal;
end architecture behavioral;

entity buffer8 is
  port (
    input_signal : in  std_logic_vector(7 downto 0);
    output_signal : out std_logic_vector(7 downto 0)
  );
end entity buffer8;

architecture behavioral of buffer8 is
begin
  output_signal <= input_signal;
end architecture behavioral;

entity top_level_8bit is
  port (
    input_data  : in  std_logic_vector(7 downto 0);
    output_data : out std_logic_vector(7 downto 0)
  );
end entity top_level_8bit;

architecture structural of top_level_8bit is
  signal intermediate_signal : std_logic_vector(7 downto 0);
  component inverter8
    port (
      input_signal : in  std_logic_vector(7 downto 0);
      output_signal : out std_logic_vector(7 downto 0)
    );
  end component;
  component buffer8
    port (
      input_signal : in  std_logic_vector(7 downto 0);
      output_signal : out std_logic_vector(7 downto 0)
    );
  end component;

begin
  inv_inst : inverter8 port map (
    input_signal => input_data,
    output_signal => intermediate_signal
  );
  buf_inst : buffer8 port map (
    input_signal => intermediate_signal,
    output_signal => output_data
  );
end architecture structural;
```

In this final scenario, `intermediate_signal` is a vector of 8 bits (`std_logic_vector(7 downto 0)`). The principle of connection remains the same: the output of the `inverter8` is connected to the input of `buffer8` through this intermediate signal. This highlights how signal declarations support the creation of multi-bit digital circuits.

For learning more in depth VHDL techniques, I recommend the textbooks "VHDL Primer" by J. Bhasker and "Digital Design: Principles and Practices" by John F. Wakerly. These resources detail the principles discussed, as well as the syntax of the language. Furthermore, manufacturer-specific documentation (e.g., Xilinx, Intel) should always be consulted for detailed information on synthesis and implementation-specific requirements. They are critical for understanding the implementation nuances and design constraints.
