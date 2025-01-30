---
title: "Why am I getting Quartus error 10170 when trying to instantiate a soft processor design?"
date: "2025-01-30"
id: "why-am-i-getting-quartus-error-10170-when"
---
Error 10170 in Intel’s Quartus Prime software, specifically “Cannot find design unit ‘<entity_name>’,” commonly arises when the Quartus compiler cannot locate the hardware description language (HDL) entity you are attempting to instantiate within your project’s scope. This situation isn't typically indicative of a design flaw in the entity itself, but rather an issue with how the project is configured, the synthesis process, or the naming conventions being used. I’ve personally encountered this frequently, particularly when working with complex hierarchical designs and soft processor cores within Intel FPGA environments. Based on my experience, several interconnected root causes contribute to this error.

The primary reason for this error is that the synthesis process, managed by Quartus, operates on a project-specific view of the design. It maintains an internal database of all recognized HDL entities. When you instantiate an entity within a top-level module, the compiler searches this database. If the entity, whether it’s a custom block or a generated processor core, has not been successfully processed into the database or if there is a discrepancy in its naming, it becomes invisible to the instantiation process, triggering Error 10170. This could stem from files not being correctly added to the project, compilation order issues, or a mismatch between declared names and instantiated names.

First, consider the project structure. Quartus needs to know about all the HDL source files that form your design. If the source file defining your soft processor core is not explicitly added to the Quartus project, the compiler won’t include it in its internal database. You can confirm this in the ‘Files’ tab within the Quartus project manager. Even if the file physically exists within the project directory on disk, it won’t be recognized unless it’s explicitly added as a design source. Additionally, the file might be added but might be specified as a simulation-only file, or even be inadvertently excluded from the compilation process during some user error.

Secondly, the order in which files are compiled can also influence this error. Quartus compiles modules in a bottom-up manner, starting with the lowest-level modules and working its way up. If you attempt to instantiate an entity in a higher-level module before the lower-level module defining the entity has been successfully synthesized, Error 10170 will appear. This is often the case with soft processor cores, which have a significant number of underlying files and a dependency structure. The core’s generation process must complete successfully before any module instantiating it can be synthesized. In more complex designs, it’s crucial to be mindful of synthesis dependencies.

Finally, and this is quite prevalent, naming inconsistencies are a common culprit. The entity name used in the instantiation must exactly match the entity declaration in the corresponding HDL file. This includes casing; for example, a module declared as 'MyProcessor' will not match an instantiation like 'myprocessor'. Further, consider the differences if an IP core’s parameters are changed, requiring a new instantiation of it in your code with parameters included. Even with parameterized entities, there is an inherent entity name, and this name needs to match the instantiation name.

To illustrate these issues with code examples, I will use VHDL. These examples focus specifically on illustrating common pitfalls with the instantiation of a soft core.

**Example 1: File not added to project**

Assume you have a soft processor core generated as an IP, and its instantiation is in the file `top_level.vhd`. The generated core itself is defined in a file called `my_soft_processor.vhd`.

```vhdl
-- top_level.vhd
library ieee;
use ieee.std_logic_1164.all;

entity top_level is
  port (
     -- Some top-level ports
  );
end entity top_level;

architecture behavioral of top_level is
  component my_soft_processor
    -- Ports of the core, typically complex
     port (
       clk : in std_logic;
       reset : in std_logic;
       data_out : out std_logic_vector(7 downto 0);
       -- Other ports
    );
  end component;
  signal core_clk : std_logic := '0';
  signal core_reset : std_logic := '1';
  signal core_data : std_logic_vector(7 downto 0);
begin
  soft_processor_inst : my_soft_processor
    port map(
     clk => core_clk,
      reset => core_reset,
      data_out => core_data
      -- Other signals
    );
  -- Rest of top-level design
end architecture behavioral;
```

Here, the entity `my_soft_processor` is instantiated, but if `my_soft_processor.vhd` is *not* added to the Quartus project, you will receive Error 10170. Quartus would have no knowledge of the existence or structure of the `my_soft_processor` entity, thus failing to synthesize it. The fix is straightforward: add `my_soft_processor.vhd` to the project using the project manager's file add function.

**Example 2: Compilation Order Issues**

Continuing with `top_level.vhd` and the core definition in `my_soft_processor.vhd`, assume both files are correctly added to the project. However, let’s consider that a complex IP generation process is needed, and the generated output of this process produces many more files which are automatically added by Quartus. A second instantiation exists in `processor_wrapper.vhd` which also uses the `my_soft_processor.vhd` definition. `top_level.vhd` instantiates `processor_wrapper.vhd`.

```vhdl
-- processor_wrapper.vhd
library ieee;
use ieee.std_logic_1164.all;

entity processor_wrapper is
  port (
      clk : in std_logic;
       reset : in std_logic;
       data_out : out std_logic_vector(7 downto 0)
       );
end entity processor_wrapper;

architecture behavioral of processor_wrapper is

  component my_soft_processor
    -- Ports of the core, typically complex
     port (
       clk : in std_logic;
       reset : in std_logic;
       data_out : out std_logic_vector(7 downto 0);
       -- Other ports
    );
  end component;
  signal core_clk : std_logic := '0';
  signal core_reset : std_logic := '1';
  signal core_data : std_logic_vector(7 downto 0);
begin
  soft_processor_inst : my_soft_processor
    port map(
     clk => core_clk,
      reset => core_reset,
      data_out => core_data
      -- Other signals
    );
end architecture behavioral;

```
If `top_level.vhd` is compiled before all of the generated output files of the IP core generation have completed, or `my_soft_processor.vhd` itself, then this error can still appear. This situation requires that Quartus has generated *all* files required by the IP and added them to the database of the design. The compilation process can be affected by dependency settings and by user interactions with the compiler. The solution would be to ensure that no compilation errors exist prior to synthesis and to allow time for the entire IP core generation process to complete.

**Example 3: Naming Discrepancy**

Again using `top_level.vhd`, assume the entity is correctly added to the project and the compilation order is correct. Let’s consider this code:

```vhdl
-- top_level.vhd
library ieee;
use ieee.std_logic_1164.all;

entity top_level is
  port (
     -- Some top-level ports
  );
end entity top_level;

architecture behavioral of top_level is
  component My_Soft_Processor
     port (
       clk : in std_logic;
       reset : in std_logic;
       data_out : out std_logic_vector(7 downto 0);
       -- Other ports
    );
  end component;
  signal core_clk : std_logic := '0';
  signal core_reset : std_logic := '1';
  signal core_data : std_logic_vector(7 downto 0);
begin
  soft_processor_inst : my_soft_processor
    port map(
      clk => core_clk,
      reset => core_reset,
      data_out => core_data
    );
end architecture behavioral;
```

In this example, the component declaration specifies `My_Soft_Processor` with a capital “M”, “S” and “P”. However, the instantiation uses `my_soft_processor` with all lowercase characters. Even though the intended design unit is present, the name mismatch will trigger Error 10170.  The fix here is to ensure that the instantiation name exactly matches the declared component name. These subtle variations in casing are a major source of error.

To mitigate Error 10170, I strongly suggest the following practices. First, carefully review the ‘Files’ tab in Quartus to ensure all required HDL files are part of the project and correctly categorized. Check the file’s settings within the project as well, making sure it is not set to simulation only or some other non-synthesis configuration. Second, examine the compilation order, particularly if you’re using complex IP cores; allow for the entire generation process to complete without user intervention. Third, double-check that component declarations and instantiations have the exact same spelling and capitalization.  Finally, after modifying project structure, it’s often beneficial to clean and recompile the entire project to eliminate any lingering cached data or partial synthesis states.

For further resources, the official Intel FPGA documentation and tutorials, specifically the Quartus Prime handbook, provide detailed information on project management, compilation, and debugging techniques, and can be found via Intel’s official website. Furthermore, the documentation for individual IP cores, such as soft processor cores, often have project integration guidelines that help mitigate these issues. Additionally, user groups, online forums, and similar channels often contain detailed troubleshooting and resolution guides for specific error cases including Error 10170. These resources are generally sufficient to identify and resolve this instantiation error.
