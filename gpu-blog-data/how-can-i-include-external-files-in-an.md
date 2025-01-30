---
title: "How can I include external files in an LPF file using Lattice Diamond?"
date: "2025-01-30"
id: "how-can-i-include-external-files-in-an"
---
Directives within Lattice Diamond's Logic Preference File (LPF) offer several avenues for integrating external file information, primarily to manage pin assignments and timing constraints across multiple design iterations or projects. I've encountered numerous scenarios where manually managing hundreds of pin declarations or complex timing constraints across different board revisions became unwieldy. Leveraging external files significantly improved design maintainability and reduced the risk of introducing errors when re-targeting similar logic across diverse hardware configurations.

The primary mechanism for including external files in an LPF file is the `INCLUDE` directive. This acts similarly to a preprocessor command, substituting the content of the referenced file directly into the current LPF at the point of inclusion. The syntax follows a simple format:

```lpf
INCLUDE "path/to/external_file.lpf";
```

This directive will process the contents of `external_file.lpf` as if it were directly written in the main LPF. Crucially, the relative paths are resolved based on the location of the *main* LPF file, not the location of any previously included LPF files. This needs careful consideration when using nested includes. It is best practice to use relative paths, rather than absolute paths to maximize portability. Absolute paths tie the project to a specific machine or file structure, which can cause issues when collaborating with others, or when migrating a project to a different development environment.

The flexibility of using `INCLUDE` allows the modularization of different aspects of the constraints. For instance, you might keep pin assignments for different board revisions in separate files, allowing for easy selection via the main LPF. Similarly, you could organize timing constraints based on clock domains, facilitating management of large, complex designs.

Beyond the basic `INCLUDE` directive, conditional inclusion can be achieved using a combination of global variables and conditional statements within the LPF. While not strictly an external file include mechanism, this approach allows different sections of the LPF to be processed based on runtime parameters, often established by the synthesis tool or through scripting. For instance, a global variable can be set through a TCL script, dictating which hardware variant should be targeted, which subsequently changes included files.

Let’s consider some specific practical examples:

**Example 1: Modular Pin Assignments:**

Suppose we have two hardware variants, ‘board_a’ and ‘board_b’ which share most of the logic but have different pinouts for some signals. To handle this, I’d set up the following file structure:

```
project_root/
    design.lpf
    board_a_pins.lpf
    board_b_pins.lpf
```

`design.lpf` would be structured as follows:

```lpf
// Global variable to select the board variant
GLOBAL board_variant "board_a";

// Conditional inclusion based on the global variable
IF {$board_variant == "board_a"} {
    INCLUDE "board_a_pins.lpf";
} ELSE {
    INCLUDE "board_b_pins.lpf";
}

// Define all other logic constraints
FREQUENCY "clk" 50 MHZ;
```
`board_a_pins.lpf` might contain:
```lpf
LOCATE  "led_r"    PIN "J12";
LOCATE  "led_g"    PIN "K10";
LOCATE  "led_b"    PIN "K11";
```

and `board_b_pins.lpf` might contain:
```lpf
LOCATE  "led_r"    PIN "L12";
LOCATE  "led_g"    PIN "M10";
LOCATE  "led_b"    PIN "M11";
```

In this case, by changing the `GLOBAL board_variant` variable, you can instantly switch between board definitions within Diamond without modifying the core logic or manually changing pin LOCATE statements. This approach greatly reduces the risk of error compared to manually editing all LOCATE statements whenever board variant changes. The conditional checks add an additional layer of control allowing the flexible choice of constraint definitions.

**Example 2: Clock Domain Specific Constraints:**

In a design with multiple clock domains, managing timing constraints is much cleaner using external files. Assume we have two clock domains, ‘clk_100’ and ‘clk_50’ driven by separate clock sources and requiring distinct timing constraints. I would use the following structure:

```
project_root/
    design.lpf
    clk_100_constraints.lpf
    clk_50_constraints.lpf
```

`design.lpf` would look like this:

```lpf
// Main LPF file including clock domain constraints
INCLUDE "clk_100_constraints.lpf";
INCLUDE "clk_50_constraints.lpf";

// Define all other LOCATE constraints and IO standards
LOCATE "clk_100" PIN "A1";
LOCATE "clk_50" PIN "B2";
```

`clk_100_constraints.lpf` would be defined as:
```lpf
// Timing constraints for the 100MHz clock domain
FREQUENCY "clk_100" 100 MHZ;
OFFSET IN "data_in" BEFORE "clk_100" 5 ns;
OFFSET OUT "data_out" AFTER "clk_100" 8 ns;
```

And `clk_50_constraints.lpf` would contain:
```lpf
// Timing constraints for the 50MHz clock domain
FREQUENCY "clk_50" 50 MHZ;
OFFSET IN "data_in_slow" BEFORE "clk_50" 10 ns;
OFFSET OUT "data_out_slow" AFTER "clk_50" 12 ns;
```

This modular approach simplifies maintenance and debugging. Each constraint is defined in a clearly labeled file, making it easier to identify and correct errors within specific clock domains. For example, if a new timing requirement was needed for the ‘clk_50’ domain, then the changes would be localized to the ‘clk_50_constraints.lpf’ file making modification fast and efficient and reducing chances of error.

**Example 3: Vendor Specific Pinouts:**

When using different package variants of an FPGA from the same family, pinouts can vary significantly. External include files can help easily re-target designs to alternate device packages. Suppose the same design is intended for a 144-pin and 256-pin package variant:

```
project_root/
    design.lpf
    package_144_pins.lpf
    package_256_pins.lpf
```

The main file `design.lpf` would look like this:

```lpf
// Select package variant
GLOBAL device_package "144";

IF {$device_package == "144"} {
   INCLUDE "package_144_pins.lpf"
} ELSE {
   INCLUDE "package_256_pins.lpf"
}

// Add constraints that are common to both package variants
FREQUENCY "sys_clk" 100 MHZ;

```

The file `package_144_pins.lpf` might contain:

```lpf
LOCATE "sys_clk" PIN "P11";
LOCATE "led_out" PIN "M10";
LOCATE "data_bus[0]" PIN "L9";
LOCATE "data_bus[1]" PIN "L10";
```
while `package_256_pins.lpf` contains:

```lpf
LOCATE "sys_clk" PIN "A1";
LOCATE "led_out" PIN "H20";
LOCATE "data_bus[0]" PIN "F12";
LOCATE "data_bus[1]" PIN "G12";

```

Switching the design between these pin packages, when using the same core logic, is simplified to changing the `GLOBAL device_package` variable. This eliminates the manual process of carefully altering hundreds of LOCATE statements, ensuring consistency and reducing opportunities for errors.

For further detailed information, I recommend consulting the official Lattice Diamond documentation specific to constraint management and LPF syntax. Additionally, the technical notes and application notes provided by Lattice Semiconductor for various FPGA families contain specific guidance on constraint definition and handling various design scenarios. Exploring example projects distributed by Lattice can also offer real-world practical usage of external includes, providing a practical understanding and insights into complex constraint management. Finally, a thorough reading of the Lattice Diamond user manual regarding LPF constraints should be considered essential to effectively manage external files within project.
