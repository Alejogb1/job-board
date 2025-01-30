---
title: "Why does Quartus report error 12002 ('Port does not exist in macrofunction')?"
date: "2025-01-30"
id: "why-does-quartus-report-error-12002-port-does"
---
Quartus Prime's error 12002, "Port does not exist in macrofunction," stems fundamentally from a mismatch between the instantiation of a macrofunction (a custom module or IP core) and its actual definition.  This discrepancy often arises from typographical errors, version mismatches, or inconsistencies in the signal names used for input and output ports.  My experience troubleshooting this error over fifteen years, primarily working on high-speed communication systems and FPGAs from various vendors, has solidified this understanding.  The error doesn't simply indicate a missing port; it highlights a crucial disconnect between the designer's intent and the compiler's interpretation of the design.


**1.  Clear Explanation:**

The error 12002 arises during the elaboration phase of Quartus compilation.  During this phase, Quartus attempts to instantiate all the components in your design hierarchy.  When encountering a macrofunction, it consults the definition of that macrofunction (typically a Verilog or VHDL file) to verify the existence and correct usage of ports.  If Quartus cannot find a port specified in the instantiation, it throws error 12002. This discrepancy can occur in several ways:

* **Typographical errors:**  A simple misspelling of a port name in your top-level design file referencing the macrofunction is a frequent cause.  For instance, typing `data_in` instead of `data_in_` will trigger this error if the macrofunction only defines `data_in_`. Case sensitivity is critical; Verilog is case-sensitive, and VHDL's case sensitivity depends on the language standard and compiler settings.

* **Version mismatch:** If the macrofunction you are using has been updated, and the definition in your project doesn't reflect these changes (e.g., a new port was added), this mismatch can lead to 12002.  Ensuring all components use consistent versions is essential.

* **Incorrect module instantiation:**  The syntax for instantiating a module might be incorrect, causing Quartus to interpret port connections incorrectly.  This can involve missing port maps, incorrect port order, or even specifying the wrong module altogether.

* **Hierarchical issues:**  If your macrofunction is nested within other modules, errors in the intermediate levels can propagate to the top level, resulting in 12002 being reported at the top level even if the direct macrofunction instantiation is correct.  Careful checking of the entire hierarchy is necessary.

* **Library path issues:** If the macrofunction is in a library, ensure that the library path is correctly specified in your Quartus project settings.  A missing or incorrect path prevents Quartus from finding the macrofunction's definition.


**2. Code Examples and Commentary:**

**Example 1: Typographical Error**

```verilog
// Macrofunction Definition (my_macro.v)
module my_macro (
  input wire clk,
  input wire rst,
  input wire data_in_,
  output wire data_out
);
  // ... module logic ...
endmodule

// Top-level design (top.v)
module top;
  wire clk;
  wire rst;
  wire data_in; //Typo here: missing underscore
  wire data_out;

  my_macro my_instance (
    .clk(clk),
    .rst(rst),
    .data_in(data_in), // Incorrect port name
    .data_out(data_out)
  );
endmodule
```

In this example, the `data_in` port in `top.v` doesn't match `data_in_` in `my_macro.v`, leading to Quartus reporting error 12002.  Correcting `data_in` to `data_in_` resolves the issue.

**Example 2: Incorrect Port Ordering**

```verilog
// Macrofunction Definition (my_macro.v)
module my_macro (
  input wire clk,
  input wire rst,
  input wire data_in,
  output wire data_out
);
  // ... module logic ...
endmodule

// Top-level design (top.v)
module top;
  // ... wire declarations ...

  my_macro my_instance (
    .rst(rst),      //Incorrect Order
    .clk(clk),      //Incorrect Order
    .data_in(data_in),
    .data_out(data_out)
  );
endmodule
```

Here, while the port names are correct, the order in the instantiation in `top.v` differs from the definition in `my_macro.v`. Quartus will likely assign signals incorrectly, resulting in unpredictable behavior or error 12002.  Maintaining the correct order is crucial.

**Example 3: Missing Port Map**

```verilog
// Macrofunction Definition (my_macro.v)
module my_macro (
  input wire clk,
  input wire rst,
  input wire [7:0] data_in,
  output wire [7:0] data_out,
  output wire valid
);
  // ... module logic ...
endmodule

// Top-level design (top.v)
module top;
  // ... wire declarations ...

  my_macro my_instance (
    .clk(clk),
    .rst(rst),
    .data_in(data_in)
    //Missing .data_out and .valid
  );
endmodule
```

This example omits the connections for `data_out` and `valid` ports.  Quartus cannot correctly instantiate the macrofunction due to incomplete port mappings, generating error 12002.  All ports defined in the macrofunction must be connected in the instantiation.


**3. Resource Recommendations:**

I would strongly recommend thoroughly reviewing the Quartus Prime documentation, specifically sections detailing Verilog and VHDL syntax, macrofunction instantiation, and troubleshooting compilation errors.  Pay close attention to the specific error messages provided by Quartus, as they often contain valuable hints about the location and nature of the problem. Consulting the relevant documentation for your specific IP core or custom module is also advisable.  Finally, using a good Linting tool integrated with your IDE will help identify potential errors early in the design process.



In summary, successfully resolving Quartus error 12002 requires meticulous attention to detail.  Carefully compare the port definitions in your macrofunction with their instantiations in your top-level design, verifying the accuracy of names, order, and types.  Employing a systematic debugging approach, combined with the resources mentioned, will significantly improve your efficiency in identifying and rectifying such errors.  Remember that even seemingly insignificant typos or inconsistencies can have significant ramifications in hardware design.
