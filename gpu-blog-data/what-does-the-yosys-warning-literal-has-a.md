---
title: "What does the Yosys warning 'Literal has a width of 8 bits' mean?"
date: "2025-01-30"
id: "what-does-the-yosys-warning-literal-has-a"
---
The Yosys warning "Literal has a width of 8 bits" signifies a type mismatch between a literal value used in your HDL code and the expected width of the associated signal or variable.  This is not necessarily an error, but it indicates a potential source of unintended behavior, particularly in synthesis and subsequent hardware implementation.  I've encountered this frequently during my years developing FPGA-based systems, most notably while migrating designs between different synthesis tools and hardware platforms.  The core issue stems from implicit type conversions and the way Yosys infers the width of signals based on their usage.

**1. Explanation**

HDL languages like Verilog and SystemVerilog are strongly typed, but they also exhibit a degree of type flexibility through implicit width inference.  When you assign a literal value (e.g., `8'b10110001`, `10`, `-5`) to a signal or variable, Yosys needs to determine if the literal's width matches the declared width of the target. If the literal's width differs, the warning is triggered.  The width of a literal is determined by its prefix.  `8'b10110001` explicitly defines an 8-bit binary literal.  However, if you simply write `10`, Yosys must infer the width, often leading to an 8-bit default depending on the specific synthesis tool's settings and the context of its usage.  This default may not always align with the declared width of your signal, causing the warning.

The consequence of this mismatch can be subtle.  In some instances, Yosys may implicitly extend or truncate the literal to match the target signal width.  Extension usually involves sign extension (for signed values) or zero-extension (for unsigned values). Truncation simply removes the most significant bits.  While this may seem benign, it can lead to unexpected behavior, especially if you intended a specific width for your literal.  Consider a scenario where an 8-bit unsigned literal is implicitly truncated to 4 bits: `0b11110000` becomes `0b0000`. The data is corrupted. Similarly, if a signed value is widened without sign extension, the sign bit might not be properly carried over leading to incorrect results in arithmetic operations.   A more critical outcome involves a logic mismatch.  An 8-bit literal might, in some contexts, end up treated as a 32-bit value, potentially causing a significant increase in logic resource utilization.

The warning itself is not a hard error, but it demands attention because it exposes a potential design flaw that could manifest during implementation.  It highlights areas where your code might be less portable and might lead to discrepancies between simulation and hardware behavior.


**2. Code Examples with Commentary**

**Example 1: Implicit Width Inference**

```verilog
module width_inference;
  reg [7:0] data;
  reg [3:0] data_short;

  initial begin
    data = 10; // Warning: Literal has a width of 8 bits.
    data_short = 10; // Warning: Literal has a width of 8 bits.
    $display("data: %h", data);
    $display("data_short: %h", data_short);
  end
endmodule
```

This example shows a clear instance of implicit width inference. Although `10` is assigned to both `data` (8-bit) and `data_short` (4-bit), Yosys will typically infer an 8-bit width for the literal `10`. This will result in the warning because the inferred width doesn't match `data_short`.  The assignment to `data` will probably not result in a warning since the default 8-bit width matches the variable width.  However, depending on Yosys's configuration, it might still trigger a warning for consistency.  In any case, this is bad practice as you lack explicit control over the literal's width, leaving it to potentially unpredictable behavior across different tools.

**Example 2: Explicit Width Specification**

```verilog
module explicit_width;
  reg [7:0] data;
  reg [3:0] data_short;

  initial begin
    data = 8'd10; // No warning.
    data_short = 4'd10; // No warning.
    $display("data: %h", data);
    $display("data_short: %h", data_short);
  end
endmodule
```

This example demonstrates the best practice – explicitly defining the width of your literals.  The `8'd10` and `4'd10` explicitly declare the literal's width as 8 bits and 4 bits, respectively. This completely eliminates the width mismatch and thus the warning.  This approach ensures both portability and predictable behavior, which are crucial for hardware design.


**Example 3:  Potential for Sign Extension Issues**

```systemverilog
module sign_extension;
  reg signed [7:0] signed_data;
  reg signed [15:0] signed_data_wide;

  initial begin
    signed_data = -10; // Warning might appear, depends on Yosys configuration
    signed_data_wide = -10; // Warning might appear, depends on Yosys configuration
    $display("signed_data: %d", signed_data);
    $display("signed_data_wide: %d", signed_data_wide);
  end
endmodule
```

In this SystemVerilog example, the assignment of `-10` to signed variables highlights a potential issue regarding sign extension. While Yosys might infer a width for `-10`, it should perform sign extension when widening the value.  However, the precise behavior depends on the Yosys version and its settings. A mismatch in how `-10` is handled (e.g., an implicit zero-extension) between the 8-bit and 16-bit variables could lead to inaccurate results. Explicitly specifying the width of the literals (`8'sd-10` and `16'sd-10`) eliminates this ambiguity.


**3. Resource Recommendations**

For a deeper understanding of HDL coding best practices, I recommend consulting the language reference manuals for Verilog and SystemVerilog.  Furthermore, thoroughly review the documentation for your chosen synthesis tool (e.g., Yosys, Vivado, Quartus Prime).  Finally, studying advanced digital design textbooks will greatly enhance your ability to anticipate and resolve these kinds of warnings effectively.  Understanding the synthesis process itself is vital for interpreting warnings from synthesis tools accurately.
