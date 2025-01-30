---
title: "Can Verilog pure functions be referenced across modules?"
date: "2025-01-30"
id: "can-verilog-pure-functions-be-referenced-across-modules"
---
Verilog's support for pure functions, while conceptually straightforward, presents subtle complexities concerning their cross-module instantiation and accessibility.  My experience designing high-performance ASICs for a major semiconductor manufacturer highlighted a crucial limitation: while the *declaration* of a pure function can be globally visible,  the function's *definition* must be available within the module that intends to utilize it. This is fundamentally due to Verilog's hierarchical design structure and its compilation process.

**1. Clear Explanation:**

Verilog's module system is inherently hierarchical.  Each module encapsulates a specific functionality, defining its interfaces (ports) and internal behavior.  A pure function, in essence, is a self-contained block of code intended to perform a calculation based purely on its input arguments, returning a deterministic output without side effects or altering global state.  The key here is *deterministic* –  for the same input, the function always produces the same output.  This deterministic nature is crucial for verification and synthesis.

The problem arises when attempting to utilize a pure function declared in one module from another. The Verilog compiler operates on a module-by-module basis.  When compiling a specific module, the compiler only has access to the code within that module and its immediate parent modules (through hierarchical instantiation).  Consequently, even if a pure function is declared in a header file or a globally accessible scope (though this is generally discouraged for larger designs for maintainability), its *definition* — the actual code that implements the function — remains unavailable to other modules unless explicitly included within their scope.  Simply declaring the function's signature (its inputs and outputs) without its body is insufficient for functional invocation.

A common misunderstanding lies in assuming that the `function` keyword provides automatic global visibility.  This is incorrect. The `function` keyword simply declares a procedural block with specific properties (no side effects, deterministic), but it does not inherently enable cross-module referencing in the same way that, for example, a `typedef` might influence global data types.  Thus, a proper understanding of scope and the compilation process is pivotal.

**2. Code Examples with Commentary:**

**Example 1: Incorrect Cross-Module Reference:**

```verilog
// module1.v
module module1;
  function integer pure_add;
    input integer a;
    input integer b;
    pure_add = a + b;
  endfunction
endmodule

// module2.v
module module2;
  integer sum;
  module1 m1; // Instantiate module1

  initial begin
    sum = m1.pure_add(5, 3); // Incorrect: pure_add not directly accessible
    $display("Sum: %d", sum);
  end
endmodule
```

This example attempts to call `pure_add` directly from `module2`.  This will result in a compilation error because `module2` does not have access to the *definition* of `pure_add`.  While `module1`'s declaration might be visible (depending on the compilation and include directives), the compiler needs the complete function body to generate the necessary code.


**Example 2: Correct Use Within the Same Module:**

```verilog
// module3.v
module module3;
  function integer pure_multiply;
    input integer a;
    input integer b;
    pure_multiply = a * b;
  endfunction

  integer product;

  initial begin
    product = pure_multiply(4, 6);
    $display("Product: %d", product);
  end
endmodule
```

This demonstrates the correct usage. The `pure_multiply` function is both declared and defined within `module3`, allowing seamless invocation within the module.


**Example 3: Correct Cross-Module Reference using Instantiation and Port Connection:**

```verilog
// module4.v
module module4 (input integer a, input integer b, output integer result);
  function integer pure_subtract;
    input integer a;
    input integer b;
    pure_subtract = a - b;
  endfunction

  always @(*) begin
    result = pure_subtract(a,b);
  end
endmodule

// module5.v
module module5;
  integer x = 10;
  integer y = 5;
  integer diff;
  module4 m4 (x, y, diff);

  initial begin
    #10 $display("Difference: %d", diff);
  end
endmodule
```

In this example, we encapsulate the pure function within `module4`.  `module5` instantiates `module4`, effectively gaining access to the subtraction functionality indirectly. The `pure_subtract` function is hidden within `module4`'s internal scope, ensuring modularity and encapsulation, a key tenet of good Verilog design. This is the recommended approach for accessing effectively 'pure' functionality across modules.

**3. Resource Recommendations:**

*   The Verilog Language Reference Manual:  This provides the definitive specification of the language, clarifying semantic details and resolving ambiguities.
*   A comprehensive Verilog textbook:  A well-structured text focusing on advanced topics will provide deeper understanding of module hierarchies, scope rules, and effective design practices.  Pay close attention to sections on procedural blocks, functions, and tasks.
*   HDL Simulation and Verification Tools Documentation:  Understanding the simulator's behavior in handling hierarchical designs is essential for debugging cross-module interactions.


In summary, while Verilog supports the concept of pure functions, their cross-module usage necessitates careful attention to modularity and hierarchical design.  Direct cross-module referencing of the pure function definition is not supported; instead, encapsulating the function within a module and instantiating that module is the correct method. This approach maintains encapsulation, improves design clarity, and facilitates verification and synthesis.  Ignoring these fundamental aspects of Verilog's structure can lead to significant compilation errors and design flaws.
