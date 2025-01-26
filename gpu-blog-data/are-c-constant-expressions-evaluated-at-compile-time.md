---
title: "Are C constant expressions evaluated at compile time?"
date: "2025-01-26"
id: "are-c-constant-expressions-evaluated-at-compile-time"
---

Based on my experience optimizing embedded systems code, the evaluation of constant expressions in C at compile time is not merely an optional optimization; it's a fundamental aspect of the language's design, enabling significant performance benefits and allowing for crucial compile-time error checking. Specifically, while C mandates that constant expressions, when used in contexts that require them, *must* be evaluable at compile time, the extent to which the compiler aggressively performs this evaluation is dependent on the compiler itself and its optimization level. This means that while C guarantees constant evaluation *when needed*, not all expressions which could be evaluated at compile-time necessarily will be by all compilers at all settings.

The definition of a constant expression is crucial. These expressions consist of literal values (e.g., integers, floating-point numbers, characters, string literals), constant identifiers, and operators that, when combined, result in a value that is known at compile time. This contrasts sharply with run-time expressions, which involve variables, function calls, and other operations whose results are determined only during the execution of the program.  The requirement for compile-time evaluation for constant expressions stems from their use in contexts where the resulting value must be known during compilation, such as array sizes, `case` labels in `switch` statements, initializers for static or global variables, bit-field sizes, and preprocessor directives such as `#if` and `#elif`. Failure to evaluate such expressions during compilation would fundamentally break these language features.

Let's illustrate with a few code examples:

**Example 1: Array Size**

```c
#include <stdio.h>

#define ARRAY_SIZE (5 * 2 + 1) // Constant expression

int main() {
    int myArray[ARRAY_SIZE];   // ARRAY_SIZE *must* be evaluated at compile-time

    for(int i = 0; i < ARRAY_SIZE; i++) {
        myArray[i] = i * 2;
        printf("%d ", myArray[i]);
    }

    printf("\n");
    return 0;
}
```

In this first example, `ARRAY_SIZE` is defined using a preprocessor macro. The expression `(5 * 2 + 1)` is a constant expression. The compiler, when processing the line `int myArray[ARRAY_SIZE];`, requires the size of the array to be known at compile time. It is not permitted to defer the calculation of `5 * 2 + 1` to run-time. Therefore, the expression *must* be evaluated during compilation resulting in a size of 11 for the array. The program then proceeds to initialize the array with even numbers and print them to the console. A compiler that did not evaluate this at compile time would reject the code with a compilation error. This underscores the necessity of compile-time evaluation in this specific context.

**Example 2: `switch` Statement Cases**

```c
#include <stdio.h>

const int CASE_A = 10;    // Constant integer variable
const int CASE_B = CASE_A + 5;  // Constant expression

int main() {
    int value = 15;
    switch (value) {
        case CASE_A:
            printf("Case A\n");
            break;
        case CASE_B:
            printf("Case B\n");
            break;
        default:
           printf("Default case\n");
    }
    return 0;
}
```

Here, `CASE_A` and `CASE_B` are declared as `const int` variables, but their initializers are constant expressions.  Specifically, `CASE_A` is initialized with a literal value (10), and `CASE_B` is initialized with a constant expression `CASE_A + 5`. During compilation, when the compiler processes the `switch` statement, these case labels must be resolved to their actual integer values before any run-time code can be generated for the `switch` statement. In this instance, the program evaluates `CASE_B` as 15 at compile time. Because value is 15, it will print "Case B" when the program is run. The C language mandates compile time evaluation for switch case labels as run time evaluation will fail to produce the necessary jump table.

**Example 3:  `#if` Preprocessor Directive**

```c
#include <stdio.h>

#define CONFIG_OPTION 1

#if CONFIG_OPTION == 1
    #define MY_VALUE 100
#elif CONFIG_OPTION == 2
    #define MY_VALUE 200
#else
    #define MY_VALUE 300
#endif

int main() {
    printf("My value: %d\n", MY_VALUE);
    return 0;
}
```

This example utilizes a preprocessor `#if` directive.  The condition within the `#if` (`CONFIG_OPTION == 1`) is a constant expression.  The preprocessor *must* evaluate this condition during compilation, and based on the result, the preprocessor will then selectively compile the code. Here, since `CONFIG_OPTION` is 1, the `MY_VALUE` is defined as 100 and is evaluated at compile time. This evaluation is not done during runtime as the preprocessor does not exist during runtime. The C language dictates that this expression must be evaluated during compilation, otherwise, the preprocessor would not be able to determine what code should be compiled.

In the examples provided, it's clear that these calculations are completed during the compilation phase, ensuring that all necessary parameters are resolved before run time. Compilers may, however, perform these calculations using different optimization techniques. Some may perform more complex constant folding compared to others. In most cases, the developer does not need to be concerned with the specifics of these optimizations as all standards conforming compilers will handle constant expression evaluation automatically.

For further exploration of C language semantics, I would recommend these resources: the ISO/IEC 9899 standard (the official C language specification), "Modern C" by Jens Gustedt, and "C Programming: A Modern Approach" by K.N. King. These resources provide a solid foundation for understanding the language's core principles and will provide you with a good theoretical understanding of the nuances behind compile-time evaluation.
