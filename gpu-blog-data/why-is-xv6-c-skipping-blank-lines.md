---
title: "Why is xv6 C skipping blank lines?"
date: "2025-01-30"
id: "why-is-xv6-c-skipping-blank-lines"
---
The apparent skipping of blank lines observed in xv6’s C code parsing arises not from C itself ignoring empty lines, but from how the `cpp`, the C preprocessor, operates during the compilation process. This preprocessor, a distinct stage before actual compilation, significantly impacts the source code passed to the compiler, and it is within this stage that blank lines are largely rendered irrelevant.

Understanding this requires examining the typical toolchain involved in compiling a C program, specifically in a simplified environment such as xv6. Source files (.c files) initially undergo preprocessing, managed by the `cpp` tool. This tool handles directives starting with `#`, such as `#include` and `#define`, and importantly for this case, whitespace, which includes blank lines. The output of `cpp` is a single, expanded C source file, where macros have been replaced, included files have been inserted, and extraneous whitespace, including blank lines, has been stripped away to a large degree. The resulting output is then fed to the actual C compiler.

Blank lines in source code primarily serve human readability purposes, allowing programmers to visually separate sections of code. The C compiler itself cares very little for this extra formatting. It parses the code based on syntactical rules and semantics, requiring tokens and statements to be delimited correctly using semicolons, brackets, parentheses, and other syntax elements. The presence or absence of blank lines generally does not change the meaning of a valid program and is not directly parsed by the C compiler itself.

The preprocessor reduces redundant blank lines to a single blank line or sometimes removes them entirely before the compiler receives it. In cases of adjacent blank lines, often the preprocessor will reduce these to only a single newline character in the preprocessed output. This is a common practice to reduce file size and to avoid introducing superfluous whitespace in the output. The compiler is therefore presented with a cleaned version of the code which it can then translate into object code without the distractions of unnecessary blank lines. This process maintains the logical structure of the code while removing some of the whitespace formatting. The compiler ignores any trailing spaces or newline characters that don't impact the actual program semantics. This is standard compiler behavior and not specific to xv6, which is implemented in a standard C programming environment.

Let's consider three code examples that illustrate this.

First, let's assume `example1.c` contains the following:

```c
#include <stdio.h>

int main()

{


  printf("Hello, world!\n");




  return 0;

}
```

The preprocessed output will look very similar, with potentially less whitespace:

```c
# 1 "example1.c"
# 1 "<built-in>"
# 1 "<command-line>"
# 1 "/usr/include/stdio.h" 1 3 4
# 31 "/usr/include/stdio.h" 3 4
# 1 "/usr/include/_stdio.h" 1 3 4
# 24 "/usr/include/_stdio.h" 3 4
# 1 "/usr/include/sys/cdefs.h" 1 3 4
# 43 "/usr/include/sys/cdefs.h" 3 4
# 1 "/usr/include/bits/wordsize.h" 1 3 4
# 44 "/usr/include/sys/cdefs.h" 2 3 4
# 25 "/usr/include/_stdio.h" 2 3 4
# 42 "/usr/include/_stdio.h" 3 4
# 1 "/usr/include/bits/types.h" 1 3 4
# 27 "/usr/include/bits/types.h" 3 4
# 1 "/usr/include/bits/wordsize.h" 1 3 4
# 28 "/usr/include/bits/types.h" 2 3 4
# 1 "/usr/include/bits/typesizes.h" 1 3 4
# 29 "/usr/include/bits/types.h" 2 3 4
# 43 "/usr/include/_stdio.h" 2 3 4
# 47 "/usr/include/_stdio.h" 3 4
# 1 "/usr/include/libio.h" 1 3 4
# 32 "/usr/include/libio.h" 3 4
# 1 "/usr/include/_G_config.h" 1 3 4
# 14 "/usr/include/_G_config.h" 3 4
# 1 "/usr/include/wchar.h" 1 3 4
# 49 "/usr/include/wchar.h" 3 4
# 1 "/usr/include/bits/wchar.h" 1 3 4
# 49 "/usr/include/wchar.h" 2 3 4
# 15 "/usr/include/_G_config.h" 2 3 4
# 33 "/usr/include/libio.h" 2 3 4
# 53 "/usr/include/libio.h" 3 4
# 1 "/usr/include/_stdio_impl.h" 1 3 4
# 67 "/usr/include/_stdio_impl.h" 3 4
# 1 "/usr/include/bits/stdio_lim.h" 1 3 4
# 68 "/usr/include/_stdio_impl.h" 2 3 4
# 54 "/usr/include/libio.h" 2 3 4
# 48 "/usr/include/_stdio.h" 2 3 4
# 32 "/usr/include/stdio.h" 2 3 4
# 2 "example1.c" 2
int main()
{
  printf("Hello, world!\n");
  return 0;
}
```

This shows that while the preprocessor injects the contents of `stdio.h` and adds some line number annotations, the blank lines are removed or reduced before reaching the compiler. It also shows, in case you were wondering, why using a preprocessor can be time consuming, and that it's worth using header guards to stop multiple copies of a single header getting processed.

Now consider another example, `example2.c`:

```c
int x  = 10;


int y
=
20;
int main()
{
return x+y;
}
```

The preprocessed output for this might appear similar to:

```c
# 1 "example2.c"
int x = 10;
int y = 20;
int main()
{
return x+y;
}
```

Here, the preprocessor has eliminated nearly all blank lines and even some extraneous spacing around operators like `=` and the newline before it. The code structure remains identical despite the lack of the original spacing. The compiler accepts this output without difficulty, proving that whitespace beyond what is absolutely necessary to distinguish tokens is irrelevant to the C parser.

Finally, let's take a look at an example with comments alongside blank lines, `example3.c`:

```c
int main() {
  // This is a comment


  int a = 10;

  int b = 20;  // Another comment




  return a + b;
}
```

The preprocessed version is likely to resemble:

```c
# 1 "example3.c"
int main() {
  int a = 10;
  int b = 20;
  return a + b;
}
```

The preprocessor has removed the blank lines and maintained only the line containing the comment, while retaining the comment text, as it is a vital part of the source code for human understanding. Note the compiler does not directly see the comments, they are filtered out by the preprocessor as well.

In all three examples, the commonality is that while source code may have blank lines and extra whitespace for readability, the preprocessor streamlines this output before it is given to the compiler. The C compiler, a strictly syntax-driven machine, handles this processed code and generates object code.

For further study on C compilation, I would suggest exploring resources that delve into the standard compilation process: "Compilers: Principles, Techniques, & Tools", also known as the "Dragon Book", is a comprehensive text on compiler design and construction. Other materials focusing on C language specifics, such as "The C Programming Language" by Kernighan and Ritchie, and general system programming books often include coverage of the compilation process in detail. A closer look at the implementation of `gcc` or other C compilers and its preprocessor will also be useful in building a solid understanding. Finally the POSIX specification which standardizes much of the compilation system can shed light on expected behaviour.

In summary, the phenomenon of xv6’s C code “skipping blank lines” is not an inherent aspect of C itself, but rather an artifact of the preprocessor stage of compilation. The preprocessor’s role is to simplify the input for the compiler, and blank line removal is a part of that simplification. The core purpose of blank lines is human-readability and is a coding style decision, not a language-level semantic element.
