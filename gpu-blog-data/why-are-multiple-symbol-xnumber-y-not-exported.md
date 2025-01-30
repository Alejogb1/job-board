---
title: "Why are multiple Symbol X(number Y) not exported from dependent modules on AIX 7.1?"
date: "2025-01-30"
id: "why-are-multiple-symbol-xnumber-y-not-exported"
---
Symbol resolution in AIX 7.1, particularly concerning shared libraries and their exported symbols, can present unique challenges due to the operating system’s binding and linking mechanisms. The behavior observed when multiple symbols of the form `X(number Y)` are not consistently exported from dependent modules is a direct consequence of AIX's use of the import/export control file within archive libraries and shared objects and how those controls interact with the system's dynamic linking process. The crucial point to understand is that the symbols are not failing to be *defined*, but rather, they are failing to be made *visible* for import. This issue stems from a combination of name mangling, the export controls within AIX, and the specific manner in which the loader resolves symbols at runtime.

Let's break down the root causes and then examine some code examples and potential solutions. AIX uses a naming convention where symbols may be decorated with type information or other metadata, commonly involving parentheses. This is distinct from C++ name mangling and affects how symbols are located in shared objects and archives. The presence of `(number Y)` in a symbol name like `X(number Y)` doesn't inherently cause an issue; the issue arises in how the export list, typically stored in the `.exp` file within archive libraries ( `.a` files), and the export list within shared object files are managed and subsequently used by the dynamic linker. When a shared object is built, only explicitly declared exported symbols are considered for external linkage, unless you utilize a wild-card symbol export.

The typical process involves the following:

1.  **Compilation:** Source code is compiled into object files (`.o`).
2.  **Archiving or Shared Object Creation:** Object files are combined into either an archive library (`.a`) or a shared object (`.so`). In both cases, an export list dictates which symbols are visible for linking against.
3.  **Linking:** An executable is linked against libraries (either static archive libraries or dynamic shared objects). This is where the import process occurs and the dynamic loader will try to resolve the needed symbols using the export lists available at load time.
4.  **Runtime Loading:** When the executable is run, the dynamic linker resolves the required symbols from shared objects and loaded into memory.

The problem occurs when multiple symbols like `X(1)`, `X(2)`, `X(3)` are present within the object files within archive libraries or shared objects, and only one or a subset is specified explicitly in the export list. Consider a scenario where a library, `libA.a`, contains object files defining symbols `X(1)`, `X(2)`, and `X(3)`. The archive's default export list `.exp` only contains `X(1)`. If a dependent shared object, `libB.so`, is linked against `libA.a` and attempts to use `X(2)` and `X(3)`, the linker will not see them as exportable symbols. Consequently, during the link phase, those symbols will not resolve unless an explicit export list of the shared object includes these symbols as exported to be resolved by the executable at run-time. Furthermore, even if the archive contains multiple objects with those symbols, without the explicit export, the linker will most likely choose to resolve the first one to find a valid match, and ignore the rest due to the linker's inherent one-symbol-one-address policy.

Here are code examples to illustrate the problem:

**Example 1: Archive library export issues**

Let's say we have three C files, `a.c`, `b.c`, and `c.c`.

```c
// a.c
int X(1) { return 1; }
```

```c
// b.c
int X(2) { return 2; }
```

```c
// c.c
int X(3) { return 3; }
```

These files are compiled into object files, `a.o`, `b.o`, and `c.o`. These object files are then archived into `libA.a`.

```shell
# Compilation
xlc -c a.c b.c c.c
ar -qv libA.a a.o b.o c.o
# An incomplete export control list. Only X(1) is declared exportable.
echo "X(1)" > libA.exp
# Generate the import list in the archive
ar -X32_64 -m libA.a libA.exp
```
In this case, only symbol `X(1)` is declared exportable. In a separate application using `libA.a`, any attempt to utilize `X(2)` or `X(3)` will result in a link error. This error may not necessarily happen at compile time, but potentially at the executable link time. The application needs an explicit declaration of the dependencies within its linking flags.

**Example 2: Shared Object Export issues**

Consider a shared object library `libB.so` that depends on `libA.a` (created in the previous example).

```c
// libB.c
int use_x_2() {
  return X(2); // Uses symbol X(2) from libA.a
}

int use_x_3() {
    return X(3); // Uses symbol X(3) from libA.a
}
```

```shell
# Create shared library B using symbols in archive A
xlc -o libB.so -shared -Wl,-bexp:libB.exp libB.c libA.a
# Create the export list for libB.so. It only exports use_x_2 and use_x_3
echo "use_x_2" > libB.exp
echo "use_x_3" >> libB.exp
```

Here, `libB.so` depends on `libA.a`'s symbols `X(2)` and `X(3)`. Even though `libB.so` is compiled successfully, it cannot be used by any application because the symbols `X(2)` and `X(3)` are not visible during the initial executable link phase, neither at the shared object link phase. The symbols are not made public through an export control declaration, so the executable will not be able to find them.

**Example 3: Addressing the issue through explicit exports**

To resolve this, we must explicitly export all the relevant `X(n)` symbols from the library. We will modify Example 1 to include additional symbols in the archive export control list. The compilation phase for the library and shared object remains the same as in the examples 1 and 2.

```shell
# Modification to export file: explicitly include X(1), X(2), and X(3)
echo "X(1)" > libA.exp
echo "X(2)" >> libA.exp
echo "X(3)" >> libA.exp
ar -X32_64 -m libA.a libA.exp
```
By modifying the export control list of `libA.a` to include all the symbols used by library `libB.so` we can overcome the missing symbol resolution issue at link-time.
Furthermore, the `libB.exp` can be modified to specify `*` if the goal is to export all symbols present in the shared library.

```shell
# Modified export control for shared object B
echo "*" > libB.exp
```

By correctly specifying the export controls, the symbols are correctly resolved at runtime.

To summarize, the root issue isn't the definition of multiple symbols named `X(n)`, but rather, the *visibility* of those symbols during linking, governed by the export lists of libraries and shared objects. The AIX linker does not automatically expose all symbols and needs an explicit instruction.

For further research and understanding of dynamic linking on AIX, I recommend studying these resources. First, examine the AIX documentation on archive libraries, which covers the format of the `.exp` export control file and its management. Second, review the documentation pertaining to shared libraries and their export mechanisms. Third, study the documentation on the AIX linker which covers the usage of export control, options, and behaviors. These resources, although technical, provide crucial insights into why this particular scenario can occur and how to prevent it, and are essential for developers working within the AIX environment.
