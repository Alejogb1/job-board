---
title: "How do ++i and i=i+1 differ in assembly language?"
date: "2025-01-26"
id: "how-do-i-and-ii1-differ-in-assembly-language"
---

Modern optimizing compilers often treat `++i` (pre-increment) and `i = i + 1` as functionally equivalent at the source code level for integer types, leading to the same assembly instructions in many scenarios. However, examining their underlying mechanisms reveals subtle distinctions that are crucial for performance and understanding compiler behavior, especially when dealing with non-trivial data types or embedded systems. I've personally encountered these differences in embedded system development where optimizing for each instruction becomes paramount due to hardware limitations.

Fundamentally, both constructs achieve the same logical outcome: increasing the value of variable `i` by one. The crucial divergence emerges in how they express this intent and, consequently, how a compiler might translate them into low-level instructions. The expression `++i` is a prefix increment operation. It explicitly specifies the intent of incrementing the variable *before* its value is used in the enclosing expression. Conversely, `i = i + 1` expresses a sum that is then assigned to `i`. This difference, while seemingly pedantic in pure functional terms, can enable a compiler to make different optimization decisions, particularly on architectures with limited register space or specific instruction sets.

To understand the nuances, consider typical scenarios in x86-64 assembly language, the most common architecture I deal with daily. With integers residing in registers, both `++i` and `i = i + 1` can translate into similar, often identical, instructions. If ‘i’ is a register, for example, both might map to `inc reg` or `add reg, 1`. However, scenarios change if ‘i’ is not directly a register but a memory location. In that case, the compiler might opt for different sequences based on the operation’s context. Consider these cases, focusing on how they translate to assembly:

**Case 1: Simple Integer Increment**

Assume 'i' is a 32-bit integer variable located at memory address pointed to by rdi.

```c
// C Code
int i = 5;
++i; // Or, i = i + 1; for the purpose of this example
```

The assembly output (using gcc) might look like this, compiled without aggressive optimization flags for illustrative purpose:

```assembly
mov     dword ptr [rdi], 5   ; Initialize i with 5
inc     dword ptr [rdi]    ; Increment i using inc instruction
```

Here, both `++i` and `i = i + 1`, with no optimization on, typically translate to `inc dword ptr [rdi]` on x86. The compiler directly modifies the memory location, using an instruction optimized for incrementing, directly reflecting the intent of prefix increment. The key point is that the `inc` instruction directly modifies the memory without needing to read it into a register first, increment in the register, and then write the incremented value back to memory.

**Case 2: Increment within a Larger Expression**

Now, let's look at how an expression utilizing the increment can impact the assembly:

```c
// C Code
int i = 5;
int j = ++i;  // Using prefix increment
// Or
int k = i;
i = i + 1; // Corresponding sequence with i = i + 1;
```
The assembly output is as follows (with -O0):

```assembly
; Prefix Increment ++i; j = ++i
mov     dword ptr [rsp-4], 5 ; Initialize i to 5
inc     dword ptr [rsp-4]     ; Increment i first
mov     eax, dword ptr [rsp-4]    ; Load i into register eax
mov     dword ptr [rsp-8], eax  ; Store eax in j

; Corresponding sequence with i=i+1, k = i;
mov     dword ptr [rsp-12], 5 ; Initialize i to 5
mov     eax, dword ptr [rsp-12] ; Load i into register eax
add     eax, 1                ; Increment i in register eax
mov     dword ptr [rsp-12], eax ; Store the incremented value back into i
mov     eax, dword ptr [rsp-12]  ; Load the new value of i into a register
mov     dword ptr [rsp-16], eax  ; Store i in k
```

In this case, with `-O0` optimization, we see two key differences:
1.  For prefix `++i`, `inc dword ptr [rsp-4]` directly modified the memory location, followed by a load into register, while `i = i + 1` first loaded the value in a register, then added 1, and then wrote the value back.
2. Although `k = i;` is also performed after `i = i + 1;`, in the case of `++i`, the result in `j` is the new value, same as what is the result after `i = i + 1`, `k = i`. However, this is achieved with fewer instructions.

Now, with `-O1` flag, a compiler can optimize this. The assembly looks like this with `-O1` flag:

```assembly
mov     dword ptr [rsp-4], 5    ; Initialize i
inc     dword ptr [rsp-4]       ; Increment i using inc
mov     eax, dword ptr [rsp-4]  ; Load new i value into eax
mov     dword ptr [rsp-8], eax  ; Store into j

mov     dword ptr [rsp-12], 5   ; Initialize i
add     dword ptr [rsp-12], 1   ; Increment i using add
mov     eax, dword ptr [rsp-12] ; Load new i value into eax
mov     dword ptr [rsp-16], eax ; Store into k
```

Now, both incrementing strategies using `++i` and `i = i + 1` use the `add` and `inc` operator, while the rest of the operation are the same.

**Case 3:  Overloaded Operators with Complex Objects**

This is where the differences become more pronounced. When dealing with non-primitive types, such as custom classes with overloaded increment operators, `++i` might resolve to different function calls than `i = i + 1` might.

Consider a simple class with an overloaded `++` operator and assignment operator.

```cpp
// C++ Code
class MyClass {
public:
  int value;
    MyClass(int v) : value(v) {}
    MyClass& operator++() {
      value++;
      return *this;
    }

    MyClass operator+(int other) const{
      return MyClass(value + other);
    }
    MyClass& operator=(const MyClass& other){
        value = other.value;
        return *this;
    }
};

int main() {
  MyClass obj(5);
  ++obj;

  MyClass obj2(5);
  obj2 = obj2 + 1;
  return 0;
}
```

The assembly output corresponding to `++obj` call will invoke `operator++` on the object `obj` which will increment the `value` field within the object, using potentially different instructions or memory access patterns. The assembly output of `obj2 = obj2 + 1` would invoke `operator+` which will create a copy of `obj2` that increments the `value` field by 1. This temporary object is then assigned to `obj2`, therefore, invoking `operator=`. These scenarios involve function calls and potentially additional instructions for object construction and copy. This highlights that in presence of overloaded operations, even though the result might be same, the underlying call stacks might differ significantly.

```assembly
; ++obj
call    MyClass::operator++()

; obj2 = obj2 + 1
mov     esi, 1   ; 1 is pushed into rsi
lea     rax, [rsp-16]
mov     rdi, rax    ; address of obj2 pushed to rdi
call    MyClass::operator+(int) const  ; function call to operator+ which returns a copy
lea     rdi, [rsp-16]
mov     rsi, rax ; address of temporary pushed to rsi
call    MyClass::operator=(MyClass const&)  ; function call to operator= to assign the temporary to obj2
```

These cases demonstrate that the seemingly simple choice between `++i` and `i = i + 1` can have varying assembly-level outcomes. While simple increment operations on integers might result in the same or similar code, complexities arise when optimization flags change, when dealing with memory access rather than register access, and when involving overloaded operators for custom classes or data structures. Understanding these differences becomes crucial for performance tuning, especially in low-level or resource-constrained environments.

To deepen one's understanding, examining compiler explorer tools allows users to observe the generated assembly output of different code snippets in real time with different optimization flags. The compiler documentation for your specific compiler is an invaluable resource; it explains optimization strategies and the expected behavior of various constructs. Assembly language books for specific target architectures are also necessary to see the details of actual instruction sets. Finally, practice writing simple code snippets and observing their assembly translations is important.
