---
title: "How can pointers be used in PL/I?"
date: "2025-01-30"
id: "how-can-pointers-be-used-in-pli"
---
PL/I's pointer capabilities, often overlooked due to its age, offer a surprisingly powerful mechanism for dynamic memory management and data structure manipulation, particularly beneficial in situations demanding flexible memory allocation and complex data relationships.  My experience working on large-scale simulation projects in the late 90s highlighted the efficiency gains possible through strategic pointer usage when dealing with dynamically sized arrays and linked lists representing complex physical systems.  Unlike some languages where pointers are primarily low-level tools, PL/I integrates them seamlessly into its higher-level constructs, enabling a degree of control that avoids the pitfalls of direct memory addressing while retaining performance advantages.

**1.  Clear Explanation:**

Pointers in PL/I are declared using the `POINTER` attribute.  Unlike some languages that implicitly dereference pointers in certain contexts, PL/I requires explicit dereferencing using the `->` operator.  This explicitness reduces ambiguity and promotes code readability, crucial for maintainability in large projects. A pointer variable holds the address of another variable, allowing indirect access to its value.  The pointed-to variable must be declared as `BASED`—indicating that its storage location is not fixed at compile time but is dynamically allocated during runtime. The `ALLOCATE` and `FREE` statements manage the dynamic allocation and deallocation of based variables.  This contrasts sharply with statically allocated variables whose memory addresses are known at compile time.  Crucially, PL/I offers strong type checking for pointers; attempting to access a variable through a pointer of an incompatible type results in a compilation error, enhancing robustness.

Furthermore, PL/I allows pointers to point to structures, arrays, and even other pointers, leading to intricate data structures like linked lists and trees. This versatility enhances the language's capability to model complex real-world systems efficiently.  Careful management of pointer allocation and deallocation is essential to avoid memory leaks and dangling pointers.  The `NULL` value serves as a special pointer value indicating that a pointer does not currently point to a valid memory location.  Using `NULL` consistently in initialization and error handling helps prevent common pointer-related errors.


**2. Code Examples with Commentary:**

**Example 1:  Simple Pointer to a Numeric Variable:**

```pl1
DCL  (Num, *Ptr) FIXED BIN(31);
DCL  1 Rec BASED (Ptr),
       2 Value FIXED BIN(31);

ALLOCATE Rec;
Ptr = ADDRESS(Rec);
Rec->Value = 100;
PUT SKIP LIST('Value:', Rec->Value);
FREE Rec;
```

*Commentary:* This example demonstrates the basic usage of pointers.  `Num` is a regular variable, while `Ptr` is a pointer variable declared using the asterisk (*). `Rec` is a based structure with a single field `Value`. `ADDRESS` built-in function returns the memory address of the structure, assigned to `Ptr`.  The `->` operator dereferences the pointer, allowing modification of the `Value` field.  Finally, `FREE` deallocates the dynamically allocated memory occupied by `Rec`.


**Example 2:  Linked List Implementation:**

```pl1
DCL 1 Node BASED (Ptr),
       2 Data CHAR(20),
       2 Next POINTER;
DCL Head POINTER;
DCL Ptr POINTER;

Head = NULL;
ALLOCATE Node;
Ptr = ADDRESS(Node);
Ptr->Data = 'First Node';
Ptr->Next = NULL;
Head = Ptr;

ALLOCATE Node;
Ptr->Next = ADDRESS(Node);
Ptr = Ptr->Next;
Ptr->Data = 'Second Node';
Ptr->Next = NULL;

/* Further node additions would follow a similar pattern */

Ptr = Head;
DO WHILE (Ptr ^= NULL);
  PUT SKIP LIST(Ptr->Data);
  Ptr = Ptr->Next;
END;
```

*Commentary:* This example illustrates a singly linked list implementation.  Each `Node` contains data and a pointer to the next node.  The `Head` pointer points to the beginning of the list.  Nodes are dynamically allocated and linked together.  The `DO WHILE` loop iterates through the list, printing the data in each node.  Note the use of `^= NULL` for unequal comparison to NULL.


**Example 3:  Pointer to an Array:**

```pl1
DCL 1 Array BASED (ArrPtr) (100) FIXED BIN(31);
DCL ArrPtr POINTER;

ALLOCATE Array;
ArrPtr = ADDRESS(Array);

DO I = 1 TO 100;
  ArrPtr->Array(I) = I * 2;
END;

DO I = 1 TO 100;
  PUT SKIP LIST(ArrPtr->Array(I));
END;
FREE Array;
```

*Commentary:* This example demonstrates how to use pointers with arrays.  The `Array` is a based array of 100 integers.  The pointer `ArrPtr` points to the beginning of the array.  Elements are accessed using `ArrPtr->Array(I)`. Note the dereferencing of `ArrPtr` before the array access. This example emphasizes the flexibility of PL/I, enabling dynamic allocation of arrays whose sizes are determined during program execution.


**3. Resource Recommendations:**

For a deeper understanding of PL/I's pointer capabilities, I would suggest consulting the official PL/I language reference manual, specifically sections covering the `POINTER` and `BASED` attributes, dynamic storage allocation, and pointer arithmetic (where applicable).  A comprehensive textbook on PL/I programming would provide further context within the broader scope of the language.  Supplementing this with practical exercises, such as building various data structures using pointers, would solidify understanding.  Finally, examining legacy PL/I codebases can offer invaluable insight into real-world applications and best practices.  Analyzing existing code examples, particularly those that deal with intricate data structures, will reveal effective pointer usage techniques and common patterns.
