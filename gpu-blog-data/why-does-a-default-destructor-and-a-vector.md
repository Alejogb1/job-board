---
title: "Why does a default destructor and a vector member prevent a class from being nothrow movable constructible?"
date: "2025-01-30"
id: "why-does-a-default-destructor-and-a-vector"
---
In C++, a class's move construction semantics are intimately tied to its members and their associated move operations. Specifically, when a class contains a member that itself isn't no-throw movable, or if the class utilizes a compiler-generated (default) destructor, the move constructor of that class is implicitly defined as potentially throwing, thus disqualifying it from being considered no-throw move constructible. The standard library's `std::is_nothrow_move_constructible` type trait will therefore evaluate to `false` for such classes.

The core issue stems from the way the compiler synthesizes move constructors and destructors. If a class doesn't explicitly define a move constructor, the compiler attempts to create one that performs a member-wise move operation. Crucially, this synthesized move constructor has a `noexcept` specifier only if *all* the members of the class are themselves no-throw movable and if the destructor of the class is also implicitly no-throw. The compiler-generated default destructor is implicitly `noexcept` only when all non-static data members have a no-throw destructor as well. This is a crucial condition. When a member like `std::vector` is present without explicit control of move construction and destruction, complications arise.

`std::vector`'s move constructor, while efficient, can potentially throw an exception under specific circumstances, such as when an allocator fails during memory reallocation. Importantly, whether a `std::vector` move constructor actually *does* throw is allocator-dependent; many implementations may not throw, yet the standard dictates that it has the *potential* to throw unless explicitly declared otherwise. This possibility prevents the compiler from marking the synthesized move constructor of the containing class `noexcept`. Similarly, the default compiler generated destructor for a `std::vector` is also not `noexcept`. I have personally encountered several scenarios where an application assumed move operations were no-throw, only to crash because a potentially throwing move operation of a member was triggered, during operations like reallocations of `std::vector`, or during vector reordering when sorting. I have learned that careful analysis of members' move construction is critical to avoid these cases.

Here are three code examples to further clarify the issue.

**Example 1: Class with a `std::vector` member, no custom move or destructor, demonstrating the lack of no-throw move construction**

```cpp
#include <iostream>
#include <vector>
#include <type_traits>

class MyClass {
public:
    std::vector<int> data;
};

int main() {
  std::cout << std::boolalpha;
  std::cout << "Is MyClass nothrow move constructible? "
            << std::is_nothrow_move_constructible<MyClass>::value << std::endl;
  return 0;
}
```

In this example, `MyClass` has a single member, `data`, which is a `std::vector<int>`. We haven't defined a move constructor or a destructor. The compiler will thus generate defaults. The standard specifies that the move constructor and destructor for `std::vector` are not `noexcept` due to potential allocator failure. Consequently, the synthesized move constructor for `MyClass` is also not `noexcept`. `std::is_nothrow_move_constructible<MyClass>` will evaluate to `false`. This aligns with the experience of observing exceptions during move operations involving `std::vector` which resulted in crashing processes.

**Example 2: Class with a `std::vector` member, custom `noexcept` move constructor, but still no `noexcept` move construction because of default destructor**

```cpp
#include <iostream>
#include <vector>
#include <type_traits>

class MyClass {
public:
    std::vector<int> data;
    MyClass(MyClass&& other) noexcept : data(std::move(other.data)) {}

};

int main() {
  std::cout << std::boolalpha;
  std::cout << "Is MyClass nothrow move constructible? "
            << std::is_nothrow_move_constructible<MyClass>::value << std::endl;
  return 0;
}
```
Here, I have explicitly defined a move constructor for `MyClass` and tagged it with `noexcept`. This move constructor performs a move operation on `std::vector<int>` which is still not itself guaranteed to be no-throw. However, even though we mark the move constructor with noexcept, the destructor is implicitly defined by the compiler and is also not `noexcept`. Consequently, `std::is_nothrow_move_constructible<MyClass>` *still* evaluates to `false`. This demonstrates that a `noexcept` move constructor is insufficient, the destructor must also be `noexcept`.

**Example 3: Class with a `std::vector` member, custom `noexcept` move constructor and destructor, achieves no-throw move construction**

```cpp
#include <iostream>
#include <vector>
#include <type_traits>

class MyClass {
public:
    std::vector<int> data;
    MyClass(MyClass&& other) noexcept : data(std::move(other.data)) {}
    ~MyClass() noexcept {}
};

int main() {
  std::cout << std::boolalpha;
  std::cout << "Is MyClass nothrow move constructible? "
            << std::is_nothrow_move_constructible<MyClass>::value << std::endl;
  return 0;
}
```

In this final example, I've defined both a `noexcept` move constructor *and* a `noexcept` destructor. The destructor is empty as no resource management or cleanup is required outside of `std::vector`. This is the key: by providing explicit `noexcept` implementations, the compiler can deduce that all the parts of a move operation are no-throw, including the destruction phase which the compiler has control over and which impacts the class moveability by being part of the implicit move constructor of the class. Therefore, `std::is_nothrow_move_constructible<MyClass>` now evaluates to `true`. This configuration is optimal for scenarios where high performance is needed and where strong exception guarantees are a must. I have used techniques similar to this to optimize code in time critical systems by avoiding move operations that could throw and cause significant performance issues.

It's important to note that while the empty destructor in the third example satisfies the compiler, in more complex scenarios one might need to ensure that the underlying resources being managed have their own no-throw destructors or handle potential exceptions in custom destructors.

**Resource Recommendations**

For a deeper understanding of move semantics, exception guarantees, and type traits in C++, I would recommend exploring:

1.  **The C++ Standard:** The actual wording of the standard provides the most accurate and detailed information on these topics. Focusing on sections related to move constructors, destructors, `noexcept` specifier, and type traits (`std::is_nothrow_move_constructible`) will be helpful. The draft standard or specific compiler documentation are excellent resources.
2.  **"Effective Modern C++" by Scott Meyers:** This book offers a pragmatic and in-depth treatment of modern C++ features, including move semantics and exception handling. It's particularly helpful in understanding the implications of choosing specific C++ constructs.
3.  **C++ Core Guidelines:** These guidelines provide a valuable perspective on best practices in C++ development. The section on exception handling and move semantics is particularly relevant to this problem and provides practical suggestions to avoid common pitfalls when working with these features. They also include a section on type safety and type traits.
4. **"Effective C++" and "More Effective C++" by Scott Meyers:** These works also provide a foundation that will further solidify one's understanding of exception handling and object construction in C++.

In summary, a default destructor coupled with a member that has the *potential* to throw when being moved (such as `std::vector`) will prevent a class from being no-throw move constructible. This stems from the compiler's synthesized move constructor requiring all members and the destructor of the class to be no-throw in order to achieve `noexcept` move construction. Explicitly defining a `noexcept` move constructor and a `noexcept` destructor is necessary to ensure that such classes meet the requirement of no-throw move construction. The choice of implementation for the move constructor and destructor will ultimately depend on the specific needs of an application.
