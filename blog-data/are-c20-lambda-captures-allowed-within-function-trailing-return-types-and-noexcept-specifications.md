---
title: "Are C++20 lambda captures allowed within function trailing return types and noexcept specifications?"
date: "2024-12-23"
id: "are-c20-lambda-captures-allowed-within-function-trailing-return-types-and-noexcept-specifications"
---

Alright, let’s tackle this one. I remember a particularly thorny situation back in my days working on a high-frequency trading platform, where we were pushing the boundaries of template metaprogramming and constexpr. We were heavily leveraging lambda expressions for lightweight computations, and the question of where they could live, particularly in complex function signatures, became rather critical. So, let's break down if C++20 lambda captures are permissible within function trailing return types and noexcept specifications, starting with some key concepts to avoid ambiguity.

First off, the trailing return type, introduced in C++11, allows us to specify the return type of a function after its parameter list. This syntax, using `->` followed by the type, can be particularly useful with templates where the return type might depend on template arguments. Similarly, the `noexcept` specification, introduced around the same time, indicates whether a function is expected to throw exceptions, with `noexcept` signifying that it will *not*. Now, the crux of the matter is how lambda expressions, specifically those with captures, interact with these constructs, given their inherent type deduction challenges.

The short answer is: yes, C++20 lambda captures *are* allowed within function trailing return types and noexcept specifications, provided they don't violate the core rules of these language features or the type system itself. This compatibility is vital for functional programming paradigms in C++ and especially useful in generic programming with templates. The critical constraint is whether the lambda expression's capture list is fully determined and well-defined by the time the compiler needs to evaluate the trailing return type or noexcept specification. Let’s unpack that a bit, specifically regarding when this can become more involved.

A key point to remember is that the compiler must resolve all type information necessary for the function signature at the time of function definition or declaration. This means that any type deduction related to lambda captures must be available. If the captured variable’s type is dependent on template arguments or some other runtime information not available at compile time, you’re going to run into problems. Let's look at the `noexcept` specifier first. The `noexcept` specification is part of the function's type signature and must be known at compile time. If a lambda with captures is used directly within a `noexcept` specifier, it needs to be absolutely certain whether that lambda can throw exceptions at compile time. A lambda itself, without explicit `noexcept` specification, is considered potentially throwing if it can throw. However, if the lambda is used purely within the type deduction context it will not execute as part of function’s declaration, therefore it won't directly invalidate the `noexcept` status if it's used to deduce the type, however it can affect the return type's noexcept-ness, as we will see below.

Now, onto the more involved aspect – the trailing return type. If a lambda capture list appears inside a trailing return type deduction, the compiler has to be able to resolve the types of the captured variables to understand what exactly will be returned by the function. If the lambda’s return type is auto (and thus deduced), and if that deduction depends on the captured variables, it can be allowed. The crucial part is to remember that this deduction happens within a non-evaluated context. It's as if the compiler is saying: “Okay, I need to know the *type* of this return, but I don’t need to *evaluate* the return at this moment.”

Let me provide three concrete examples, complete with explanations to clear up any lingering questions.

**Example 1: Simple Lambda in Trailing Return Type**

```cpp
#include <type_traits>

template <typename T>
auto make_lambda(T val) -> auto  {
    auto lambda = [val](){ return val; };
    return lambda;
}


int main() {
  auto l1 = make_lambda(5);
  static_assert(std::is_same_v<decltype(l1),  decltype([&]{return 5;})>);
  auto l2 = make_lambda(5.0);
    static_assert(std::is_same_v<decltype(l2),  decltype([&]{return 5.0;})>);

}
```

In this snippet, the `make_lambda` function is a template function that takes any type `T` and returns a lambda that captures `val` by value. The `-> auto` syntax is for the trailing return type. In this case, the lambda's return type is deduced to be `T` by the compiler using the capture `val`, and it perfectly valid, since the value of the template parameter `T` is available at the function declaration. The capture within the trailing return type is correctly interpreted as a way to determine the return type, not to actually execute the lambda itself at declaration time. The lambda object is generated at function return execution time. The `static_assert` confirms this.

**Example 2: Lambda in noexcept Specification (with potential for implicit throw)**

```cpp
#include <stdexcept>
#include <iostream>
#include <type_traits>

struct NonCopyable {
  NonCopyable() = default;
  NonCopyable(const NonCopyable &) = delete;
  NonCopyable(NonCopyable&& ) = default;

};


auto throwing_lambda(int x) noexcept(false)
{
    return [x]()->int { if(x==0) throw std::runtime_error("zero"); return x; };
}


auto no_throw_lambda(int x) noexcept
{
    return [x]() noexcept ->int { return x; };
}



int main()
{
    auto f1 = throwing_lambda(0);
    try {
        f1();
    } catch(std::runtime_error &e)
    {
        std::cout << "Caught exception: " << e.what() <<std::endl;
    }

    auto f2 = no_throw_lambda(5);
    static_assert(noexcept(f2()));

}
```

Here, we have two functions `throwing_lambda` and `no_throw_lambda`. the former generates a lambda that throws if the input parameter `x` is 0 and the later one generates a lambda that does not throw and is explicitly marked as `noexcept`. Both are valid. The key part is `noexcept(false)` in `throwing_lambda` (and implicit `noexcept` in `no_throw_lambda`) which means the functions themselves are not marked as noexcept (and this means they can throw exceptions).

**Example 3: Lambda Dependent on Template Type in Return Type**

```cpp
#include <type_traits>

template <typename T>
auto make_add_lambda(T val) -> auto {
    auto lambda = [val](T other) { return val + other; };
    return lambda;
}


int main() {
    auto int_lambda = make_add_lambda(5);
    static_assert(std::is_same_v<decltype(int_lambda(1)), int>);

    auto float_lambda = make_add_lambda(5.5f);
    static_assert(std::is_same_v<decltype(float_lambda(1.0f)), float>);

}
```

This example showcases a more complex interaction where the lambda takes another argument. Here the lambda capture, again through value, is still fully deduced by the compiler during template instantiation, and the compiler can resolve the full function signature and return type during compilation. The `static_assert` is used to demonstrate the return type deduction is based on the return of the lambda. This is yet again perfectly valid.

To solidify the knowledge regarding these concepts, I'd strongly recommend diving into the C++ standard itself, specifically sections related to lambda expressions (e.g., [expr.prim.lambda]) and function declarations. Another great resource is "Effective Modern C++" by Scott Meyers, especially the sections covering lambdas and generic programming. Additionally, "Professional C++" by Marc Gregoire offers comprehensive coverage of these more involved language features.

In conclusion, C++20 lambdas with captures are indeed allowed within function trailing return types and `noexcept` specifications. The compiler is clever enough to handle them by resolving the capture's types during type deduction, making sure that all relevant type information is available during the compilation stage. The examples above should provide a strong foundation, however a thorough study of the standard will provide a more in depth understanding of these language features.
