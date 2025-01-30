---
title: "Why can't Idris solve the constraint after case splitting?"
date: "2025-01-30"
id: "why-cant-idris-solve-the-constraint-after-case"
---
The inability of Idris to automatically solve constraints after case splitting often stems from the inherent limitations of its unification algorithm when faced with dependent types, particularly those involving complex function application or data constructors within the type. Idris employs a unification process that seeks to equate two types or terms to make a program type-safe. While effective in many cases, this process can falter when case splitting introduces new variables or equations which the unifier cannot readily infer.

Here’s a breakdown of why this happens, coupled with practical code examples from my experiences building type-safe applications:

**The Challenge of Dependent Unification**

Idris, like other dependently-typed languages, relies on unification to deduce types at compile time. Unification, at its core, is the process of finding a substitution for type variables such that two type expressions become equal. The problem escalates when the types in question are dependent – that is, they contain values or are parametrised by values. Case splitting, a core mechanism in functional programming, inherently introduces new equations and potentially, new variables related to these values which must be satisfied via the unification process. These are not always readily apparent or easily solved using pattern matching alone.

Consider a type defined as the length of a vector and used as a constraint.  The unifier must, ideally, determine the correct values of these length parameters to satisfy constraints after a function is applied to vectors with varying length. When the lengths are not directly present in the type definition or pattern match, the unifier encounters ambiguity. It often cannot 'look ahead' or perform symbolic reasoning required to bridge the information gap introduced by the case split, particularly when dependent functions or data constructors obscure the underlying relationships. The unifier prefers to remain as general as possible, and this can lead to unresolvable constraints even when a human can see a clear solution.

**Illustrative Examples**

Let’s examine practical scenarios. I have frequently encountered this while writing functions that operate on length-indexed vectors, often encountering similar issues as described below.

*Example 1: Incomplete Information on Case Split Variables*

```idris
data Vec : Nat -> Type -> Type where
    Nil : Vec Z a
    (::) : a -> Vec n a -> Vec (S n) a

vecHead : Vec (S n) a -> a
vecHead (x :: xs) = x

--This will compile, as `S n` is explicit.

badHead : Vec m a -> a
badHead Nil = ?impossible-- This does not compile, as the return type doesn't match a.
badHead (x :: xs) = x -- Idris cannot infer `m = S n`

```
Here `vecHead` compiles successfully. The explicit pattern match with `(x :: xs)` automatically provides the type checker with sufficient information to infer the value of the `Vec`’s index. However, in `badHead`, the type `Vec m a` offers less information, the value of `m` has to be constrained to `S n`, Idris does not allow direct pattern matching and deduction that would result in `m = Z` from `Nil` to work out the different cases of `m`, hence the error. Idris needs an explicit constraint or `with` pattern to help.

*Example 2: Hidden Dependencies Within Function Application*

```idris
addOne : Nat -> Nat
addOne n = S n

vecTail : Vec (S n) a -> Vec n a
vecTail (x :: xs) = xs


manipulateVec : (n : Nat) -> Vec (addOne n) a -> Vec n a
manipulateVec n v = vecTail v --This also does not compile

```
Here the problem lies in that Idris cannot directly see that `addOne n` is equivalent to `S n` to allow `vecTail` to act upon it. Even though we humans can infer that `addOne n` always returns `S n`, it is non-trivial for the type checker to equate these two as they are function applications with an argument that is a variable. A more explicit constraint would need to be added here, potentially with the use of `with` or by providing an explicit `rewrite` rule.

*Example 3: Constructors Obscuring Equality*

```idris
data Fin : Nat -> Type where
    FZ : Fin (S n)
    FS : Fin n -> Fin (S n)


--This is fine, `Fin (S n)` is the type required, `n` can be used to construct it.
safeIndex : Fin (S n) -> Vec (S n) a -> a
safeIndex FZ (x :: xs) = x
safeIndex (FS f) (x :: xs) = safeIndex f xs

--Here the issue is that `Fin m` is being pattern matched, without revealing whether it's FZ or FS form.
badIndex : (m : Nat) -> Fin m -> Vec m a -> a
badIndex _ FZ  (x :: xs) = x -- Does not compile - Idris cannot relate type to S n
badIndex _ (FS f) (x::xs) = badIndex _ f xs  -- Does not compile - Idris cannot relate type to S n

```

The `safeIndex` function performs as expected, because the `Fin` type is explicitly associated with the vector index in the type signature, and the pattern matches specify the exact constructors. In the `badIndex` example, the same problem as example 1 arises.  Case splitting on `Fin m` doesn't expose enough information for Idris to deduce that `m` must be `S n` for the pattern match on `Vec m a` to work when matching on either `FZ` or `FS f`, thus leading to unresolvable constraints.

**Strategies for Resolving Constraint Failures**

When encountering these issues, several tactics can help:

1. **Explicit Constraints with `with`:** Using `with` clauses allows you to introduce new equations and pattern match on intermediate values, giving the unifier more explicit information to work with. This often involves rewriting a type expression to a form the unifier can recognize. In example 2 we can achieve a successful type check with:

```idris
manipulateVec : (n : Nat) -> Vec (addOne n) a -> Vec n a
manipulateVec n v with (addOne n)
  manipulateVec n v | (S n) = vecTail v
```
2. **Helper Functions with Specific Types:** Break down complex functions into smaller pieces, each with more specific type signatures. This can reduce the complexity of the constraints the unifier has to solve at any single point. Often this involves creating a helper function with a constraint that maps the desired result types onto a known type. This is especially useful if the compiler fails with an intermediate function.

3. **Type-Level Proofs:** In more challenging cases, you might need to prove type equivalencies using explicit type-level functions or data types. For instance, you might demonstrate that `addOne n` is indeed equivalent to `S n`. Idris does have features for type level proof automation, but the user may often have to define the type equalities, especially with dependent functions.

4.  **Rewrite Rules:** Using `rewrite` rules allow you to define specific type level equality rules that the compiler can apply during type checking. These can be useful to equate results from functions, which are non-trivial for the compiler to equate without guidance. This is similar to having helper functions, but can be more concise.

**Resource Recommendations**

For a deeper understanding, I would recommend exploring resources that delve into dependent type theory, specifically focusing on:

*   **Idris Tutorials and Documentation:** The official Idris documentation is an invaluable resource, and there are several tutorials available which explore the intricacies of type checking and unification in Idris.

*   **Books on Dependent Type Theory:** Works that discuss the theoretical underpinnings of dependent types, unification, and constraint solving. Look for texts that cover the use of pattern matching and type-level computation in type checking.

*   **Research Papers on Dependent Type Implementations:** Examining the implementation details of dependent type checkers, though advanced, can be insightful. Pay particular attention to papers that address unification algorithms and constraint solving in the presence of dependent types.

In conclusion, the issues with constraint resolution after case splitting in Idris are not arbitrary quirks, but rather a consequence of the challenges inherent to dependently-typed programming. Understanding the unification process, identifying the sources of ambiguity during pattern matching, and applying appropriate workarounds are essential skills for anyone working with dependent types. This allows for effective type safe programming using the core type system provided by Idris.
