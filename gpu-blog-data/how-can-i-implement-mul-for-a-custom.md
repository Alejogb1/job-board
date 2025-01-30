---
title: "How can I implement Mul for a custom struct to support both left and right multiplication?"
date: "2025-01-30"
id: "how-can-i-implement-mul-for-a-custom"
---
Implementing multiplication (`Mul`) for a custom struct to handle both left and right multiplication requires a nuanced understanding of trait implementation and operator overloading in Rust. The crucial detail is that Rust's `Mul` trait, defined in the `std::ops` module, operates on a single type, implying that the left-hand side (LHS) and right-hand side (RHS) operands must generally be the same type. To support multiplication where the RHS is a different type, typically involving scalar multiplication, we must implement `Mul` multiple times, each time with a distinct RHS type. This multi-implementation approach can significantly complicate the process compared to languages that implicitly coerce types.

I've personally wrestled with this challenge while building a custom linear algebra library, where the core types were vectors and matrices. The goal was to permit multiplication between vectors and scalars, and matrices and vectors, all while maintaining strong type safety and avoiding any run-time overhead from dynamic dispatch. The issue manifests when trying to define operations where the order of operands is significant. For instance, in mathematical terms, multiplying a matrix by a vector is not commutative; i.e., `Matrix * Vector` and `Vector * Matrix` represent distinct operations. The key lies not in modifying the behavior of the `Mul` trait itself but in providing multiple implementations, taking different types as arguments.

The core of the problem is Rust's type system enforcing clear type distinctions and the static nature of operator overloading. The `Mul` trait's signature is `fn mul(self, rhs: Rhs) -> Self::Output`. Therefore, when dealing with a custom struct, say `MyStruct`, we need to provide implementations for cases where `rhs` could be `MyStruct` itself, or any other type with which `MyStruct` is logically multiplied. For example, if we have a custom struct representing a 2D vector, we might want to be able to multiply it by a scalar. We will address this with concrete code in the examples below.

**Example 1: Implementing Multiplication with Self**

Let's start with a `Vec2` struct. We will first implement `Mul` so that we can multiply two `Vec2` structures, using a component-wise approach. The resulting vector will be another vector of the same type.

```rust
#[derive(Debug, Copy, Clone, PartialEq)]
struct Vec2 {
    x: f64,
    y: f64,
}

impl std::ops::Mul for Vec2 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        Vec2 {
            x: self.x * rhs.x,
            y: self.y * rhs.y,
        }
    }
}

fn main() {
    let v1 = Vec2 { x: 2.0, y: 3.0 };
    let v2 = Vec2 { x: 4.0, y: 5.0 };
    let v3 = v1 * v2;
    println!("{:?}", v3); // Output: Vec2 { x: 8.0, y: 15.0 }
}
```
This implementation is straightforward: `Mul` is implemented for `Vec2` with `Self` as the RHS type, and the output is also `Self`. This enables us to multiply two `Vec2` instances component-wise. The `main` function demonstrates the usage. Note that if the goal was to obtain a dot product the implementation of `mul` would have to be different, this is left as an exercise to the reader.

**Example 2: Implementing Multiplication with a Scalar**

Now, let's enable scalar multiplication where we multiply a `Vec2` by a float. Crucially, we need to provide a *separate* `Mul` implementation for this.
```rust
#[derive(Debug, Copy, Clone, PartialEq)]
struct Vec2 {
    x: f64,
    y: f64,
}

impl std::ops::Mul for Vec2 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        Vec2 {
            x: self.x * rhs.x,
            y: self.y * rhs.y,
        }
    }
}


impl std::ops::Mul<f64> for Vec2 {
    type Output = Self;

    fn mul(self, rhs: f64) -> Self {
        Vec2 {
            x: self.x * rhs,
            y: self.y * rhs,
        }
    }
}


fn main() {
    let v1 = Vec2 { x: 2.0, y: 3.0 };
    let scalar = 2.0;
    let v2 = v1 * scalar;
    println!("{:?}", v2);  // Output: Vec2 { x: 4.0, y: 6.0 }
}
```
Here, we've added a second `impl` block: `impl std::ops::Mul<f64> for Vec2`. This indicates that we're implementing `Mul` for `Vec2` specifically with `f64` as the RHS. The resulting code is an instance of `Vec2`. This gives us left-hand scalar multiplication. The `main` function showcases how this is used.

**Example 3: Implementing Right-Hand Scalar Multiplication**

The previous example permits us to write code like `vector * scalar` but not `scalar * vector`, which is also a desired mathematical operation. To achieve that, we need to provide a `Mul` implementation for the scalar.
```rust
#[derive(Debug, Copy, Clone, PartialEq)]
struct Vec2 {
    x: f64,
    y: f64,
}

impl std::ops::Mul for Vec2 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        Vec2 {
            x: self.x * rhs.x,
            y: self.y * rhs.y,
        }
    }
}


impl std::ops::Mul<f64> for Vec2 {
    type Output = Self;

    fn mul(self, rhs: f64) -> Self {
        Vec2 {
            x: self.x * rhs,
            y: self.y * rhs,
        }
    }
}

impl std::ops::Mul<Vec2> for f64 {
    type Output = Vec2;

    fn mul(self, rhs: Vec2) -> Vec2 {
        Vec2 {
            x: self * rhs.x,
            y: self * rhs.y,
        }
    }
}


fn main() {
    let v1 = Vec2 { x: 2.0, y: 3.0 };
    let scalar = 2.0;
    let v2 = scalar * v1;
    println!("{:?}", v2);  // Output: Vec2 { x: 4.0, y: 6.0 }
}
```
Now we have an additional `impl`: `impl std::ops::Mul<Vec2> for f64`.  This defines how `f64` types are multiplied by `Vec2` types. The output type is `Vec2` as well. The `main` function illustrates this capability, showcasing the commutative behavior for scalar multiplication. This also permits us to perform `scalar * vector`.

This multi-implementation strategy is not limited to scalars; you can extend this to support matrix-vector and vector-matrix multiplication. Each new type combination that represents a new valid operation must have a dedicated `impl` of `Mul`.

Several key aspects of `Mul` implementation must be kept in mind.  Error handling should be properly implemented where needed. For example, if implementing matrix multiplication, it’s necessary to check if the matrix dimensions are compatible before conducting the operation. Similarly, when implementing other mathematical operations, ensure that appropriate validation and handling of exceptional cases is in place.  For advanced mathematical operations, explore libraries that provide well-tested and optimized implementations, but this detailed approach using `Mul` will permit the creation of fine-grained custom types with custom operations.

**Resource Recommendations**

For deeper understanding of the trait system and operator overloading, refer to the official Rust book. The Rust documentation also serves as a crucial reference for details on the standard library and the `std::ops` module. Numerous online communities and forums, like the Rust subreddit and the official forums, can offer support and perspectives. Examining existing crates that implement math libraries, especially the source code of commonly used crates, will enhance your grasp of real-world application of these techniques.
