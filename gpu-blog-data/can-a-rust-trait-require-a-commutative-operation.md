---
title: "Can a Rust trait require a commutative operation?"
date: "2025-01-30"
id: "can-a-rust-trait-require-a-commutative-operation"
---
Rust traits, by their nature, define *capabilities*, not *properties*. They specify what a type *can do*, not what inherent characteristics it possesses.  Consequently, a trait *cannot directly* enforce a commutative property on an associated operation. This is a fundamental limitation stemming from Rust's type system and its focus on static dispatch. My experience building numerical libraries in Rust has repeatedly highlighted this challenge. While we can't force commutativity at the trait level, we can utilize techniques to mitigate its absence and enforce it in specific contexts.

The core issue lies in how traits are used.  A trait defines a set of methods that a type must implement to conform to that trait. For example, a `Add` trait specifies an `add` method.  The trait definition itself doesn’t understand the semantics of addition, such as commutativity (a + b = b + a). Instead, the specific implementations of `add` on concrete types are responsible for any semantic correctness, including fulfilling commutativity where it applies. The trait merely dictates the syntax of the operation (an `add` method accepting an `Rhs` type and returning a `Self`).

Let's examine three different situations encountered in my work and how we typically handle this. The first involves a simple numerical addition scenario, the second considers custom data structures, and the third addresses type-level encoding using marker traits.

**Example 1: Implementing Numerical Addition**

The standard library provides the `std::ops::Add` trait. Suppose you're working with a custom numerical type `Complex`.

```rust
use std::ops::Add;

#[derive(Debug, Clone, Copy, PartialEq)]
struct Complex {
    real: f64,
    imag: f64,
}

impl Complex {
    fn new(real: f64, imag: f64) -> Self {
        Complex { real, imag }
    }
}

impl Add for Complex {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Complex {
            real: self.real + other.real,
            imag: self.imag + other.imag,
        }
    }
}


fn main() {
    let c1 = Complex::new(1.0, 2.0);
    let c2 = Complex::new(3.0, 4.0);
    let c3 = c1 + c2;
    let c4 = c2 + c1;
    println!("c1 + c2 = {:?}", c3); // Output: c1 + c2 = Complex { real: 4.0, imag: 6.0 }
    println!("c2 + c1 = {:?}", c4); // Output: c2 + c1 = Complex { real: 4.0, imag: 6.0 }
    assert_eq!(c3,c4);
}
```
This example implements the `Add` trait for our `Complex` number type.  The trait itself does not guarantee commutativity. In this case, due to the properties of floating-point addition (which is commutative), and the way I defined the addition, the resulting `add` operation on `Complex` is indeed commutative. However, this is specific to the *implementation* of the `add` method and not a trait-level guarantee. If we changed `add` to perform a non-commutative operation on `real` and `imag` parts, Rust wouldn't raise any errors - it is entirely up to us to provide a coherent implementation. The `assert_eq!` at the end of main shows that commutativity *holds* for our complex number implementation.

**Example 2: Non-Commutative Matrix Multiplication**

Consider matrix multiplication. This operation is notably not commutative, except in special cases. Suppose we have a simplified `Matrix` struct.

```rust
use std::ops::Mul;

#[derive(Debug, Clone, PartialEq)]
struct Matrix {
    rows: usize,
    cols: usize,
    data: Vec<f64>,
}

impl Matrix {
    fn new(rows: usize, cols: usize, data: Vec<f64>) -> Self {
        assert_eq!(rows * cols, data.len());
        Matrix { rows, cols, data }
    }

    fn get(&self, row: usize, col: usize) -> f64 {
         self.data[row * self.cols + col]
    }

    fn set(&mut self, row: usize, col: usize, val: f64){
        self.data[row * self.cols + col] = val;
    }

    fn get_size(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }
}


impl Mul for Matrix {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        assert_eq!(self.cols, other.rows, "Matrix dimensions not compatible");
        let (rows, _) = self.get_size();
        let (_, cols) = other.get_size();

        let mut result = Matrix::new(rows, cols, vec![0.0; rows * cols]);


        for i in 0..rows{
            for j in 0..cols{
                let mut sum = 0.0;
                for k in 0..self.cols{
                    sum += self.get(i, k) * other.get(k, j);
                }
                result.set(i,j, sum)
            }
        }

        result
    }
}


fn main(){
    let m1 = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
    let m2 = Matrix::new(2, 2, vec![5.0, 6.0, 7.0, 8.0]);

    let m3 = m1.clone() * m2.clone();
    let m4 = m2 * m1;

    println!("m1 * m2 = {:?}", m3);
    println!("m2 * m1 = {:?}", m4);
    assert_ne!(m3, m4);

}
```

The `Mul` trait is implemented for `Matrix`. The implementation adheres to matrix multiplication rules, which, generally, are not commutative.  As such, the computed `m3` and `m4` values are different. Again, there is no trait-level enforcement or constraint requiring or preventing commutativity. The trait only ensures that a `mul` method accepting `Self` exists and returns a type conforming to `Self::Output`. The commutativity of the operation is defined by the implementation details of the `mul` method.

**Example 3:  Type-Level Enforcement with Marker Traits (Partial Solution)**

While a trait cannot *directly* enforce commutativity, we can use marker traits and generics to achieve some level of type-level encoding.  Suppose we have a mathematical concept where some, but not all, operations are commutative. We can define marker traits to tag commutative and non-commutative types, then leverage generics in type bounds to enforce operation compatibility only for commutative types. While this doesn't force the trait itself to *be* commutative, it prevents using non-commutative types in a context that *expects* commutativity.

```rust
use std::ops::Add;

// Marker trait to identify a commutative type
trait Commutative {}

// Struct representing a commutative numerical type
#[derive(Debug, Clone, Copy, PartialEq)]
struct  CommutativeNum(f64);

impl Commutative for CommutativeNum{}

impl Add for CommutativeNum{
    type Output = Self;
    fn add(self, other: Self) -> Self {
        CommutativeNum(self.0 + other.0)
    }
}



// Struct representing a non-commutative type
#[derive(Debug, Clone, Copy, PartialEq)]
struct NonCommutative(f64);

impl Add for NonCommutative {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        NonCommutative(self.0 - other.0) // Intentionally non-commutative
    }
}


// Function expecting a commutative operand type.
//  This is where the `Commutative` marker trait is useful.
fn commutative_add<T: Add<Output = T> + Commutative>(a: T, b: T) -> T {
    a + b
}


fn main(){
    let c1 = CommutativeNum(1.0);
    let c2 = CommutativeNum(2.0);

    let n1 = NonCommutative(1.0);
    let n2 = NonCommutative(2.0);

    let c3 = commutative_add(c1, c2);
    println!("{:?}", c3); // Output: CommutativeNum(3.0)


    // The next line would not compile, because NonCommutative does not implement Commutative trait
    // let n3 = commutative_add(n1, n2); // Compiler error: NonCommutative does not implement Commutative

    let n3 = n1+ n2; // This still works, because non commutative type *can be added*, but not in the context of `commutative_add`
    println!("{:?}", n3); //Output: NonCommutative(-1.0)

}
```

Here, `Commutative` acts as a marker trait. `CommutativeNum` implements both `Add` and `Commutative`.  `NonCommutative` implements `Add` but not `Commutative`.  The `commutative_add` function leverages this. It is generic with a type bound requiring the type to implement both `Add` and `Commutative`.  This prevents us from passing a `NonCommutative` to this particular function, although we can still use the plain add method for `NonCommutative` types in other contexts. Note that this method only provides *partial* solution. This example encodes a type-level constraint using the traits. The actual commutativity of addition has to be guaranteed by the implementation details of the `add` method of `CommutativeNum`, which is commutative, but not by the trait definition itself.

**Resource Recommendations**

To deepen your understanding of traits and type system in Rust, I recommend consulting the following resources:

1.  The official Rust Book: This is a comprehensive guide covering the basics of traits and generics.
2.  "Programming Rust" by Jim Blandy et al: This book explores Rust in detail, offering insights into its more complex features.
3.  The Rust Reference: For the most detailed explanations of Rust’s language semantics. Focus on sections about traits, generics, and operator overloading.

In conclusion, while Rust traits cannot directly enforce commutativity, we use careful implementations and complementary techniques like marker traits and generic constraints to manage and, where appropriate, enforce commutativity in our code. The responsibility for correctness lies with the implementors of the trait methods rather than the trait definition itself. This reflects Rust’s philosophy of providing mechanisms without imposing policies regarding the semantic behavior of those mechanisms.
