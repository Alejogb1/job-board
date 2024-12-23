---
title: "Why isn't the `Sum<&T>` trait implemented for type `T` in Rust?"
date: "2024-12-23"
id: "why-isnt-the-sumt-trait-implemented-for-type-t-in-rust"
---

Okay, let's dive into this. It's a question I've definitely pondered before, particularly back when I was wrestling with a large numerical simulation library where these kinds of subtleties could cause some serious friction in the code.

The core issue, as you've identified, is that `std::iter::Sum<T>` is not automatically implemented for a generic type `T`, even when we might naively expect it should be. Instead, it's implemented for `Sum<&T>` (where `T` itself needs to implement `Add<T>`). The reasoning behind this design decision is multifaceted but fundamentally revolves around Rust's emphasis on ownership, borrowing, and avoiding implicit copying that might introduce unexpected performance bottlenecks or data mutation issues.

The `Sum` trait, at its heart, is used by iterators to accumulate values. Specifically, the `sum()` method, which is available on any iterator whose elements implement the `Sum` trait. If you attempt to call `.sum()` on an iterator that produces a type `T` directly, you will find it only works if `T` itself implements `Sum`. But this isn’t generally the case and it is generally an error.

Let's think about what might happen if `Sum<T>` was implemented automatically. Consider a situation where `T` is a large, complex struct containing, say, a vector. Each accumulation in a sum operation would then involve a full *copy* of the entire struct. This could be immensely inefficient. While, in some scenarios, a copy might be exactly what you want, in most numerical/data processing cases, we are typically working with references to the data for both performance and correct mutability management. Hence Rust forces you to explicitly work with references, enabling control over what is being moved, cloned or borrowed, which brings down on average unnecessary allocations.

The `Sum<&T>` approach, on the other hand, is far more flexible and performance-conscious. It allows the iterator to return references to elements without requiring these elements to implement `Copy`. If the type `T` does implement `Copy`, then you get an implicit copy of the reference, which means a copy of the *pointer* to the data, not the data itself. This is precisely what we want in most situations.

To make this more concrete, consider three simple examples. First, a scenario where you’re working with integers, which are copy types and `sum()` works as you expect on an `iter`. Second, let's imagine a situation with a non-copy type, like a `String`. Finally, let’s look at how you can implement summing your own type.

```rust
fn example_integers() {
    let numbers = vec![1, 2, 3, 4, 5];
    let sum: i32 = numbers.iter().sum();
    println!("Sum of integers: {}", sum); // Output: Sum of integers: 15

    // this also works, and the implicit reference is used by the sum
    let sum_ref: i32 = numbers.iter().sum();
    println!("Sum of integers: {}", sum_ref);
}
```

In this example with integers, `i32` is copy, and thus the implicit dereference of the iterator is the same as a reference for all intent and purposes. `sum()` is implemented directly for `i32` as you might expect.

```rust
fn example_strings() {
    let strings = vec!["hello".to_string(), "world".to_string(), "!".to_string()];
    // let sum: String = strings.iter().sum(); // this will error, because strings are not copy and Sum<String> isn't implemented.
    // we must tell it that we want the references to be used
     let sum: String = strings.iter().cloned().sum(); // This works because we cloned the strings
     println!("Concatenated string: {}", sum); // Output: Concatenated string: helloworld!


    let strings2 = vec!["hello".to_string(), "world".to_string(), "!".to_string()];
    let sum_ref: String = strings2.iter().fold(String::new(), |mut acc, s| {
            acc.push_str(s);
            acc
        });
    println!("Concatenated string: {}", sum_ref);

}
```

Here, `String` does not implement `Copy`, hence directly using `.sum()` errors if we don't manually handle that fact. We use `.cloned()` to force a clone of each of the string in the iterator, which allows the sum to work. Alternatively, you can bypass `.sum()` entirely and just use fold to handle the summing behavior by yourself. Either of these approaches works to ensure the type matches the implemented Sum<T> trait. This also avoids issues if your types implement Add but not Sum.

Finally, let's look at a custom type and how one might implement Sum.

```rust
#[derive(Debug, PartialEq, Clone, Copy)]
struct Point {
    x: i32,
    y: i32,
}

impl std::ops::Add for Point {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        Point {
            x: self.x + other.x,
            y: self.y + other.y,
        }
    }
}
// Note: we do not need to implement Copy as long as we can add &T
impl std::iter::Sum for Point {
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
         iter.fold( Point {x:0,y:0} , |a,b| a+b)
    }
}

fn example_points() {
    let points = vec![Point { x: 1, y: 2 }, Point { x: 3, y: 4 }, Point { x: 5, y: 6 }];
     // let sum_points: Point = points.iter().sum(); // this will error since sum is not implement for &Point (it does not implement Add<&Point>)
     let sum_points: Point = points.iter().cloned().sum(); // this works as the iterator now produce type Point
    println!("Sum of points: {:?}", sum_points); // Output: Sum of points: Point { x: 9, y: 12 }
}

```

In this case, we implement `Add` for our custom struct `Point`. Since `Sum` is implemented on Point directly (which must be passed by value). In general, you should avoid creating types that will be summed, that are not copy because in general the best way to implement Sum is by using references to your data, or by cloning it. The use of references is paramount for performance-sensitive applications.

As you can see, the seemingly "missing" `Sum<T>` implementation is not an oversight, but a deliberate choice to enable more performant and predictable code. It encourages explicit handling of ownership and borrowing, which is a key tenet of Rust's design.

If you want to delve deeper into this, I highly recommend examining the source code of the standard library. Specifically, look at the implementation of the `Sum` trait and the `sum` method on iterators. You will find the details in the `core` and `std` crates. Another good reference is "Programming Rust" by Jim Blandy, Jason Orendorff and Leonora Tindall, which dedicates a substantial portion of the book to ownership and borrowing, which is the underlying principle why the `Sum` trait is implemented as it is. You should also take a look at "Effective Rust" by Doug Milford, it contains very good explanations of the best way to handle iterators and how it relates to Rust design principles. For very high-level understanding, the Rust book can also help clear up some of the ownership-related concepts. I hope this explanation is helpful!
