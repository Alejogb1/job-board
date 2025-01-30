---
title: "Why isn't a `const` associated with a trait in Rust behaving as expected?"
date: "2025-01-30"
id: "why-isnt-a-const-associated-with-a-trait"
---
In Rust, associating a `const` with a trait doesn't inherently create a constant that's directly accessible through the trait itself. Instead, such a `const` defines a *requirement* for any type implementing the trait. This distinction is crucial for understanding why a `const` defined within a trait doesn't act as a concrete, trait-associated constant, but rather an interface that implementors must satisfy. I’ve often seen newcomers to Rust, and even seasoned developers occasionally, struggle with this specific characteristic of trait-associated `const`s.

The key point is that a trait defines a set of *obligations* for any type that chooses to implement it. The `const` item within a trait declaration is not a concrete value provided by the trait, rather it’s a declaration of a constant that *must* exist and have a specific type for each implementing type. You aren’t declaring a default value, like you might in other languages. Each implementing type has the responsibility to define its own `const` value which fulfills the trait's requirement. In simpler terms, the trait is a blueprint and each implementor builds their version of it with its own concrete constants. It does not inherit some default implementation.

To illustrate, imagine a hypothetical trait `Measurable` designed to allow comparisons between different units of measure. This trait might define a constant representing a conversion factor:

```rust
trait Measurable {
    const CONVERSION_FACTOR: f64;
    fn value(&self) -> f64;
    fn to_base_unit(&self) -> f64 {
       self.value() * Self::CONVERSION_FACTOR
    }
}
```

Here, `CONVERSION_FACTOR` is *not* a concrete value provided by the `Measurable` trait, it is a *requirement* to define a concrete value when implementing the trait. The method `to_base_unit` shows that a constant member is accessed using `Self::CONSTANT_NAME`, which is the standard method for accessing constants associated with types. The `Self` here refers to the concrete type that implements the trait and, therefore, carries its concrete implementation of the trait constant. Let's look at concrete implementations now.

Consider the implementation of `Measurable` for a type representing meters:

```rust
struct Meters(f64);

impl Measurable for Meters {
    const CONVERSION_FACTOR: f64 = 1.0;
    fn value(&self) -> f64 {
       self.0
    }
}
```

Here, the `Meters` struct declares the specific value of `CONVERSION_FACTOR` as 1.0. This is the concrete value that satisfies the requirement introduced by the `Measurable` trait for the `Meters` struct. Another implementor could be a `Feet` struct:

```rust
struct Feet(f64);

impl Measurable for Feet {
  const CONVERSION_FACTOR: f64 = 0.3048;
    fn value(&self) -> f64 {
       self.0
    }
}
```

The `Feet` struct provides its own concrete value for `CONVERSION_FACTOR`, namely 0.3048. This value is distinct from the value provided by the `Meters` implementation. Each implementation provides a concrete implementation for that specific type and the implementation within the trait is not inherited.

The critical point to understand is that if I attempt to access `Measurable::CONVERSION_FACTOR`, I will find that this is an error. The trait `Measurable` itself does not have a value associated with `CONVERSION_FACTOR`.

```rust
// This will cause a compiler error
fn print_conversion_factor() {
    println!("{}", Measurable::CONVERSION_FACTOR);
}
```

This will generate an error like "`associated constant `CONVERSION_FACTOR` not found for trait `Measurable``. This error illustrates the core principle that `const` within a trait define a requirement, not a concrete value. You must access the concrete value via a concrete type:

```rust
fn print_meters_conversion_factor() {
    let m = Meters(1.0);
    println!("{}", Meters::CONVERSION_FACTOR);
    println!("{}", m.to_base_unit());
}

fn print_feet_conversion_factor() {
   let f = Feet(1.0);
   println!("{}", Feet::CONVERSION_FACTOR);
    println!("{}", f.to_base_unit());
}
```

In these examples, `Meters::CONVERSION_FACTOR` and `Feet::CONVERSION_FACTOR` resolve to the concrete `const` values defined within the respective implementations. Calling `m.to_base_unit()` or `f.to_base_unit()` utilizes the respective constant correctly because the method is called on an instance of a concrete type.

Contrast this with other languages that might allow some form of static or default values on interfaces, and it becomes clear why this can be initially counterintuitive. In languages with more direct object inheritance and class-based models, we might expect an interface's constants to be available on the interface itself. Rust’s traits behave more as contracts. They outline the capabilities an implementing type *must* possess, and each implementor fulfills the contract in its own way.

The behavior of trait constants helps encourage a design that’s more flexible and less tied to assumptions. By requiring each type to define its constants, the code becomes more explicit and avoids issues that can arise with default or inherited values which don't fit the context. In fact, having each type define its constants allows the same trait to be implemented for wildly differing data types and use cases.

To fully grasp this behavior and the broader principles behind Rust traits, I recommend exploring several resources. The official Rust Book provides a comprehensive explanation of traits and their role in the language. The documentation also dives into details about associated types and consts. Additionally, numerous online articles and blogs written by Rust experts can provide additional insights, as well as worked examples of various uses of trait constants.

In summary, a trait-associated `const` isn’t an inherited value accessible from the trait, but a *requirement* for each type that implements the trait. This is fundamental to how Rust implements polymorphism via traits and avoids the pitfalls associated with more class-based or inheritance-based object models. Each implementor has the freedom to define the const according to their own specific needs and this is by design in order to guarantee flexibility and freedom of implementation. It emphasizes the role of traits as blueprints rather than sources of static values. Understanding this distinction is essential for effectively using traits in Rust.
