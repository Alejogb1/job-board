---
title: "Why can't I collect an `Iterator<Item = Box<Dog>>` into a `Vec<Box<dyn Animal>>` in Rust?"
date: "2024-12-23"
id: "why-cant-i-collect-an-iteratoritem--boxdog-into-a-vecboxdyn-animal-in-rust"
---

,  I've seen this scenario come up quite a few times over the years, and it’s a common point of confusion when working with Rust’s ownership system and traits. You're trying to collect an iterator of `Box<Dog>` into a `Vec<Box<dyn Animal>>`, which seems intuitive at first, but the compiler is rightfully flagging it. The core issue revolves around type variance and the inherent safety guarantees that Rust provides. We’re not just battling syntax here; we’re dealing with foundational concepts.

Let’s break this down. When you declare `dyn Animal`, you're creating a trait object, which is a dynamically sized type. This type doesn’t represent a concrete struct like `Dog`; it represents any type that implements the `Animal` trait. `Box<dyn Animal>` is a heap-allocated pointer to such an instance. Crucially, each trait object carries a hidden "vtable"—a lookup table containing the address of the implementation of methods declared in the `Animal` trait for that concrete type.

The reason an `Iterator<Item = Box<Dog>>` doesn't directly convert to a `Vec<Box<dyn Animal>>` is that Rust needs to ensure type safety at compile time. `Box<Dog>` is a specific, concrete type, while `Box<dyn Animal>` is a type that can represent potentially numerous implementing types. The compiler can’t implicitly convert from a specific type to a more generic one without the programmer explicitly stating this intention. This is where type variance comes in. In this specific situation, `Box` is invariant over its contained type. This means that `Box<Dog>` is *not* a subtype of `Box<dyn Animal>`, even if `Dog` implements `Animal`.

To illustrate this better, think of it this way. Suppose we have another type, `Cat`, that also implements the `Animal` trait. If the conversion was automatically allowed, you could have a `Vec<Box<dyn Animal>>` which, at runtime, contains `Box<Dog>` and `Box<Cat>` instances. The vtable for each would be different. When you iterate through this vector, the correct vtable would need to be accessed for each element to correctly call methods declared in `Animal`. Rust's ownership and type systems must prevent situations where you have a `Box<dyn Animal>` referring to a `Dog` when the compiler might expect a `Cat`, and vice versa. Such situations could lead to memory corruption and unsafe behaviors.

Now, how can you achieve your goal? You need to explicitly tell Rust how to convert the `Box<Dog>` instances to `Box<dyn Animal>` instances. This involves a process called "upcasting," where you convert from a specific type to a more general type by creating a trait object. You do this through an explicit conversion. We need to iterate over `Box<Dog>` and convert them individually to `Box<dyn Animal>`.

Let me give you a few code examples to show this process. I'll set up the basic `Animal` trait and a `Dog` struct first:

```rust
trait Animal {
    fn make_sound(&self);
}

struct Dog {
    name: String,
}

impl Animal for Dog {
    fn make_sound(&self) {
        println!("Woof! My name is {}", self.name);
    }
}
```

Here's the first working example, which makes use of explicit mapping:

```rust
fn example_one() {
    let dogs = vec![
        Box::new(Dog { name: "Buddy".to_string() }),
        Box::new(Dog { name: "Bella".to_string() }),
    ];

    // Explicitly map each Box<Dog> to a Box<dyn Animal>
    let animals: Vec<Box<dyn Animal>> = dogs
        .into_iter()
        .map(|dog| dog as Box<dyn Animal>)
        .collect();

    for animal in animals {
        animal.make_sound();
    }
}

```

In this example, the `.map(|dog| dog as Box<dyn Animal>)` is the key. We are explicitly casting each `Box<Dog>` into a `Box<dyn Animal>`. The `as` keyword here performs the type coercion to construct the trait object on the heap, including setting the vtable.

Here's a second example, slightly more verbose, but perhaps more explicit:

```rust
fn example_two() {
    let dogs = vec![
        Box::new(Dog { name: "Charlie".to_string() }),
        Box::new(Dog { name: "Lucy".to_string() }),
    ];

    let mut animals: Vec<Box<dyn Animal>> = Vec::new();
    for dog in dogs {
        animals.push(dog as Box<dyn Animal>);
    }

    for animal in animals {
        animal.make_sound();
    }
}
```

This second snippet does the same thing but uses a loop rather than a functional map. It's functionally equivalent to the first example, but explicitly demonstrates the type coercion within a loop, offering a bit more clarity for those who might be newer to functional programming approaches.

Finally, here is a slightly more generic solution that abstracts this upcasting pattern into a function that could be reused:

```rust
fn upcast_vec<T: Animal + 'static>(items: Vec<Box<T>>) -> Vec<Box<dyn Animal>> {
    items.into_iter().map(|item| item as Box<dyn Animal>).collect()
}

fn example_three() {
    let dogs = vec![
        Box::new(Dog { name: "Max".to_string() }),
        Box::new(Dog { name: "Daisy".to_string() }),
    ];

    let animals = upcast_vec(dogs);

    for animal in animals {
        animal.make_sound();
    }
}
```

The `upcast_vec` function demonstrates how to abstract the pattern of converting a vector of specific `Box` types into a vector of `Box<dyn Animal>`. Note the `'static` bound—this means that the type must have no references that could potentially outlive it.

From these examples, you can see the core principle: you can't directly treat a `Vec<Box<Dog>>` as a `Vec<Box<dyn Animal>>`. You need to explicitly iterate through the concrete types and coerce them into the trait objects, creating the vtable in the process. This ensures type safety and allows Rust to correctly handle dynamic dispatch when dealing with trait objects.

If you’d like to delve into this further, I recommend reading about type variance and trait objects in "Programming Rust" by Jim Blandy, Jason Orendorff, and Leonora F.S. Tindall, or, for a more academic look, "Types and Programming Languages" by Benjamin C. Pierce. Understanding the intricacies of these concepts will significantly enhance your proficiency in Rust and explain why these seemingly frustrating compiler errors are essential for building safe and correct programs. Specifically focus on the chapter covering subtyping and variance. They provide a more detailed and formal explanation of this topic and its importance in language design. These texts are quite rigorous and provide deep insight into these topics.
