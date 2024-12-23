---
title: "How can I create a vector containing items that implement two traits?"
date: "2024-12-23"
id: "how-can-i-create-a-vector-containing-items-that-implement-two-traits"
---

Alright, let's tackle this. I've bumped into this scenario a good number of times over the years, particularly when dealing with systems that require composable behaviors. The challenge, as you've presented it, is how to construct a `Vec` (or similar data structure) that holds items conforming to *two* distinct traits. It's not about concrete types; it’s about ensuring that the items we store can exhibit behaviors defined by multiple interfaces. This might seem straightforward at first glance but can become a bit nuanced.

The core issue here is that rust doesn't support multiple inheritance in the way some object-oriented languages do. You can't just say "this type implements both trait a and trait b, and I want a `vec` of those" as a direct type declaration. We need to approach this problem with some careful consideration of rust's type system.

My first experience with this was when building a plugin system. I had various plugins each needing to implement both `Loadable` (responsible for initializing) and `Renderable` (responsible for displaying on the screen). A `Vec` of simply either `Loadable` or `Renderable` wouldn't do, because what we really needed was the ability to manage things that can do *both*.

Let's walk through the strategies I found effective, coupled with code snippets to illustrate the concepts.

**Strategy 1: Trait Objects with Compound Bounds**

This is likely the most common and flexible approach. We utilize trait objects with a compound `dyn` bound. A trait object is a dynamically dispatched type whose exact concrete type is not known at compile time. Think of it like a pointer to an object that implements a specific trait or set of traits. This allows us to create collections of varied underlying types as long as they adhere to the interface defined by the traits.

Here's how it plays out:

```rust
trait Loadable {
    fn load(&self) -> bool;
}

trait Renderable {
    fn render(&self);
}

struct PluginA;
impl Loadable for PluginA {
    fn load(&self) -> bool { println!("PluginA loaded"); true }
}
impl Renderable for PluginA {
    fn render(&self) { println!("PluginA rendered"); }
}

struct PluginB;
impl Loadable for PluginB {
    fn load(&self) -> bool { println!("PluginB loaded"); true }
}
impl Renderable for PluginB {
   fn render(&self) { println!("PluginB rendered"); }
}

fn main() {
    let plugins: Vec<Box<dyn Loadable + Renderable>> = vec![
        Box::new(PluginA),
        Box::new(PluginB),
    ];

    for plugin in &plugins {
        plugin.load();
        plugin.render();
    }
}
```

In this example, `dyn Loadable + Renderable` specifies that the objects in the `Vec` must implement *both* `Loadable` and `Renderable`. Notice the `Box` is necessary for trait objects as they are dynamically sized. The code iterates through the `Vec`, calling methods defined on the traits. It's runtime polymorphism in action. This method is powerful and allows you to work with varied types. However, it incurs the cost of dynamic dispatch, which can have a performance impact in very hot loops.

**Strategy 2: Enums with Associated Data**

When you know all the possible concrete types that need to implement both traits beforehand, or when your set of types is limited and known at compile time, an enum can be a performant alternative. An enum allows us to wrap various structs within it, as long as they all fulfill our trait requirements.

Here's how this strategy looks:

```rust
trait Loadable {
    fn load(&self) -> bool;
}

trait Renderable {
    fn render(&self);
}

struct PluginA;
impl Loadable for PluginA {
    fn load(&self) -> bool { println!("PluginA loaded"); true }
}
impl Renderable for PluginA {
    fn render(&self) { println!("PluginA rendered"); }
}

struct PluginB;
impl Loadable for PluginB {
    fn load(&self) -> bool { println!("PluginB loaded"); true }
}
impl Renderable for PluginB {
   fn render(&self) { println!("PluginB rendered"); }
}

enum Plugin {
    A(PluginA),
    B(PluginB),
}

impl Loadable for Plugin {
    fn load(&self) -> bool {
        match self {
            Plugin::A(a) => a.load(),
            Plugin::B(b) => b.load(),
        }
    }
}

impl Renderable for Plugin {
   fn render(&self) {
        match self {
            Plugin::A(a) => a.render(),
            Plugin::B(b) => b.render(),
        }
    }
}


fn main() {
    let plugins: Vec<Plugin> = vec![
        Plugin::A(PluginA),
        Plugin::B(PluginB),
    ];

    for plugin in &plugins {
        plugin.load();
        plugin.render();
    }
}
```

This strategy avoids the runtime cost of dynamic dispatch. We define `Plugin` as an enum, where each variant holds a specific concrete type. We then implement `Loadable` and `Renderable` on the enum by pattern matching to the correct variant and calling the methods on the underlying struct. This works great for known, bounded sets. If you need to be able to extend the system later with new plugins, the enum is less convenient.

**Strategy 3: Generic Data Structures with Trait Constraints**

If you want to go a slightly different direction, and the goal is not really storing a mixed vector of differing types implementing your traits, you could consider a generic struct that imposes trait bounds on its internal members. This is particularly useful when your data structure needs to work with different concrete types that implement the same traits but does not need a heterogenous collection.

Here's a conceptual snippet to illustrate:

```rust
trait Loadable {
    fn load(&self) -> bool;
}

trait Renderable {
    fn render(&self);
}

struct PluginA;
impl Loadable for PluginA {
    fn load(&self) -> bool { println!("PluginA loaded"); true }
}
impl Renderable for PluginA {
    fn render(&self) { println!("PluginA rendered"); }
}

struct PluginB;
impl Loadable for PluginB {
    fn load(&self) -> bool { println!("PluginB loaded"); true }
}
impl Renderable for PluginB {
   fn render(&self) { println!("PluginB rendered"); }
}

struct PluginManager<T: Loadable + Renderable> {
    plugin: T,
}

impl <T: Loadable + Renderable> PluginManager<T> {
    fn manage(&self) {
        self.plugin.load();
        self.plugin.render();
    }
}


fn main() {
    let manager_a = PluginManager { plugin: PluginA };
    manager_a.manage();

    let manager_b = PluginManager { plugin: PluginB };
    manager_b.manage();
}
```

Here `PluginManager` is parameterized by a generic type `T` which is constrained to implement both `Loadable` and `Renderable`. This means that `PluginManager` instances can hold any concrete struct which implements both those traits. This method is suitable when you need static dispatch and prefer type safety over dynamic flexibility. It's not about storing a `Vec` of different types. It’s about managing the logic around *one* concrete type that has those traits.

**Resources for further exploration:**

For deepening your understanding of trait objects and dynamic dispatch, I'd highly recommend the chapter on trait objects in "The Rust Programming Language" by Steve Klabnik and Carol Nichols (available online as well as in print). It provides an excellent, in-depth treatment. Additionally, for advanced usage, consider diving into "Programming Rust" by Jim Blandy, Jason Orendorff, and Leonora F.S. Tindall; the sections on generics, traits, and lifetimes are essential. Finally, for a more theoretical grounding in type systems, particularly around sum types (as seen in the enum approach) and bounded polymorphism, "Types and Programming Languages" by Benjamin C. Pierce provides a good foundation.

In closing, choosing the correct strategy depends on your specific needs and constraints. Trait objects provide flexibility at the cost of some runtime overhead, enums offer static dispatch when your types are known at compile time, and generic data structures provide yet another way when you need type safety and compile-time dispatch. Hopefully, this gives you the tools and understanding to navigate similar challenges effectively in the future.
