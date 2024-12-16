---
title: "How do I create generic objects by type with a function in Rust?"
date: "2024-12-16"
id: "how-do-i-create-generic-objects-by-type-with-a-function-in-rust"
---

Alright, let's talk about dynamically creating objects in Rust based on their type, which is a surprisingly common requirement when you're building anything beyond basic applications. It's a problem I've faced multiple times, most memorably when I was developing a plugin system for a large data processing pipeline. We needed to load and instantiate various data handler plugins at runtime, and the types of these plugins weren't known at compile time. This situation forced me to explore some of Rust's more powerful features.

The crux of the matter lies in Rust's strong static typing. At compile time, Rust needs to know the exact type of the values it's working with. Generic functions and traits do a great job of abstraction, but they don't directly let us generate new instances of a type based on runtime information alone. Essentially, the question becomes: how do we inform the Rust compiler about the type we intend to instantiate, when that type is only determined at runtime?

The solution, in many cases, involves the use of traits and trait objects, coupled with a factory pattern, which I’ve found to be the most versatile approach. Here’s how it usually pans out:

First, we define a trait that all objects we want to dynamically create must implement. This trait needs at least a method or associated function to instantiate an instance of itself. For simplicity, let's use an `impl` function that can return a boxed trait object:

```rust
trait Creatable {
    fn create() -> Box<dyn Creatable>;
}

struct ConcreteTypeA;

impl Creatable for ConcreteTypeA {
    fn create() -> Box<dyn Creatable> {
        Box::new(ConcreteTypeA)
    }
}

struct ConcreteTypeB;

impl Creatable for ConcreteTypeB {
    fn create() -> Box<dyn Creatable> {
        Box::new(ConcreteTypeB)
    }
}
```

In this example, we have a trait called `Creatable`. Both `ConcreteTypeA` and `ConcreteTypeB` implement this trait, each with their own instantiation logic. This allows us to create instances of `Creatable` as boxed trait objects. This allows us to return boxed trait objects. Notice, I am using `Box<dyn Creatable>`, which is a trait object. This trait object is where the magic happens. The key is that this `dyn` keyword tells the compiler that this box could hold anything that implements the `Creatable` trait.

The next step involves creating a registry where we can store the methods used to generate our objects. This is the “factory” part of the pattern. We'll use a `HashMap` to map a string representing the type to the respective function. This would allow users to specify the type they want to instantiate as a string, providing the dynamic aspect.

```rust
use std::collections::HashMap;

type CreatorFn = fn() -> Box<dyn Creatable>;

struct ObjectFactory {
    creators: HashMap<String, CreatorFn>,
}

impl ObjectFactory {
  fn new() -> Self {
        ObjectFactory {
            creators: HashMap::new(),
        }
    }

    fn register(&mut self, name: &str, creator: CreatorFn) {
        self.creators.insert(name.to_string(), creator);
    }

    fn create(&self, name: &str) -> Option<Box<dyn Creatable>> {
        self.creators.get(name).map(|creator| creator())
    }
}
```

The `ObjectFactory` struct holds a `HashMap` that maps string keys to creator functions. The `register` method adds creator functions to this map, and the `create` method uses that to instantiate objects. Note that the `create` method returns an `Option<Box<dyn Creatable>>` which handles the scenario when the type name is not present in the registry. This is critical for robustness. The creator function in this case is a zero-argument function which returns a `Box<dyn Creatable>`.

Finally, let’s combine this with the initial example:

```rust
use std::collections::HashMap;

trait Creatable {
    fn create() -> Box<dyn Creatable>;
}

struct ConcreteTypeA;

impl Creatable for ConcreteTypeA {
    fn create() -> Box<dyn Creatable> {
        Box::new(ConcreteTypeA)
    }
}

struct ConcreteTypeB;

impl Creatable for ConcreteTypeB {
    fn create() -> Box<dyn Creatable> {
        Box::new(ConcreteTypeB)
    }
}

type CreatorFn = fn() -> Box<dyn Creatable>;

struct ObjectFactory {
    creators: HashMap<String, CreatorFn>,
}

impl ObjectFactory {
    fn new() -> Self {
        ObjectFactory {
            creators: HashMap::new(),
        }
    }

    fn register(&mut self, name: &str, creator: CreatorFn) {
        self.creators.insert(name.to_string(), creator);
    }

    fn create(&self, name: &str) -> Option<Box<dyn Creatable>> {
      self.creators.get(name).map(|creator| creator())
    }
}


fn main() {
    let mut factory = ObjectFactory::new();
    factory.register("TypeA", ConcreteTypeA::create);
    factory.register("TypeB", ConcreteTypeB::create);

    if let Some(obj) = factory.create("TypeA") {
      // We can call methods on the created objects, if they were declared on the Creatable trait.
        println!("Successfully created TypeA object");
        // Type casting can be done at runtime if the concrete type is known here, but that is outside the current scope of this solution.
    }

    if let Some(obj) = factory.create("TypeB"){
        println!("Successfully created TypeB object");
    }

   if let None = factory.create("NotAType"){
      println!("Could not create object with name: NotAType.");
   }
}
```

In `main`, we create our factory, register the types along with their `create` function and attempt to create a couple of `Creatable` objects. The `main` function also illustrates how the `Option` returned from `create` allows the user to handle the case when no type is associated with the string specified.

This pattern allows for dynamic creation, though, it does have some overhead associated with the trait object, and virtual function calls. Depending on the performance characteristics you are dealing with, there are ways to further optimize this pattern, but for most applications, this technique works really well. For deeper understanding of the performance considerations of trait objects and virtual dispatch, I recommend "Programming Rust" by Jim Blandy, Jason Orendorff, and Leonora F.S. Tindall, specifically chapters related to traits and dynamic dispatch. Also, understanding the concept of *monomorphization* as discussed in the Rust documentation will help you better navigate Rust's generic programming and its performance.

In summary, dynamic object creation via type in Rust is handled via trait objects and a form of a factory pattern that register instantiation functions, making it runtime type creation possible. When implemented correctly, it allows Rust to handle runtime type discovery, which is usually a static language feature.
