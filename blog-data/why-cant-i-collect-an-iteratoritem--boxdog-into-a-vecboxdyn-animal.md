---
title: "Why can't I collect an `Iterator<Item = Box<Dog>>` into a `Vec<Box<dyn Animal>>`?"
date: "2024-12-16"
id: "why-cant-i-collect-an-iteratoritem--boxdog-into-a-vecboxdyn-animal"
---

,  I've seen this sort of frustration many times, especially when developers are coming to grips with Rust's ownership system and trait objects. It’s not uncommon to expect an iterator of `Box<Dog>` to neatly transform into a `Vec<Box<dyn Animal>>`, but the compiler throws a wrench into the works, and for good reason.

The core issue lies in how Rust handles type erasure and memory layout with trait objects, specifically when boxed. To understand this, we first have to appreciate that `Dog` is a concrete type, with a known size at compile time, and `Box<Dog>` is just a pointer to memory holding that `Dog`. However, `dyn Animal` is a *trait object*, which means it’s essentially a pointer, not just to the data, but also to a virtual table (vtable) of the concrete type's methods that implement the `Animal` trait. This vtable is what enables runtime polymorphism.

When you try to directly collect an `Iterator<Item = Box<Dog>>` into a `Vec<Box<dyn Animal>>`, the compiler encounters a type mismatch that it cannot easily resolve. You're essentially trying to store a pointer to a `Dog` (which also implies the type information) into a slot that expects a pointer to a data structure that includes a vtable, which is not inherent to `Box<Dog>`. The compiler isn't going to implicitly create this vtable pointer, and forcing it to do so would be unsound and require additional runtime machinery. It's a type mismatch, and more specifically, a *semantic* mismatch.

Let’s put this in the context of a past project. I worked on a virtual pet simulation a while back, and I wanted to have a collection of different animals – dogs, cats, hamsters – all stored in a vector. My initial attempt looked something like the scenario you’re describing: I had a separate function that returned an iterator of `Box<Dog>` (or `Box<Cat>`, etc.), and I tried to collect that iterator directly into a `Vec<Box<dyn Animal>>`. Needless to say, that didn't work out, resulting in a classic type error.

The solution, like many things in Rust, involved a bit of explicit casting, or more specifically, coercion of the concrete type to a trait object. The important step is that you need to explicitly create a `Box<dyn Animal>` from the `Box<Dog>`. Here’s the approach I took:

```rust
trait Animal {
    fn make_sound(&self);
}

struct Dog {
    name: String,
}

impl Animal for Dog {
    fn make_sound(&self) {
        println!("Woof!");
    }
}


fn create_dogs() -> impl Iterator<Item = Box<Dog>> {
    vec![Box::new(Dog { name: "Fido".to_string() }), Box::new(Dog { name: "Buddy".to_string() })].into_iter()
}

fn collect_animals() -> Vec<Box<dyn Animal>> {
    create_dogs()
        .map(|dog_box| -> Box<dyn Animal> { dog_box }) // the magic happens here
        .collect()
}


fn main() {
    let animals = collect_animals();
    for animal in animals {
        animal.make_sound();
    }
}
```

Notice the crucial `.map(|dog_box| -> Box<dyn Animal> { dog_box })` part? This is where the type coercion happens. Rust is smart enough to know that `dog_box` is a `Box<Dog>`, which implements the `Animal` trait, and it can therefore perform the necessary conversion into `Box<dyn Animal>`.

The second example highlights why this is necessary and explores more of the underlying memory considerations. Imagine `Dog` and `Cat` both implement `Animal`, and they have different internal sizes.

```rust
trait Animal {
    fn make_sound(&self);
    fn get_size(&self) -> usize;
}

struct Dog {
    name: String,
    breed: String,
}

impl Animal for Dog {
    fn make_sound(&self) { println!("Woof!"); }
    fn get_size(&self) -> usize { std::mem::size_of::<Self>() }
}

struct Cat {
    name: String,
    fur_color: String,
    claw_count: u8,
}

impl Animal for Cat {
    fn make_sound(&self) { println!("Meow!"); }
    fn get_size(&self) -> usize { std::mem::size_of::<Self>() }
}


fn create_animals() -> impl Iterator<Item = Box<dyn Animal>> {
    let dogs = vec![
        Box::new(Dog { name: "Fido".to_string(), breed: "Golden Retriever".to_string() }),
        Box::new(Dog { name: "Rover".to_string(), breed: "Labrador".to_string() })
    ];
    let cats = vec![
        Box::new(Cat { name: "Whiskers".to_string(), fur_color: "Gray".to_string(), claw_count: 18 }),
        Box::new(Cat { name: "Mittens".to_string(), fur_color: "Black".to_string(), claw_count: 20 })
    ];
    dogs.into_iter().map(|x| -> Box<dyn Animal> {x}).chain(cats.into_iter().map(|x| -> Box<dyn Animal> {x}))
}


fn main() {
    let animals: Vec<Box<dyn Animal>> = create_animals().collect();
    for animal in &animals {
        println!("Size: {}", animal.get_size());
        animal.make_sound();
    }
}

```

In this case, both the `Dog` and the `Cat` structs differ in size and data layout, yet they both implement the `Animal` trait. Without the trait object coercion, it would be impossible to place them in the same `Vec`. The vtable, which is part of the `dyn Animal`, facilitates the correct dispatch to the `make_sound` and `get_size` methods of the concrete types at runtime.

Finally, let’s look at a third scenario. Sometimes you might not have the concrete type readily available but rather wrapped in an enum, and you'll still want to box it and make it part of a list of trait objects.

```rust

trait Animal {
    fn make_sound(&self);
}

struct Dog {
    name: String,
}
impl Animal for Dog {
    fn make_sound(&self) {
        println!("Woof!");
    }
}
struct Cat {
    name: String,
}
impl Animal for Cat {
    fn make_sound(&self) {
        println!("Meow!");
    }
}

enum AnimalType {
    Dog(Dog),
    Cat(Cat),
}


fn create_enums() -> impl Iterator<Item = AnimalType> {
  let enums = vec![
    AnimalType::Dog(Dog { name: "Fido".to_string() }),
    AnimalType::Cat(Cat { name: "Whiskers".to_string() }),
    AnimalType::Dog(Dog {name: "Buddy".to_string()})
    ];
  enums.into_iter()
}


fn collect_animals() -> Vec<Box<dyn Animal>> {
    create_enums()
      .map(|animal_type| match animal_type {
        AnimalType::Dog(dog) => Box::new(dog) as Box<dyn Animal>,
        AnimalType::Cat(cat) => Box::new(cat) as Box<dyn Animal>,
    })
        .collect()
}



fn main() {
  let animals = collect_animals();
  for animal in animals {
      animal.make_sound();
  }
}
```

Here, we are explicitly matching on the enum, converting to a boxed concrete type and then coercing that box to `Box<dyn Animal>`. Each case is essential. We're not simply trying to shoehorn the raw `AnimalType` into a trait object.

In conclusion, you can't directly collect `Iterator<Item = Box<Dog>>` to a `Vec<Box<dyn Animal>>` due to Rust’s strict type system and the memory layout differences between concrete types and trait objects. The solution involves explicit coercion or matching and casting to `Box<dyn Animal>`. Understanding this coercion, and how it’s tied to the vtable and runtime polymorphism, is essential for mastering Rust’s powerful type system. For further information on this, I highly recommend checking out "Programming Rust" by Jim Blandy, Jason Orendorff, and Leonora F. S. Tindall. It’s a great resource for understanding Rust’s internals. And for a deep dive into type systems and polymorphism, “Types and Programming Languages” by Benjamin C. Pierce is an excellent theoretical resource. You can also dive into the Rust documentation for more details, specifically on trait objects and dynamic dispatch. These are all excellent resources that have certainly helped me navigate such issues.
