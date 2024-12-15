---
title: "How to have a static trait from_str method return a trait?"
date: "2024-12-15"
id: "how-to-have-a-static-trait-fromstr-method-return-a-trait"
---

alright, so you're hitting that classic rust wall, where you want a static method on a trait, specifically `from_str`, to return another trait, and yeah, it's a bit of a head-scratcher at first. i’ve spent more hours than i care to count banging my head against similar type system puzzles. it's like rust is saying "i know what you want, but also, i don't like it".

first things first, the core issue is that traits in rust are not concrete types, they're like blueprints. you can't directly instantiate a trait, because the compiler doesn't know the actual size or behavior associated with that blueprint until it knows the concrete type that implements that trait. when you want to return something from a function or method, rust needs a concrete type at compile time, not a trait.

let’s go over some things. if you’re trying to do this:

```rust
trait MyTrait {
    fn do_something(&self);
}

trait Stringy {
    fn from_str(s: &str) -> Self;
}

impl Stringy for dyn MyTrait {
    fn from_str(s: &str) -> Self {
        // this is wrong, doesn't compile
        todo!()
    }
}
```

this won't work, because as i stated `dyn MyTrait` is not a type; it's like saying "anything that implements `MyTrait`" or a pointer to something that implements it. the compiler needs something with an actual, known size. think about it like this: if i give you a box labeled "something that can hold water," you can't just pour water into that label. you need a real container, like a cup or a bucket, before you can do that, and those containers (concrete types) implement a trait like `can_hold_water`.

i remember when i first stumbled upon this. i was building a parser for a custom configuration file format. each configuration section could be handled by a different type, all sharing a common `ConfigSection` trait. i naively tried to have a `from_str` that could just spit out a `dyn ConfigSection` based on the section name, and boy was that a learning experience. my code looked very similar to the example above and the compiler was not happy. this took me several hours trying to work around the borrow checker until i realized i was trying to do something that was not the intended way to use rust.

the solution usually involves a few techniques. first, we can't return a bare `dyn MyTrait` directly from a static method; instead we have to return some concrete type that implements the desired trait. a common way is to use `Box<dyn MyTrait>`. `Box` is a heap allocated pointer. because `Box` has a size (the size of a pointer), the compiler can work with it. let’s show an example:

```rust
trait MyTrait {
    fn do_something(&self);
}

struct ConcreteTypeA;
impl MyTrait for ConcreteTypeA {
    fn do_something(&self) {
        println!("i'm concrete type a");
    }
}

struct ConcreteTypeB;
impl MyTrait for ConcreteTypeB {
    fn do_something(&self) {
        println!("i'm concrete type b");
    }
}

trait Stringy {
    fn from_str(s: &str) -> Box<dyn MyTrait>;
}

struct MyStruct;

impl Stringy for MyStruct {
    fn from_str(s: &str) -> Box<dyn MyTrait> {
        match s {
            "a" => Box::new(ConcreteTypeA),
            "b" => Box::new(ConcreteTypeB),
            _ => panic!("unknown type"),
        }
    }
}


fn main() {
    let a = MyStruct::from_str("a");
    a.do_something(); // outputs: "i'm concrete type a"
    let b = MyStruct::from_str("b");
    b.do_something(); // outputs: "i'm concrete type b"
}
```

in this example, `from_str` now returns a `Box<dyn MyTrait>`. the function is now responsible for actually creating the concrete types and then boxing them. this is a very common pattern. the `match` statement in `from_str` is how we select what the concrete type will be.

another common approach when the set of concrete types is known at compile time is to use enums. enums in rust are a sum type that can contain different variants, each variant being a different type, therefore the compiler has all the information needed.

here is an example:

```rust
trait MyTrait {
    fn do_something(&self);
}

struct ConcreteTypeA;
impl MyTrait for ConcreteTypeA {
    fn do_something(&self) {
        println!("i'm concrete type a");
    }
}

struct ConcreteTypeB;
impl MyTrait for ConcreteTypeB {
    fn do_something(&self) {
        println!("i'm concrete type b");
    }
}

enum MyTraitEnum {
    A(ConcreteTypeA),
    B(ConcreteTypeB),
}

impl MyTrait for MyTraitEnum {
    fn do_something(&self) {
        match self {
            MyTraitEnum::A(a) => a.do_something(),
            MyTraitEnum::B(b) => b.do_something(),
        }
    }
}

trait Stringy {
    fn from_str(s: &str) -> MyTraitEnum;
}

struct MyStruct;

impl Stringy for MyStruct {
    fn from_str(s: &str) -> MyTraitEnum {
        match s {
            "a" => MyTraitEnum::A(ConcreteTypeA),
            "b" => MyTraitEnum::B(ConcreteTypeB),
             _ => panic!("unknown type"),
        }
    }
}


fn main() {
    let a = MyStruct::from_str("a");
    a.do_something(); // outputs: "i'm concrete type a"
    let b = MyStruct::from_str("b");
    b.do_something(); // outputs: "i'm concrete type b"
}
```

in this example, we define an enum, `MyTraitEnum` that holds our concrete types and implements the `MyTrait` for them. the `from_str` function now returns that enum. i often found that this approach makes it easier to handle all the cases, but it can get verbose with more variants, just something to keep in mind. i once spent a few hours debugging a large enum and it was not great so a lot of small variants might not be the best approach.

and one last pattern that i found can be very useful (but be cautious, as it can add complexity) is generics. generics let you make a function or struct work with different types, but still be type safe. when i was working on a project involving a lot of different database interactions i often had to handle different types that implemented the same trait and i found this method to work very well:

```rust
trait MyTrait {
    fn do_something(&self);
}

struct ConcreteTypeA;
impl MyTrait for ConcreteTypeA {
    fn do_something(&self) {
        println!("i'm concrete type a");
    }
}

struct ConcreteTypeB;
impl MyTrait for ConcreteTypeB {
    fn do_something(&self) {
        println!("i'm concrete type b");
    }
}


trait Stringy<T: MyTrait> {
    fn from_str(s: &str) -> T;
}

struct MyStructA;

impl Stringy<ConcreteTypeA> for MyStructA {
    fn from_str(s: &str) -> ConcreteTypeA {
        match s {
            "a" => ConcreteTypeA,
            _ => panic!("unknown type"),
        }
    }
}

struct MyStructB;
impl Stringy<ConcreteTypeB> for MyStructB {
   fn from_str(s: &str) -> ConcreteTypeB {
       match s {
           "b" => ConcreteTypeB,
           _ => panic!("unknown type"),
       }
   }
}

fn main() {
    let a = MyStructA::from_str("a");
    a.do_something(); // outputs: "i'm concrete type a"
    let b = MyStructB::from_str("b");
    b.do_something(); // outputs: "i'm concrete type b"
}
```

in this snippet `Stringy` is a generic trait which needs to be implemented with a specific concrete type that implements `MyTrait`, this will give the compiler all the info that needs to work. the advantage of this method is that you have a stronger type system and more safety since you are not boxing or using enums and in many cases it will have better performance, because you are not allocating on the heap.

as for resources. you should look into the "the rust programming language" book, specifically the chapters on traits, generics and enums. this is always a good place to start and a very well written book (no pun intended). it also has great explanation and reasoning behind all these concepts. "programming rust" by jim blandy, jason orendorff, leonora f.s. tindall is another great book that will go deeper on rust. if you're looking for something slightly more academic and formal look into "types and programming languages" by benjamin c. pierce, this is not a rust specific book and covers theoretical computer science but the knowledge can be translated to rust problems.

that's pretty much all there is to it. this is a common problem with rust trait's static methods returning traits and there isn’t a way to have them return a trait directly but hopefully with one of the methods outlined here you can accomplish what you need. it's all about understanding rust’s type system and how traits work. it seems complicated now but eventually it will just click and the borrow checker will be your friend (or at least not your worst enemy). and remember to not overthink your solutions sometimes it is best to go step by step and go with the simpler solution first. sometimes it helps to take a break and come back later with a fresh mind, it's like trying to debug code after a long day, your eyes start crossing and everything looks confusing. good luck out there!
