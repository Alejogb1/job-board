---
title: "How can I create a new object of any type passed to a Rust function?"
date: "2024-12-23"
id: "how-can-i-create-a-new-object-of-any-type-passed-to-a-rust-function"
---

Alright, let's tackle this. It's a question that surfaces quite frequently, and for good reason, because Rust's ownership system and static typing make it a bit more nuanced than, say, dynamic languages. I've encountered similar challenges in various projects, particularly when dealing with serialization and generic data processing. I recall one specific instance involving a plugin system where I needed a way to instantiate various types of plugin components based on configuration data at runtime. This wasn't something that Rust would let me do implicitly, of course.

The core issue is that Rust requires concrete types at compile time. We can't directly tell a function to "create an object of whatever type comes my way," because the compiler must know the size and layout of the type to allocate memory and perform necessary operations. However, all is not lost. There are several ways we can achieve the desired outcome with various trade-offs.

The most straightforward method, and often the most appropriate for many scenarios, involves using traits. Traits allow us to define a common interface that different types can implement. We can then leverage this trait to create new instances. We’ll start with a simple, effective pattern and then progress to other techniques if the first one proves insufficient.

Let's define a trait called `Creatable` that has a single method for creating a new instance:

```rust
trait Creatable {
    fn create() -> Self;
}
```

Now any type that implements this trait can be instantiated through the `create` method. For example, let’s implement `Creatable` for a simple struct:

```rust
struct MyStruct {
    value: i32
}

impl Creatable for MyStruct {
    fn create() -> Self {
        MyStruct { value: 0 } // Or any default instantiation needed
    }
}
```

Here's how you’d use it:

```rust
fn create_and_use<T: Creatable>() -> T {
   let new_object = T::create();
   // ... further use of the new object
    new_object
}

fn main() {
    let my_struct: MyStruct = create_and_use();
    println!("Created value: {}", my_struct.value); //outputs "Created value: 0"
}
```

This approach works well if you know, at compile time, the types you will need to create, and those types can implement the `Creatable` trait. However, this pattern relies on generic functions and requires explicit type annotations for the function `create_and_use()`, and that limits its flexibility slightly. The trait `Creatable` must be implemented for each type.

A more dynamic but also more verbose approach is to use trait objects and dynamic dispatch. We can redefine our trait to include the `Box<dyn Creatable>` return type in our create method. This allows us to return any type that implements the `Creatable` trait while also working with a trait object instead of having to specify a generic type:

```rust
trait Creatable {
  fn create_boxed() -> Box<dyn Creatable>;
}

struct ConcreteType {
    data: String
}
impl Creatable for ConcreteType {
   fn create_boxed() -> Box<dyn Creatable> {
       Box::new(ConcreteType{data: String::from("Initial Data")})
   }
}

struct AnotherConcreteType {
    number: i32,
}

impl Creatable for AnotherConcreteType{
  fn create_boxed() -> Box<dyn Creatable> {
      Box::new(AnotherConcreteType{ number: 42 })
  }
}
```

And now, the factory usage.

```rust
fn use_boxed_creatable(create_func: fn() -> Box<dyn Creatable>) -> Box<dyn Creatable> {
    let new_object = create_func();
    //... use new_object as a trait object
    new_object
}

fn main() {
   let boxed_struct = use_boxed_creatable(ConcreteType::create_boxed);
   let boxed_another = use_boxed_creatable(AnotherConcreteType::create_boxed);
   println!("Boxed struct created with data: {}", match boxed_struct.as_any().downcast_ref::<ConcreteType>(){
    Some(concrete) => &concrete.data,
    None => "Type Mismatch"
   });
   println!("Boxed another struct created with data: {}", match boxed_another.as_any().downcast_ref::<AnotherConcreteType>(){
    Some(concrete) => &concrete.number.to_string(),
    None => "Type Mismatch"
   });
}

```
Notice how the usage of `.as_any().downcast_ref::<>()` lets us actually use the inner types.

This approach uses `Box<dyn Creatable>` to erase the concrete type at the function level. While more flexible, it does incur a performance overhead due to dynamic dispatch, and you need to use `as_any()` and `downcast_ref` if you wish to access the fields of the actual types, adding verbosity. However, this does allows us to create a new object without knowing at compile time what type it will be.

Finally, let’s look at another option using enums. This is a pattern I’ve found useful in situations where the set of possible types is well-defined and fixed at compile time. Instead of a generic function, we define an enum representing all the types we could be creating. We implement the `Creatable` trait for each enum variant, and then we can use pattern matching to create the required concrete type based on a variant we pass to a function.

```rust
enum ObjectType {
    First,
    Second
}

struct FirstType {
    value_a: i32
}
struct SecondType {
  value_b: String
}

impl Creatable for ObjectType {
  fn create() -> Self {
      ObjectType::First //Default
  }
}

impl ObjectType {
  fn create_object(&self) -> Box<dyn std::any::Any> {
      match self {
          ObjectType::First => Box::new(FirstType{ value_a: 10}),
          ObjectType::Second => Box::new(SecondType{ value_b: "hello".to_string() }),
      }
  }
}
```

And its usage:
```rust
fn create_and_process(object_type: ObjectType) {
    let new_object = object_type.create_object();
    //...further use based on matching using downcast_ref
    match new_object.downcast_ref::<FirstType>() {
      Some(first_type) => println!("Created First with data: {}", first_type.value_a),
      None => match new_object.downcast_ref::<SecondType>() {
        Some(second_type) => println!("Created Second with data: {}", second_type.value_b),
        None => println!("Type mismatch"),
      }
    }
}


fn main() {
  create_and_process(ObjectType::First);
  create_and_process(ObjectType::Second);
}

```
This strategy neatly avoids the dynamic dispatch overhead, while still providing flexibility in that, we can now create any of the enum's variants using a `create_object()` function and pattern matching, without generic function arguments. We do still need to use the `downcast_ref()` method which can be considered verbose.

For further study on these topics, I'd highly recommend looking into the *Rust Programming Language* book by Steve Klabnik and Carol Nichols. Also, exploring the Rust documentation on traits and generics is key. For more in-depth understanding of dynamic dispatch, I would recommend looking into academic work on type systems and object-oriented programming with dynamic dispatch.

In summary, the best approach depends greatly on the specifics of your project and the kinds of types you expect to handle. The trait approach with generics is excellent if you know your types at compile time, the trait object solution is good if you need runtime flexibility with some performance trade-offs, and the enum approach if you have a set number of object types you wish to instantiate. Each approach is a viable solution, each with its own particularities. It's about choosing the right tool for the job, weighing flexibility against performance.
