---
title: "How do I implement the Sized trait?"
date: "2024-12-16"
id: "how-do-i-implement-the-sized-trait"
---

Alright, let’s tackle the *Sized* trait implementation. It’s one of those core concepts in rust that often seems straightforward until you actually need to get hands-on with it, and then it throws some interesting curveballs. I recall my first encounter with this back when I was optimizing a data structure library; unexpected compile errors kept popping up regarding type sizes. It took a solid afternoon of debugging before it clicked. So, let’s break this down step-by-step, avoiding overly theoretical jargon.

The `Sized` trait in rust fundamentally denotes whether the compiler knows the size of a type at compile time. Most types we use regularly, like integers, floats, structures, and enums, implicitly implement `Sized`. The compiler needs to know this size to allocate memory correctly and perform operations. However, types with dynamic sizes, like trait objects and slices, are generally *not* `Sized`. Instead, they’re often used behind pointers or references.

Now, you might ask, when do you specifically need to implement `Sized` manually? The answer is almost never, directly. It's more likely you'll encounter situations where the *absence* of `Sized` becomes an issue, typically when you’re working with generics or traits. The compiler is quite clever; it automatically infers `Sized` bounds for many situations. It's when you're working with types that *might not* be `Sized` that these complexities surface, making it seem like you're implementing `Sized` yourself.

Let's first look at scenarios where `Sized` is implicitly enforced. Most commonly, a function using a generic type assumes that type is `Sized`:

```rust
fn print_value<T>(value: T) {
    println!("Value: {:?}", value);
}

fn main() {
    print_value(10);  // '10' is a Sized type
    print_value("hello"); // "&str" is also a Sized type, though behind reference
}
```

In this example, even though `T` is a generic type parameter, the compiler implicitly adds a `T: Sized` bound to the `print_value` function. It knows it needs to be able to reason about the size of whatever `T` is at compile time. The string literal `"hello"` implicitly converts to a `&str`, a reference which has a known size, and thus, conforms to the `Sized` bound.

However, here’s the crux: If you are dealing with trait objects or unsized types within a generic context, you must explicitly state if they are *not* Sized. This is achieved via `?Sized` bound. Let's illustrate with an example using trait objects:

```rust
trait Printable {
    fn print(&self);
}

struct Number(i32);

impl Printable for Number {
    fn print(&self) {
        println!("Number: {}", self.0);
    }
}

struct Text(String);

impl Printable for Text {
    fn print(&self) {
        println!("Text: {}", self.0);
    }
}


fn print_something<T: Printable>(value: &T) {
    value.print();
}

fn print_trait_object(value: &dyn Printable) {
    value.print();
}

fn main() {
    let num = Number(42);
    let text = Text("Example".to_string());

    print_something(&num);
    print_something(&text);
    print_trait_object(&num);
    print_trait_object(&text);

    // Example of &dyn trait where Sized is not assumed by default
}

```

Notice that `print_something` has a generic parameter, but we are passing references to concrete types. The compiler implicitly includes `T: Sized`, which is why we have no compilation errors. In contrast, the function `print_trait_object` explicitly takes a trait object, which is unsized. It's the `dyn` keyword that signifies that `Printable` has been converted into a trait object behind a reference, making it `!Sized`. We cannot use the type `dyn Printable` directly by value in most contexts as the size of the concrete type implementing the trait is unknown at compile time. We *can* use `&dyn Printable` or `Box<dyn Printable>` because pointers are `Sized`, despite the pointed-to type not being.

Finally, let's examine the impact of `?Sized` on a generic function. Let's adapt the `print_something` function to accept potentially unsized `T` with the `?Sized` trait bound:

```rust
trait Drawable {
    fn draw(&self);
}

struct Circle { radius: f64 }

impl Drawable for Circle {
    fn draw(&self) { println!("Drawing a circle with radius {}", self.radius); }
}

struct Square { side: f64 }

impl Drawable for Square {
    fn draw(&self) { println!("Drawing a square with side {}", self.side); }
}

fn draw_shape<T: Drawable + ?Sized>(shape: &T) {
  shape.draw();
}

fn main() {
    let circle = Circle { radius: 5.0 };
    let square = Square { side: 10.0 };

    draw_shape(&circle);
    draw_shape(&square);
    draw_shape(&circle as &dyn Drawable); //Explicit cast to trait object
    draw_shape(&square as &dyn Drawable); //Explicit cast to trait object

}
```

Here, the `draw_shape` function can accept both concrete types that implement `Drawable` and, importantly, trait objects. The `?Sized` annotation makes the compiler aware that `T` might not be `Sized`. This allows the function to accept references to trait objects like `&dyn Drawable`, which have unknown sizes at compile time. The compiler also will accept `&Circle` or `&Square` because they both implement `Drawable`, are implicitly `Sized`, and thus conform to `Drawable + ?Sized`. The `?Sized` trait bound is a special marker that essentially means, “I do not require this generic type to be `Sized`.”

In essence, `Sized` isn't something you implement manually in most scenarios; rather, it’s an implied requirement that affects how you handle generic types and trait objects. `?Sized` is the tool for removing this implicit bound, enabling functions to handle types where size is not known at compile time. Remember, when using generic types, the compiler assumes they are `Sized` unless you explicitly indicate otherwise with `?Sized`, particularly when dealing with trait objects or custom dynamically-sized types. Understanding how the compiler views size is essential to navigating these intricacies. For in-depth study, I recommend diving into the “Rustonomicon,” specifically its chapters on unsized types, and “Programming Rust” by Jim Blandy, Jason Orendorff, and Leonora F.S. Tindall, which has a good section on traits, especially trait objects and their limitations.
