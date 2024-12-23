---
title: "Why can't a `Box<dyn Trait>` be passed to a function expecting a `&mut Trait`?"
date: "2024-12-23"
id: "why-cant-a-boxdyn-trait-be-passed-to-a-function-expecting-a-mut-trait"
---

, let’s unpack this. I've certainly stumbled upon this particular roadblock more than once in my career, usually mid-refactor when trying to apply some good object-oriented principles to Rust code. The core issue here boils down to ownership, borrowing, and the specific semantics of `Box<dyn Trait>` versus `&mut Trait`. It’s not about arbitrary limitations; it's about maintaining Rust's core tenets around memory safety. Let's get into it.

The crux of the problem is that `Box<dyn Trait>` represents an *owned* dynamically dispatched trait object, allocated on the heap. This `Box` owns the data. The `&mut Trait`, conversely, is a *mutable borrow* of a trait object; it doesn’t own the underlying data. These are fundamentally different types with distinct responsibilities. Trying to pass one to a function expecting the other is akin to trying to fit a square peg into a round hole.

Specifically, a function accepting a `&mut Trait` expects a mutable reference to an existing object that already lives elsewhere. It anticipates having temporary, exclusive access to modify that object. Crucially, the function *doesn't* expect to take ownership of that object. The lifetime of this mutable borrow is usually tied to the scope where the borrowed object exists. Now consider a `Box<dyn Trait>`. It holds a pointer to a heap-allocated object, which it is solely responsible for managing, and it will deallocate this object when the box goes out of scope. Passing the `Box`'s contents as a `&mut Trait` would require the function to operate under two contradictory assumptions: the function would believe it has a borrowed mutable reference and not own the memory while, in reality, it would be accessing an owned value. This can lead to double frees and undefined behavior if the function does anything that would invalidate that memory.

Let’s look at this from a different angle. `Box<dyn Trait>` itself is a concrete type; it is a pointer to a data structure containing both the data of the specific underlying concrete type that implements the `Trait` *and* a vtable. The vtable contains the pointers to the trait methods. However, the `&mut Trait` is a *trait object*, meaning it is a fat pointer containing not just the address of the data but also a reference to the vtable, at the time that the borrow is made. Importantly, the data pointed to by `&mut Trait` has to already live somewhere on the stack or on the heap *outside* of the `Box`. The lifetime of this `&mut Trait` is usually shorter than the lifetime of the data. Passing a `Box` to a `&mut` reference would mean that the lifetime of the `&mut` reference would be tied to that of the `Box`, which is the heap allocated pointer. This also doesn't align with the rust ownership model.

To illustrate this, let’s consider a practical example. Imagine we have a trait `Drawable` and a concrete struct `Square` that implements it:

```rust
trait Drawable {
    fn draw(&mut self);
}

struct Square {
    size: u32,
}

impl Drawable for Square {
    fn draw(&mut self) {
        println!("Drawing a square of size: {}", self.size);
    }
}

fn modify_drawable(drawable: &mut dyn Drawable) {
    drawable.draw();
}
```

Now, let's examine the problematic scenario:

```rust
fn main() {
    let square = Square { size: 5 };
    let boxed_drawable: Box<dyn Drawable> = Box::new(square);

   // This won't work and will raise a compilation error:
   // modify_drawable(&mut *boxed_drawable);
   // It is not possible to convert a Box<dyn Drawable> to a &mut dyn Drawable directly

   // We need to create the &mut dyn Drawable from something that already exists
   let mut square2 = Square { size: 10 };
   modify_drawable(&mut square2); // This works

   // Or equivalently:

   let mut boxed_drawable2: Box<dyn Drawable> = Box::new(Square{size: 12});
   // The line below also won't work.
   // modify_drawable(&mut *boxed_drawable2);

    // Instead, if you really need to work with a `Box<dyn Trait>`, you can 'move' its contained value out,
    // which transfers ownership and gives you a `&mut Trait`
    let mut boxed_drawable3: Box<dyn Drawable> = Box::new(Square{size: 15});
    let mut owned_drawable = *boxed_drawable3;
    modify_drawable(&mut owned_drawable); // This works, but `boxed_drawable3` is no longer valid.

    // One last example: we can create the `&mut` reference at the beginning
    let mut square4 = Square{size: 20};
    let mutable_drawable: &mut dyn Drawable = &mut square4;
    modify_drawable(mutable_drawable); // Also works

    // Notice that in the last example, `mutable_drawable` is now a mutable borrow of `square4`.
}

```

The commented-out line `modify_drawable(&mut *boxed_drawable);` is the exact issue we're discussing. The compiler prevents us from making a `&mut dyn Drawable` reference directly out of the `Box<dyn Drawable>`.

The third code example shows a potential (but often discouraged) way around the issue. By taking the value out of the `Box` using `*` operator and creating a new mutable reference to this value, it is possible to call the function. However, this takes ownership away from the box, making it invalid after this line. It is usually preferable to use the fourth example, where the `&mut dyn Drawable` reference is created on the stack from the beginning. The last example shows an equivalent approach that emphasizes the fact that `&mut Trait` is always a borrow of existing data.

So, how should one approach this practically? It depends on the context. If you intend to *modify* the object through a trait, a mutable reference is often the appropriate abstraction. If you need to pass the responsibility for ownership of the data, a `Box` is necessary. I have found that sometimes the best option is to design functions to accept generics instead of trait objects. A generic function can accept any type that implements a given trait and will be statically dispatched, providing performance and flexibility. Here is an example:

```rust
fn modify_drawable_generic<T: Drawable>(drawable: &mut T) {
  drawable.draw();
}

fn main() {
    let mut square = Square { size: 5 };
    modify_drawable_generic(&mut square);

    let mut square2 = Square { size: 10 };
    let boxed_drawable: Box<Square> = Box::new(square2);
    modify_drawable_generic(&mut *boxed_drawable);
}

```

Here, we can accept any type that implements the `Drawable` trait, including a `Box<Square>` after dereferencing it. Notice that the performance will be better in this case as there is no dynamic dispatch involved.

In summary, the inability to directly cast a `Box<dyn Trait>` to a `&mut Trait` isn’t an arbitrary limitation. It’s a fundamental aspect of Rust’s ownership and borrowing system. The two represent different ways of handling memory and modifying the objects they point to. Misunderstanding this difference leads to memory safety issues that Rust aims to prevent.

To dive deeper, I'd recommend reading *The Rust Programming Language* by Steve Klabnik and Carol Nichols, particularly the chapters on ownership, borrowing, and trait objects. Also, *Programming Rust* by Jim Blandy, Jason Orendorff, and Leonora F.S. Tindall is an invaluable resource for understanding the practicalities of these concepts. Additionally, reviewing the RFCs (Request for Comments) on trait objects and dynamic dispatch can provide a more granular understanding of the motivation and design behind these features. These resources will provide a thorough understanding of the nuances, helping to prevent future head-scratching and memory-related headaches.
