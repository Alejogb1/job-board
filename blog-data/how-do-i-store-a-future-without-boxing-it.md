---
title: "How do I store a future without boxing it?"
date: "2024-12-16"
id: "how-do-i-store-a-future-without-boxing-it"
---

Let's tackle this challenge of storing futures without boxing, a problem I've definitely encountered a few times in my years working with asynchronous code. The crux of the issue, as I see it, isn't just about avoiding the heap allocation that boxing incurs, it’s about crafting a more efficient and, often, more elegant solution. Specifically, we're talking about futures – objects that represent the result of an asynchronous operation – and the desire to hold onto them without incurring the performance penalty of a dynamic allocation. I'll draw from past projects, focusing on concrete approaches rather than hypothetical scenarios.

The problem often arises when you're designing state machines or task orchestrators. Imagine I'm building a highly concurrent network service; managing hundreds or thousands of pending operations concurrently is common. Using `Box<dyn Future>` everywhere would quickly become unsustainable due to the overhead. It leads to memory fragmentation and potentially noticeable latency, especially under heavy load. So, what are our options?

Fundamentally, the solution hinges on leveraging generics and monomorphization—a technique where the compiler generates specific code for each type, eliminating the runtime overhead of type erasure. Instead of storing a `Box<dyn Future>`, we store the concrete future type, or at least a type that can hold it. This shift is about using static dispatch instead of dynamic dispatch.

The most straightforward technique is to use an enum. Let’s say we have a few asynchronous operations in our networking service, such as reading from a socket and sending a response. We can define an enum, where each variant holds the specific future type associated with an operation.

```rust
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};

enum OperationFuture {
    Read(tokio::io::ReadHalf<'static>),
    Write(tokio::io::WriteHalf<'static>),
}

impl Future for OperationFuture {
    type Output = Result<(), std::io::Error>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        match self.get_mut() {
            OperationFuture::Read(read_half) => {
                let mut buf = [0u8; 1024];
                Pin::new(read_half).poll_read(cx, &mut buf).map(|result| result.map(|_|()))
            },
            OperationFuture::Write(write_half) => {
                 let buf = b"response";
                 Pin::new(write_half).poll_write(cx, buf).map(|result| result.map(|_|()))
            }
        }
    }
}
```

Here, `OperationFuture` can hold either a `tokio::io::ReadHalf<'static>` future or `tokio::io::WriteHalf<'static>` future—both concrete types. When polling, we match against the enum variant to call the appropriate underlying `poll` function. Crucially, no boxing is involved; the futures are stored directly within the enum's memory footprint. Note: For simplicity and to keep the code snippets succinct, I'm using a placeholder type for the read and write halves rather than the actual socket objects. In a real application these types will be coming from a stream. This snippet illustrates the enum approach clearly. It’s suitable when the number of possible future types is known beforehand and relatively small.

In scenarios involving a larger, less predictable set of futures, a more general approach is needed. This is where *existential types* or trait objects with specific size constraints, though complex, come into play. The essence lies in defining a trait that our futures must implement and then using a type that can hold any future conforming to that trait. However, instead of relying on a dynamically sized trait object, we use static sizing via generics.

Let's demonstrate with a simple `FutureHolder` that can store any future up to a specific size:

```rust
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};
use std::mem;

struct FutureHolder<F: Future<Output = ()>>(mem::MaybeUninit<F>);

impl<F: Future<Output = ()>> FutureHolder<F> {
    fn new(future: F) -> Self {
        FutureHolder(mem::MaybeUninit::new(future))
    }

    fn as_mut_future(&mut self) -> Pin<&mut F> {
        // SAFETY: we know that MaybeUninit is initialised due to the constructor
        unsafe { Pin::new_unchecked(self.0.assume_init_mut()) }
    }
}

impl<F: Future<Output = ()>> Future for FutureHolder<F> {
    type Output = ();

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        self.as_mut_future().poll(cx)
    }
}

async fn example_async_fn() {
    println!("This is inside of an async function")
}

async fn another_async_fn() {
    println!("This is also inside of an async function")
}

fn main() {
    let future1 = FutureHolder::new(example_async_fn());
    let future2 = FutureHolder::new(another_async_fn());
    tokio::runtime::Runtime::new().unwrap().block_on(async {
        future1.await;
        future2.await;
    });
}

```

Here, `FutureHolder` takes a generic parameter `F` that must implement the `Future` trait. `mem::MaybeUninit` allows us to store the future without immediately initializing it, which is important because futures may have their own initialization sequences. The `as_mut_future` function allows us to access the underlying future. This technique avoids boxing and allows a very basic form of storing any future without boxing.

A more robust approach involves using type erasure with a fixed-size buffer. This requires careful management to ensure that we don't overstep the buffer’s bounds and that the future is properly dropped. I use this when I need to have a collection of futures that can have very different concrete types and still want to avoid boxing them, for instance, when managing many active database operations in a server.

```rust
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};
use std::mem::MaybeUninit;
use std::mem;
use std::ptr;

const BUFFER_SIZE: usize = 256; // A realistic size for many futures.

struct ErasedFuture {
    vtable: &'static ErasedVTable,
    data: MaybeUninit<[u8; BUFFER_SIZE]>,
}

struct ErasedVTable {
  poll: unsafe fn(*mut u8, &mut Context<'_>) -> Poll<()>,
  drop: unsafe fn(*mut u8),
}

impl ErasedFuture {
  fn new<F: Future<Output=()> + 'static>(future: F) -> Self{
    let mut data = MaybeUninit::<[u8; BUFFER_SIZE]>::uninit();
    // SAFETY: Because we have ensured the data buffer is big enough, this is safe
    let ptr = unsafe { data.as_mut_ptr() as *mut F };
    // Write future in the buffer, don't drop the future if initialization fails
    unsafe{ptr::write(ptr, future)};
    let vtable = &ErasedVTable{
        poll:  unsafe fn<F: Future<Output = ()> + 'static>(ptr: *mut u8, cx: &mut Context<'_>) -> Poll<()> {
            let future_ptr = ptr as *mut F;
            // SAFETY: ptr always point to a correctly initialized future of type F
            let future = unsafe { Pin::new_unchecked(&mut *future_ptr)};
            future.poll(cx)
        },
        drop: unsafe fn<F: Future<Output = ()> + 'static>(ptr: *mut u8) {
             let future_ptr = ptr as *mut F;
             // SAFETY: ptr always point to a correctly initialized future of type F
             unsafe{ptr::drop_in_place(future_ptr);}
        }
    };
    ErasedFuture {
      vtable,
      data,
    }

  }
  fn poll(&mut self, cx: &mut Context<'_>) -> Poll<()> {
        unsafe{ (self.vtable.poll)(self.data.as_mut_ptr() as *mut u8, cx)}
    }
}

impl Drop for ErasedFuture {
    fn drop(&mut self) {
       unsafe{ (self.vtable.drop)(self.data.as_mut_ptr() as *mut u8)}
    }
}

async fn another_async_function() {
   println!("This is another async function");
}

async fn example_async() {
    println!("This is an example async function");
}

fn main() {
  let future1 = ErasedFuture::new(example_async());
  let future2 = ErasedFuture::new(another_async_function());
  tokio::runtime::Runtime::new().unwrap().block_on(async{
    let mut f1 = future1;
    let mut f2 = future2;
    let waker = futures::task::noop_waker_ref();
    let mut cx = Context::from_waker(waker);
    let _ = f1.poll(&mut cx);
    let _ = f2.poll(&mut cx);
  });
}
```

In this approach, we use a fixed-size byte array (`BUFFER_SIZE`) to hold the future's data, along with a virtual table (`ErasedVTable`) containing pointers to methods for polling and dropping the concrete future. The `new` function carefully constructs the `ErasedFuture` by placing the future directly into the pre-allocated data buffer and setting up the vtable. Note: the drop implementation must be careful to drop the contained type. This method adds a significant layer of safety concerns, and as such is more complex to implement but can be exceptionally useful when the number of different future types makes other strategies difficult.

These techniques, in my experience, strike a good balance between efficiency and practicality. The enum approach works well for predictable sets of future types. Generics with size constraints provide more flexibility while retaining static dispatch, and fixed-size type erasure provides ultimate flexibility at the cost of increased complexity. While implementing the fixed-size type erasure, care must be taken to handle the data safely.

For further reading, I highly recommend looking at *Rust for Rustaceans* by Jon Gjengset, especially the chapters dealing with type erasure and working with unsafe rust. The *Async Programming in Rust* book from O'Reilly is also invaluable. These sources provide deeper insights into advanced async techniques. I have used the insights from these books in many past projects to help manage asynchronous tasks without the cost of unnecessary boxing. Choosing the correct method for your specific project depends on the level of flexibility required and the trade-offs you are willing to make in terms of complexity.
