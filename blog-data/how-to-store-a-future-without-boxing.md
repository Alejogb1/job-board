---
title: "How to store a future without boxing?"
date: "2024-12-16"
id: "how-to-store-a-future-without-boxing"
---

Okay, let's tackle this. I've spent a fair amount of time navigating the complexities of asynchronous operations, and the question of "how to store a future without boxing" is one that resonates deeply. It’s a critical concern, particularly when optimizing performance in systems dealing with concurrency. We often hear about the overhead that boxing introduces, and it's not just theoretical— I've seen it degrade response times in high-throughput services firsthand. So, let’s unpack this, focusing on practical strategies I've applied over the years.

The fundamental challenge arises from the need to store the *result* of an asynchronous operation – a future – somewhere. Traditionally, you might think of placing it in a data structure, or passing it around, and the most straightforward way is to box it—that is, allocating memory on the heap and storing a pointer to that memory. Boxing introduces indirection, which incurs a performance cost. If you’re dealing with many futures, particularly small ones, this overhead can become noticeable. And, depending on your language and runtime environment, that heap allocation and deallocation can add pressure on the garbage collector or memory management system, impacting overall throughput and latency.

Let's talk about avoiding this. The core idea is to work directly with the underlying type of the future, but this is only achievable under certain circumstances, typically when you know the *concrete type* of that future at compile time. The general concept revolves around using techniques that allow for direct storage rather than relying on a type-erased, heap-allocated container such as box.

Firstly, `Pin<T>` and its relatives like `Pin<&mut T>`. This is a fundamental approach in Rust's asynchronous ecosystem. If you are working in a language that has a similar concept, the ideas are transferable even if the syntax is different. The gist is that once a future starts executing, it shouldn't be moved. It can invalidate internal pointers if it gets moved within memory while running. `Pin` acts as a safeguard, preventing it from being moved and thus ensuring stability. To understand this better, the Rust documentation is a great start, specifically related to `std::pin` and `std::future`. The principle translates across languages, though implementation might vary. Consider the following example in Rust that showcases how to implement a future without boxing that executes on a threadpool.

```rust
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};
use std::thread;
use std::sync::{Arc, Mutex};
use futures::task::waker_ref;

// A simple future that computes a value on a thread
struct ThreadPoolFuture<T: Send + 'static>(
    Arc<Mutex<Option<T>>>,
    Option<thread::JoinHandle<()>>
);

impl<T: Send + 'static> ThreadPoolFuture<T> {
    fn new(f: impl FnOnce() -> T + Send + 'static) -> Self {
        let result = Arc::new(Mutex::new(None));
        let result_clone = result.clone();
        let handle = thread::spawn(move || {
            let computed = f();
            *result_clone.lock().unwrap() = Some(computed);
        });

        ThreadPoolFuture(result, Some(handle))
    }
}


impl<T: Send + 'static> Future for ThreadPoolFuture<T> {
    type Output = T;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        if self.1.is_some() {
                if self.1.as_ref().unwrap().is_finished() {
                    let mut result = self.0.lock().unwrap();
                    Poll::Ready(result.take().unwrap())

                } else{
                    cx.waker().wake_by_ref();
                    Poll::Pending
                }

        } else {
          //panic!("Future was already polled")
          let mut result = self.0.lock().unwrap();
          Poll::Ready(result.take().unwrap())

        }

    }
}


fn main() {
    let future = ThreadPoolFuture::new(|| {
        println!("Calculating...");
        std::thread::sleep(std::time::Duration::from_secs(2));
        42
    });


    futures::executor::block_on(future);
    println!("Future finished executing.");

}

```

In this example, the `ThreadPoolFuture` stores all information about its execution directly inside its struct, and doesn't require a `Box<dyn Future>`. We can control the type and therefore storage directly. We must pay careful attention to the lifetime and ownership of the data that the future is referencing to ensure it doesn't get invalidated, which is why the mutex is used here.

Secondly, leveraging generic programming, template meta-programming, or similar techniques specific to your programming language helps greatly. If you have the luxury of knowing the specific type of the future at compile time, you can store that directly in your container. This strategy is about working with specific types rather than generic ones. For example, a templated class or a struct might store a specific kind of future object directly, instead of needing a `Box<dyn Future>`. Consider the following C++ example using templates to avoid boxing, which is functionally similar to the previous example:

```cpp
#include <future>
#include <iostream>
#include <thread>

template <typename T>
class ThreadPoolFuture {
public:
    using OutputType = T;

    ThreadPoolFuture(std::function<T()> f) : future_(std::async(std::launch::async, f)) {}


    T get() {
        return future_.get();
    }


private:
    std::future<T> future_;
};

int main() {
    ThreadPoolFuture<int> future([]() {
        std::cout << "Calculating...\n";
        std::this_thread::sleep_for(std::chrono::seconds(2));
        return 42;
    });

    int result = future.get();
    std::cout << "Future finished executing. Result: " << result << "\n";


    ThreadPoolFuture<std::string> future2([]() {
        std::cout << "Calculating string...\n";
        return "Hello, world!".to_string();
    });

    std::string result2 = future2.get();
    std::cout << "Future finished executing. Result: " << result2 << "\n";
}
```
Here, we use templates in C++ to store the specific type of future (e.g., `std::future<int>` or `std::future<std::string>`) rather than using a generic pointer. This provides type safety and avoids the boxing overhead.

Thirdly, if you're operating in an environment like Javascript/Typescript, `async/await` offers a form of control flow without explicit future objects. While technically there are promises being generated under the hood, these are often managed by the javascript runtime and optimized internally. The programmer rarely needs to consider how to store these, as control flow is handled by compiler transformation. We can demonstrate this using a similar example as above using asynchronous functions and promises to perform thread-like operations in the javascript event loop.

```javascript
function delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

async function performCalculation() {
    console.log("Calculating...");
    await delay(2000);
    return 42;
}


async function performStringCalculation() {
    console.log("Calculating string...");
    return "Hello, world!";
}


async function main() {
    const result = await performCalculation();
    console.log("Future finished executing. Result:", result);


    const result2 = await performStringCalculation();
    console.log("Future finished executing. Result:", result2);
}

main();

```

This example shows that you can create an asynchronous program without worrying about the storage or boxing of a future. The `async` keyword on the functions implies that promises are used behind the scenes, which are managed by the runtime itself.

In summary, the approach to "storing a future without boxing" largely depends on the specific programming environment and constraints. The core idea revolves around using type-specific storage and/or language mechanisms such as compiler transforms that abstract away the future storage. If you have a good understanding of the specific type of your future and need to maximize performance, you should explore the alternatives to boxing. These strategies require a good grasp of the type system, ownership rules, and the asynchronous primitives in your environment. I'd recommend diving into the documentation for `std::future` and `std::pin` in Rust, the chapter on templates in “Effective C++” by Scott Meyers if working in C++, and the documentation on promises and async/await in javascript for better understanding. By utilizing these approaches, one can write performant, efficient asynchronous code, without the overhead associated with boxing.
