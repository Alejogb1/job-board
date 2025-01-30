---
title: "How can I safely share an immutable reference to a `Send` resource in Rust?"
date: "2025-01-30"
id: "how-can-i-safely-share-an-immutable-reference"
---
A crucial challenge in concurrent Rust programming involves safely providing shared, read-only access to resources that can move between threads (`Send` types), while maintaining immutability to prevent data races. Naively sharing a mutable reference across threads violates Rust's ownership rules and leads to undefined behavior. However, several mechanisms in the standard library allow us to achieve this safely.

The core problem stems from the fact that a mutable reference (`&mut T`) guarantees exclusive access. This guarantee is fundamental for data race prevention. Consequently, a direct `&mut T` cannot be safely sent between threads. If multiple threads held such a reference, modifications from different threads would create a data race. Thus, immutable sharing requires a different approach. The solution involves wrapping the shared resource in a type that manages access and guarantees immutability, even across thread boundaries.

We can accomplish immutable sharing in multiple ways, each suited for slightly different use cases. Primarily, we employ `Arc` coupled with interior mutability. `Arc<T>`, an Atomic Reference Counted pointer, allows shared ownership of a resource. The counter tracks how many owners exist, deallocating the memory when the last owner goes out of scope. `Arc` itself is `Send` and `Sync` when `T` is `Send` + `Sync` – an important property for multi-threaded applications.

However, `Arc<T>` by itself, where `T` is not `Sync`, does not allow for interior mutability. `T` needs to allow read-only shared access. This means you cannot obtain a `&mut T` from an `Arc<T>`. To facilitate interior mutability within an `Arc`, a wrapper like `Mutex` or `RwLock` is necessary if you need to modify that resource from any threads. However, if `T` is read-only, we can achieve immutable sharing with an `Arc` alone. We simply clone the `Arc`, which increments the reference count and provides shared ownership.

Let’s first examine a scenario where the type we want to share is already immutable. If our resource implements `Copy` and is `Send` + `Sync` or it’s already immutable, we simply wrap that in an `Arc`. Let’s illustrate this with a simple integer.

```rust
use std::sync::Arc;
use std::thread;

fn main() {
    let shared_value: i32 = 10;
    let shared_arc: Arc<i32> = Arc::new(shared_value);

    let mut handles = vec![];

    for _ in 0..3 {
        let cloned_arc = Arc::clone(&shared_arc);
        let handle = thread::spawn(move || {
            println!("Thread says: {}", cloned_arc);
        });
        handles.push(handle);
    }

    for handle in handles {
       handle.join().unwrap();
    }
}

```

In this example, we create an `Arc` holding the integer 10. We clone the `Arc` in the loop. The important aspect is that `i32` implements `Copy`, allowing for simple, immutable shared access. Each thread receives its own copy of the `Arc`, referring to the same underlying integer value. Since the value is immutable, there is no danger of data races or other concurrency issues.

Now, let’s consider a situation where we have a complex data structure that we want to share immutably, but it is not `Copy`. For this scenario, it’s crucial that the resource we share is designed for read-only access. Let's assume we have a configuration struct that doesn’t allow its internal fields to be modified after initialization.

```rust
use std::sync::Arc;
use std::thread;

#[derive(Debug)]
struct Configuration {
    server_address: String,
    port: u16,
}

impl Configuration {
    fn new(address: String, port: u16) -> Self {
        Configuration { server_address: address, port: port }
    }

    fn server_info(&self) -> String {
        format!("Server: {}, Port: {}", self.server_address, self.port)
    }
}

fn main() {
    let config = Configuration::new("127.0.0.1".to_string(), 8080);
    let shared_config: Arc<Configuration> = Arc::new(config);

    let mut handles = vec![];

    for _ in 0..3 {
        let cloned_config = Arc::clone(&shared_config);
        let handle = thread::spawn(move || {
            println!("Thread says: {}", cloned_config.server_info());
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }
}

```

Here, the `Configuration` struct is not `Copy`, but it’s designed to be immutable after creation. The `server_info` method allows read-only access to the struct’s fields. We encapsulate the configuration object within an `Arc`. Each thread receives a cloned `Arc`, pointing to the same read-only data structure. The `server_info` method allows multiple threads to concurrently read the data safely.

Finally, if you need to share an object that initially requires a single, mutable setup (non-`Sync` type) but can then transition to shared read-only mode, the following pattern might be valuable. Consider sharing a cached value. Initially, the cache needs to be set. After that, its read-only values are shared. Using `OnceLock` can ensure that the initialization happens exactly once.

```rust
use std::sync::Arc;
use std::sync::OnceLock;
use std::thread;

#[derive(Debug)]
struct CachedData {
    data: String,
}

static CACHED_DATA: OnceLock<Arc<CachedData>> = OnceLock::new();

fn get_shared_cache() -> Arc<CachedData> {
    CACHED_DATA.get_or_init(|| {
       println!("Initializing cache!");
       Arc::new(CachedData { data: "Initial Cache Data".to_string()})
    }).clone()
}


fn main() {

    let mut handles = vec![];

    for i in 0..3 {
       let cloned_cache = get_shared_cache();
       let handle = thread::spawn(move || {
          println!("Thread {} sees cache with: {}", i, cloned_cache.data);
       });
       handles.push(handle);
    }

   for handle in handles {
       handle.join().unwrap();
   }

}

```

In this third example, `CachedData` does not implement `Sync`, and thus could not directly be used in a multi-threaded context. The `OnceLock` manages its initialization such that it happens only once, regardless of how many times `get_shared_cache` is called across different threads. We then take a clone of the cached data. Each thread can read the content, but since it is wrapped in an `Arc`, they do not have a mutable reference, guaranteeing that no data races can occur.

In summary, the key to safely sharing an immutable reference to a `Send` resource in Rust is using `Arc` and making sure that the underlying resource is either inherently immutable after creation, has internal read-only access functions, or is only modified once. For resources that need to be modified, mechanisms like Mutex or RwLock are necessary to manage that access. The choice depends heavily on the specific requirements of your application – whether the data is truly immutable, whether it needs any form of mutability, and how frequently you access it.

For further learning and mastery of Rust's concurrency features, I recommend reading the official Rust documentation on the standard library, paying special attention to sections on concurrency primitives and smart pointers. The “Rust Programming Language” book covers many of these topics in depth. Additionally, practice with various concurrency scenarios will significantly increase your understanding and skill. Studying common patterns and antipatterns in concurrent programming is also extremely beneficial. I often find that experimenting with different concurrency tools while solving practical problems reveals the nuances of each approach and aids in developing safe and performant concurrent software.
