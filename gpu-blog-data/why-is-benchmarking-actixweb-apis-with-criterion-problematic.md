---
title: "Why is benchmarking actix_web APIs with Criterion problematic?"
date: "2025-01-30"
id: "why-is-benchmarking-actixweb-apis-with-criterion-problematic"
---
Benchmarking Actix Web APIs using Criterion presents unique challenges stemming from Actix's asynchronous, actor-based architecture.  My experience optimizing high-throughput services built on Actix Web revealed that Criterion's synchronous nature fundamentally clashes with the asynchronous operation model, leading to inaccurate and misleading benchmark results.  The core issue lies in Criterion's inability to accurately capture the true performance characteristics of an asynchronous framework under realistic load.

Criterion excels at measuring the execution time of synchronous code blocks.  It achieves this by repeatedly running the code and averaging the execution times.  This methodology works well for purely synchronous applications where a single thread executes the entire task.  However, Actix Web's asynchronous runtime utilizes a multi-threaded event loop, where numerous tasks are concurrently handled by a thread pool.  Criterion's synchronous approach fails to accurately reflect this concurrency.  It essentially measures the time taken for the *first* request to complete, often including significant initialization overhead, rather than the sustained throughput achievable under concurrent load.  This becomes increasingly problematic as the complexity and resource usage of the API increase.

This inaccuracy manifests in several ways. Firstly, the time taken to establish the initial connection to the Actix server significantly skews the benchmark results, especially for simple requests.  The setup time is a one-time cost not representative of subsequent requests. Secondly, Criterion's single-threaded measurement ignores the inherent parallelism of Actix Web. It doesn't account for the optimized handling of multiple concurrent requests by the runtime, leading to an underestimation of the server's true capacity.  Thirdly, the lack of explicit control over the number of concurrent connections in Criterion's default execution means the benchmark is not run under a realistic load profile.  A poorly designed benchmark might only stress a single worker thread, obscuring the server's true scalability.

To illustrate, consider these examples. I will focus on the impact of these issues in three different scenarios, emphasizing the limitations of relying on Criterion for benchmarking Actix Web APIs.


**Code Example 1: Simple Endpoint Benchmarking**

```rust
use actix_web::{App, HttpServer, Responder, web};
use criterion::{criterion_group, criterion_main, Criterion};

async fn simple_handler() -> impl Responder {
    "Hello, world!"
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    HttpServer::new(|| {
        App::new()
            .route("/", web::get().to(simple_handler))
    })
    .bind(("127.0.0.1", 8080))?
    .run()
    .await
}

fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("simple_handler", |b| b.iter(|| {
        // This is inherently flawed.  It doesn't simulate concurrent requests.
        // The time includes connection setup, which dominates the result.
        // It's essentially a single-threaded measurement of a multi-threaded system.
        let client = reqwest::blocking::Client::new();
        let response = client.get("http://127.0.0.1:8080/").send().unwrap();
    }));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
```

This example showcases the fundamental problem: the benchmark executes a single request in a blocking manner.  The `reqwest::blocking::Client` prevents concurrent requests. The result reflects the combined cost of establishing the connection, processing the request, and closing the connection, a grossly inaccurate measure of the API's true performance under realistic load.


**Code Example 2:  Introducing Concurrent Requests (still flawed)**

```rust
use actix_web::{App, HttpServer, Responder, web};
use criterion::{criterion_group, criterion_main, Criterion};
use rayon::prelude::*;

// ... (simple_handler remains the same) ...

fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("concurrent_requests", |b| b.iter(|| {
        (0..10).into_par_iter().for_each(|_| {
            // Rayon parallelism doesn't interact with Actix's event loop effectively.
            // This achieves apparent concurrency but doesn't model real-world conditions.
            let client = reqwest::blocking::Client::new();
            let response = client.get("http://127.0.0.1:8080/").send().unwrap();
        });
    }));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
```

While this example attempts to simulate concurrent requests using Rayon, it's still insufficient. Rayon provides thread-level parallelism, which doesn't interact seamlessly with Actix's asynchronous event loop.  The requests are still handled serially within each Rayon thread, failing to accurately reflect the true concurrency within the Actix runtime.  The benchmark still suffers from the connection overhead issue, and importantly, the interaction with the Actix event loop isn't being profiled.


**Code Example 3:  A More Accurate (but not Criterion-based) Approach**

This example highlights the need for a different benchmarking tool.  While not directly using Criterion, it demonstrates a more appropriate method.

```rust
use actix_web::{App, HttpServer, Responder, web};
use tokio::time::{sleep, Duration};
use reqwest::Client;
use std::time::Instant;

// ... (simple_handler remains the same) ...

#[tokio::main]
async fn main() -> std::io::Result<()> {
    let server = HttpServer::new(|| {
        App::new()
            .route("/", web::get().to(simple_handler))
    })
    .bind(("127.0.0.1", 8080))?;

    let server_handle = server.run();

    let client = Client::new();
    let num_requests = 1000;
    let start = Instant::now();

    let mut handles = Vec::new();
    for _ in 0..num_requests {
        handles.push(tokio::spawn(async move {
            client.get("http://127.0.0.1:8080/").send().await.unwrap();
        }));
    }

    futures::future::join_all(handles).await;
    let elapsed = start.elapsed();
    println!("Time for {} requests: {:?}", num_requests, elapsed);

    server_handle.await?;
    Ok(())
}
```

This demonstrates a more accurate approach using `tokio` and `reqwest`. We explicitly control the number of concurrent requests using `tokio::spawn` and measure the total time for all requests to complete.  This still doesn't provide the sophisticated statistical analysis that Criterion offers, but it produces a more realistic measure of the API's throughput under controlled concurrent load.

In summary, while Criterion is a valuable tool for benchmarking synchronous code, its limitations become apparent when applied to asynchronous frameworks like Actix Web.  Its synchronous nature hinders its ability to accurately capture the performance characteristics of a system designed for concurrency.  For benchmarking Actix Web APIs, dedicated tools or custom solutions that account for the asynchronous runtime are necessary to obtain meaningful and reliable results.  Exploring alternatives like `wrk`, `hey`, or developing custom benchmarking scripts utilizing `tokio` and a suitable HTTP client library provides significantly more accurate results for asynchronous applications.  Careful consideration of request concurrency, connection management, and the inherent overhead of the asynchronous runtime are crucial in achieving a robust and insightful benchmark.  Remember to consider the impact of resource contention in a realistic deployment environment.  These factors are not well captured by simply measuring the execution time of a single request.
