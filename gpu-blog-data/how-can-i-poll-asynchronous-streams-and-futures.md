---
title: "How can I poll asynchronous streams and futures derived from stream items within a single loop?"
date: "2025-01-30"
id: "how-can-i-poll-asynchronous-streams-and-futures"
---
Asynchronous stream processing frequently requires handling futures generated from individual stream items while maintaining overall stream consumption. Failure to manage this interleaving efficiently can lead to stalls, resource contention, or incorrect results. I've encountered this several times during development of high-throughput data pipelines and have developed a reliable approach utilizing `futures::stream::FuturesUnordered`.

The challenge lies in the inherently asynchronous nature of both streams and futures. A stream provides a sequence of items over time, while each item might trigger a future representing a potentially lengthy operation. To process these futures effectively, a simple `for await` loop iterating over the stream is insufficient. This approach would handle one future to completion before even acknowledging the next stream item, creating a severe bottleneck. The ideal solution permits the concurrent resolution of multiple futures originating from different stream elements, and that's where `FuturesUnordered` proves invaluable.

`FuturesUnordered` is a collection that stores a series of futures, allowing concurrent execution. Unlike a simple `Vec` of futures, `FuturesUnordered` is itself a stream, yielding the results of the futures as they complete. Crucially, it automatically removes completed futures, preventing resource leaks. This characteristic makes it ideally suited for polling futures generated within a stream's consumption loop. The core idea is to: 1) initialize an empty `FuturesUnordered` collection; 2) iterate over the asynchronous stream; 3) for each stream item, convert it into a future and insert it into the `FuturesUnordered`; 4) then simultaneously poll the `FuturesUnordered` in the same loop until it’s empty. This mechanism manages multiple in-flight asynchronous operations without explicitly blocking on any single one.

This ensures that we are always simultaneously processing the next available stream item and any futures that have become available in `FuturesUnordered`. Because we're treating `FuturesUnordered` as a stream, it automatically yields completed futures without further interaction. This greatly simplifies management compared to manual polling.

Consider a scenario where each stream item contains text that needs to be processed via an external service, represented by an asynchronous function. Here’s how one can accomplish that:

```rust
use futures::{stream, StreamExt, Future, future, stream::FuturesUnordered};

async fn process_text(text: String) -> String {
  // Simulate asynchronous work
  tokio::time::sleep(std::time::Duration::from_millis(rand::random::<u64>() % 500)).await;
  format!("Processed: {}", text)
}

async fn process_stream() {
    let text_stream = stream::iter(vec!["Hello".to_string(), "World".to_string(), "Async".to_string(), "Streams".to_string()]);
    let mut futures = FuturesUnordered::new();

    let mut stream = text_stream.enumerate();
    loop {
      // Poll for available values in FuturesUnordered first.
      match futures.next().await {
        Some(result) => {
          println!("Future Result: {}", result);
        },
         None => {
           // Now check for stream items.
           match stream.next().await{
              Some((_index, item)) => {
                  futures.push(process_text(item));
              }
              None if futures.is_empty() => {
                break;
              },
              None => {} // stream is done, but FuturesUnordered still has work.
            }
         }
      }
    }
}


#[tokio::main]
async fn main() {
    process_stream().await;
}
```

This example first constructs a simple `text_stream`. Within the loop, we first examine if any futures in the `futures` collection have completed via `futures.next().await`. If yes, we print the result. If not, we get the next stream item and push the created future into `futures`. Note the need to handle the cases where the stream has terminated but futures still remain in the `futures` collection. The loop terminates when both the input stream and `FuturesUnordered` are empty. This ensures no futures are left unmanaged, a vital point in robust asynchronous handling. Note the usage of `stream.enumerate()` to demonstrate access to the index which could be valuable for error handling or logging.

A more advanced approach would involve error handling during future processing. Consider a situation where `process_text` might return a `Result`. `FuturesUnordered` does not inherently handle errors. We would need to propagate it from the returned future.

```rust
use futures::{stream, StreamExt, Future, future, stream::FuturesUnordered};

async fn process_text(text: String) -> Result<String, String> {
    // Simulate asynchronous work, some of which might fail
    tokio::time::sleep(std::time::Duration::from_millis(rand::random::<u64>() % 500)).await;
    if rand::random::<f64>() < 0.2 {
        Err(format!("Processing failed for: {}", text))
    } else {
       Ok(format!("Processed: {}", text))
    }
}

async fn process_stream() {
  let text_stream = stream::iter(vec!["A".to_string(), "B".to_string(), "C".to_string(), "D".to_string(), "E".to_string()]);
  let mut futures = FuturesUnordered::new();

  let mut stream = text_stream.enumerate();
    loop {
      // Poll for available values in FuturesUnordered first.
      match futures.next().await {
        Some(result) => {
          match result {
              Ok(processed) => println!("Future Result: {}", processed),
              Err(err) => println!("Future Error: {}", err),
            }
        },
        None => {
          // Now check for stream items.
          match stream.next().await{
            Some((_index, item)) => {
                futures.push(process_text(item));
            }
            None if futures.is_empty() => {
              break;
            },
            None => {} // stream is done, but FuturesUnordered still has work.
          }
        }
      }
    }
}

#[tokio::main]
async fn main() {
    process_stream().await;
}
```
In this revised version, `process_text` returns a `Result`. The processing loop now matches on `result` before printing. We've converted the error case to be printed as well instead of an immediate `panic`, a more graceful failure case. This illustrates that `FuturesUnordered` only handles the concurrency aspect, and error propagation must be handled by the user.

Furthermore, handling large streams can be optimized by introducing a limit on the number of concurrent futures. This prevents overwhelming the system, a necessary consideration in high-load scenarios. We can introduce a "buffer" by simply limiting the number of items we insert to `FuturesUnordered` at a given point.

```rust
use futures::{stream, StreamExt, Future, future, stream::FuturesUnordered};

async fn process_text(text: String) -> String {
    // Simulate asynchronous work
    tokio::time::sleep(std::time::Duration::from_millis(rand::random::<u64>() % 500)).await;
    format!("Processed: {}", text)
}


async fn process_stream() {
    let text_stream = stream::iter((0..100).map(|i| format!("Text {}", i)).collect::<Vec<_>>());
    let mut futures = FuturesUnordered::new();
    let mut stream = text_stream.enumerate();
    let concurrent_limit = 10; //Limit the number of concurrent futures.

    loop {
      match futures.next().await {
        Some(result) => {
            println!("Future Result: {}", result);
        },
        None => {
            // Fill up FuturesUnordered until limit is hit
           while futures.len() < concurrent_limit {
            match stream.next().await {
              Some((_index, item)) => {
                futures.push(process_text(item));
              },
              None => break,
            }
          }
           // If both are empty, break
          if futures.is_empty() && stream.size_hint().0 == 0{
            break;
          }

        }
      }
    }

}
#[tokio::main]
async fn main() {
    process_stream().await;
}
```

Here, `concurrent_limit` dictates the maximum number of concurrently executing futures, creating a controllable flow. In each loop iteration, we attempt to add new futures to `FuturesUnordered` until this limit is met. This prevents an unchecked buildup of futures when dealing with very large or even infinite input streams, safeguarding resource consumption.

For more in-depth knowledge, I recommend studying the official documentation for the `futures` crate, particularly focusing on the `Stream`, `Future`, and `FuturesUnordered` types. The Rust documentation also contains detailed explanations of asynchronous programming in general. Additionally, numerous online blogs and tutorials are available, which walk through concrete use cases of asynchronous programming in Rust. Mastering these resources provides the necessary foundation for understanding and efficiently utilizing these powerful tools. This technique of using `FuturesUnordered` with stream polling has consistently proven effective in my experience, and is a pattern I frequently employ for asynchronous processing.
