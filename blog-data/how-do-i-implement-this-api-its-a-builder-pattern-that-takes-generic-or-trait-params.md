---
title: "How do I implement this API? It's a Builder pattern that takes generic (or trait) params."
date: "2024-12-23"
id: "how-do-i-implement-this-api-its-a-builder-pattern-that-takes-generic-or-trait-params"
---

,  I’ve definitely been down this road before – implementing a builder pattern with generic parameters can get tricky fast, but it's an incredibly powerful technique once you nail the fundamentals. In one project, we were constructing complex data structures for a simulation engine, and leveraging a generic builder significantly simplified the process. The key is to maintain type safety while providing a flexible interface.

The core of your challenge stems from marrying the builder pattern’s fluent, chained method calls with the flexibility of generics or traits. You're essentially asking for a type-safe way to configure your target object step-by-step using methods that accept different types—that's the essence of the builder pattern, and the generics add that extra dimension of variability. The trick is to establish a clear boundary for these generic parameters at compile time while also making the API as ergonomic as possible for the end-user.

Let's break down a potential implementation strategy, focusing on clarity and practicality:

First, we need to define what our builder should look like at a high level. We’ll aim for a builder struct or class that holds a reference to the object it is constructing. Then we’ll have methods, each accepting a single generic parameter constrained by a trait, if necessary, which modify the internal state before returning a modified builder instance, thus enabling the chaining mechanism. The final method would be the build function which creates the desired object.

Let's start with a Rust example, demonstrating traits and generics in a real-world context. Suppose you have a system that deals with different types of processing pipelines. These pipelines all need a way to load data, but the data loading mechanisms can vary significantly (file, network, etc.).

```rust
trait DataSource {
    fn load(&self) -> String;
}

struct FileSource {
    path: String,
}

impl DataSource for FileSource {
    fn load(&self) -> String {
        format!("Loading from file: {}", self.path)
    }
}

struct NetworkSource {
    url: String,
}

impl DataSource for NetworkSource {
    fn load(&self) -> String {
        format!("Loading from network: {}", self.url)
    }
}

struct Processor<T: DataSource> {
    source: Option<T>,
    processing_steps: Vec<String>,
}

impl<T: DataSource> Processor<T> {
    fn new() -> Self {
        Processor {
            source: None,
            processing_steps: Vec::new(),
        }
    }
}

struct ProcessorBuilder<T: DataSource> {
    processor: Processor<T>,
}


impl<T: DataSource> ProcessorBuilder<T> {
    fn new() -> ProcessorBuilder<T> {
        ProcessorBuilder {
            processor: Processor::new(),
        }
    }

    fn with_source(mut self, source: T) -> Self {
         self.processor.source = Some(source);
         self
    }


    fn add_step(mut self, step: String) -> Self {
        self.processor.processing_steps.push(step);
        self
    }

    fn build(self) -> Processor<T> {
         self.processor
    }
}

fn main() {
    let file_processor = ProcessorBuilder::new()
        .with_source(FileSource {path: "data.txt".to_string()})
        .add_step("Step 1".to_string())
        .add_step("Step 2".to_string())
        .build();

    let network_processor = ProcessorBuilder::new()
        .with_source(NetworkSource {url: "http://example.com".to_string()})
        .add_step("Pre-process".to_string())
        .build();

   if let Some(source) = file_processor.source {
     println!("File Processor: {}", source.load());
    }

    if let Some(source) = network_processor.source {
       println!("Network Processor: {}", source.load());
    }


   println!("File processor Steps {:?}",file_processor.processing_steps);
   println!("Network processor Steps {:?}", network_processor.processing_steps);
}

```

In this snippet, `DataSource` is our trait. `FileSource` and `NetworkSource` are concrete implementations of this trait. The `Processor` struct holds a `source` of type `Option<T>`, where `T` is a type that implements the `DataSource` trait. The builder provides methods to set the source and add processing steps, showing how the generics are used, the `.with_source` method ensures that a proper `DataSource` implementation can be passed to it. The `build` method creates the fully configured `Processor` object.

Let’s now look at a slightly different approach using Kotlin, which can also work well for generics within a builder.

```kotlin
interface DataSource {
    fun load(): String
}

class FileSource(val path: String) : DataSource {
    override fun load(): String = "Loading from file: $path"
}

class NetworkSource(val url: String) : DataSource {
    override fun load(): String = "Loading from network: $url"
}

data class Processor<T : DataSource>(var source: T? = null, val processingSteps: MutableList<String> = mutableListOf())

class ProcessorBuilder<T : DataSource> {

    private var processor: Processor<T> = Processor()

    fun withSource(source: T): ProcessorBuilder<T> {
        processor.source = source
        return this
    }


    fun addStep(step: String): ProcessorBuilder<T> {
         processor.processingSteps.add(step)
        return this
    }

    fun build(): Processor<T> = processor
}

fun main() {
    val fileProcessor = ProcessorBuilder<FileSource>()
        .withSource(FileSource("data.txt"))
        .addStep("Step 1")
        .addStep("Step 2")
        .build()

    val networkProcessor = ProcessorBuilder<NetworkSource>()
        .withSource(NetworkSource("http://example.com"))
        .addStep("Pre-process")
        .build()


    println("File Processor: ${fileProcessor.source?.load()}")
     println("Network Processor: ${networkProcessor.source?.load()}")

    println("File processor Steps ${fileProcessor.processingSteps}")
    println("Network processor Steps ${networkProcessor.processingSteps}")
}

```

Here, `DataSource` is an interface, and `FileSource` and `NetworkSource` implement it. The `ProcessorBuilder` uses generics `<T : DataSource>` to ensure the type T parameter passed has to implement `DataSource`, and it provides a similar fluent interface. This example illustrates that generics in the builder pattern can translate to different languages using the same underlying concept. Kotlin also benefits from type inference making the generic type for the builder sometimes obvious.

Finally, let's consider a Python approach to implementing a similar builder pattern using abstract base classes which works well for many cases. While Python doesn't have generics in the same way as Rust or Kotlin, we can use type hints and abstract base classes (ABCs) to impose some level of type constraints and achieve similar goals.

```python
from abc import ABC, abstractmethod

class DataSource(ABC):
    @abstractmethod
    def load(self):
        pass

class FileSource(DataSource):
    def __init__(self, path):
        self.path = path

    def load(self):
        return f"Loading from file: {self.path}"

class NetworkSource(DataSource):
    def __init__(self, url):
        self.url = url

    def load(self):
         return f"Loading from network: {self.url}"

class Processor:
    def __init__(self, source: DataSource = None, processing_steps: list = None):
        self.source = source
        self.processing_steps = processing_steps if processing_steps else []

    def add_step(self, step):
        self.processing_steps.append(step)


class ProcessorBuilder:
    def __init__(self):
        self.processor = Processor()

    def with_source(self, source: DataSource):
        self.processor.source = source
        return self

    def add_step(self, step: str):
        self.processor.add_step(step)
        return self

    def build(self) -> Processor:
        return self.processor


if __name__ == "__main__":
    file_processor = ProcessorBuilder() \
        .with_source(FileSource("data.txt")) \
        .add_step("Step 1") \
        .add_step("Step 2") \
        .build()

    network_processor = ProcessorBuilder() \
        .with_source(NetworkSource("http://example.com")) \
        .add_step("Pre-process") \
        .build()

    print(f"File Processor: {file_processor.source.load()}")
    print(f"Network Processor: {network_processor.source.load()}")
    print(f"File processor Steps {file_processor.processing_steps}")
    print(f"Network processor Steps {network_processor.processing_steps}")
```
In this example, `DataSource` is an abstract base class, and `FileSource` and `NetworkSource` are implementations. `ProcessorBuilder` takes a `DataSource` as input for its source step using a type hint which offers some level of dynamic type safety. While not as strict as compile-time generics, ABCs and type hints provide valuable guidance during development and allow you to write code that is easy to understand and reason about.

In terms of further learning and solidifying your understanding, I’d recommend looking into:

1.  **“Effective Java” by Joshua Bloch:** While Java specific, the chapters on builders and generics provide invaluable insights into these design patterns, which are generally applicable.
2.  **"Programming in Scala" by Martin Odersky, Lex Spoon, and Bill Venners:** Although it focuses on Scala, it has great coverage of generics and the techniques employed translate very well when implementing this pattern.
3.  **"Rust Programming Language" by Steve Klabnik and Carol Nichols:** This is your primary source for rust specifics, with excellent explanations of generics and traits.
4.  **"Fluent Python" by Luciano Ramalho:** For a deeper dive into Python, specifically focusing on idioms and techniques you might find useful when working with ABCs in this context.

The essence of implementing a generic builder pattern is to decouple the object construction logic from the object's representation, allowing you to construct objects with varying configurations with type-safety. By focusing on traits or interfaces to constrain your generic parameters and understanding how to chain method calls correctly, you will be able to build powerful and expressive APIs. The examples above show just a small sample, and the specific implementation will likely depend on your chosen language and specific requirements, but the core concept remains universal.
