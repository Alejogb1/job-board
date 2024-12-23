---
title: "How to close files RAII-style using SEastar.io coroutines?"
date: "2024-12-23"
id: "how-to-close-files-raii-style-using-seastario-coroutines"
---

Alright, let's talk about something that’s always near the top of my mind when juggling asynchronous I/O in a high-performance context: resource management, particularly file handles. In the context of Seastar, where performance is king and we're leaning heavily into coroutines, the need for deterministic resource cleanup—specifically closing files—becomes paramount. Trying to rely solely on garbage collection or similar mechanisms often introduces unpredictability, and that’s just something we can't afford when we're trying to eke out every ounce of performance. So, instead of hoping for the best, we should actively manage resources using RAII – resource acquisition is initialization – a fundamental C++ programming idiom.

I’ve had my fair share of encounters with resource leaks back in the day, specifically in a distributed storage system I worked on. We were dealing with thousands of concurrent requests, each involving file operations. In the initial design, we weren't strictly enforcing RAII with our asynchronous file access patterns; we’d just open a file, pass the file descriptor around, and trust that eventually someone would remember to close it. Well, you can imagine how that ended. We were consistently hitting the limits on the number of open file descriptors, which brought the whole system down, sometimes quite dramatically, and diagnosing it was like trying to find a specific grain of sand on a beach after a storm. I can tell you firsthand that debugging resource leaks in asynchronous systems is far from ideal, which led to a lot of late nights.

So how do we get around this with Seastar’s coroutines? The key lies in crafting objects that encapsulate file descriptors and ensure they are closed when the object goes out of scope. This is what RAII is all about. We want our resource to be tied to the lifetime of an object. When that object is destroyed, the resource is released. Let’s start with a simple example:

```cpp
#include <seastar/core/file.hh>
#include <seastar/core/coroutine.hh>
#include <seastar/core/task.hh>
#include <seastar/core/app-template.hh>
#include <iostream>

namespace my {

class scoped_file {
public:
    scoped_file(seastar::file f) : _file(std::move(f)) {}

    ~scoped_file() {
        if (_file.is_valid()) {
            _file.close().get(); //wait for the close to complete.
        }
    }

    seastar::file& get() { return _file; }
    
    scoped_file(const scoped_file& other) = delete;
    scoped_file& operator=(const scoped_file& other) = delete;

    scoped_file(scoped_file&& other) noexcept : _file(std::move(other._file)) {}
    scoped_file& operator=(scoped_file&& other) noexcept {
        if(this != &other){
          if (_file.is_valid()){
              _file.close().get();
          }
          _file = std::move(other._file);
        }
        return *this;
    }


private:
    seastar::file _file;
};

seastar::future<> process_file(const seastar::sstring& filename) {
  auto file_open_result =  co_await seastar::open_file_dma(filename, seastar::open_flags::ro);
  if (file_open_result.has_error()){
     std::cerr << "Failed to open " << filename << ": " << file_open_result.error() << std::endl;
    co_return;
  }

  scoped_file sf(file_open_result.value());
  auto &file = sf.get();
  
  //Perform operations on the file using 'file'
    char buffer[1024];
    auto read_result = co_await file.read_dma(0, seastar::temporary_buffer<char>(buffer, 1024));

    if (read_result.has_error()){
        std::cerr << "Failed to read from " << filename << ": " << read_result.error() << std::endl;
        co_return;
    }

  std::cout << "Successfully read from the file." << std::endl;
}

} // namespace my


int main(int argc, char** argv) {
    seastar::app_template app;
    return app.run(argc, argv, [] {
        return my::process_file("test.txt"); // Create 'test.txt' in the directory where you run it before executing this code, and add some content to it.
    });
}
```

In this snippet, `scoped_file` takes ownership of a `seastar::file` object in its constructor, using `std::move`. Crucially, its destructor attempts to close the file only if the file descriptor is valid. This is very important because if the file was never successfully opened, then it will not try to close an invalid file, which would result in an error. Inside `process_file`, we use it to ensure our file gets closed when `sf` goes out of scope, regardless of how `process_file` exits (normally or with an exception). The `get()` function is how you get a reference to the underlying `seastar::file` to use for read/write operations. This approach encapsulates the file's lifecycle.

Now, you might think that’s the end of the story, but in my experience, resource handling rarely is that straightforward. We also need to think about error handling, particularly when dealing with asynchronous operations like file I/O. For instance, what happens if we encounter an error opening the file? We can’t just leave it hanging and leaking resources. Therefore, we can use some of C++'s capabilities for exception handling and resource cleanup to refine the previous approach.

```cpp
#include <seastar/core/file.hh>
#include <seastar/core/coroutine.hh>
#include <seastar/core/task.hh>
#include <seastar/core/app-template.hh>
#include <iostream>

namespace my {

class scoped_file {
public:
    explicit scoped_file(seastar::file f) : _file(std::move(f)) {}
    ~scoped_file() {
        if (_file.is_valid()) {
            try{
                _file.close().get();
            } catch(const std::exception& e){
                std::cerr << "Exception closing file: " << e.what() << std::endl;
            }
        }
    }

    seastar::file& get() { return _file; }
    
    scoped_file(const scoped_file& other) = delete;
    scoped_file& operator=(const scoped_file& other) = delete;

    scoped_file(scoped_file&& other) noexcept : _file(std::move(other._file)) {}
    scoped_file& operator=(scoped_file&& other) noexcept {
        if(this != &other){
          if (_file.is_valid()){
              try {
                  _file.close().get();
              } catch(const std::exception& e){
                  std::cerr << "Exception closing file in move assign: " << e.what() << std::endl;
              }

          }
          _file = std::move(other._file);
        }
        return *this;
    }
    

private:
    seastar::file _file;
};

seastar::future<> process_file(const seastar::sstring& filename) {
    auto file_open_result =  co_await seastar::open_file_dma(filename, seastar::open_flags::ro);
    if (file_open_result.has_error()){
      std::cerr << "Failed to open " << filename << ": " << file_open_result.error() << std::endl;
      co_return;
    }
    scoped_file sf(file_open_result.value());
  
    auto& file = sf.get();
    char buffer[1024];
    auto read_result = co_await file.read_dma(0, seastar::temporary_buffer<char>(buffer, 1024));
    if (read_result.has_error()){
        std::cerr << "Failed to read from " << filename << ": " << read_result.error() << std::endl;
        co_return;
    }

    std::cout << "Successfully read from the file." << std::endl;
}

} // namespace my


int main(int argc, char** argv) {
    seastar::app_template app;
    return app.run(argc, argv, [] {
        return my::process_file("test.txt"); // Create 'test.txt' in the directory where you run it before executing this code, and add some content to it.
    });
}
```

Here, the destructor of `scoped_file` now catches any potential exceptions that arise during the closing process. This may seem a bit overkill, however in practice, file operations, like all I/O, are fallible and can sometimes throw, especially in complex environments. Therefore, it’s essential that we capture any issues that arise and deal with them as appropriately as possible. Now, even if the close operation throws, the file's descriptor will still be considered closed, preventing a resource leak. This adds another layer of robustness to our code. Furthermore, we also wrap the close in the move assignment operator, as moving the object may require a cleanup operation.

Finally, let’s consider a slightly more complex scenario where you might want to perform some additional cleanup operations related to the file before it’s definitively closed.

```cpp
#include <seastar/core/file.hh>
#include <seastar/core/coroutine.hh>
#include <seastar/core/task.hh>
#include <seastar/core/app-template.hh>
#include <iostream>


namespace my {

class scoped_file {
public:
    explicit scoped_file(seastar::file f, std::function<seastar::future<>> cleanup_handler = [] { return seastar::make_ready_future<>(); })
        : _file(std::move(f)), _cleanup_handler(std::move(cleanup_handler)) {}


    ~scoped_file() {
       
        seastar::future<> cleanup_fut = seastar::make_ready_future<>();
        if (_file.is_valid()) {
            cleanup_fut = _cleanup_handler();
        }

         try {
              cleanup_fut.get();
              if(_file.is_valid()){
                _file.close().get();
              }
        } catch(const std::exception& e){
                std::cerr << "Exception during cleanup or close: " << e.what() << std::endl;
            }
       
    }
  
    seastar::file& get() { return _file; }
    
    scoped_file(const scoped_file& other) = delete;
    scoped_file& operator=(const scoped_file& other) = delete;

    scoped_file(scoped_file&& other) noexcept : _file(std::move(other._file)), _cleanup_handler(std::move(other._cleanup_handler)) {}
    scoped_file& operator=(scoped_file&& other) noexcept {
        if(this != &other){
           seastar::future<> cleanup_fut = seastar::make_ready_future<>();
           if(_file.is_valid()){
              cleanup_fut = _cleanup_handler();
           }
            
          try {
                cleanup_fut.get();
                if (_file.is_valid()){
                    _file.close().get();
                }
              
          } catch(const std::exception& e){
                std::cerr << "Exception during cleanup or close in move assign: " << e.what() << std::endl;
            }
          
           _file = std::move(other._file);
           _cleanup_handler = std::move(other._cleanup_handler);

        }

        return *this;
    }


private:
    seastar::file _file;
    std::function<seastar::future<>> _cleanup_handler;
};


seastar::future<> process_file(const seastar::sstring& filename) {
    auto file_open_result = co_await seastar::open_file_dma(filename, seastar::open_flags::rw | seastar::open_flags::create);
    if (file_open_result.has_error()) {
        std::cerr << "Failed to open " << filename << ": " << file_open_result.error() << std::endl;
        co_return;
    }


    scoped_file sf(file_open_result.value(), []{
           std::cout << "Cleanup function ran" << std::endl;
        return seastar::make_ready_future<>();
     });


    auto& file = sf.get();
    char buffer[] = "Hello, World!";
    auto write_result = co_await file.write_dma(0, seastar::temporary_buffer<char>(buffer, sizeof(buffer) - 1));
      if (write_result.has_error()) {
        std::cerr << "Failed to write to " << filename << ": " << write_result.error() << std::endl;
        co_return;
    }
     auto read_result = co_await file.read_dma(0, seastar::temporary_buffer<char>(buffer, sizeof(buffer) - 1));
     if (read_result.has_error()){
        std::cerr << "Failed to read from " << filename << ": " << read_result.error() << std::endl;
        co_return;
    }

    std::cout << "Successfully read and wrote from the file. Content: " << buffer << std::endl;
}

} // namespace my


int main(int argc, char** argv) {
    seastar::app_template app;
    return app.run(argc, argv, [] {
        return my::process_file("test.txt"); // Create 'test.txt' in the directory where you run it before executing this code, and add some content to it.
    });
}
```

Here, we've added a `cleanup_handler` which is a `std::function<seastar::future<>>` that you can set when constructing the `scoped_file`. This handler is executed before the file is closed in the destructor, allowing you to perform things like flushing buffers, logging, etc. It may be particularly useful in certain circumstances. Again, similar to the previous approach, we ensure that we handle exceptions appropriately.

For further reading on resource management in modern C++, I'd recommend Scott Meyers' "Effective C++" and "Effective Modern C++" and Herb Sutter's "Exceptional C++." These books dive deep into best practices and offer invaluable insights. For deeper understanding of RAII and its use in async programming, exploring patterns and practices outlined in various concurrency-focused books, like "C++ Concurrency in Action" by Anthony Williams, will be helpful. Additionally, studying patterns like the scope guard pattern will be useful. In our examples we have implemented similar ideas but using a class instead of a generic function based approach.

In summary, using RAII with Seastar's coroutines is critical for achieving predictable resource management. By encapsulating resources within objects and ensuring proper cleanup in their destructors, we not only prevent leaks but also write cleaner, more robust code. While the initial setup requires a bit of extra thought, the benefits in terms of stability and predictability are significant in the context of high-performance asynchronous programming.
