---
title: "parallel_for_each tbb threading library?"
date: "2024-12-13"
id: "parallelforeach-tbb-threading-library"
---

Okay so you're asking about `parallel_for_each` from the Intel Threading Building Blocks TBB library yeah I've been there done that got the t-shirt and probably a few compiler warnings too let's unpack this

First off `parallel_for_each` it's a beast it's like the workhorse of TBB when you need to apply the same operation to a collection of items in parallel It's all about efficiency avoiding those nasty bottlenecks and making full use of your multi-core processor instead of the usual single-thread slow poke approach I've spent years wrestling with these things and seen the good bad and the ugly believe me

So basically you've got some container let's say a `std::vector` or a custom data structure or anything that supports iterators and you need to do something with every element right a transformation a computation some kind of action and the straightforward way is a `for` loop a classic loop that sequentially goes through one element at a time It's simple enough but it won't use all your cores and you're just wasting CPU power

`parallel_for_each` takes that loop and parallelizes it it breaks the work down into chunks and distributes them across multiple threads each core does a chunk of work and the library takes care of all the complex stuff like load balancing synchronizing the threads dealing with the thread pool things that would make a regular programmer cry it's a beautiful thing when you use it right

I remember back in my early days working on this image processing project I had this huge image array like megapixels worth and I needed to apply a filter to each pixel I first naively used a normal `for` loop it took forever like I left my computer running for 30 mins on one image It was a terrible experience I learned then that single core processing was a bad idea Then I discovered TBB specifically `parallel_for_each` and it changed everything it was like the image processing went from snail speed to rocket speed the rendering of each frame was almost instantaneous

Here is a simple example let's say you have a vector of integers you need to square every number this is how you'd do it with `parallel_for_each`

```cpp
#include <vector>
#include <iostream>
#include <tbb/tbb.h>

int main() {
    std::vector<int> numbers = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

    tbb::parallel_for_each(numbers.begin(), numbers.end(), [](int& number) {
        number = number * number;
    });

    for (int number : numbers) {
        std::cout << number << " ";
    }
    std::cout << std::endl;

    return 0;
}
```

See the lambda function there `[](int& number) { number = number * number; }` this is what's applied to each element this function will be executed in parallel on different cores it's pretty simple and straight forward

Now let's say you need to do something more complex something that requires accessing data from different positions within the container this is where you need to be a bit more careful with TBB because it's parallel and things can get a bit tricky fast if you are not careful about it

Here's an example of summing two adjacent elements lets assume that you know how to handle cases of boundary of the container let's say you have an even size container for simplicity reasons

```cpp
#include <vector>
#include <iostream>
#include <tbb/tbb.h>

int main() {
    std::vector<int> numbers = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    std::vector<int> results(numbers.size()/2);

     tbb::parallel_for(tbb::blocked_range<size_t>(0, numbers.size()/2),
            [&](const tbb::blocked_range<size_t>& r)
            {
               for(size_t i=r.begin(); i!=r.end(); ++i) {
                 results[i] = numbers[2*i] + numbers[2*i+1];
               }
            }
        );


    for (int number : results) {
        std::cout << number << " ";
    }
    std::cout << std::endl;

    return 0;
}

```

Now the important thing to note about the previous example is that we used `tbb::parallel_for` which is a slightly different TBB function but it can be used in the same fashion as `parallel_for_each` I am including it because of its versatility in indexing and accessing elements. It takes a range object rather than iterators and it allows us to more easily access the elements using integer indices this also gives us more fine grained control over the distribution of tasks to different threads. Now the fun part the joke "Why did the programmer quit his job? Because he didn't get arrays".

One more thing I have seen is using complex objects as parameters to process by `parallel_for_each` if you happen to be using a complicated object inside of your container you also need to take care of thread safety

```cpp
#include <vector>
#include <iostream>
#include <tbb/tbb.h>
#include <mutex>

class ComplexData {
public:
    int value;
    std::mutex mutex;

    ComplexData(int val) : value(val) {}

    void increment() {
        std::lock_guard<std::mutex> lock(mutex);
        value++;
    }
};

int main() {
    std::vector<ComplexData> data;
    for (int i = 0; i < 10; ++i) {
        data.emplace_back(i);
    }

    tbb::parallel_for_each(data.begin(), data.end(), [](ComplexData& item) {
        item.increment();
    });

    for (const auto& item : data) {
        std::cout << item.value << " ";
    }
    std::cout << std::endl;

    return 0;
}
```
Here you can see the inclusion of a `mutex` which is to avoid race conditions when multiple threads try to change the value of `value` of different objects at the same time if you're using an object that contains shared data always always protect it with mutexes or some other concurrency control mechanisms it will save you headaches down the road believe me I've seen a lot of headaches caused by race conditions

Now there are some things to keep in mind when using `parallel_for_each` or similar tools. The first is overhead sometimes parallelization can have a slight overhead it can take time to setup the threads and distribute tasks especially for very small tasks the overhead can outweigh the benefits of parallelization you need to find the right balance. Second data dependencies if one element needs the result of another element you need to be very careful and you can't just blindly parallelize everything It can create problems if you write to an item and some other item is trying to read from the first one it can become a mess of race conditions. Also debugging parallel code is generally harder than debugging sequential code so get familiar with the debugging tools because things can get complicated when you have many threads running at the same time

Resources to learn more about this I suggest starting with the book "Intel Threading Building Blocks" by James Reinders it's a deep dive into the TBB world it's an essential read if you plan on using this library a lot Also "Programming Concurrency on the Java Virtual Machine" by Venkat Subramaniam and "Java Concurrency in Practice" by Brian Goetz can provide good insights about multi threading in general and "C++ Concurrency in Action" by Anthony Williams also gives a good overview of multi threading in c++

So that's it in a nutshell `parallel_for_each` is a very powerful tool for performing parallel computations on collections but it requires careful thought and understanding of the problem. It's not a magic bullet it will not instantly make your code fast you also need to analyze your problem well and figure out if it is worth parallelizing your code. You also need to watch out for possible performance bottlenecks due to how you are going to access your data inside of the parallel processing context.
