---
title: "microbenchmarking code performance measurement?"
date: "2024-12-13"
id: "microbenchmarking-code-performance-measurement"
---

Alright let's talk microbenchmarking I've been down this rabbit hole more times than I care to admit and it's always a wild ride honestly

The question of microbenchmarking code performance measurement is deceptively simple It's not just about slapping a timer around some code and calling it a day oh no It's about understanding the nuances of your machine your compiler your language and the way your code interacts with them all

First things first what do I mean by microbenchmarking? We're not talking about measuring the performance of an entire application or a large system No this is about getting down to the nitty-gritty of individual functions loops even individual lines of code We want to see how fast this tiny part of the program is executing and whether it's doing it efficiently

Why is this important? Well sometimes the seemingly smallest change can have a surprisingly large impact especially in critical sections of your application A few extra nanoseconds here or there might not seem like much but when you're doing that calculation millions or billions of times those nanoseconds add up to real seconds minutes or even hours So we need to know where the time is going and identify bottlenecks

I remember one particularly nasty case I had back in my early days I was working on some graphics processing code and this one particular function was just killing my frame rate like seriously a slideshow I mean like I would not see such frame rates even on a Commodore 64 if it had the graphic capabilities I spent ages looking for an algorithm issue optimizing loops rewriting code like a maniac I mean I even asked my mom if she knew about optimization and performance but sadly she just recommended that I should go to the doctor because she thought that I was overdoing with my work load I eventually I used microbenchmarks and I found that the function itself wasn't the issue it was a single line of code an innocent-looking allocation that was happening repeatedly and was the issue My memory was a mess because of it and it cost me a lot of performance

So what does that experience mean for us? Well it means that proper microbenchmarking is a must and I mean a must not only to understand but to optimize our code and it has to be done with care and there are some important things to consider

First you need to isolate your code You can't just throw some random code into a test environment and hope for reliable results You need to create a controlled environment and you need to make sure that no other external factors influence the performance results Try to isolate the specific piece of code you're testing from the rest of your application

You need to think about the time you are measuring You need a high-resolution timer to measure small time differences You need to make sure that the timer is accurate and precise and that has a low overhead

Then there's the warmup issue your code might run slower the first time it executes due to things like JIT compilation and caching So you need to perform multiple runs and throw away the initial results The only thing you need to measure is the steady-state performance

Also be aware of the compiler optimizations the compiler can sometimes perform optimizations on your code that can affect the performance of the code You have to be careful when you measure performance in a development mode of your compiler rather than in release mode of the compiler

And don't forget about hardware dependencies microbenchmarks can be very sensitive to hardware specific behaviors So run your benchmarks on the target hardware or the platform you want to run this code

Now let's get to the actual code examples I will give some very simple examples and keep the examples concise so you understand the idea

```python
import time

def function_to_benchmark():
    sum_ = 0
    for i in range(1000000):
        sum_ += i

def benchmark(func):
    start_time = time.perf_counter()
    func()
    end_time = time.perf_counter()
    return end_time - start_time

if __name__ == "__main__":
    time_taken = benchmark(function_to_benchmark)
    print(f"Time taken: {time_taken:.6f} seconds")

```

This is very basic Python example of microbenchmarking and uses `time.perf_counter()` for a high-resolution timer You can modify it to include a warmup loop and average multiple runs. Also there are python libraries that are much better for microbenchmarking such as `timeit` and `perf`

Here's an example of a C++ microbenchmark:

```cpp
#include <iostream>
#include <chrono>

using namespace std;
using namespace chrono;

void functionToBenchmark() {
    long long sum_ = 0;
    for (long long i = 0; i < 1000000; ++i) {
        sum_ += i;
    }
}

double benchmark(void (*func)()) {
    auto start_time = high_resolution_clock::now();
    func();
    auto end_time = high_resolution_clock::now();
    duration<double> time_taken = end_time - start_time;
    return time_taken.count();
}

int main() {
    double time_taken = benchmark(functionToBenchmark);
    cout << "Time taken: " << time_taken << " seconds" << endl;
    return 0;
}
```

This C++ example uses the chrono library for high-resolution timing Just like in python we need to modify this code to include a warmup and multiple runs also use the compiler optimizations for release mode not development mode. As you can see the idea remains the same between both of the examples use a good timer measure the function that you want to benchmark

Now for one in javascript

```javascript
function functionToBenchmark() {
    let sum = 0;
    for(let i = 0; i < 1000000; i++){
        sum += i;
    }
}


function benchmark(func){
  const start_time = performance.now();
  func();
  const end_time = performance.now();
  return (end_time - start_time) / 1000;
}


if (typeof window !== 'undefined') {
  console.log("Time taken:", benchmark(functionToBenchmark), "seconds");
} else if (typeof process !== 'undefined') {
  console.log("Time taken:", benchmark(functionToBenchmark), "seconds");
}
```

This javascript code is similar to the other two examples and it uses `performance.now()` for high-resolution timing the idea is the same you need to do a warmup loop and multiple runs and to run this code in a node environment or a browser.

Also be careful when comparing the measurements between different languages or technologies there are many factors that you have to take into consideration and you can not do a 1 to 1 comparison and you can not say one is faster than the other

Microbenchmarking is not just about running a benchmark you need to understand your results and interpret them correctly I mean if you run your benchmarks and you have the results now what? you have to be able to understand what it means in terms of your code If your benchmarks are fluctuating and they are not steady there might be something going wrong

And here's a little tech joke for you why did the microbenchmark break up with the code because it said there is not enough time for this

Now a big point is not just to optimize individual code segments but also to optimize your algorithms and data structures that is a bigger task but it is part of microbenchmarking but not in the micro detail but it will influence your microbenchmarks.

I recommend "Agile development methods" by Craig Larman and "Code Complete" by Steve McConnell for general software engineering principles that might be helpful also for optimizing your algorithms I recommend "Introduction to Algorithms" by Thomas H Cormen that is a must for data structures I would recommend "Data structures and algorithm analysis in C++" by Mark Allen Weiss.

So I know this a long response and has many information but microbenchmarking is a big topic and there are many nuances. Always remember to isolate your code use high-resolution timers warm-up your code and think about all of the factors that I have mentioned I mean microbenchmarks might take a lot of your time but the information is very valuable and it might save you hours of optimizing in the wrong place.
