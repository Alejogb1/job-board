---
title: "broadwell vs skylake cpu architecture?"
date: "2024-12-13"
id: "broadwell-vs-skylake-cpu-architecture"
---

 so you're asking about Broadwell versus Skylake CPU architectures right been there done that more times than I care to remember honestly These two Intel architectures they're like siblings who constantly argue over who's got the better toys It's a classic upgrade path debate and trust me I've been on both sides of this fence

Let's get the basics down first Broadwell that's the one that's a bit like the awkward middle child It was supposed to be the big leap to 14nm but it got delayed It kinda just existed for a short time and then Skylake just steamrolled its way in with a vengeance Skylake on the other hand that was the more planned out long term architecture It was a real shift after Haswell and even after Broadwell

From what I recall first time I bumped into this issue it was back in my university days 2015 or so I was working on a project that needed some serious processing power I remember thinking I'm gonna get a sweet laptop setup and I had two choices Broadwell and Skylake I'd been hyped up about this new smaller 14 nm process and this efficiency but in reality performance differences were minimal for the most part

I ended up going with a Broadwell laptop initially because it was a bit cheaper then the Skylake option but man did it get hot under heavy loads I was crunching some complex numerical simulations for fluid dynamics and the thermal throttling made my life hell It felt like the whole machine was going to melt on me I started seeing performance drops way sooner than I expected Then I got to see my friend's Skylake machine and oh boy the difference was noticeable Skylake just handled thermal throttling better it’s weird honestly since the die sizes are kinda the same and everything else is almost the same

So that pushed me into experimenting with this I wasn't happy with the performance I was getting I started digging deeper and looking into the architectural differences it wasn't just about the process shrink I started reading through Intel's documentation and then got into some academic papers on CPU architecture to really grasp what was going on

Let's look at some code examples just to illustrate the point Here's a simple C++ snippet for vector addition you know just to have something to work with

```cpp
#include <iostream>
#include <vector>
#include <chrono>
#include <numeric>

std::vector<double> vectorAddition(const std::vector<double>& a, const std::vector<double>& b) {
    if(a.size() != b.size()) {
        throw std::invalid_argument("Vectors must be the same size");
    }
    std::vector<double> result(a.size());
    for(size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] + b[i];
    }
    return result;
}

int main() {
    size_t size = 10000000; // Large vectors
    std::vector<double> vectorA(size, 1.0);
    std::vector<double> vectorB(size, 2.0);

    auto start = std::chrono::high_resolution_clock::now();
    std::vector<double> result = vectorAddition(vectorA, vectorB);
    auto stop = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

    std::cout << "Vector addition time: " << duration.count() << " ms" << std::endl;

     return 0;
}
```

Now run this same code on both a Broadwell and a Skylake machine under similar conditions and you'll probably see Skylake pull ahead It's subtle but it's there The key architectural improvements in Skylake like a more efficient memory controller and enhanced instruction fetch and decode units and the better thermal solution all play their part Now before someone says "But it's only vector addition" it's a great example that showcases micro architectural improvements and it's easy to set up.

Another thing I was dealing with was some python scripting for data analysis I used pandas a lot and things like groupby operations took longer on the Broadwell machine This was my next project that I was working on then

```python
import pandas as pd
import numpy as np
import time

# Creating some data
data = {'group': np.random.randint(0, 100, 1000000),
        'value': np.random.rand(1000000)}
df = pd.DataFrame(data)

# Timing the groupby aggregation
start_time = time.time()
aggregated_data = df.groupby('group').agg({'value': 'sum'})
end_time = time.time()

print(f"Pandas groupby aggregation time: {end_time - start_time:.4f} seconds")
```

Again you would see Skylake performing better in this scenario it felt like it was running on a better memory system and this makes a big difference when you are analyzing big data I remember running this script so many times to try and optimize my code on both my laptop with Broadwell and the desktop I managed to get my hands on with a Skylake CPU

I spent so much time trying to optimize the code I wrote for Broadwell but in the end I learned that at some point you have to accept that the limitations lie within the hardware itself That's the whole point of using better and more modern hardware the micro-optimizations could only go so far especially when it came to large data sets the hardware made the difference

And just for some more fun here's a simple example using some basic linear algebra in python I know this isn't groundbreaking but this will show the difference between different hardware and just because its simple doesn't mean its not important

```python
import numpy as np
import time

# Size of matrix
size = 2000
# Creating random matrices
matrix_a = np.random.rand(size, size)
matrix_b = np.random.rand(size, size)

start_time = time.time()
matrix_c = np.dot(matrix_a, matrix_b)
end_time = time.time()

print(f"Matrix multiplication time: {end_time - start_time:.4f} seconds")
```
Matrix math again a classic problem right? You’d probably think any modern CPU could handle it but no Skylake was able to make it run faster thanks to better memory bandwidth and faster execution of floating point operations. Broadwell struggled a bit more than its successor.

So what’s the takeaway you're not asking about the specifics but the overall feel here for the general user and developer and in my experience Skylake was definitely the more mature and efficient architecture While Broadwell was important stepping stone it never really hit the sweet spot that Skylake did Skylake had a more refined process much better power efficiency and some meaningful architectural improvements over Broadwell That better architecture allowed it to reach higher clock speeds with better thermals and also improve memory performance.

If you want to deep dive I suggest reading “Computer Architecture A Quantitative Approach” by Hennessy and Patterson it is like the bible for computer architecture stuff It covers all kinds of architectures and is a must-read if you're serious about this Another good one is “Modern Processor Design Fundamentals of Superscalar Processors” by John Paul Shen and Mikko H Lipasti its more focused on the micro architecture details of modern CPU’s It gets into the weeds of instruction pipelining and out-of-order execution which is something to check out if you are curious. There are also lots of academic papers you can find on IEEE Xplore and ACM Digital Library but thats a whole different rabbit hole

So yeah Broadwell versus Skylake Skylake wins hands down Its better in almost every single way the only exception being that Broadwell was available before Skylake and was cheaper at the time but the performance benefits of Skylake for the most part were worth the price upgrade But hey at least Broadwell kept my coffee warm for a little bit.

Hope this helps
