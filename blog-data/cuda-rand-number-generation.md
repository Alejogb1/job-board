---
title: "cuda rand number generation?"
date: "2024-12-13"
id: "cuda-rand-number-generation"
---

so you’re asking about CUDA random number generation eh Been there done that probably more times than I care to admit. Let me tell you it’s not as straightforward as a simple `random()` call you’d find in say Python or Javascript. You’re dealing with parallel computation here and that changes everything. You can't just blindly generate random numbers in parallel without thinking about potential issues like repeatability and statistical quality. Believe me I’ve stumbled on these problems hard back in my early CUDA days.

I remember vividly this project it was a computational fluid dynamics thing where we needed a ton of random numbers to simulate turbulent flows. I just tossed `rand()` calls in my kernel all willy nilly. Man was that a disaster. The results were all over the place the simulation didn’t converge and everything looked weird. Took me like two days of head scratching and coffee guzzling to figure out that my “random” numbers were basically synchronized garbage. Different threads were getting correlated random sequences and that messed everything up. I learned my lesson that day trust me.

Ok so first things first standard C rand function? Avoid it at all costs. It’s just not designed for parallel execution on a GPU. You’ll get those correlations I mentioned plus performance will be terrible. CUDA provides a few different ways to get good quality random numbers on the GPU and let’s go through some of them.

The go-to library for most people is the cuRAND library. It’s designed specifically for CUDA and gives you a ton of different random number generators to choose from like Mersenne Twister XORWOW and Philox. The Mersenne Twister is a popular choice because its quality is pretty good. The Philox is another good option especially if you want performance but be wary that they are not all equivalent each has its strengths and weaknesses.

Here's a very basic code snippet showing how to use cuRAND for generating some random floats on the GPU using the Mersenne Twister:

```c++
#include <curand.h>
#include <cuda_runtime.h>
#include <iostream>

__global__ void generateRandomFloats(float* output, size_t count) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < count) {
       curandState_t state;
       curand_init(1234 + index, 0, 0, &state); // Seed needs to be different per thread

       output[index] = curand_uniform(&state);
    }
}

int main() {
    size_t count = 1024; // Number of random numbers
    float* hostOutput = new float[count];
    float* deviceOutput;
    cudaMalloc((void**)&deviceOutput, count * sizeof(float));


    int threadsPerBlock = 256;
    int blocksPerGrid = (count + threadsPerBlock - 1) / threadsPerBlock;


    generateRandomFloats<<<blocksPerGrid, threadsPerBlock>>>(deviceOutput, count);


    cudaMemcpy(hostOutput, deviceOutput, count * sizeof(float), cudaMemcpyDeviceToHost);

    for(int i=0; i < 10; i++){
        std::cout << hostOutput[i] << std::endl;
    }


    cudaFree(deviceOutput);
    delete[] hostOutput;

    return 0;
}
```

A few key things to notice in this example. First you need to initialize a `curandState_t` per thread using `curand_init`. And that seed there? Very very important. You can not just use a global seed. You’ll have the same problem I had before threads will be synchronized. You have to seed each state differently for each thread to get actual independent random sequences that is the reason why we add `index` to the seed. Otherwise you’ll get the same sequence of numbers from every thread. Then you can call `curand_uniform` or other functions like `curand_normal` or whatever distribution you need. Finally you need to remember to cleanup after you're done.

Another thing is that when you generate a sequence of random numbers from a fixed seed the sequence is actually deterministic. This is important because for debugging you can replicate sequences. If you’re doing Monte Carlo simulations or something like that that's invaluable.

Now lets say you need more control over the random number generation process. You want to implement a custom RNG that is not part of cuRAND. No problem. The simplest option you can go to implement is a linear congruential generator LCG. It’s old school but fast and easy to implement and it’s not bad if you know what you're doing and you don't require high quality numbers. Here is how you could implement something like that in CUDA:

```c++
__device__ unsigned int lcg(unsigned int& seed) {
    unsigned int a = 1664525;
    unsigned int c = 1013904223;
    seed = a * seed + c;
    return seed;
}

__global__ void generateLCGIntegers(unsigned int* output, size_t count, unsigned int seedBase) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if(index < count){
       unsigned int seed = seedBase + index;
       output[index] = lcg(seed);
    }


}
```
```c++
#include <cuda_runtime.h>
#include <iostream>
#include <vector>


__device__ unsigned int lcg(unsigned int& seed);
__global__ void generateLCGIntegers(unsigned int* output, size_t count, unsigned int seedBase);



int main() {
    size_t count = 1024;
    std::vector<unsigned int> hostOutput(count);
    unsigned int* deviceOutput;
    cudaMalloc((void**)&deviceOutput, count * sizeof(unsigned int));
    unsigned int seedBase = 1234; // Base seed

    int threadsPerBlock = 256;
    int blocksPerGrid = (count + threadsPerBlock - 1) / threadsPerBlock;

    generateLCGIntegers<<<blocksPerGrid, threadsPerBlock>>>(deviceOutput, count, seedBase);

    cudaMemcpy(hostOutput.data(), deviceOutput, count * sizeof(unsigned int), cudaMemcpyDeviceToHost);


    for(int i=0; i < 10; i++){
         std::cout << hostOutput[i] << std::endl;
    }

    cudaFree(deviceOutput);
    return 0;
}
```
Again we're seeding each thread with different values. The quality of LCGs is not as good as Mersenne Twister though but for some simple uses cases you can get away with them and they are fast to calculate.

Finally if you are feeling very ambitious you could implement a PCG generator. It has good statistical properties and it’s also very fast to compute. Here is a simple example:
```c++
__device__ unsigned int pcg32(unsigned long long& state) {
    unsigned long long oldstate = state;
    state = oldstate * 6364136223846793005ULL + 1442695040888963407ULL;
    unsigned int xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
    unsigned int rot = oldstate >> 59u;
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

__global__ void generatePCGIntegers(unsigned int* output, size_t count, unsigned long long seedBase) {
   int index = blockIdx.x * blockDim.x + threadIdx.x;

    if(index < count){
        unsigned long long state = seedBase + index;
        output[index] = pcg32(state);
    }

}
```
```c++
#include <cuda_runtime.h>
#include <iostream>
#include <vector>


__device__ unsigned int pcg32(unsigned long long& state);
__global__ void generatePCGIntegers(unsigned int* output, size_t count, unsigned long long seedBase);

int main() {
    size_t count = 1024;
    std::vector<unsigned int> hostOutput(count);
    unsigned int* deviceOutput;
    cudaMalloc((void**)&deviceOutput, count * sizeof(unsigned int));
    unsigned long long seedBase = 1234; // Base seed


    int threadsPerBlock = 256;
    int blocksPerGrid = (count + threadsPerBlock - 1) / threadsPerBlock;

    generatePCGIntegers<<<blocksPerGrid, threadsPerBlock>>>(deviceOutput, count, seedBase);
    cudaMemcpy(hostOutput.data(), deviceOutput, count * sizeof(unsigned int), cudaMemcpyDeviceToHost);


    for(int i =0; i < 10; i++){
          std::cout << hostOutput[i] << std::endl;
    }

    cudaFree(deviceOutput);

    return 0;
}
```
This is a slightly more complex RNG and it’s slightly slower than the LCG but it has a better statistical quality than LCG. As a rule of thumb if you don’t know what you are doing cuRAND is the way to go but if you are feeling adventurous you can try to roll your own but make sure you know what you are doing and your numbers are not too correlated.

Now there's a lot more stuff that can be said about this topic but it’s getting a bit lengthy. You can explore the cuRAND documentation which has tons of details for all of its functions and some really good advice. Also if you want to understand these things more deeply try "Numerical Recipes" by Press et al it's a classic on computational methods and has a good chapter on random number generation. For the GPU specific aspects "CUDA by Example" by Sanders and Kandrot is also a great resource.

Oh one last thing there was a guy who tried using the thread id as his random number once he said it was surprisingly uniform but then when we went to another machine he said that all his numbers were the same it was a classic case of the “what do you mean it works on my machine” problem haha. Always check your random numbers and have a good day.
