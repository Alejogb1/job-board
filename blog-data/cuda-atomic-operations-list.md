---
title: "cuda atomic operations list?"
date: "2024-12-13"
id: "cuda-atomic-operations-list"
---

so you wanna know about CUDA atomic operations right Been there done that got the t-shirt and probably a few scars from debugging those little beasts Let's break it down because honestly they're crucial for anything serious with CUDA and they are not just a simple `++` operation on shared memory

I remember way back when I was first messing with CUDA I tried to do some kind of histogram calculation naively with shared memory It was a complete mess race conditions everywhere values all over the place I thought I was going crazy The debugger was useless because the problems were non-deterministic and well what can you say about non-deterministic problems They're hard I quickly learned that if you want to avoid data corruption in parallel operations you have to use atomics plain and simple

So what exactly are these things Well basically atomic operations are special instructions that guarantee a read-modify-write sequence to memory is performed in an indivisible manner What this means in simple terms is that no other thread can interfere in the middle of an atomic operation preventing race conditions that corrupt our data For example when multiple threads try to increment the same memory address at the same time a normal increment operation would have each thread load the current value compute its increment then write the new value but multiple threads might do this at the same time so one or more increments would be discarded causing an error atomic operations solve this with a single read modify write operation this cannot be interrupted guaranteeing consistency

Now CUDA offers a bunch of them for different data types and purposes and they all have these similar characteristics let's have a look

First off the most common ones are probably `atomicAdd` `atomicSub` `atomicExch` `atomicMin` and `atomicMax` you have them for integers `int` `unsigned int` `long long` `unsigned long long` and floating points `float` and `double` too

Here's how you use a simple integer atomic add

```cpp
__global__ void myKernel(int* sharedData, int valToAdd) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < arraySize) {
      atomicAdd(&sharedData[0], valToAdd);
    }
}
```

Here `sharedData` is a device memory address and `atomicAdd(&sharedData[0], valToAdd)` increments the value at memory address `sharedData[0]` by `valToAdd` in an atomic way it will work across all threads running this kernel this ensures that the increment is actually done and that the value will be the expected one

Now let's move to `atomicExch` this function swaps a value in memory with a new one so you can think of it like this `old_value = mem_value; mem_value = new_value;` this is all done as one single uninterrupted atomic operation This is useful for tasks like implementing simple locks for resources.

```cpp
__global__ void lockTest(int *lock, int *output)
{
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;

    if(thread_id == 0){
        int oldValue = atomicExch(lock, 1); // Try to acquire the lock
        if (oldValue == 0) {
            output[0] = 1;
        }
    } else {
        int oldValue = atomicExch(lock, 1); // Try to acquire the lock
        if (oldValue == 0) {
            output[1] = 1;
        }
    }
}

```

In this example the first thread that gets the lock will write the value to `output[0]` so if you are not using atomics this would not work correctly as both threads would execute and write a value but using atomics ensures that only one thread is writing to this resource or both depending on your problem

I remember once spending hours trying to debug some locking issue that was very similar to this and not using atomic operations correctly a very very long time ago haha

Then you have `atomicMin` and `atomicMax` they do what you'd expect they compare the value in memory with a new value and store either the minimum or maximum depending on which function you use this is useful for tasks like finding the min or max in a large array in parallel

```cpp
__global__ void minKernel(int* data, int* min_val) {
   int idx = threadIdx.x + blockIdx.x * blockDim.x;
   if (idx < arraySize) {
       atomicMin(min_val, data[idx]);
   }
}
```

This code snippet finds the minimum value of a data array in an atomic and parallel way

Now these atomics all work with global and shared memory locations There is a slightly different function `__syncwarp` that works for thread warp synchronization but that one is a completely different beast and a more complex topic on its own which is not in the scope of this question

One crucial note about atomics performance is that they are not free especially if there are a lot of threads accessing the same memory location This might result in bottlenecks because the device has to serialized all these operations this might also mean that if you have a huge block of threads trying to atomically add to a single address it might become slow consider this when designing your kernels avoid atomics when it's not absolutely necessary This is because atomics are usually done using the L2 cache so the device might serialize the operations leading to lower performance but is guaranteed to be correct

So when do you use atomic operations Well usually anytime that you have concurrent threads updating the same memory locations or resource If you want to create a histogram or implement a lock you will need atomics If you have a reduction operation and you have threads updating the same reduced value you will need atomics If you need to update a global variable and each thread will increment it then you will need atomics otherwise race conditions will appear and your program will result in a mess of non-deterministic behaviour It can be hard to pinpoint these bugs so try to use atomics when needed and avoid when you are not sure if you actually need them

The most common mistake is thinking that just because you have memory access there is no race condition in the code I've made that mistake way more than once and those are by far the worst bugs to debug trust me And I still make mistakes sometimes we all do so use those static analysis tools to help you prevent bugs like these they can be a real lifesaver.

Now another important thing to consider is the atomics operation support based on your GPU architecture There are some differences between the different architectures older architectures will lack some more recent atomic operation support for example some floating point atomic operations were not available in older devices so you should keep that in mind and use your device query system in CUDA to check the availability of atomic operations if you have some older device

If you want to dive deeper into atomic operations I would recommend reading the CUDA programming guide there you will find all the details and the latest features but also some interesting books like "CUDA by Example" and "Programming Massively Parallel Processors" these are good starting points to dig more into this area and to improve your GPU programming skills

Ok so that's about it for now hope this explanation helps you with your CUDA journey and feel free to ask anything else you need good luck with your code and remember debugging is a science an art and a very long dark tunnel
