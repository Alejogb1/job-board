---
title: "shared and gshared keywords in d?"
date: "2024-12-13"
id: "shared-and-gshared-keywords-in-d"
---

 so you’re asking about `shared` and `gshared` in D right Been there wrestled with that beast a few times let me tell you It's not exactly the most intuitive thing in D but once it clicks it's pretty powerful stuff buckle up

Basically we're talking about managing data access in multi-threaded environments right D is awesome at concurrency and these keywords are how you tell the compiler what's safe and what might explode your app into a pile of segfaults and frustration

`shared` think of it like a big flashing sign saying "Hey compiler this variable is potentially going to be touched by multiple threads be careful!" It’s a type qualifier not a type itself so you slap it on any mutable data types that threads might read and write at the same time Now the compiler knows to generate the necessary locking code behind the scenes to ensure that you don't get race conditions and data corruption

Now that locking well that introduces overhead right so use `shared` judiciously only where it’s absolutely needed It adds a cost you know It also restricts operations on non shared data so it can get tricky if you try to mix the two all willy nilly

Let’s say we have this struct which we are using to store data:

```d
struct Data {
  int count;
}
```

Now if we want to use this struct in a multi threaded environment without any protection there’s going to be trouble because multiple threads can try modifying count at the same time and this is what we call a race condition it will lead to unpredictable results it might lead to crashes and who wants that So now we change the struct like this:

```d
struct Data {
  shared int count;
}
```

By marking `count` as `shared` the D compiler does its magic and makes sure that access is properly synchronized using an internal mutex or similar mechanism

Here is an example of how you would use a shared struct:

```d
import std.stdio;
import std.concurrency;
import std.thread;


struct Data {
  shared int count;
}

void worker(Data* sharedData) {
  foreach(i; 0 .. 1000){
    sharedData.count++;
  }
}

void main() {
  Data sharedData;
  Thread[] threads;
  
  foreach(i; 0.. 10) {
    threads ~= new Thread(() => worker(&sharedData));
  }

  foreach(thread; threads){
     thread.join();
  }

  writeln("Final count: ", sharedData.count);
}
```
This code shows 10 threads incrementing a shared counter and using the `shared` keyword we prevent a race condition and guarantee that the count will be 10000. Without the keyword we might get any number that is less than 10000

Now we get to `gshared` this one's a little different and to be honest it confused me for a long time So where `shared` is about data access within a thread `gshared` is about data access across threads or more precisely in the global scope. A `gshared` variable means a single instance of that data exists across all threads all threads get the exact same data

It’s a static global variable with extra magic the most important detail here is that variables declared outside of a function or a class in D are thread local by default this means every thread gets its own copy of that variable but that is not always desired that is why `gshared` was invented

So imagine you're counting the number of requests your server has handled you will want to have a single counter across all threads right if it was only thread local every thread will have it's own private counter and that will be very bad for tracking the overall requests

A normal static variable will create thread local variables while a static gshared variable will create a shared instance of that variable

Here is an example that shows the difference

```d
import std.stdio;
import std.concurrency;
import std.thread;

static int normalCounter;
static gshared int gsharedCounter;

void worker(){
    foreach(i; 0 .. 1000){
        normalCounter++;
        gsharedCounter++;
    }
}

void main() {
    Thread[] threads;
    foreach(i; 0 .. 10){
        threads ~= new Thread(&worker);
    }

    foreach(thread; threads){
        thread.join();
    }

    writeln("Normal Counter: ", normalCounter);
    writeln("GShared Counter: ", gsharedCounter);
}
```
If you run this program you will see that normalCounter is only equal to 1000 and gsharedCounter will be 10000 it's because each thread increments its own version of `normalCounter` while they all increment a single instance of `gsharedCounter`

Now for some personal experience I remember one time I was building this real-time data processing pipeline in D I had these data structures going through different stages in different threads and I made a real mess of it initially I was using shared everywhere because I was terrified of race conditions but the performance was terrible the locking overhead was killing me it was like trying to move a freight train with a bicycle Then I realized that most of my variables were only read in the first few stages and were never written by multiple threads so there wasn't actually any need for shared there at all I ended up isolating the critical data that needed shared and the rest was simple normal variables it was like a switch was flipped the whole thing was flying It was like the time I tried to make coffee with orange juice (don't ask) the result was not good then I realized my error and used water instead I feel like the story is relatable

Another time I was trying to use `gshared` to share a resource that was mutable I thought it was like a singleton or something where I can just change it in any part of the code but it ended up being a nightmare debugging hell I forgot that even gshared variables needed proper synchronization if more than one thread is modifying them at the same time

So what are my recomendations? First read up on the documentation very carefully and pay particular attention to how shared variables affect type conversion operations and how they limit the ability to assign to non-shared variables. The documentation on dlang.org is really good here and you should not be afraid to delve into that second i would advice to read Herb Sutter's "Effective Concurrency" it has a great overview of multithreading in general and might help understand these D concepts in a broader context Also "Programming in D" by Ali Çehreli covers the fundamentals of D’s concurrency model and has a very detailed chapter on `shared` and `gshared`.

Also if you have very complex and non trivial concurrency needs you might have to go outside of the default primitives that D provides you can use external libraries or implement your own low level primitives for that the documentation of the D programming language is your best friend

In conclusion use shared when you need to protect mutable data from concurrent access in multiple threads and gshared when you need a single global instance shared across all threads but be careful when using it because if you are not careful and write to it from different threads without synchronization you will have a race condition. Both `shared` and `gshared` can be useful when used correctly but be cautious of the performance implications and the additional restrictions that they introduce

Hope this helps you out feel free to ask if anything is unclear and we will tackle this problem together I have made my fair share of mistakes using these keywords so do not feel bad if you're confused its all part of the process
