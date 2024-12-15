---
title: "If ten people call a function that counts up by 1 every time it is called, how are these requests handled if they're submitted to the same block?"
date: "2024-12-15"
id: "if-ten-people-call-a-function-that-counts-up-by-1-every-time-it-is-called-how-are-these-requests-handled-if-theyre-submitted-to-the-same-block"
---

alright, so we're talking about a classic concurrency issue, specifically with a shared resource – that counter – and how multiple threads or processes hammer on it at the same time. i've been down this rabbit hole more than a few times, and it's always a great reminder of why we need to think carefully about how we handle shared state.

first off, if we're talking about ten separate function calls all hitting the same block of code (let’s assume this block is part of a process), the outcome isn't going to be a consistent count of ten.  the reason is race conditions. imagine each call is trying to increment the counter, essentially:

1.  read the current value
2.  add one to that value
3.  write the new value back

now, this looks simple enough. but in the real world, processors don't do things serially unless forced to. let me give you a little bit of my history here, back in my early days as a junior engineer, i once wrote a really naive web app that kept track of visitor counts.  i remember thinking, "hey, it's just a single int, should be easy," which is the classic newbie move, and then i deployed it and noticed the counts were all wrong. sometimes it would be off by a little, other times by a lot. turns out, that "easy" increment operation wasn't so easy at all, in real life. so, let’s visualize, this is what actually happens: if call one reads the counter, say its zero and before it can write back, call two reads it also zero. call one writes back one. call two writes back one too, so now the count is one instead of two because the operation is not atomic. multiple threads can read the initial value at approximately the same time, and then increment and write back causing the values to get overwritten and lost. a classical case of a race condition. that code was like a clown show, frankly.

if it is within the same thread, it's not that much of a problem since threads are mostly executed sequentially within the main process although some interleaving of the code can happen but if ten independent threads or processes are calling that counter function, the problem gets much worse, like much much worse, the results are quite often unpredictable. so you can have threads taking turns but in no particular order or completely overwriting the values. that simple increment operation isn't atomic. you have multiple steps involved. the processor and the memory manager are working all at the same time and that read, add, and write can be interrupted by other threads or processes all trying to do the same thing. this is where bad stuff happens, and a great way to debug this is the dreaded “print statements”, which usually makes it even worse, like an observer effect kind of thing. we have to somehow force those threads to have a more civilized way to access the resource.

to deal with this problem, we need some kind of synchronization mechanism that forces all threads to access the counter only when its their turn, like a really polite queue. the main ideas here, are the mutual exclusion principle, meaning only one thread can access the resource at any given moment, and atomic operations, so the whole read-increment-write operation becomes uninterruptible and it happens entirely on its own without the risk of being overwritten.

here's a quick example of a not thread-safe code in python, it will illustrate the issue:

```python
counter = 0

def increment_counter():
  global counter
  counter += 1

if __name__ == '__main__':
    import threading
    threads = []
    for _ in range(10):
        t = threading.Thread(target=increment_counter)
        threads.append(t)
        t.start()
    for t in threads:
       t.join()
    print(f'counter value: {counter}') # the result will likely not be 10

```

if you run that code you will see that the counter is rarely ten. sometimes it will be nine, eight, and even seven. try running this a few times, each time you will probably get a different value less than ten, i mean less than ten most of the time. this is because of the race condition, multiple threads were racing to write to the same memory location at the same time and overwriting each other's values.

here's how you can fix it with locks:

```python
import threading

counter = 0
lock = threading.Lock()

def increment_counter():
  global counter
  with lock:
    counter += 1

if __name__ == '__main__':
    threads = []
    for _ in range(10):
        t = threading.Thread(target=increment_counter)
        threads.append(t)
        t.start()
    for t in threads:
       t.join()
    print(f'counter value: {counter}') # should now always be 10

```

here, the `threading.lock()` part ensures that only one thread can be within the `with lock:` block at any given time.  it's a way of saying "hold on everyone else, i'm using this counter now!".  this ensures that the increment operation happens atomically and without interruptions.

and you can also use atomic operations if the language and os support it:

```python
import multiprocessing
import multiprocessing.managers

class CounterManager(multiprocessing.managers.BaseManager):
    pass
CounterManager.register('get_counter')

if __name__ == '__main__':
    m = multiprocessing.Manager()
    counter = m.Value('i', 0)

    def increment_counter(shared_counter):
      shared_counter.value += 1

    processes = []
    for _ in range(10):
      p = multiprocessing.Process(target=increment_counter, args=(counter,))
      processes.append(p)
      p.start()
    for p in processes:
       p.join()
    print(f'counter value: {counter.value}') # should be 10

```

here, `multiprocessing.Value` creates a shared memory area where the counter is stored and it supports atomic operations for basic data types like integers. this avoids the need of a lock.

these are very simple examples, but the concept of synchronization is fundamental. it's why, in complex systems, we often see things like databases with their transaction systems, and other things like message queues. that counter is just a simple illustration of a much bigger issue.

as for resources, i recommend “operating system concepts” by silberschatz, galvin, and gagne, it's a classic in the operating system field and has excellent explanations for all these issues and all its implementations, you’ll probably find it in any university library. and if you want a deeper dive into concurrency patterns, "java concurrency in practice" by goetz et al is a solid resource, despite focusing on java, the principles are universal and it is like a bible on the matter. and, before i forget, it's like my dad used to say: “concurrency is like a bunch of kids fighting over a single toy, it’s better to have individual toys (single-threaded) than this mess”.
