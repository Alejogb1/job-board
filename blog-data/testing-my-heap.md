---
title: "testing my heap?"
date: "2024-12-13"
id: "testing-my-heap"
---

Okay so you're asking about testing your heap right I've been there done that more times than I care to admit let's break it down and get you sorted out heaps can be tricky little beasts if you're not careful

First things first let’s assume we're talking about a binary heap the one you typically use for priority queues or heap sort you know the classic min or max heap structure I'm not going to bore you with the theory because you probably know that already what I'm going to talk about is the nitty gritty of how to *actually* test it

Alright so you've implemented a heap presumably in C++ python java go or some language like that and you’re now facing the daunting task of verifying it works as expected right Been there dude trust me the debugging sessions are legendary we all have those stories of spending hours chasing down a simple off by one error in a heap implementation I remember once I spent a whole afternoon debugging a heap where the percolate up was broken because of a misplaced i++ it was epic in the worst way

So what are the common issues and how do we test for them let’s start with the basics

1 **Insertion Testing**

You need to check that after each insert the heap property is maintained This is the cornerstone of a good heap a simple insert and see what happens wont cut it so lets have that in mind

Here’s a typical test sequence you'd do in your test suite I’m assuming for this example that you want to do a simple max heap (the largest element at root)

```python
def test_insert_basic():
    heap = MaxHeap()
    heap.insert(10)
    assert heap.peek() == 10  # Root should be the inserted value
    heap.insert(5)
    assert heap.peek() == 10  # Root should still be 10 since its max heap
    heap.insert(20)
    assert heap.peek() == 20 # now 20 is bigger should be the root
    heap.insert(15)
    assert heap.peek() == 20 #still 20
    # ... more assertions after each insert
```

You'll want to check for edge cases too like inserting duplicates or very large very small numbers or negative numbers and don't forget about testing an empty heap insertion in the first place

2 **Extraction Testing**

This is the other part where heaps get really tested extracting the right element and keeping the tree intact after that is critical

Remember that in a max heap the `extract()` method should return the max element of the heap and then heapify again lets say you have the following initial heap \[20, 15, 10, 5] the first extraction should give you 20 then after the adjustment we should expect \[15, 10, 5] as well I once thought my `extract` was working only to later find out I was mixing up the parent and children index in the swap down part causing it to be an utterly broken heap It's the little things that get you

```python
def test_extract_basic():
    heap = MaxHeap([10, 5, 20, 15]) # Initialize with some values
    assert heap.extract() == 20
    assert heap.peek() == 15  #New max after extract
    assert heap.extract() == 15
    assert heap.peek() == 10
    assert heap.extract() == 10
    assert heap.peek() == 5
    assert heap.extract() == 5
    assert heap.is_empty() # Check if empty after all elements are removed
    # test empty heap removal

```

Make sure you test removing from an empty heap or a heap with only one element or try removing a large number of items that will tell you about your internal re-adjustments in the implementation

3 **Heapify Testing**

This is the function you call to "fix" the heap after insertion and removal or if you create a heap directly from a random array if this part is wrong the whole thing falls apart I tell you I had many nights trying to figure out why I could not get a correct heap from an initial array. I started to understand why my professor kept telling me that understanding pointers and indices is vital for these kind of data structures

```python
def test_heapify():
    arr = [3, 10, 1, 5, 7, 2]
    heap = MaxHeap(arr) # Create from an unsorted array
    assert heap.peek() == 10
    #you may add checks to verify the order of the rest of the elements
    #but its harder to assert all elements are in the right place just test the order and the property

```

Okay so this test uses an initial unsorted array and uses the heapify procedure to create a heap from it. Make sure to test with different array sizes, sorted array, reverse sorted arrays and arrays with duplicates

**Stress Tests**

Basic tests are fine to start but you need some stress tests after that We need to push the heap to its limits so how about inserting and extracting a very large number of random numbers? like a million numbers or more see how it handles a lot of insertions and extractions If the heap crashes or gets the wrong values then we know there are problems there I remember one time I ran a stress test with millions of elements and my program took about 10 minutes to finish and started swapping like hell it turns out I was using a recursive heapify that went deep and caused a stack overflow.

```python
def test_stress_test():
    heap = MaxHeap()
    import random
    for _ in range(100000):
         heap.insert(random.randint(1, 1000000))
    for _ in range(50000):
        heap.extract()

    # you can add checks after this
```

**Checking for Heap Property**

Another thing is after inserting or extraction and any other modification you want to be sure the heap is always following the basic principles a parent is always greater than the child in a max heap or smaller in a min heap. You should use an auxiliary function for it

```python
def is_max_heap(arr):
    n = len(arr)
    for i in range(n // 2):
        left_child = 2 * i + 1
        right_child = 2 * i + 2
        if left_child < n and arr[i] < arr[left_child]:
            return False
        if right_child < n and arr[i] < arr[right_child]:
            return False
    return True
```

You should implement it in a test suite method where you check this property every now and then after a complex sequence of instructions This will help you catch edge cases and wrong index calculations in your implementations I once spent a few hours doing that

**Testing Edge Cases**

We talked about inserting and removing from empty heaps or heaps with one element or duplicates but also consider some more cases consider testing with a very long input array or testing what happens when you try inserting and extracting the max of integer type consider using very large int or very low int types

**Testing for Time Complexity**

Heaps are supposed to be fast log(n) time is what you expect so it is not only about functionality but also speed of insertion extraction and heapify so you can use some timers in your tests to see the time it takes for example with the stress test I shared you can measure the insertion and extraction time and if its way off the expected complexity there is probably an issue in the implementation

**Resources**

Avoid the trap of relying solely on online tutorials While they can be helpful to get you started you will need a more robust knowledge on the concepts so go to actual books

*   **Introduction to Algorithms by Thomas H Cormen Charles E Leiserson Ronald L Rivest and Clifford Stein** is a classic in computer science. It has detailed explanations of heap data structure and algorithms and time complexity analysis this is a must read if you’re serious about your CS fundamentals
*   **Algorithms by Robert Sedgewick and Kevin Wayne** also is a must-have book this one is more practical and has actual implementation and detailed explanations on the subject of data structures including heaps
*  **Data Structures and Algorithm Analysis in C++ by Mark Allen Weiss** if you are using c++ then this book is very very helpful it's practical and theoretical at the same time. It's written by a professor in Computer Science

These resources will help you develop a good understanding of data structures and algorithms and will make your implementation of data structures more robust including testing and debug sessions

**Final Note**

Don’t just treat your tests as something you do after you finished implementing a data structure. In software engineering they will tell you to do test driven design and while I find that to be a very philosophical discussion that can be debated forever it is true that when you think about how you will test your component first it is easier to build a better component

Oh also one more thing the other day i was trying to test my heap and it was all good until I realize I was actually testing the python built in heapq module because I messed up with the name import you know like the "I've been there" meme type of thing. Be careful about it.
