---
title: "reverse a vector without rev?"
date: "2024-12-13"
id: "reverse-a-vector-without-rev"
---

 so you wanna reverse a vector right without using the built in `rev` function I've been there dude more times than I'd like to admit You'd think its simple right but then you get some weird edge case or some crazy performance bottleneck I remember back in college when I was doing a project on image processing we had to do a lot of matrix manipulations and reversing vectors came up like all the time It was this stupid algorithm where we had to flip a color array back and forth for some convolution filter thing and well using the standard reverse was like super slow especially when working with huge images plus we were trying to be cool by implementing it ourselves you know the whole "not needing libraries" phase

Anyway so the first thing you think of is a loop right just iterate backwards and copy everything to a new vector that’s fine simple easy to read maybe you use a for loop or maybe a while loop whatever floats your boat something like this works

```cpp
#include <iostream>
#include <vector>
#include <algorithm> // for std::copy
#include <iterator> // for std::back_inserter

std::vector<int> reverseVectorBasic(const std::vector<int>& input) {
    std::vector<int> reversed;
    for (int i = input.size() - 1; i >= 0; --i) {
        reversed.push_back(input[i]);
    }
    return reversed;
}


int main() {
    std::vector<int> original = {1, 2, 3, 4, 5};
    std::vector<int> reversed = reverseVectorBasic(original);

    std::cout << "Original vector: ";
    for (int val : original) std::cout << val << " ";
    std::cout << std::endl;

    std::cout << "Reversed vector: ";
    for (int val : reversed) std::cout << val << " ";
    std::cout << std::endl;
    return 0;
}
```
This is like the most basic solution possible works good for small vectors but it's kinda inefficient it creates a new vector allocates memory for each element and then copies stuff this is  for most use cases but when things get big or you have tight loops like I had back with my image processing thing then this solution suffers from the unnecessary memory allocations so it slows things down a lot especially when you have a giant vector like the pixels of a huge image

Then I was messing around with pointers you know trying to get fancy you could iterate two pointers one from the start and one from the end then swap them and keep moving them towards the middle that way you can reverse the vector in place you're not allocating new memory at all like so:

```cpp
#include <iostream>
#include <vector>
#include <algorithm>

void reverseVectorInPlace(std::vector<int>& vec) {
    int start = 0;
    int end = vec.size() - 1;
    while (start < end) {
        std::swap(vec[start], vec[end]);
        start++;
        end--;
    }
}

int main() {
    std::vector<int> original = {1, 2, 3, 4, 5};
    std::vector<int> original2 = {1, 2, 3, 4, 5, 6};

    reverseVectorInPlace(original);
    reverseVectorInPlace(original2);

    std::cout << "Reversed vector 1: ";
    for (int val : original) std::cout << val << " ";
    std::cout << std::endl;

        std::cout << "Reversed vector 2: ";
    for (int val : original2) std::cout << val << " ";
    std::cout << std::endl;

    return 0;
}
```

This method is usually faster cause it is not creating a copy no memory allocations and it reverses things in place I ended up using something similar for my image processing project although I was dealing with some raw memory but the concept was the same This method is good for large vectors as it is more performant especially when you are working in a memory constraint environment or when you need to have the best possible speed for your algorithm

Now I remember I was reading some book on algorithms and I came across this recursive solution for reversing things I was like why would you even do this its the worst possible solution right because recursion is usually bad when it comes to performance and dealing with stacks and stuff but sometimes it can be very elegant and show you how the underlying logic works so heres the code

```cpp
#include <iostream>
#include <vector>
#include <algorithm>


void reverseVectorRecursiveHelper(std::vector<int>& vec, int start, int end) {
    if (start >= end) {
        return;
    }
    std::swap(vec[start], vec[end]);
    reverseVectorRecursiveHelper(vec, start + 1, end - 1);
}

void reverseVectorRecursive(std::vector<int>& vec) {
    reverseVectorRecursiveHelper(vec, 0, vec.size() - 1);
}

int main() {
    std::vector<int> original = {1, 2, 3, 4, 5};
    reverseVectorRecursive(original);

    std::cout << "Reversed vector: ";
    for (int val : original) std::cout << val << " ";
    std::cout << std::endl;
    return 0;
}
```

This recursive implementation uses a helper function its kinda a divide and conquer situation you keep swapping the elements from the ends and then make a recursive call to do the same for the rest of the vector it does not need extra storage like our basic first method since it modifies the input vector and it has a very nice simplicity to it when you look at it I never use it though I admit but its good to know. My boss however thought it was the best thing since slice bread for code readability.

So  which method to use I would say it depends on the specific case for most situations the in place swapping is the fastest and most memory efficient method I would use that for any production code or when dealing with large datasets the basic looping copy method is good for when you need a copy of the reversed vector and don’t want to modify the original so it is a very reasonable approach and the recursive solution is only useful if you are doing academic stuff or wanna flex in front of someone who probably is less knowledgeable about this kind of things

I have another funny story from when I was doing some kernel programming the compiler was acting all funny and kept reversing the vectors I needed to have in order I was going nuts thinking my code was the problem only to find out that it was some optimization bug with the compiler after wasting an entire afternoon on the issue that was my funny code story for the day.

Now for the resources if you are diving deep in algorithms I would really recommend "Introduction to Algorithms" by Thomas H Cormen et al that is a must have for all CS people or for general coding performance I would look up "Effective C++" by Scott Meyers or any of the books he wrote they are all amazing in explaining those finer details of C++ that can make or break your code in terms of optimization but for general algorithms knowledge then go for Cormen's book it will give you a lot of very important and deep understanding on data structures and algorithms. There is a paper but I cannot find it right now it is some old research about the memory layout and how it affects performance but generally the book resources should be good enough

So yeah that's my take on reversing a vector without using `rev` it’s a simple problem with a surprisingly lot of different ways to tackle it hopefully this answer helps someone down the road I had to learn all this stuff the hard way you know and there was no stackoverflow back then
