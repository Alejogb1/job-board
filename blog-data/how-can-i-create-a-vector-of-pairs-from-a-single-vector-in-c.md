---
title: "How can I create a vector of pairs from a single vector in C++?"
date: "2024-12-23"
id: "how-can-i-create-a-vector-of-pairs-from-a-single-vector-in-c"
---

Okay, let's tackle this. It's a common enough task, transforming data structures, and one I've certainly encountered more than once in my time. Back in my early days working on a large data processing pipeline, we had a raw data feed that arrived as a flat vector of integers, and we frequently needed to process them as adjacent pairs for subsequent analysis. It felt cumbersome dealing with individual elements, so I had to figure out a clean way to consistently turn that flat vector into a vector of pairs. The core idea here is to iterate through the original vector, taking elements two at a time and creating `std::pair` objects. Simple in theory, but you need to handle boundary conditions and decide what to do with odd-sized vectors.

The approach we'll take here relies heavily on the standard library, particularly `std::vector` and `std::pair`. We’ll also be using iterators which are the backbone of generic algorithms in c++.

First, let's assume a basic scenario where you want to pair up consecutive elements, and ignore the last element if the original vector has an odd number of elements. This is the simplest implementation.

```cpp
#include <iostream>
#include <vector>
#include <utility> // for std::pair
#include <stdexcept> // for exceptions

std::vector<std::pair<int, int>> create_pairs_simple(const std::vector<int>& input) {
    std::vector<std::pair<int, int>> result;
    for (size_t i = 0; i < input.size() - 1; i += 2) {
        result.push_back(std::make_pair(input[i], input[i+1]));
    }
    return result;
}

int main() {
  std::vector<int> numbers = {1, 2, 3, 4, 5, 6};
  auto pairs = create_pairs_simple(numbers);

  for(const auto& pair : pairs) {
    std::cout << "(" << pair.first << ", " << pair.second << ") ";
  }
    std::cout << std::endl; // outputs: (1, 2) (3, 4) (5, 6)
  
  std::vector<int> odd_numbers = {1, 2, 3, 4, 5};
  auto odd_pairs = create_pairs_simple(odd_numbers);

  for(const auto& pair : odd_pairs) {
      std::cout << "(" << pair.first << ", " << pair.second << ") ";
  }
  std::cout << std::endl; // outputs: (1, 2) (3, 4)

  return 0;
}
```

In this `create_pairs_simple` function, we iterate through the vector with a step of two, taking `input[i]` and `input[i+1]` and packaging them into a `std::pair`. Notice the `i < input.size() - 1` condition. This ensures that we don't attempt to access an element beyond the vector's bounds. This implementation is straightforward and suitable for many cases where you are fine with discarding the odd element. I used this particular variation for a data preprocessing stage which used only even number of samples.

Now, let's say we *don't* want to discard the last element in an odd-sized vector. Instead, we'd want to pair it with something default or handle it as a special case. We can adapt the function. Suppose that in this new scenario, when the vector has an odd number of elements I want to pair the last element with the element immediately before it.

```cpp
#include <iostream>
#include <vector>
#include <utility>

std::vector<std::pair<int, int>> create_pairs_with_last(const std::vector<int>& input) {
    std::vector<std::pair<int, int>> result;
    size_t size = input.size();
    for (size_t i = 0; i < size - 1; i += 2) {
         result.push_back(std::make_pair(input[i], input[i+1]));
    }

    if(size % 2 != 0) {
        result.push_back(std::make_pair(input[size - 2], input[size-1]));
    }

    return result;
}

int main() {
    std::vector<int> numbers = {1, 2, 3, 4, 5};
    auto pairs = create_pairs_with_last(numbers);
    for(const auto& pair : pairs) {
        std::cout << "(" << pair.first << ", " << pair.second << ") ";
    }
    std::cout << std::endl; // outputs: (1, 2) (3, 4) (4, 5)
    return 0;
}
```

Here, I added a conditional after the main loop to check if the vector size is odd. If so, I create a pair containing the second to last element and the last element before adding it to the result vector. This approach ensures that no element is left behind, albeit with a choice on how to handle the odd element. This adaptation I found useful in cases where the order of the data was very important, like a time-series data where having the last data point is meaningful.

But, what if you wanted to pair the final element with some default value, such as zero? or perhaps you want to throw an exception? This is a common practice if the data must be even sized, and you want the program to immediately exit should you receive an odd-sized vector. Let's look at that version, which also incorporates basic error handling to make it a more production-ready utility.

```cpp
#include <iostream>
#include <vector>
#include <utility>
#include <stdexcept>

std::vector<std::pair<int, int>> create_pairs_with_default(const std::vector<int>& input, int default_val) {
    std::vector<std::pair<int, int>> result;
    size_t size = input.size();

    if(size % 2 != 0){
        throw std::runtime_error("Input vector must have an even number of elements");
    }

    for (size_t i = 0; i < size - 1; i += 2) {
         result.push_back(std::make_pair(input[i], input[i+1]));
    }

    return result;
}

int main() {
    std::vector<int> numbers = {1, 2, 3, 4, 5, 6};
    auto pairs = create_pairs_with_default(numbers, 0);
    for(const auto& pair : pairs) {
        std::cout << "(" << pair.first << ", " << pair.second << ") ";
    }
    std::cout << std::endl; // Outputs: (1, 2) (3, 4) (5, 6)

    std::vector<int> odd_numbers = {1, 2, 3, 4, 5};
    try{
      auto odd_pairs = create_pairs_with_default(odd_numbers, 0);
    } catch (const std::runtime_error& e) {
      std::cerr << "Error: " << e.what() << std::endl; // Outputs: Error: Input vector must have an even number of elements
    }

    return 0;
}

```

In this final version, I've added an exception check which forces us to be mindful of incorrect data being processed. The function `create_pairs_with_default` throws an exception if the size of the vector is odd, making it clear when the input doesn't conform to the expectations. These approaches handle edge cases such as odd-sized vectors explicitly, ensuring predictable behavior in your programs. This approach is especially helpful when building applications where robustness and data correctness are critical.

For a deeper dive into the standard library components, I'd recommend 'Effective Modern C++' by Scott Meyers. It provides a practical guide to modern C++, including best practices for using `std::vector`, iterators, and generic programming techniques. Also, for anyone looking to get better at algorithmic complexity I cannot recommend enough the book 'Introduction to Algorithms' by Thomas H. Cormen, Charles E. Leiserson, Ronald L. Rivest, and Clifford Stein. It has been a bedrock in my understanding of algorithms.

In my experience, the most important thing is to choose the implementation that best suits the context and to always consider edge cases and potential errors. Choosing the method which does not discard data or throw errors may be more important than outright simplicity for a specific use case. These approaches I've discussed here should provide a solid starting point. I've used variations of all of them in various projects with success.
