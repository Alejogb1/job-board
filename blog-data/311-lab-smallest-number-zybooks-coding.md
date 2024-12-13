---
title: "3.11 lab smallest number zybooks coding?"
date: "2024-12-13"
id: "311-lab-smallest-number-zybooks-coding"
---

Okay so you’re looking at the classic "find the smallest number" problem specifically from Zybooks coding lab 311 I've been there done that more times than I care to admit let me tell you

It’s usually one of the first few things people tackle when they're learning to code like a rite of passage and it might seem deceptively simple but there are definitely some nuances I've seen people trip up on over the years

First off what you're typically given is a sequence of numbers and your goal is to identify which one is the smallest It's important to remember that smallest means numerically smallest it’s not about character count or anything weird like that

Now let's get into how I tackled this back in the day when I was first starting and how I’ve seen others approach it over the years I’ll give you the rundown in a few common languages you’re likely to run into

**Python approach**

When I first grappled with this I naturally gravitated towards Python its super readable nature made it easier for me to wrap my head around logic

```python
def find_smallest(numbers):
    if not numbers: #handling empty list edge case always important!
        return None # or raise an exception your choice depending on context

    smallest = numbers[0] #start off assume first is the smallest always works

    for num in numbers: #iterate through all others and compare
        if num < smallest:
            smallest = num

    return smallest

#example input
my_numbers = [5, 2, 8, 1, 9, 4]
smallest_num = find_smallest(my_numbers)
print(f"The smallest number is: {smallest_num}") #should output 1 obviously


my_numbers = [-5,-2,-8,-1,-9,-4] #works with negative numbers and edge cases
smallest_num = find_smallest(my_numbers)
print(f"The smallest number is: {smallest_num}") #should output -9 obviously
```

So yeah the basic idea is to go through the list you initialize a variable usually called "smallest" with the first number in the list and you compare each subsequent number to that "smallest" variable if you find something smaller you update that smallest variable until you’ve checked every element

That's the python way clean simple and readable I've spent so many late nights in college chasing python bugs I feel like I can practically speak the language and yes I know using f strings wasn’t an option in earlier versions I still have python 2.7 scars from dealing with those problems

**JavaScript approach**

Okay moving onto JavaScript I ran into this one a lot when I was doing front end development and some NodeJS stuff so similar logic but a bit different syntax obviously

```javascript
function findSmallest(numbers) {
  if (numbers.length === 0) { //same empty array check
    return null;  //or throw an error your call
  }

  let smallest = numbers[0];  //same initial starting point assumption

  for (let i = 1; i < numbers.length; i++) { //using index based loop this time for some variety
    if (numbers[i] < smallest) {
      smallest = numbers[i];
    }
  }

  return smallest;
}

//example usage
const myNumbers = [5, 2, 8, 1, 9, 4];
const smallestNum = findSmallest(myNumbers);
console.log("The smallest number is:", smallestNum); //prints 1

const myNumbersNegative = [-5, -2, -8, -1, -9, -4];
const smallestNumNegative = findSmallest(myNumbersNegative);
console.log("The smallest number is:", smallestNumNegative); //prints -9

```

The core logic remains unchanged you're just expressing it in javascript syntax instead of python Notice the differences in handling the list lengths and the loop variable declarations I've spent so much time switching between these two languages I can practically write the code in both simultaneously I once dreamt in a combination of python and javascript syntax I probably need a break from coding lol

One thing to note here and something I’ve definitely seen beginners stumble on is edge cases empty arrays what should happen then in most cases you should return null or throw an exception its an important consideration in good code

**C++ approach**

Finally lets look at C++ Its more verbose but it gives a different perspective and gives you an understanding of different code styles that are important in the real world

```cpp
#include <iostream>
#include <vector>
#include <limits> //to deal with the edge cases where the vector is empty

int findSmallest(const std::vector<int>& numbers) {
    if (numbers.empty()) {
        return std::numeric_limits<int>::min(); // handling an empty vector by returing the lowest number in int
        // throw std::runtime_error("Vector is empty"); //alternatively you can throw an exception for handling an empty vector
    }

    int smallest = numbers[0];

    for (size_t i = 1; i < numbers.size(); ++i) {
        if (numbers[i] < smallest) {
            smallest = numbers[i];
        }
    }
    return smallest;
}

int main() {
    std::vector<int> myNumbers = {5, 2, 8, 1, 9, 4};
    int smallestNum = findSmallest(myNumbers);
    std::cout << "The smallest number is: " << smallestNum << std::endl; //output 1

    std::vector<int> myNumbersNegative = {-5,-2,-8,-1,-9,-4};
    int smallestNumNegative = findSmallest(myNumbersNegative);
    std::cout << "The smallest number is: " << smallestNumNegative << std::endl; //output -9

   return 0;
}

```

C++ uses a vector instead of a list and here I use the range based loop I think I'm starting to like it here The use of `std::numeric_limits<int>::min()` is a clever trick to handle the edge case of an empty list by returning the smallest possible integer you know always got to think about all the edge cases not only is it good practice but it is also a way to learn how to be better

Also you see here the handling of size is a little different also it’s important to know what to expect from different languages and their implementations that's why I’m going over these languages

Okay so that's pretty much the gist of it for Zybooks coding lab 311 remember the important thing here is not just getting a solution that works for the sample test cases they give you but also that your code is robust and handle edge cases and well-documented for anyone to read through your code and understand why you have coded something in a certain way

As for learning resources if you are interested in more in depth understanding of algorithms and data structures and how they are implemented I would recommend you grab a copy of "Introduction to Algorithms" by Thomas H Cormen Charles E Leiserson Ronald L Rivest and Clifford Stein its a great reference text and has a lot of detail on the different approaches to these kind of problems and it is considered the bible of algorithms for many CS students and professionals Also If you are interested in more detail with C++ I highly recommend getting a copy of "Effective Modern C++" by Scott Meyers its a great resource to dive deep into all the features and best practices of modern C++ but if you are learning python I’d recommend starting with "Python Crash Course" by Eric Matthes it is a book that teaches you the basics of python and has a lot of real-world coding examples.
