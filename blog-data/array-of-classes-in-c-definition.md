---
title: "array of classes in c++ definition?"
date: "2024-12-13"
id: "array-of-classes-in-c-definition"
---

 so you're asking about arrays of classes in C++ yeah got it been there done that a few times let me spill the tea based on my experiences It's a fairly straightforward concept but it's also a place where a lot of folks stumble initially especially with memory management and initialization so lets break it down

First thing's first when you say "array of classes" you're basically talking about creating a contiguous block of memory where each element is an instance of your class That's it nothing fancy under the hood its just objects stacked one after the other Just like an array of `int` or `float` except now instead of storing those primitive types you’re storing objects

Let's start with the most basic scenario declaration of a simple class array I'm assuming you have a class like this:

```cpp
class MyClass {
public:
    int myValue;
    MyClass(int value) : myValue(value) {}
    ~MyClass() {
        // some cleanup maybe
    }
};
```

Now here's how you declare a static array of `MyClass` objects

```cpp
MyClass myArray[10]; // array of 10 MyClass objects
```

Simple enough right But wait there's a catch You just declared an array but those objects inside the array they don’t have initialized members yet In this specific case the constructor without parameters needs to exist. The program does not know yet the value for `myValue` We need to call the constructor somehow I remember the first time I used this I got a segmentation fault because I tried to access `myArray[0].myValue` before I gave it a proper value the hard way the compiler said "dude you are trying to read some garbage memory"

So lets fix this. If you try to declare with parameterized constructor you need to declare explicitly each element in the following way:

```cpp
MyClass myArray[3] = {MyClass(1),MyClass(2),MyClass(3)};
```
This is correct now.  

But what if you want to initialize all elements to the same initial value? For this you can do the following:

```cpp
MyClass myArray[10] = {1} // All objects set with myValue = 1
```

That's cool and handy isn't it? But a static array like this has a fixed size at compile time What if you need a dynamically sized array? Well here's where dynamic memory allocation comes in to play

For dynamic arrays you use `new` and `delete[]`

```cpp
MyClass* myArray = new MyClass[20]; // Create 20 MyClass objects in the heap
// always initialize the elements to make sure the object has valid state
for(int i=0; i<20; i++){
    myArray[i] = MyClass(i+1);
}

// ... do stuff with myArray

delete[] myArray;  // Don't forget to clean up
myArray = nullptr; // Always good to null it after deleting
```

Few important things to notice here:

*   `new MyClass[20]` allocates memory for 20 `MyClass` objects on the heap.
*   `new` returns a pointer `MyClass*` so you need to store it in a pointer variable.
*   You need to explicitly loop to construct the elements with parameters in case you do not want to use the copy constructor this is tedious but necessary if you do not want to initialize all elements with the same value
*   The `delete[] myArray;` part is critical. If you forget it you get memory leak because the heap allocated memory will never be deallocated the operating system will take it back eventually but it is very important you deallocate the memory by hand since it is your responsability. Also the `myArray = nullptr` at the end is good practice you never know what could happen with the pointer if you use it after it was deleted so make it a habit to point it to nowhere using `nullptr`
*   You must `delete[]` if you allocate with `new[]` I have seen so much code with just a `delete` for array of objects it's insane and it is so wrong.

Now the dynamically allocated version of array allows for variable size but this is a bit error prone with all this `new` and `delete[]` mess This is why `std::vector` is way better. It handles all the memory management stuff for you it's like it is a good friend that takes care of your messy memory deallocation and initialization

```cpp
#include <vector>
#include <iostream>

std::vector<MyClass> myVector;

int main(){
    //lets add a few objects into our vector
    for(int i=0; i<10; i++){
        myVector.push_back(MyClass(i+1)); // using the push_back that adds at the end
    }
    //access using normal index operator
    std::cout << myVector[0].myValue << std::endl; // should print 1

    //access using at
    std::cout << myVector.at(5).myValue << std::endl; // should print 6

    // iterate with iterators
    for (std::vector<MyClass>::iterator it = myVector.begin(); it != myVector.end(); ++it){
         std::cout << it->myValue << " ";
    }
    std::cout << std::endl;
    //iterate with range-based for loop
    for(const auto& item: myVector){
        std::cout << item.myValue << " ";
    }
    std::cout << std::endl;
    
    return 0;

}
```

The vector automatically allocates memory when you push data to it and it takes care of deallocating it for you when the vector goes out of scope The `push_back` adds elements to the end of the vector dynamically making it grow and shrink without a single line of code for memory management You can also access the elements in the vector using indexes like a normal array or with `at` and iterate in it using iterators or ranged based for loops I find the ranged based for loops the easiest to read and write.

One last thing a few years ago I wrote a bunch of code using dynamic arrays directly with `new` and `delete` because I didn't understand the value of `std::vector` and I can tell you the headache and the number of bugs was crazy. That's a good code I wish to forget but at the end I learnt the hard way that it is way better using `std::vector` or other STL containers rather than reimplementing everything by hand.

**Memory and Performance Considerations**

Now let's get a bit more techy about what is going on.

*   **Stack vs Heap**: When you declare a static array like `MyClass myArray[10];` the memory is allocated on the stack. Stack memory is fast but it's limited and determined during compilation. Dynamic arrays are stored in the heap which is much larger but it's slower to access and you need to manage its lifetime yourself with the help of `new` and `delete[]`. `std::vector` will be also in the heap since it dynamically manages memory allocation.
*   **Initialization**: Initializing the array is crucial. If you don't properly initialize your objects you can end up with unexpected behavior. Compiler generated default constructors can lead to garbage memory values in fields of the class since they are not initialized with a constructor with specific parameters or a explicit initialization of the variable inside the class body.
*   **Copy Constructor**: If your class has non-trivial resources make sure you define the copy constructor properly. If you don't your copy will be shallow and you will end up with double deallocation or memory leaks. The same goes for assignment operator. This is part of the Rule of Five or Rule of Zero. If you do not use the copy operator your code could have very hard to find bugs. The compiler auto generate a shallow copy constructor and this could lead to many errors.
*   **Vector Growth**: When `std::vector` needs more space it will reallocate more memory and copy all the content to the new block of memory This is why `std::vector` adds elements at the end of the allocated memory block for performance reasons, if it adds at the beginning all the data should be copied to the right one position in memory, this is expensive in terms of execution time. This reallocation operation is slow so you need to try to minimize it by setting initial vector size if you know how much data you will store beforehand using `std::vector::reserve(number)` function to reserve the needed memory in advance.

**Resources**

If you want to deep dive into the subject I recommend a couple of books that helped me a lot in the beginning:

*   *Effective C++* by Scott Meyers: This is a classic that goes into best practices in C++. Essential for understanding how to write efficient code.
*   *C++ Primer* by Stanley B Lippman: It's a comprehensive guide covering the whole C++ language in depth. This is like the C++ bible.
*   *Thinking in C++* by Bruce Eckel: A good approach for learning C++ with the practical use cases.

Anyways I've seen so many folks getting confused with array of objects I hope this has cleared some doubts and gave some food for thought. Oh and by the way did you hear about the programmer who got stuck in the shower? He couldn't figure out how to `exit()` without getting wet. So yeah its always good to use `std::vector` when possible.
