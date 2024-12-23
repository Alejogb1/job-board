---
title: "low level and high level data types?"
date: "2024-12-13"
id: "low-level-and-high-level-data-types"
---

 so low level versus high level data types right Been there done that got the t-shirt a couple of times actually Let’s break it down from my experience you know the trenches of debugging

First off let’s talk about what we actually mean by data types Because without that we're just throwing terms around like confetti I'm going to assume we're talking about programming here not you know astrophysics data types or something though that's cool too but for different reasons

 so at the core level your computer a hunk of silicon and electricity only groks bits and bytes That's it Zeroes and ones Now a low level data type is one that's very close to this hardware representation think of it as a direct mapping to how the CPU and memory view data These are things like integers characters floating point numbers at their most basic representations

For example in C or C++ we have `int` `char` `float` and `double` Those are low level data types `int` for example is typically represented as a fixed number of bits usually 32 or 64 which directly maps to a range of numerical values The same with `char` being a byte mapping to a character representation usually ASCII or UTF-8 and floating points are stored using the IEEE 754 standard its a detailed process believe me i spent a few nights wrestling with bit manipulation debugging these things a long time ago when I was building a custom rendering engine way before unity or unreal existed think mid 2000’s I mean we are talking real hardcore low level stuff no external libraries just custom code that barely worked sometimes I am just glad I am still alive after all that

Here is a quick C code example to illustrate what I am talking about this will print the size of each primitive data type using `sizeof`

```c
#include <stdio.h>

int main() {
    printf("Size of int: %zu bytes\n", sizeof(int));
    printf("Size of char: %zu byte\n", sizeof(char));
    printf("Size of float: %zu bytes\n", sizeof(float));
    printf("Size of double: %zu bytes\n", sizeof(double));
    return 0;
}

```

Run it and see how sizes vary depending on your architecture you see that’s the low level part about it we are literally working with bits and bytes

Now enter the high level data types These are abstractions built on top of the low level stuff They are designed to make your life as a programmer easier by giving more structure meaning and handling of data behind the curtains Think of it like this your basic car engine is low level with its pistons and valves and stuff but a high level car like your phone with a navigation system is built on that the low level stuff still powers the high level things

These high level types can include things like strings lists objects and dictionaries that are very common today in most languages They might look simple but under the hood they involve dynamic memory allocation often pointer manipulation and a bunch of bookkeeping that the programming language or its runtime environment is taking care of for you

Take Python for example strings are not just a bunch of bytes like in C its complex objects that can grow in size and support various string operations and this is the power of high-level data types

```python
my_string = "Hello world"
my_list = [1 2 3 4 5]
my_dictionary = {"name": "John" "age": 30}

print(type(my_string))
print(type(my_list))
print(type(my_dictionary))
```

See how they are not just primitive types they are objects with methods and functionalities These are high level data types at work

I remember I was implementing some networking protocol in the early days you know for a project that had to transfer large amounts of structured data between servers My initial thought was to pack everything into raw byte arrays and you know parse them manually at the other end Talk about a nightmare lots of bit shifts and masking you know bitwise operations I spent a week debugging a single byte that was out of place because of a single wrong bit operation I ended up rewriting the whole thing using custom objects with specific data structure rules and using a serialization library and it was literally 1000 times easier to debug and maintain So in essence instead of dealing with individual bits I was dealing with data that made sense to my application and I was not losing my sanity trying to trace down the source of an error which was really a good decision I can tell you that.

The trade-off here is that while high level data types are super convenient they often come with some performance overhead there is more processing and memory management behind the scenes So if you are working on something that's really performance critical like embedded systems real time processing or the lower level parts of a game engine or a super optimized library you might end up using more low level data types and libraries directly because you need to squeeze every bit of performance you can. Or sometimes you just like to mess with bits its fun once in a while not all the time though trust me.

Here is an example in C that shows how you might do some bit manipulation to pack multiple values into a single integer this is a common technique in low level programming

```c
#include <stdio.h>
#include <stdint.h>

int main() {
    uint32_t packed_value;
    uint8_t value1 = 10;
    uint8_t value2 = 20;
    uint16_t value3 = 1000;

    packed_value = (value1 << 24) | (value2 << 16) | value3;

    printf("Packed value: %u\n", packed_value);
    printf("Value 1: %u\n", (packed_value >> 24) & 0xFF);
    printf("Value 2: %u\n", (packed_value >> 16) & 0xFF);
    printf("Value 3: %u\n", packed_value & 0xFFFF);
    return 0;
}

```

Notice the bit shifting `<<` and bitwise AND `&` I am doing some masking there to extract the individual values from the packed integer. It's not something I do daily anymore but it's a good example of what low level manipulation is like and what goes under the hood before some higher level system abstracts this complexity for you. I will be honest sometimes I find myself missing this type of low level coding now that things are way easier there was something special about bit manipulation I do not know perhaps a little bit of chaos in my brain made me enjoy it I am not sure now that I am saying it out loud

So basically low level gives you precise control of the hardware but requires more effort and more brain power a.k.a more debugging sometimes way too much and high level provides easier development and better abstraction but might come with some performance trade-offs. It's a balancing act you always need to choose the right tool for the job based on your requirements and this comes with experience and with a lot of debugging so keep that in mind the next time you are building something and also make sure you get lots of coffee and sleep because coding when you are tired can lead you to make a lot of mistakes I learned this the hard way I ended up with code that looked like a random generated text file for a week a very painful week.

If you want to dive deeper into this I would suggest checking out “Computer Organization and Design” by David A. Patterson and John L. Hennessy it goes deep into the hardware side and how data is represented there also “Modern Operating Systems” by Andrew S. Tanenbaum is great for understanding how operating systems manage memory and different data types This combination will give you a solid understanding of the subject a lot of my own experience is based on the information in these books. They helped me a lot back in the day and still do.
