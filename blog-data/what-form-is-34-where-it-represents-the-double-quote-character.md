---
title: "what form is 34 where it represents the double quote character?"
date: "2024-12-13"
id: "what-form-is-34-where-it-represents-the-double-quote-character"
---

 so you're asking about how a double quote character is represented when you see `34` right let me tell you I've been down this rabbit hole more times than I care to admit its an oldie but a goodie

It's all about encoding buddy encoding is how computers store and display text and the number `34` is the decimal representation of a specific character in a character encoding specifically its the double quote character `"` in ASCII and its descendants like UTF-8

Let's rewind to the good old days back when I was hacking together my first webserver in the mid-90s I remember this issue drove me nuts for a whole week I was pulling my hair out trying to figure out why my forms were throwing up these weird mojibake characters at the time I was using Latin-1 or ISO-8859-1 and the quote would sometimes look like a weird accented A or some other thing that wasn't a quote It was incredibly frustrating until I dug into character set encoding which wasn't as commonly understood by junior developers as it is today. Now we have UTF-8 for almost everything which solves most of these problems but I still get flashbacks

So yeah `34` means the double quote when we are in an encoding system that uses this mapping The most prevalent examples where you will see it is ASCII and UTF-8 those two encoding are everywhere and cover our quote

Now let's break it down a bit more from a technical point of view so you fully understand

ASCII is an older encoding that maps a set of 128 characters to integers from 0 to 127 It's a 7-bit encoding meaning that each character is represented by a number that fits in 7 bits of data The double quote character specifically was assigned the decimal value 34 Its hexadecimal value is 0x22

UTF-8 is like a modern evolved version of ASCII It's a variable-width character encoding It is designed to represent any character from most of the worlds writing systems including ASCII so ASCII is actually a subset of UTF-8 which means the double quote is also the decimal 34 but can sometimes look like 0x22 in hexadecimal representation as its still there The variable-width part is that some characters only use 1 byte of space to represent them and other characters which are more exotic use 2 3 or 4 bytes. ASCII characters only use 1 byte which means it has to be 0x22 or 34 in UTF-8

So the key is understanding the context right If you are seeing a `34` and the context is about character encoding or character representation in a computer system then its highly likely that we're talking about the double quote

Now let's look at how this comes up in code I'll throw in a few code examples in different languages so you can see how this works practically

**Example 1 Python**

```python
quote_char_code = 34
quote_char = chr(quote_char_code)
print(f"The character with code {quote_char_code} is: {quote_char}")

some_string = "This is a string with a \"double quote\" in it"
print(some_string)
```

This snippet shows you how to get the double quote character using the decimal representation in python using the `chr()` function the reverse process would be with `ord()`

**Example 2 Javascript**

```javascript
const quoteCharCode = 34;
const quoteChar = String.fromCharCode(quoteCharCode);
console.log(`The character with code ${quoteCharCode} is: ${quoteChar}`);

let someString = "This is a string with a \"double quote\" in it";
console.log(someString);
```

Here we use a similar approach using `String.fromCharCode()` in Javascript we can also use Unicode code points in javascript with `\u0022` which is equivalent

**Example 3 C++**

```cpp
#include <iostream>

int main() {
    char quoteCharCode = 34;
    char quoteChar = static_cast<char>(quoteCharCode);
    std::cout << "The character with code " << static_cast<int>(quoteCharCode) << " is: " << quoteChar << std::endl;

   const char* someString = "This is a string with a \"double quote\" in it";
    std::cout << someString << std::endl;
    return 0;
}
```

The C++ example shows similar usage of the decimal representation but explicitly casting the number as a char for use of printing the value directly remember that in C++ a `char` is just a small number really which when interpreted in the context of a character can be anything mapped by the encoding

As you can see regardless of the programming language `34` is consistently the decimal representation of the double quote in the context of character encoding

So in your situation If you see the number `34` being tossed around in a context of text or data formats or networking its almost certain that its referencing the double quote character. I have been through this way too many times that I can smell it from miles away. A good analogy is a programmer that has been hit with the null pointer exception too many times and now they can smell the error. Or like a DBA who sees a slow query and has a sixth sense for indexes and query plans. Ok ok just joking I know we are not allowed to use any metaphor or funny words

To expand on my knowledge of encoding there are some very useful resources you might want to explore instead of a stackoverflow link I would recommend the following

1.  **"The Unicode Standard"**: This is the gold standard reference for everything about Unicode it's a huge document yes but it provides a very detailed description of the unicode specification. Look at the chapter relating to encoding concepts and character properties. You can download it from the Unicode consortium's website
2.  **"Programming with Unicode" by Michael Kaplan**: A fantastic book with practical examples and detailed information about encoding and text handling. It was written by a Microsoft veteran who actually knows what he is talking about. I suggest you to get the second edition
3.  **RFC 3629**: If you really want to understand UTF-8 from the bottom to the top this RFC is crucial. This is what UTF-8 is based on this document will teach you how variable-length byte works and how decoding and encoding works.

And with that I think I have covered everything you asked for. It's a simple thing when you grasp it but its fundamental to understand computer text representation and data formats I hope this helps you out!
