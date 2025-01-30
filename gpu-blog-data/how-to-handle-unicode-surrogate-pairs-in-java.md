---
title: "How to handle Unicode surrogate pairs in Java strings?"
date: "2025-01-30"
id: "how-to-handle-unicode-surrogate-pairs-in-java"
---
Unicode surrogate pairs, specifically, represent characters outside the Basic Multilingual Plane (BMP) using two code units. Java, which utilizes UTF-16 encoding for its `String` objects, stores these surrogate pairs as two distinct `char` values. Properly handling these pairs is critical for applications processing text from a wide range of scripts and symbols, as simply iterating over `char` values can lead to incorrect display, processing, or storage of these extended characters. I've repeatedly encountered this issue during data migration tasks involving internationalized databases, underscoring its importance.

To address this, one must understand the underlying structure of surrogate pairs within the UTF-16 encoding. A surrogate pair consists of a high surrogate (leading) code unit and a low surrogate (trailing) code unit, both occupying 16 bits. High surrogates fall in the range U+D800 to U+DBFF, while low surrogates range from U+DC00 to U+DFFF. These specific ranges are reserved, not assigned to specific characters, specifically for use in representing supplementary characters. In Java, a single `char` variable, which is 16-bits wide, cannot represent the entire character; therefore the two-`char` approach is adopted. Any processing that assumes a one-to-one mapping between `char` values and displayed characters will fail when encountering surrogate pairs.

Correct handling involves recognizing surrogate pairs and treating them as a single logical unit, known as a code point. Java provides methods within the `Character` class to help with this process. The key is to utilize the `codePointAt` and `codePointCount` methods rather than treating a string as a simple `char` array. Specifically, `codePointAt(int index)` returns the integer representation of the code point at a given index within the string, considering both single characters and surrogate pairs. The method `codePointCount(int beginIndex, int endIndex)` will accurately provide the length of the string in terms of code points, accounting for surrogate pairs. In contrast, the `length()` method of the String object returns the number of `char` values in the String, which may not reflect the actual number of characters.

Let's illustrate this with some code examples:

**Example 1: Iterating through Code Points**

This snippet demonstrates how to correctly iterate through a string containing a surrogate pair, in contrast to a naive `char` based iteration:

```java
String text = "Hello \uD83D\uDE0A world!"; // U+1F60A is a smiling face with smiling eyes, a supplementary character
System.out.println("String Length (char): " + text.length()); // Incorrect Length
System.out.println("Code Point Length: " + text.codePointCount(0, text.length())); // Correct Length

System.out.print("Incorrect char based iteration: ");
for (int i = 0; i < text.length(); i++) {
    System.out.print(text.charAt(i) + " ");
}
System.out.println();

System.out.print("Correct code point based iteration: ");
for (int i = 0; i < text.length();) {
    int codePoint = text.codePointAt(i);
    System.out.print(Character.toChars(codePoint) + " "); // ToChars converts code point to char array
    i += Character.charCount(codePoint); // Increment index by length of the codepoint
}
System.out.println();
```

The output will clearly show the discrepancy. The `length()` call returns 15, whereas there are only 14 logical characters. When looping using `charAt`, the emoji is split into two separate garbled characters. Using `codePointAt` and incrementing the index by `Character.charCount`, which accurately returns the number of `char` values required for the code point, will correctly loop and print the emoji.

**Example 2: Extracting Substrings**

This example demonstrates how to correctly extract a substring that may contain a surrogate pair. Incorrectly slicing based on `char` indices could produce invalid results or break a surrogate pair.

```java
String text = "ABC\uD83D\uDE01DEF"; // U+1F601 is a grinning face, another supplementary character
System.out.println("Original text: " + text);

//Incorrect Substring extraction
String incorrectSubstring = text.substring(1,5); // Incorrectly split the surrogate pair
System.out.println("Incorrect substring: " + incorrectSubstring);

//Correct Substring extraction using code points
int startIndex = 1;
int endIndex = 4;

int correctStartCharIndex = 0;
int codePointCount = 0;
while (codePointCount < startIndex) {
   correctStartCharIndex += Character.charCount(text.codePointAt(correctStartCharIndex));
   codePointCount++;
}

int correctEndCharIndex = correctStartCharIndex;
codePointCount = startIndex;
while (codePointCount < endIndex) {
   correctEndCharIndex += Character.charCount(text.codePointAt(correctEndCharIndex));
   codePointCount++;
}

String correctSubstring = text.substring(correctStartCharIndex, correctEndCharIndex);
System.out.println("Correct substring: " + correctSubstring);
```

In this example, slicing using `substring(1,5)` will result in the surrogate pair being split into two invalid parts. The code to get the correct indices using the codepoints iterates over the string, counting logical code points until the desired indexes, then extracting the `substring` based on `char` position. The resulting substring correctly extracts the intended sequence "BC\uD83D\uDE01" as expected.

**Example 3: Replacing Characters**

This example demonstrates the importance of code point manipulation when replacing characters in strings, ensuring surrogate pairs are treated as single entities.

```java
String text = "Testing\uD83D\uDE30with\uD83D\uDE32"; // U+1F630 is a face with open mouth and cold sweat and U+1F632 is an astonished face
System.out.println("Original text: " + text);

String updatedText = new String(text.codePoints()
    .map(codePoint -> {
        if (codePoint == 0x1F630) { // Replace 1F630 with 1F60E, a smiling face with sunglasses
           return 0x1F60E;
        } else if (codePoint == 0x1F632) { // Replace 1F632 with 1F60A, a smiling face with smiling eyes
           return 0x1F60A;
        }
        return codePoint;
    }).toArray(),0,text.codePointCount(0,text.length()));
System.out.println("Updated text: " + updatedText);
```

The `codePoints()` method creates a stream of code points for the text. This allows for proper replacement of Unicode values. The `map()` function iterates over this stream, applying a transformation that replaces specific code points. The altered code points are then converted back into a `String` using the appropriate constructor to produce the altered text. If we were to attempt a char-by-char replace, it would lead to broken surrogate pairs when attempting to replace or match only one half of the pair.

In my experience, failing to recognize and handle surrogate pairs correctly leads to subtle and often difficult-to-debug issues. For instance, during a project involving data from various sources, including legacy systems and external APIs, incorrectly parsed strings with surrogate pairs caused inconsistencies in data analysis and reporting. The consequences ranged from incorrect character displays in user interfaces to data corruption during database updates. Proper implementation of the described methods was essential to resolve these issues, highlighting the importance of understanding this aspect of the Java string representation.

For further reading and guidance, I recommend consulting resources like the official Java API documentation for the `Character` class, particularly the sections detailing code point handling. The Unicode Consortium's website provides in-depth technical details of the Unicode standard and surrogate pairs. Books on Java internationalization can offer detailed explanations and patterns for working with text in globalized applications. Furthermore, any text focusing on Unicode and character encoding will be helpful. The key is to look for documentation and resources that emphasize code points and specifically call out surrogate pair handling as part of the discussion.
