---
title: "Why do UTF-16 surrogate pairs for emojis display incorrectly in console.log() using yarn on Windows?"
date: "2025-01-30"
id: "why-do-utf-16-surrogate-pairs-for-emojis-display"
---
Windows terminals, particularly older versions, traditionally struggled with complete UTF-16 support, presenting a nuanced issue regarding surrogate pairs, especially when dealing with complex Unicode characters like emojis. My experience working on a cross-platform JavaScript application revealed this problem when logging emoji output via `console.log()` during development on Windows using Yarn. The mismatch originates from how the Windows console, the JavaScript engine (Node.js in this case), and the terminal's output encoding interact with each other, each having its own interpretation of Unicode codepoints.

The core issue stems from the fact that Unicode characters beyond the Basic Multilingual Plane (BMP), which includes most common characters, require more than 16 bits (two bytes) to represent. UTF-16 addresses this by employing *surrogate pairs*. These are two 16-bit code units, a high surrogate and a low surrogate, that together represent a single Unicode character. Emojis, often existing outside the BMP, fall into this category, utilizing surrogate pairs for encoding. While JavaScript internally handles strings as UTF-16, meaning it correctly stores emojis as surrogate pairs, the `console.log()` function's rendering behavior is context-dependent, particularly problematic when dealing with Windows terminals that might not interpret or display UTF-16 characters accurately.

Node.js itself works with UTF-16 internally. This means that when your JavaScript code manipulates an emoji string, it is operating on the underlying surrogate pair representation correctly. The problem isn't within the JavaScript string itself but in the way that the Windows console handles the output from Node.js after it processes the `console.log()` command. When a string containing surrogate pairs is passed to `console.log()`, Node.js sends that data to the standard output stream. The Windows console then receives this output and attempts to display it. Here lies the problem: older versions of the Windows console, or misconfigured console settings, might not correctly interpret these surrogate pairs, and would often display them as two separate glyphs, or more commonly, as a pair of question marks within squares ('�'). This occurs because it tries to interpret each 16-bit code unit of the surrogate pair individually, rather than recognizing them as a single unit. Furthermore, the interaction with Yarn adds a layer of complexity because Yarn is the process executing Node.js, and any encoding inconsistencies in how Yarn handles the command invocation can contribute to the issue, though in the vast majority of cases, Yarn does not directly influence this problem.

To illustrate, consider these examples demonstrating how these issues arise, and how one might work around them. First, let's examine what happens when we directly log an emoji:

```javascript
// Example 1: Direct emoji output
const emoji = "😀"; // U+1F600 (Grinning Face)
console.log(emoji); //May display as '?' or incorrectly in console.
console.log("Length:", emoji.length); //Will always output 2 as it's a surrogate pair.
console.log("Codepoints:", Array.from(emoji).map(char => char.codePointAt(0).toString(16)));
```
This code directly outputs an emoji and its length, which will be 2 since it utilizes a surrogate pair. It then also shows the code points in hexadecimal, which would reveal `d83d` and `de00`. On a correctly configured terminal this will display the correct emoji, but on an affected Windows setup, the output is likely to be two question marks in boxes, or some other incorrect representation. It's key to note that the string length is two; JavaScript sees the string as two 16-bit units, correctly stored internally, yet the console fails at visual representation.

Next, let's explicitly construct the surrogate pair within JavaScript and log the result:

```javascript
// Example 2: Surrogate pair construction.
const highSurrogate = 0xD83D; // High surrogate for U+1F600
const lowSurrogate = 0xDE00; // Low surrogate for U+1F600
const emojiFromSurrogates = String.fromCharCode(highSurrogate, lowSurrogate);

console.log(emojiFromSurrogates); //Likely same incorrect output on affected terminals.
console.log("Length:", emojiFromSurrogates.length); //Will output 2, expected.
console.log("Codepoints:", Array.from(emojiFromSurrogates).map(char => char.codePointAt(0).toString(16)));
```
Here, the surrogate pair is constructed manually using `String.fromCharCode()`. While the string's internal encoding remains correct, resulting in a length of two, the visual representation on a misconfigured Windows console remains incorrect, similar to the previous example. The correct codepoints can be shown again.

Finally, we can demonstrate how to leverage a potential workaround: using an intermediate buffer that might prompt the console to interpret the bytes correctly. However, this approach isn't universally reliable as console output behaviour is often inconsistent.
```javascript
// Example 3: Attempted workaround with buffer

const emoji = "😀";
const buffer = Buffer.from(emoji, "utf16le"); //UTF16 Little Endian encoding.
console.log(buffer.toString("utf8")); //Attempt to display as UTF8, may correct console output in some setups.
console.log("Length:", buffer.length);
```
This code transforms the emoji string into a buffer encoded using UTF16 Little Endian and then attempts to output it using UTF8, often the encoding that Windows consoles use. While this can sometimes correct the display issue, it's not a reliable or recommended long term solution. A much more reliable solution is to use a modern terminal that handles UTF-16 surrogate pairs correctly, or to use a console that defaults to a modern UTF8 encoding which would usually not suffer from the problem in the first place.

Recommendations for further learning, apart from the usual online search and community boards, include detailed explanations of Unicode encoding, especially UTF-8 and UTF-16, often found in textbooks on character encoding or internationalization. Additionally, exploration of the nuances of terminal emulator behavior, specifically concerning Unicode representation, can help in developing a deeper understanding of the complex interactions that ultimately dictate whether emojis display correctly. Resources focusing on the internals of the Windows console and its history are also quite beneficial. Furthermore, documentation relating to the Node.js `console` API and it’s implementation will prove useful, as this highlights how data is passed from a Node.js process to a terminal. Examination of JavaScript's internal string representation, often covered in more advanced JavaScript material, is equally useful. Finally, practical experience with different terminals and operating systems is indispensable in grasping the subtleties of this subject.
