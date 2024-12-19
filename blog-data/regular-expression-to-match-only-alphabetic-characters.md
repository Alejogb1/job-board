---
title: "regular expression to match only alphabetic characters?"
date: "2024-12-13"
id: "regular-expression-to-match-only-alphabetic-characters"
---

Alright so you wanna match only alphabetic characters with a regular expression right Been there done that Probably a hundred times Seems simple enough but trust me its easy to trip up on the edge cases Lets break it down and Ill share some of my war stories

First things first I mean the most basic approach if you are dealing with standard english alphabet is going to be something like `[a-zA-Z]` That seems like a no-brainer right And most of the time it works perfectly fine But here's where things get interesting You start throwing in real-world text and suddenly you're dealing with a whole zoo of characters

I remember this one project I was working on way back when It was a web scraper scraping product names and reviews from some dodgy e-commerce site I figured `[a-zA-Z]` would be enough for name validation I mean who uses non-alphabetic characters in product names right I was so wrong So so wrong Turns out some products had names with all sorts of accented characters foreign letters you name it My simple regex choked and spat out garbage So I learned my lesson the hard way always test your regex against real data not just some textbook example

Now you might be thinking okay I need to handle unicode What about `\p{L}` Yeah thats a good start `\p{L}` is a unicode property that matches any letter from any alphabet Its more flexible than `[a-zA-Z]` and will cover you against those nasty é á ü and all the other non-ascii characters This works wonders most of the time Its a pretty solid workhorse

But its also not bulletproof and if you are working on extremely sensitive things like password validation you might need to be very strict about what you accept in the input Remember that you can also use different character sets like ASCII which will help to validate that the input characters are actually what you think they are and not some weird character encoded with unicode

Let me give you some examples in different languages to see how they work in the real world These are quick and dirty examples not production ready just to give you the idea

**Python:**

```python
import re

def is_alpha_python(text):
    return bool(re.fullmatch(r'[a-zA-Z]+', text))

def is_alpha_unicode_python(text):
    return bool(re.fullmatch(r'\p{L}+', text, re.UNICODE))

# Basic ASCII
print(is_alpha_python("HelloWorld")) # True
print(is_alpha_python("Hello World")) # False (space not alpha)
print(is_alpha_python("123Hello")) # False (number present)

# Unicode
print(is_alpha_unicode_python("Héllo")) # True
print(is_alpha_unicode_python("你好")) # True (Chinese is considered letter by unicode)
print(is_alpha_unicode_python("你好123")) # False
```
The first python example uses the basic ascii alphabet range the second one uses the unicode to match any letter in any alphabet and also uses the fullmatch to make sure it matches the whole string. The `re.UNICODE` flag is important here to let regex know that we want to match unicode characters.

**JavaScript:**

```javascript
function isAlphaJS(text) {
  return /^[a-zA-Z]+$/.test(text);
}

function isAlphaUnicodeJS(text) {
  return /^\p{L}+$/u.test(text);
}

// Basic ASCII
console.log(isAlphaJS("HelloWorld")); // true
console.log(isAlphaJS("Hello World")); // false
console.log(isAlphaJS("123Hello")); // false

// Unicode
console.log(isAlphaUnicodeJS("Héllo")); // true
console.log(isAlphaUnicodeJS("你好")); // true
console.log(isAlphaUnicodeJS("你好123")); // false
```
The javascript example is very similar to python the key difference is the use of `test` method for matching and the `u` flag which allows the use of unicode. Also `^` and `$` are being used to ensure the entire string is a match and not just part of it

**Java:**

```java
import java.util.regex.Pattern;
import java.util.regex.Matcher;

public class AlphaMatcher {
    public static boolean isAlphaJava(String text) {
        Pattern pattern = Pattern.compile("[a-zA-Z]+");
        Matcher matcher = pattern.matcher(text);
        return matcher.matches();
    }

     public static boolean isAlphaUnicodeJava(String text) {
        Pattern pattern = Pattern.compile("\\p{L}+");
        Matcher matcher = pattern.matcher(text);
        return matcher.matches();
    }
    public static void main(String[] args) {
        // Basic ASCII
        System.out.println(isAlphaJava("HelloWorld")); // true
        System.out.println(isAlphaJava("Hello World")); // false
        System.out.println(isAlphaJava("123Hello")); // false

        // Unicode
        System.out.println(isAlphaUnicodeJava("Héllo")); // true
        System.out.println(isAlphaUnicodeJava("你好")); // true
        System.out.println(isAlphaUnicodeJava("你好123")); // false

    }
}
```
Java example is a bit more verbose because of the need to create pattern and matcher but the core regex is still very similar to the others The `matcher.matches()` is also very important here.

Now for the serious part What resources should you look at if you want to be a regex master I would recommend Jeffrey Friedl's "Mastering Regular Expressions" Its like the bible for regular expressions Trust me its worth every penny if you work with regex often and its very good if you really want to be a regex jedi Another great one is "Regular Expression Cookbook" by Jan Goyvaerts and Steven Levithan It offers a bunch of very practical ready-to-use recipes for many common tasks so its an amazing reference book.

Oh and one last thing you might want to think about performance when using regex For simple matching it's usually not an issue but when working with large amounts of text or complex patterns performance can degrade quickly Its like trying to use a spoon to dig a swimming pool so pay attention to how you build your regular expressions. There is always room for improvement. Also remember to always use a profiler to analyze the performance it might surprise you where the real bottle necks of your code might be. And finally avoid regex if you do not really need them. If a simple algorithm using the ASCII set would be enough to validate the input then that's always going to be a better performance option in the long run.

So yeah thats pretty much all there is to it simple alphabetic matching is not that hard once you know the tricks and gotchas and what can go wrong. I mean unless you need to validate only uppercase letters or lowercase letters. Then things get a bit more complicated but not impossible of course. I mean I hope you learned something from my long rant and that it helps you with your regex adventures

I hope this helps. If I didn't understand your question correctly or I made some mistakes please let me know. If you want to know more about edge cases let me know and I will expand my answer.

One more thing. Remember to treat your regex with care or it can become a monster very fast. Like I told my boss ones that a regex I built was so complex that it had its own existential crisis and started questioning its meaning in life. That wasn't fun he asked me to rewrite it.
