---
title: "regex with variables substitution?"
date: "2024-12-13"
id: "regex-with-variables-substitution"
---

Alright alright I see you need some regex variable substitution help that's a classic one honestly I’ve been down that rabbit hole more times than I care to remember and I'm gonna break it down for you like you're a fellow dev who’s been through the coding trenches so let’s get to it

First off let's clarify what we are even talking about here You've got a regex pattern and you want to plug in some values dynamically usually from variables you have in your code or maybe from user input or even some sort of data source and you want those values to get inserted into the regex before it does its matching or replacing dance Basically your regex string isn't static it's got these little placeholders or areas where you need to do some fancy variable injection and yeah string concatenation can get ugly pretty fast

I’m not going to talk about why regular expressions are generally bad for certain tasks That's for another thread for now let's focus on your problem cause I’ve totally been there and the solutions vary based on your language or environment let's start with Python since I used a lot of that in my younger years back in the AI winter of the early 2010's when all we had was logistic regression and that was exciting (those days)

In python you can do this in a few ways but the go-to way is using f-strings these are the coolest honestly if you're running python 3.6 or later because they keep everything clear and readable you know they actually added this feature to deal with this exact problem we had in previous python versions

```python
import re

variable1 = "apple"
variable2 = r"\d+" #raw string cause we are using escape char in regex
pattern = fr"The quick brown {variable1} jumps over {variable2} lazy dogs"
text = "The quick brown apple jumps over 123 lazy dogs"

if re.search(pattern, text):
  print("Match found!")
else:
  print("No match.")

```

See what we did there? We used an f-string with the f prefix before the string and inside the string you can just drop your variables in using those curly braces `{}` it's super clean and it's easy to read and in the example I have defined the variables before the string but it can also be done in the same line using the f string feature it's the same as concatenating a string that is why I prefer f strings in python because it is easier to read and less error prone Now you can make your regexes as dynamic as your heart desires

Before f strings in python we used to use the `.format` method or even the `%` operator (oh god the % operator it's so old school I hope nobody is still using it) but those look messy compared to the f strings and it is less readable so please for the sake of your team and yourself do use f strings in python

Okay let's switch gears and go over Javascript a lot of frontend work in my life too and this is where `template literals` come to the rescue its similar to the f strings in python with one difference the backticks `` are the main heroes here they look the same but work slightly different in context of their respective languages

```javascript
const variable1 = "apple";
const variable2 = "\\d+"; // escaping needs extra attention in JS
const pattern = new RegExp(`The quick brown ${variable1} jumps over ${variable2} lazy dogs`);
const text = "The quick brown apple jumps over 123 lazy dogs";

if (pattern.test(text)) {
  console.log("Match found!");
} else {
  console.log("No match.");
}
```

Here we are using a template literal (the one with backticks) to embed variables into our regex string because this is Javascript we are also creating the regex using the `RegExp()` constructor not just writing `/pattern/` and the main difference between template literals and f strings in python is that they have different syntax but they do similar things string interpolation and also in Javascript you have to pay extra attention to escape characters

There are also ways to do this in other languages too like in bash you can use double quotes and that works similar to the javascript and python solutions

Now you need to watch out for a few gotchas here First always remember to escape characters properly especially when building regex patterns because you need to consider both the regex escape and the language escape characters and that can get confusing fast Second be careful when inserting data that could be user supplied you know things like injection attacks it's a big problem especially if you are using the user input directly in the regex so please sanitize your inputs

And if you are thinking about performance yeah that's a good question because if you are building a complex regex every time it is going to affect your performance and in some cases you can actually compile or pre-compile your regex pattern if it's not changing every time which does give you some speed benefit so its something to keep in mind

Okay here is a bash example since I did some DevOps stuff back in the day cause nobody else wanted to touch it.

```bash
variable1="apple"
variable2="\d+"
pattern="The quick brown $variable1 jumps over $variable2 lazy dogs"
text="The quick brown apple jumps over 123 lazy dogs"
if [[ "$text" =~ $pattern ]]; then
  echo "Match found!"
else
  echo "No match."
fi

```

In bash the variable substitution works almost exactly like in Javascript and python but be careful because shell scripting can be a headache for beginners to understand so be careful in bash

A word of advice though don't get too crazy with regex for simple string matching tasks use basic string operations for simple substring searches and also sometimes a regex can become a big black box and can become less readable if you start to use complex features and it will get really hard to maintain and if someone else looks at your code they will be like “what is this doing”. (insert joke here) “It's like a magic trick if you can understand it” that's why keeping it simple will help you a lot

Now if you are really into this deep dive I would suggest a few resources instead of links for you if you are interested in regexes in general I found Jeffrey Friedl’s “Mastering Regular Expressions” to be like the bible of regexes and it will take you to the next level from beginners level to master levels its worth a read or if you like the more academic side of it you can find some good papers on finite automata and formal language theory that will help understand the underlying mechanics of regular expressions it will become a second nature to you if you get the theory in the first place trust me it helped me with my coding skills a lot and last but not least if you just want to know how regex works in your specific language the official documentation for your language is actually a good place to start

So yeah that’s pretty much it for regex variable substitution and if you have any more specific questions feel free to ask I'm usually around to help.
