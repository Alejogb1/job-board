---
title: "why does 2e3 return 2000 isnt 2 to the 3rd power equal to 8?"
date: "2024-12-13"
id: "why-does-2e3-return-2000-isnt-2-to-the-3rd-power-equal-to-8"
---

 so you're tripping on `2e3` returning `2000` and not `8` right Been there done that plenty of times early in my career let me tell you It's a classic gotcha in JavaScript and many other languages actually it ain’t just JS

so the root of your confusion lies in how languages interpret the `e` notation It's not exponentiation like you're thinking with `2**3` or `Math.pow(2, 3)` This `e` is shorthand for scientific notation specifically its for "times ten to the power of"  So `2e3` actually means 2 times 10 to the power of 3 or 2 * 10<sup>3</sup>

That's why it results in 2000 See it's all about that scientific notation convention not mathematical power function

I remember vividly back in my early days working on some data processing pipelines I kept getting these wild numbers in the output that I could not for the life of me understand Took me way longer than I'd like to admit to realize I was accidentally using scientific notation where I meant to use `Math.pow` I had this nasty bug where coordinates were all messed up because I used a wrong calculation which included this syntax mistake so imagine that for like several hours you have people looking at a map with coordinates that are totally misplaced It was embarassing

let's break it down with some code examples since that's how us programmers communicate right

```javascript
// This is exponentiation as you expect it: 2 to the power of 3
console.log(2 ** 3); // Output: 8
console.log(Math.pow(2, 3)); // Output: 8

// This is scientific notation 2 times 10 to the power of 3
console.log(2e3);  // Output: 2000
console.log(2 * Math.pow(10,3)); // Output: 2000
```

See the difference  The double asterisk `**` and `Math.pow` are the true mathematical power operators while `e` is just a shortcut for scientific representation of numbers Specifically useful when you are dealing with very large or very small numbers where you do not want a very long number as input

So what about the other side of the same coin the `e` with negative sign Lets go there

```javascript
//Scientific notation using negative exponent
console.log(2e-3) // Output: 0.002
console.log(2 * Math.pow(10,-3)); // Output: 0.002
```
As expected this represents 2 times 10 to the power of negative 3 which means we are dealing with 2 times 1/1000 or 0.002

And if you are a python guy here is some code:

```python
# Python also works the same way

# Exponentiation: 2 to the power of 3
print(2 ** 3)  # Output: 8
print(pow(2, 3))  # Output: 8

# Scientific notation: 2 times 10 to the power of 3
print(2e3)  # Output: 2000
print(2 * 10**3) # Output: 2000

#Scientific notation with negative power
print(2e-3) #Output 0.002
print(2 * 10**-3) #Output 0.002
```

It is similar to what we already saw before and both languages handle scientific notation and power operations the same way

Now you might be asking Why have two different ways of doing this power thing? The `**` or `Math.pow` is how we do normal exponentiation as humans write it out in normal math equations and the scientific notation notation is very useful when dealing with scientific data or data manipulation where the size of number could become a concern

And that is a really valid question indeed

The `e` notation is actually pretty handy when you're working with really big or really small numbers because it makes the number shorter or easier to read like the distance to the moon or like the mass of an electron where those numbers get really cumbersome to write in full if you dont use scientific notation

So next time you see `e` in a number remember it's about powers of ten not a normal base exponent operation and now you'll avoid that headache I went through that time trying to get map coordinates right the hard way

You know there was a moment during that bug investigation I thought maybe my computer was just drunk I swear those numbers were coming out of nowhere

Now for further learning I'd really recommend a few resources on numerical representations and computer arithmetic First off get your hands on “What Every Computer Scientist Should Know About Floating-Point Arithmetic” by David Goldberg this is a classic paper that dives deep into floating-point numbers and the issues that can occur which are surprisingly numerous The whole thing is pretty fundamental if you’re planning to become a serious programmer or just want to know what’s going on in the engine Also the “Computer Organization and Design” book by David Patterson and John Hennessy while not specifically focused on this detail it provides the bedrock for understanding number representation at the hardware level

And of course reading the ECMAScript standard is always good if you really want to understand how numbers work in Javascript which will teach you that  scientific notation is one way of expressing number types in ECMAScript

I hope this helps and let me know if you have more questions around numbers and stuff I've probably seen it all at this point
