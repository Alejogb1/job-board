---
title: "non linear scale slider in laymans terms?"
date: "2024-12-13"
id: "non-linear-scale-slider-in-laymans-terms"
---

 so non linear sliders yeah I’ve wrestled with those beasts a few times lemme tell ya

First off in layman's terms you’re dealing with a slider where the visual distance of the slider doesn’t directly correspond to the numerical change in the value it represents Imagine a regular slider where moving it half the way goes from 0 to 50 well a non linear one might go from 0 to 10 in the first half and then from 10 to 100 in the second half the further you go the bigger the number jump get it Its not a uniform progression

 now why you'd want that and not just normal boring linear sliders well think about audio volume or say a zoom level for a map or even exposure on a camera Often these things don’t feel natural when represented linearly Small nudges at the low end should give you smaller changes while bigger nudges at the high end can give you bigger changes think about a logarithm its a classic example Youre covering more ground in a smaller travel distance

So the trick is in the math not necessarily the UI element its just a normal slider underneath You need to translate your visual slider position usually from 0 to 1 into a value on your desired non linear scale

I’ve had a fun experience debugging this I was working on a sound design tool a while back in the bad old days of Flash (yeah I know) and the users were complaining that the gain slider was way too touchy at the low end You could barely nudge it and the sound would go all over the place So of course we were using a simple linear scale for decibels and it sucked It was a pain to get right because the clients were musicians so you know how those guys are like about sound precision lol

My quick fix was a logarithmic scale its the go to for audio This makes the slider less sensitive at low levels and more sensitive at high levels So a small change at the bottom of the slider would give you a tiny dB change while a small change at the top of the slider would give a bigger dB change This matched how users actually perceived volume better

Here's a taste of the kind of code you might write for this kinda thing in say javascript which everyone seems to love nowadays

```javascript
function linearToLogarithmic(linearValue, minValue, maxValue) {
  // Ensure linearValue is within 0 and 1
  const clampedValue = Math.max(0, Math.min(1, linearValue));

  // If minValue is 0 the log10 function will give -infinity as a result which is bad
  // add a tiny offset to prevent problems
  const offsettedMin = minValue <= 0 ? 0.0001 : minValue;


  const logMinValue = Math.log10(offsettedMin);
  const logMaxValue = Math.log10(maxValue);
  const logValue = logMinValue + (logMaxValue - logMinValue) * clampedValue;
  return Math.pow(10, logValue);
}

//Example usage
const linearPosition = 0.5 // slider position between 0 and 1
const minimumValue = 1; //minimum desired value
const maximumValue = 100; //maximum desired value
const logarithmicValue = linearToLogarithmic(linearPosition, minimumValue, maximumValue);
console.log(logarithmicValue); //This will print a value between 1 and 100
```

This snippet takes a linear value from your slider (between 0 and 1) and returns the corresponding logarithmic value between your given minimum and maximum value You'll need to adjust this for your specific range but you get the idea

Now let’s talk about some other fun options for non linear sliders You don’t always need a logarithm A power function like x^n can work wonders especially for visual things like zoom levels where the perceived difference in zoom changes faster the more you zoom You'll be happy to know power functions are easier to work with and implement than logarithms because you do not have to worry about logarithm limits

Here is how you do that with a power function

```javascript
function linearToPowerScale(linearValue, minValue, maxValue, exponent) {
  // Clamp the linearValue between 0 and 1
  const clampedValue = Math.max(0, Math.min(1, linearValue));

  //Calculate the scaled value
  const scaledValue = Math.pow(clampedValue, exponent);

  //Map the scaled value to the target value range
  return minValue + (maxValue - minValue) * scaledValue;
}

//Example usage
const sliderPosition = 0.8 // slider position between 0 and 1
const minimum = 1; //minimum desired value
const maximum = 1000; //maximum desired value
const exponent = 2; //power of 2 makes changes more pronounced at the higher end of the slider
const poweredValue = linearToPowerScale(sliderPosition, minimum, maximum, exponent);
console.log(poweredValue)
```

This one is even simpler it basically applies the exponent to your slider position before scaling it to the output range The higher the exponent the bigger the difference between your low and high end

Here’s a gotcha I stumbled upon once when working on a photo editing software we needed to have a color temperature slider that had its own weird curve we wanted the slider to be extremely sensitive around the neutral color point (say 6500K) but far less sensitive in blue and orange zones so log scale wasn't going to cut it for this one We ended up using a polynomial curve to map the linear slider input to the desired color temperature This let us fine-tune the behavior in very specific parts of the scale It was a real mess to get right let me tell you the users kept giving feedback about the slider's feel and it ended up with us doing lots of tests and A/B testing sessions

So here’s how you could create a very flexible curve via a polynomial:

```javascript
function linearToPolynomialScale(linearValue, coefficients) {
  // Clamp the linearValue between 0 and 1
  const clampedValue = Math.max(0, Math.min(1, linearValue));

  let result = 0;
  for (let i = 0; i < coefficients.length; i++) {
    result += coefficients[i] * Math.pow(clampedValue, i);
  }

  return result;
}

//Example Usage
const sliderPosition = 0.5;
const polynomialCoefficients = [0, 10, -5]; // coefficients for 0th 1st and 2nd degree polynomial
const polynomialResult = linearToPolynomialScale(sliderPosition, polynomialCoefficients);
console.log(polynomialResult) // result will vary depending on the coefficients
```

This is pretty flexible all you need to do is tune the coefficient array to create any arbitrary curve you desire The coefficients here go from lower to higher degree i e `[a, b, c]` for `a + bx + cx^2` This is much more complex than our previous examples but it allows you to create any non linear behavior you want

Now some resources if you wanna dive deeper into this stuff If you are an academic type check out "Numerical Recipes in C" it might sound ancient but it goes over how to create very precise and versatile numerical algorithms it's more about the math side but it has great stuff about mapping functions In terms of books "Graphics Gems Series" those books are a gold mine for practical algorithms often used in graphical tools You'll find different ways to do all sorts of mappings in there Its a treasure trove of the tricks old software developers used to use before Stack Overflow was a thing

Oh a quick joke i saw recently at the programming camp a programmer’s cat was stuck in an infinite loop it kept going "meow meow meow".

Anyway those books will give you a very good understanding of non linear mapping for all sorts of things its not just for sliders you could use them for animation or physics simulations or pretty much anything that needs more complex than a linear mapping I would recommend you focus on experimenting to learn how to implement them but the theory behind it is equally as important so dont let it get into the background if you know what i mean

And that's pretty much it it is not rocket science I hope that helps you out in your non linear adventures it’s really just about understanding which math function fits your problem and then figuring out how to apply it
