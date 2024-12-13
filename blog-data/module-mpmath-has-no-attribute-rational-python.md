---
title: "module 'mpmath' has no attribute 'rational' python?"
date: "2024-12-13"
id: "module-mpmath-has-no-attribute-rational-python"
---

Okay so you're running into that old chestnut where `mpmath` is playing hard to get with its `rational` attribute huh? Been there done that got the t-shirt and a few sleepless nights debugging it let me tell you

First off let's break it down real quick `mpmath` yeah that's the multi-precision floating-point library the one you reach for when regular floats just won't cut it because they're like trying to measure the galaxy with a ruler right

Now about `rational` its usually in modules that deal with symbolic math or exact arithmetic stuff Think fractions not just decimal approximations So seeing that `mpmath` error is a clear sign you are trying to use that concept in the wrong library

Let me spill a bit of my history with this type of situation I was working on a project a while back where I needed to calculate some high precision values that included dividing two huge numbers and storing the result not as an approximate floating number but as an exact fraction for further symbolic processing My initial thought as it should have been yours to use the `mpmath` library because the initial calculations required very high precision However I was dumbly trying to call that specific `rational` function which of course it was not there. I banged my head on the keyboard for a solid hour before realizing that `mpmath` is not meant for exact rational representation It's about arbitrary precision *floating-point* numbers not symbolic rational numbers That project ended up using `sympy` which has much better capabilities in that regard

So yeah `mpmath` doesn't have a `rational` attribute and it's not supposed to its focus is on numerical approximations with arbitrary precision not symbolic or exact representations

Now to fix your code which we can not see of course but let me give you some code snippets for a fix

If you are dealing with just storing an exact fraction you can use Python's built-in `fractions` module and the result is like this

```python
from fractions import Fraction

numerator = 234234234234
denominator = 345345345345

exact_fraction = Fraction(numerator,denominator)

print(exact_fraction)

result = exact_fraction * 2

print(result)
```

See that simple no `mpmath` involved Here we just store the exact fraction and if required we can do more symbolic maths operations with the result

This way you get to keep the exact rational number without any floating-point shenanigans

But if you were using the arbitrary precision of `mpmath` for a real reason like calculating very precise numerical values and also need a symbolic math solution well the solution is `sympy`. `sympy` deals with symbolic calculations including exact rational numbers. Check it

```python
import sympy
from mpmath import mp

mp.dps = 50

numerator = mp.mpf("1234567890.12345678901234567890")
denominator = mp.mpf("9876543210.98765432109876543210")

mp_division = numerator/denominator
print(mp_division)

sympy_numerator = sympy.Rational(str(numerator))
sympy_denominator = sympy.Rational(str(denominator))

rational_division = sympy_numerator/sympy_denominator

print(rational_division)

result = rational_division * sympy.Rational(3,2)

print(result)

```

What did we do here We used `mpmath` first to store the numbers with their high precision and we store the division of those two numbers.

And what we did next is the most important part. We converted the `mpmath` strings into `sympy` rational numbers This way we get the precise values and also the symbolic math power that `sympy` provides for further calculations.

Now if what you need is to store the results as floating point but keep the precision use `mpmath` itself without the symbolic rational number stuff like this

```python
from mpmath import mp

mp.dps = 50

numerator = mp.mpf("1234567890.12345678901234567890")
denominator = mp.mpf("9876543210.98765432109876543210")

result_division = numerator/denominator

print(result_division)

result_multiplication = result_division * mp.mpf("2")

print(result_multiplication)

```

Here you are doing the calculations with the `mpmath` values directly and everything is high precision.

And hey don't worry this confusion with libraries is totally normal we all have to learn the hard way which one does what especially with numerical math in python

Now for some extra tips and where to get more info because you need to study more.

First about `mpmath`: its a fantastic library for arbitrary precision floating point stuff but if you want more precise information in the underlying math I recommend "NIST Handbook of Mathematical Functions" that will have detailed information about the concepts behind it. And if you need a more practical understanding with real examples then check the source code documentation of `mpmath` itself and its examples it is always good to learn from the experts.

Now about `sympy`: this one is all about symbolic math. So you will get more information in books like "Concrete Mathematics" it has a good overview on the algebra side of things. Also I found that "Computer Algebra and Symbolic Computation" is very useful if you want to dive deep in the algorithms behind symbolic calculations so check it out as well.

And now one final tip the one you are gonna appreciate the most. If you find yourself banging your head against a problem like this for too long take a break go for a walk because the answer is sometimes found in the shower. It's how we all debug right

Hope this was helpful I know it was a long answer but you had to be reminded about the details.
