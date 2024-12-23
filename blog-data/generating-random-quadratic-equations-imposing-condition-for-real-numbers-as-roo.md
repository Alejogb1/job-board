---
title: "generating random quadratic equations imposing condition for real numbers as roo?"
date: "2024-12-13"
id: "generating-random-quadratic-equations-imposing-condition-for-real-numbers-as-roo"
---

 so you wanna generate random quadratic equations but with a twist you need the roots to be real not just some imaginary numbers chilling in the complex plane I get it I've been there done that probably more times than I care to remember So let's dive in

First off let's nail down what we're dealing with A quadratic equation is your classic ax² + bx + c = 0 The roots the solutions those x values are what we're after and you need them real not some i nonsense To make that happen we need to control the discriminant the part under the square root in the quadratic formula that's b² - 4ac If that discriminant is zero or positive then bingo real roots If it's negative then we're in imaginary land and we don't wanna go there today

So how do we generate these things while guaranteeing real roots That's the fun part It's not just about random numbers you have to have some constraints to make it work Here's the deal

The most straightforward way is to generate b and then figure out what we need for a and c to make b² - 4ac >= 0 lets go with this approach

**Python Example 1: Generate with conditions**

```python
import random
import math

def generate_real_quadratic():
    b = random.uniform(-10, 10)
    # we want b^2 - 4ac >= 0 so we pick random a and random c from a range 
    # that we ensure will make 4ac <= b^2
    # Let's choose a random a in [-10,10] that is not 0 and
    # choose c in a range around 0.
    a=random.uniform(-10,10)
    while a == 0:
      a=random.uniform(-10,10)

    c_max = (b**2) / (4*abs(a))
    
    c = random.uniform(-c_max, c_max) if c_max >=0 else 0
    
    return a, b, c


for _ in range(5):
    a, b, c = generate_real_quadratic()
    print(f"Equation: {a:.2f}x² + {b:.2f}x + {c:.2f} = 0")
    discriminant = b**2 - 4*a*c
    if discriminant >= 0:
      print(f"Discriminant: {discriminant:.2f} real roots")
    else:
      print(f"Discriminant: {discriminant:.2f} complex roots something is wrong ")
```

 so what's going on here I am generating random b with some range of values and then I am trying to make the value of 4ac small enough so that it will be less than b². This can be done by choosing range of values of c centered around 0. We have to be careful here because if a and c have different signs we are guaranteed a real result. If we don't take this into account we might get many of the generated discriminants not being real and we want to make sure all are real.

Now sometimes you might want more control over the roots directly If you want a particular set of real roots r1 and r2 we could just work backwards like it's highschool math all over again we can make it so a=1 and we just need to calculate b and c

**(Python Example 2: Generate from roots)**

```python
import random

def generate_quadratic_from_roots():
    r1 = random.uniform(-10, 10)
    r2 = random.uniform(-10, 10)
    
    b = -(r1 + r2)
    c = r1 * r2

    return 1, b, c


for _ in range(5):
    a, b, c = generate_quadratic_from_roots()
    print(f"Equation: {a:.2f}x² + {b:.2f}x + {c:.2f} = 0")
    discriminant = b**2 - 4*a*c
    if discriminant >= 0:
      print(f"Discriminant: {discriminant:.2f} real roots")
    else:
      print(f"Discriminant: {discriminant:.2f} complex roots something is wrong ")
```

This code will generate a random set of roots and then calculate b and c such that when we create the equation with a = 1, the equation created will have roots r1 and r2. I've used this method many times.

I remember back in my early coding days I was working on this physics simulation and I needed to generate equations for projectile motion Well that's basically a quadratic problem. So I used the method I just talked about to get realistic trajectories. It was a pain to debug at the beginning getting a bunch of complex numbers popping out when it was supposed to be a real world trajectory

And then there is another way to guarantee that real roots exist by making sure that 4ac is always less than b² you pick random a and b and c and then you ensure that 4ac is less than b² by reducing the absolute value of the c that you have generated in each iteration and making sure that if 4ac is already less than b² no change has to be done

**(Python Example 3: Generate by scaling c)**

```python
import random

def generate_real_quadratic_scaling_c():
    a = random.uniform(-10, 10)
    while a == 0:
        a = random.uniform(-10, 10)
    b = random.uniform(-10, 10)
    c = random.uniform(-10, 10)
   
    
    if b**2 < 4 * a * c:
      c=c * ((b**2) / (4*a*c) * 0.9)
      
    return a, b, c


for _ in range(5):
    a, b, c = generate_real_quadratic_scaling_c()
    print(f"Equation: {a:.2f}x² + {b:.2f}x + {c:.2f} = 0")
    discriminant = b**2 - 4*a*c
    if discriminant >= 0:
      print(f"Discriminant: {discriminant:.2f} real roots")
    else:
      print(f"Discriminant: {discriminant:.2f} complex roots something is wrong ")
```

This method is a bit more complex since it will have to scale c at each iteration. It is important to mention that this method will generate a lot of small c values in order to make b² >= 4ac and that will influence your equation distribution.

And yes sometimes it looks like b² - 4ac is smaller than zero because of float point approximation and thats the point of that last check

So a lot of the time when i am implementing those equations I use the numpy package. I just want to mention that because I usually do not do it using plain python code. I mean why would you if you have matrix calculations and numerical problems in the numpy package? But you could use this as a base.

For diving deeper into the math behind quadratic equations and their roots I'd highly recommend "Numerical Methods That Work" by Forman S Acton it's a classic. You could also check "Introduction to Numerical Analysis" by Endre Suli and David F Mayers which covers more general numerical topics but has a solid section on root finding It's dense but it explains things rigorously. If you want to dive into root finding algorithms check out the book "Applied Numerical Methods" by Steven Chapra it's a really good resource.

Now before we wrap up here's a little joke I heard in a lecture hall once Why was the math book sad Because it had too many problems and not enough real solutions get it haha ok back to the code I swear.

that's pretty much it you've got a few ways to generate random quadratic equations that always have real roots Just pick the method that works best for your needs. Keep in mind that different methods might have different biases in the distribution of the generated equations, so be mindful of your specific application. Happy coding and remember math is fun especially when you can make it do cool things
