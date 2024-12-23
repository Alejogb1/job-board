---
title: "angle between two vectors unity definition?"
date: "2024-12-13"
id: "angle-between-two-vectors-unity-definition"
---

 so you're asking about the angle between two vectors and how unity or unit vectors play into that I gotcha Been there wrestled with this exact problem more times than I care to admit especially back in my early days building 3D engines from scratch man that was a mess let me tell you

So first things first let's break down what a unit vector is A unit vector is basically a vector that has a magnitude of one it points in a certain direction but its length is always 1 We achieve this by taking a regular vector and dividing it by its own magnitude Think of it as normalizing the vector to its core direction and losing the scaling information

Now why do we care about unit vectors when calculating the angle between two vectors Well the key is the dot product The dot product of two vectors say `a` and `b` is related to the cosine of the angle between them by this formula `a.b = ||a|| ||b|| cos(theta)` where `||a||` and `||b||` are the magnitudes of vectors a and b and theta is the angle we're after

Here's the kicker If your vectors are unit vectors `||a||` and `||b||` will be 1 so our formula simplifies to `a.b = cos(theta)` That means the dot product of two unit vectors directly gives you the cosine of the angle between them Super convenient Right

To get the actual angle we need to take the inverse cosine of that dot product i e `theta = arccos(a.b)` This is much easier to work with because we don't need to calculate the magnitudes of original vectors saving some processing time particularly if you're in a performance sensitive environment

Let me show you how this pans out in some simple code examples First let's assume we have a basic vector class I am going to show this in python but this works equally with other languages with minor syntax variations

```python
import math

class Vector:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def magnitude(self):
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def unit(self):
      mag = self.magnitude()
      if mag == 0:
        return Vector(0,0,0)
      return Vector(self.x/mag,self.y/mag,self.z/mag)


    def dot(self, other):
      return self.x * other.x + self.y * other.y + self.z * other.z
```

 that’s the basic vector stuff. Now let's write the core method for getting the angle between vectors:

```python
def angle_between(vector1, vector2):
  unit_v1 = vector1.unit()
  unit_v2 = vector2.unit()
  dot_product = unit_v1.dot(unit_v2)
  # make sure dot product is in -1 to 1 range due to floating point issues
  dot_product = max(-1, min(1, dot_product))
  return math.degrees(math.acos(dot_product))
```

See how clean that is I normalize both vectors using the unit vector method then compute the dot product This gives me `cos(theta)` and then the `acos` function (inverse cosine) gets the angle itself in radians. The degrees function converts the radians to degrees for easier interpretation of the angle

Now lets test it:

```python
v1 = Vector(1, 0, 0) # unit vector in x direction
v2 = Vector(0, 1, 0) # unit vector in y direction

angle = angle_between(v1,v2)
print(f"The angle between v1 and v2 is: {angle} degrees") # this will print 90 degrees
v3 = Vector(1, 1, 0) #diagonal vector

angle = angle_between(v1,v3)

print(f"The angle between v1 and v3 is: {angle} degrees") # this will print around 45 degrees

```
Running that will output the angle between the unit vectors of the specified vectors notice the use of unit vector for calculating the angle using the dot product The code uses the cosine angle to compute the angle between two vectors

The reason I made the vector unit before the dot product is for the reasons I described earlier avoiding the multiplication of magnitudes. This way it is computationally cheaper than using the traditional equation of dot product

Now you might be thinking what if I skip the unit vector part and use `a.b / (||a|| * ||b||)` directly right?  Yes that also works but it's extra calculations that you don't need specially if this is performance bottleneck in your code If you're doing this operation a lot for millions of vectors or in a shader then skipping the norm calculations really helps Also unit vectors have other uses it's common in graphics and physics engines to see unit vectors everywhere

Also if one of your vectors is a zero vector or very close to zero you will get an error due to dividing by zero. I have encountered this very issue back in my early days of game development where I had a spaceship that did not accelerate and was always at position 0 0 0 and when I did the vector calculations I would get NaNs everywhere so I had to add checks to make sure no zero vectors were used in calculations which added unneeded complexity to the code. This is why I recommend making sure the vector magnitude is never zero or very close to zero. In case of zero magnitude return a 0 0 0 unit vector as you can see in the code above.

And yeah I have been debugging for days trying to figure out why my object always goes to zero just to remember to add an epsilon check so yeah been there seen that. Once I spent a whole week debugging that and turned out the issue was just the number was so low that is was being converted to zero which resulted in a black hole effect of everything being pulled to origin. In programming always remember that floating point numbers have low precisions and be careful with them! (Yeah I spent that week questioning my existence and going into therapy because of it).

Also if you’re working with higher-dimensional vectors the core principles are the same You just have more coordinates and the vector magnitude will take into account all of those coordinates The dot product method doesn’t change you keep summing up the products of each corresponding coordinate

And thats the gist of it! Unit vectors are incredibly useful for angles because they give us the cosine of the angle directly via the dot product it eliminates the need for extra magnitude calculations and allows for cleaner code

If you're really interested in digging deeper into vector math and linear algebra for computer graphics I recommend reading "Mathematics for 3D Game Programming and Computer Graphics" by Eric Lengyel This will set you straight with all the concepts and the math behind the graphics algorithms. Also the book "3D Math Primer for Graphics and Game Development" by Fletcher Dunn and Ian Parberry is also a great resource to get familiar with the common operations that use vector calculations

Hope this helps clarify things good luck with your vector adventures
