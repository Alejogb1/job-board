---
title: "binary relation among prime numbers?"
date: "2024-12-13"
id: "binary-relation-among-prime-numbers"
---

Okay so you're asking about binary relations among prime numbers huh Yeah I've been down this rabbit hole before more times than I care to admit let me tell you its a fun ride a frustrating one but a fun one

Right off the bat when you say "binary relation" its pretty broad We're talking about sets of ordered pairs where each pair consists of two prime numbers Lets call that set R So R = {(p1 p2) | p1 p2 are prime numbers and some condition holds} The condition is the real meat of the problem

The simplest one probably is just the equality relation R = {(p p) | p is a prime number} you know each prime is related to itself not too exciting I've used that for some basic checks in code back in the day but not exactly groundbreaking stuff. I remember I once spent a whole weekend trying to optimise that thinking I could squeeze more perf out of it just to realise it's not really the bottleneck. Good times good times.

A slightly more useful one is a less than relation R = {(p1 p2) | p1 < p2 p1 p2 are prime numbers} this one is useful when you're dealing with prime number sequences or trying to search or something. You can imagine how to implement that its just a simple comparison. I remember coding this in early python days and it wasn't as performant as I would've liked to.

Then you have stuff like the "is a divisor of" relation but that doesn't make a lot of sense with primes since primes only have two divisors 1 and themselves. Unless you are talking about prime factorisation of non-primes but I don't think that's what you meant. Still it's a reminder to be careful what relation means in each given context. I have seen many a junior dev confuse that one.

So lets think of some more interesting relations. How about the "differ by two" relation also known as twin primes R = {(p1 p2) | |p1 - p2| = 2 p1 p2 are prime numbers} this one is cool because its not immediately obvious whether there are infinite pairs that satisfy that relation. This is actually an open math problem as the twin prime conjecture says there are infinitely many pairs but nobody has been able to prove it yet. I have seen many attempts in my time many of them wrong very very wrong but it is a fascinating area.

The "sum to another prime" relation R = {(p1 p2) | p1 + p2 = p3 p1 p2 p3 are prime numbers} is another example Its not really a binary relation in the strictest sense because its using three primes but its still a relation among primes and its quite fun to play with.

Here is a Python example that implements the "less than" relation:

```python
def is_prime(n):
  if n <= 1:
    return False
  for i in range(2, int(n**0.5) + 1):
    if n % i == 0:
      return False
  return True

def less_than_relation(limit):
    primes = [i for i in range(2,limit+1) if is_prime(i)]
    relations = []
    for p1 in primes:
        for p2 in primes:
            if p1<p2:
                relations.append((p1,p2))
    return relations

print(less_than_relation(20))
```

Here is a Python example that implements the twin primes relation:

```python
def is_prime(n):
  if n <= 1:
    return False
  for i in range(2, int(n**0.5) + 1):
    if n % i == 0:
      return False
  return True

def twin_prime_relation(limit):
    primes = [i for i in range(2,limit+1) if is_prime(i)]
    relations = []
    for i in range(len(primes)):
        for j in range(i+1,len(primes)):
            if abs(primes[i] - primes[j]) == 2:
                relations.append((primes[i] , primes[j]))
    return relations


print(twin_prime_relation(100))
```

And here is a Python example that implements the sum to a prime relation:

```python
def is_prime(n):
  if n <= 1:
    return False
  for i in range(2, int(n**0.5) + 1):
    if n % i == 0:
      return False
  return True

def sum_to_prime_relation(limit):
    primes = [i for i in range(2,limit+1) if is_prime(i)]
    relations = []
    for p1 in primes:
      for p2 in primes:
         sum_val = p1 + p2
         if sum_val in primes:
             relations.append((p1,p2,sum_val))

    return relations

print(sum_to_prime_relation(10))
```
These are just examples using python you can do this in any programming language the logic is basically the same just different syntax. You can optimize these further especially the prime checking function with some clever tricks I remember using something called a Sieve of Eratosthenes it's a way faster for generating primes.

The cool thing is that you can invent so many of these relations there's probably a new one you can invent right now and that's the beauty of maths. I remember when I was at University I thought that math was all figured out and there was nothing new I could discover but I was so wrong. It's still full of amazing ideas and open problems.

Now resources you should look at for more in depth math theory you should absolutely look at books like "Number Theory Structures Examples and Problems" by Titu Andreescu or if you like something more accessible "An Introduction to the Theory of Numbers" by Ivan Niven is a good one. Those are classic books so I am sure you will find it useful. For more programming and implementations on this type of stuff you should check out stuff by Project Euler which is a really amazing resource. Oh and if you like algorithms and performance you should look into "Introduction to Algorithms" by Thomas H Cormen thats the bible for algorithm stuff. And if you think its too hard dont worry I've spent many nights crying over that book when I was starting out. You will get there.

I hope that was useful. If you need more stuff just ask away this is the kind of thing I get excited about. I actually have a whiteboard full of prime number related relations from about 3 years ago I need to finally clean up that mess one day.

Oh and one more thing what do you call a prime number that is not willing to work with anyone? *A number with too much prime-ide*. Get it prime-ide prime-pride ah never mind. Just ignore me. I need more coffee.
