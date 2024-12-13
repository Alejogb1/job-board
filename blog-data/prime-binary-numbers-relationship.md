---
title: "prime binary numbers relationship?"
date: "2024-12-13"
id: "prime-binary-numbers-relationship"
---

Alright so you're asking about prime binary numbers and their relationships huh yeah I've been down that rabbit hole a few times let me tell you

It's not as straightforward as decimal primes obviously because binary is base-2 and primes are generally defined in terms of divisibility and factors in base-10 but there are some cool patterns and things you can explore especially when you start looking at binary representations of prime numbers

First things first remember that a prime number is a whole number greater than 1 that has no positive divisors other than 1 and itself right So a number like 2 3 5 7 11 13 they are all prime in base 10 but when converted to binary its a different ball game

We arent changing the number fundamentally but we are changing the representation I had to get that in my head back in the day it took me a while to grasp that one honestly back in my university days i spent countless hours trying to find some mathematical correlation between decimal representation of primes and the same numbers when represented in base 2 It was a dead end but I did end up with a better understanding of number theory So what I learned was that any correlation we may find would be due to numerical coincidence rather than a fundamental mathematical relationship its kind of obvious right but I was young and foolish

Now about the actual question which I understood to be about patterns and relationships between prime numbers when represented in binary I think what you are asking is if there are certain bit patterns that are more likely to represent a prime number compared to others right

Well there are no magic bit sequences or patterns that tell you if a binary number is a prime number we cant just look at a number and say oh yeah that's a prime by its binary representation that I know this for sure There is no short cut algorithm for finding prime numbers that would let you avoid the computational heavy and time consuming primality tests like Miller-Rabin you know which is probabilistic or AKS which is deterministic but very slow

However there are some basic observations you can make

Like for instance we know that any binary number ending in 0 is even and therefore not prime except for the number 2 which is '10' in binary so no pattern there

Also if a binary number has more than 2 digits it has to end with a 1 to be prime since any binary ending in a zero is divisible by two simple stuff I know but important to remember

But dont fall into the trap of thinking that all odd binary numbers are primes they are not.

Let me give you some examples of numbers in decimal and their binary equivalents

Decimal Prime Binary
2 10
3 11
5 101
7 111
11 1011
13 1101
17 10001
19 10011
23 10111
29 11101
31 11111
37 100101
41 101001
43 101011
47 101111

You might notice some stuff like the fact that many primes in binary have many 1's this is just a coincidence though you cant rely on that there are plenty of primes with more 0's than 1's

Also you might notice that many primes are almost a sequence of 1's like 7 31 this is because they represent numbers close to a power of 2 but one less than the power of 2 so if we have all ones it will be 2^n-1 and that can be prime but usually isn't

There is also another very useful way to visualize it is by counting the number of ones in the binary representation of each prime we can see if there is some sort of distribution going on some trend but I bet you won't find anything useful there except another distraction. It is fun to explore though

One thing that actually matters and is a property that has an actual effect is the mersenne primes mersenne primes are primes of the form 2^n-1 where n is also prime numbers These prime numbers in binary would always have the form of all ones ie 11111111 and so on these are very important in number theory for reasons that would be too advanced for this response but it's worth mentioning because it's one of the only tangible relations there is between prime and binary

I would strongly recommend reading 'An Introduction to the Theory of Numbers' by G H Hardy and E M Wright for a much much deeper dive into this sort of stuff if you are into that sort of thing

Now lets get to the code part because thats what most of us care about am I right Lets show how to check if a number is a prime by converting it first to binary then checking if that binary form represents a prime

Here is a python snippet to check for primality

```python
def is_prime(n):
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


def decimal_to_binary(n):
    return bin(n)[2:]  # remove '0b' prefix


def check_binary_is_prime(n):
    binary_representation = decimal_to_binary(n)
    decimal_equivalent = int(binary_representation, 2)
    return is_prime(decimal_equivalent)


#lets test with a few examples
print(check_binary_is_prime(2))
print(check_binary_is_prime(3))
print(check_binary_is_prime(5))
print(check_binary_is_prime(7))
print(check_binary_is_prime(10))

```

This example will just check if the decimal representation of the binary number is prime no relationship searching here but just checking the nature of the number I hope that is clear

Now I know you were looking for relationships between the binary representation and primality but the relationship is not as direct as we would like to be I can however show you how to generate primes and how they look in binary its not the relationship you were looking for I know but its something that could help you visualize it better

```python
def sieve_of_eratosthenes(limit):
    primes = []
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False

    p = 2
    while p * p <= limit:
        if is_prime[p]:
            for i in range(p * p, limit + 1, p):
                is_prime[i] = False
        p += 1
    for p in range(2, limit + 1):
        if is_prime[p]:
            primes.append(p)
    return primes

def display_primes_in_binary(limit):
    primes = sieve_of_eratosthenes(limit)
    for prime in primes:
       print(f"Decimal {prime} is Binary {decimal_to_binary(prime)}")


display_primes_in_binary(100)
```

This will show you how primes look in binary as a sequence nothing fancy but you might find something interesting out of it although I doubt it

Ok lets do one more example with Mersenne numbers so you can see how those look like remember those are of the form 2^p-1 and p must be a prime number

```python
def generate_mersenne_numbers(p):
  return (2**p) -1


def check_mersenne_is_prime(p):
    mersenne_number = generate_mersenne_numbers(p)
    return is_prime(mersenne_number)


for p in [2,3,5,7,13,17,19,31,61]:
    if check_mersenne_is_prime(p):
        mersenne=generate_mersenne_numbers(p)
        print (f"The Mersenne {mersenne} number for prime {p} is prime and in binary is {decimal_to_binary(mersenne)} it has {len(decimal_to_binary(mersenne))} ones")
    else:
        mersenne=generate_mersenne_numbers(p)
        print(f"The Mersenne {mersenne} number for prime {p} is not prime and in binary is {decimal_to_binary(mersenne)}")


```

This last one is a bit more specific since it will calculate the first mersenne numbers and display their binary form and check if they are prime

So basically to sum it all up there is no magic direct relationship between the binary pattern of a number and its primality other than the obvious stuff I mentioned with even and odd numbers there is not a hidden language with bits we can leverage to find primes

Prime numbers are weird and unpredictable at least for me sometimes i think i finally understand them and then they do some unexpected stuff its like the world of prime numbers is telling you I am a prime number deal with it

Its not my fault you thought there was some secret handshake between binary and primes ok I'll stop I know its not funny at all sorry

But in all seriousness if you want to understand prime numbers I suggest reading more about number theory specifically about primality tests and the properties of primes like distribution etc If there is a relationship between primes and binary representations it is very subtle and I really doubt it will have any practical use in our lifetime

I hope this helps even if it is not what you expected but i hope I made it clear there is no relationship other than the fact they are the same number represented in different bases
