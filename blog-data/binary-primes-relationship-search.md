---
title: "binary primes relationship search?"
date: "2024-12-13"
id: "binary-primes-relationship-search"
---

 so binary primes relationship search right Yeah I've been down that rabbit hole before more times than I care to admit It's one of those things that sounds deceptively simple on paper like just figure out if binary representations of prime numbers have some sort of hidden relationship right But man the devil's in the details let me tell you

First off let's level set because "relationship" is a pretty broad term We're talking about patterns right Not just any pattern a correlation a predictable connection between the binary strings of prime numbers I've spent countless late nights fueled by cold coffee and questionable pizza trying to crack this specific nut

My early attempts were pretty naive I remember back in my uni days thinking a simple for loop checking the hamming distance between consecutive prime numbers would reveal some fundamental truth I wrote this quick Python script it was embarrassingly straightforward

```python
def to_binary(n):
    return bin(n)[2:]

def hamming_distance(s1, s2):
    s1 = s1.zfill(max(len(s1), len(s2)))
    s2 = s2.zfill(max(len(s1), len(s2)))
    return sum(c1 != c2 for c1, c2 in zip(s1, s2))

def find_primes(limit):
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

limit = 100
primes = find_primes(limit)
binary_primes = [to_binary(p) for p in primes]

for i in range(len(binary_primes) - 1):
    dist = hamming_distance(binary_primes[i], binary_primes[i+1])
    print(f"Hamming distance between {primes[i]} ({binary_primes[i]}) and {primes[i+1]} ({binary_primes[i+1]}): {dist}")
```

Yeah that didn't work out spectacularly The hamming distances were all over the place no consistent trend nothing that jumped out and screamed correlation It was like looking for order in a bowl of alphabet soup

So I took a step back I had to get more rigorous I read a lot of papers on number theory specifically those dealing with prime distribution and representation I strongly recommend the book "An Introduction to the Theory of Numbers" by Niven Zuckerman and Montgomery it's dense but it's a goldmine for these kinds of problems Also the paper "The Distribution of Primes" by Heath-Brown is crucial to understand the complexities involved This wasn't about brute forcing it was about understanding the underlying mathematical landscape

Then I started looking at bit patterns not just the distance between them Maybe there were specific recurring substrings within those binary representations I wrote a little program in C this time performance was key since I was dealing with larger primes It was faster than my Python experiments for sure but it didn't reveal any obvious pattern either

```c
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>

char* toBinary(int n) {
  char* binary = malloc(sizeof(char) * 33);
  if (binary == NULL) return NULL;
  int i = 31;
  for (; i >= 0; i--) {
    binary[31 - i] = (n & (1 << i)) ? '1' : '0';
  }
  binary[32] = '\0';
  while (*binary == '0' && *(binary + 1) != '\0') binary++;
  return binary;
}

bool isPrime(int n) {
    if (n <= 1) return false;
    if (n <= 3) return true;
    if (n % 2 == 0 || n % 3 == 0) return false;
    for (int i = 5; i * i <= n; i += 6) {
        if (n % i == 0 || n % (i + 2) == 0) return false;
    }
    return true;
}

int main() {
    int limit = 100;
    for (int i = 2; i <= limit; i++) {
        if (isPrime(i)) {
            char* binary = toBinary(i);
            if (binary == NULL) return 1;
            printf("Prime: %d Binary: %s\n", i, binary);
            free(binary);
        }
    }
    return 0;
}
```
I started thinking about other ways to examine the relationship maybe focusing on the differences of consecutive primes and its effects on the binary representations instead of just the binary representations themselves This led me to try a moving average window approach to see if differences in prime numbers themselves showed a pattern and if that pattern was also somewhat reflected in the differences of their binary forms Now this is something I can get behind and it seems more promising but still no clear pattern emerged It's like they are hiding under the cover of noise

I even explored more advanced techniques autocorrelation Fourier transforms on the binary string representation hoping to find hidden frequencies or periodic patterns nothing conclusive In a moment of frustration I remember thinking if primes were a sitcom what would they be called *'The Unpredictables'* yeah not my best moment in coding humor trust me there's another level of unfunny in my coding history

Here's the thing The seemingly random nature of primes in the decimal system might also be present in their binary form It’s a feature not a bug as they say I tried focusing on specific properties of the binary strings like the number of 1s and 0s the frequency of specific subsequences the number of transitions from 0 to 1 or 1 to 0 It’s pretty complex when you start delving into it

```python
def count_bits(binary_string, bit):
    return binary_string.count(bit)

def count_transitions(binary_string):
  transitions = 0
  for i in range(len(binary_string) - 1):
    if binary_string[i] != binary_string[i + 1]:
      transitions += 1
  return transitions

limit = 100
primes = find_primes(limit)
binary_primes = [to_binary(p) for p in primes]

for i in range(len(binary_primes)):
  ones = count_bits(binary_primes[i], '1')
  zeros = count_bits(binary_primes[i], '0')
  transitions = count_transitions(binary_primes[i])
  print(f"Prime {primes[i]} Binary {binary_primes[i]} 1s: {ones} 0s: {zeros} Transitions: {transitions}")

```

The lack of a simple deterministic rule for the generation of prime numbers is the biggest roadblock The binary representation is just another lens through which we look at this challenging mathematical problem I spent way too much time trying to find a relationship a concrete pattern in the binary representations of prime numbers and for now I think it's better to conclude that the relationship is not easily revealed or as some researchers would say there may not even be a relationship that would show a concrete pattern

The relationship if it even exists might be found in the statistical properties of these binary representations instead of their deterministic structure I think the study of prime number distribution is a more fertile area to look at before trying to find some pattern that might not exist in the binary strings of prime numbers and that would be a much better approach if a relationship needs to be found or at least that's how I'm tackling it now After all those many nights I came to realize this kind of problem needs a bit more than just a simple coding solution It requires deep understanding of number theory statistical analysis and the courage to admit defeat if the data shows there is nothing there you can't force order if it's simply not there I think I will go back to that issue but for now I think it's time for a coffee break.
