---
title: "How to Quickly Tell if a Decomposition is Lossless or Lossy: A Simple Trick"
date: '2024-11-08'
id: 'how-to-quickly-tell-if-a-decomposition-is-lossless-or-lossy-a-simple-trick'
---

```
# Decomposition 1: R1(AB), R2(ADE), R3(BCD)

# Check if AB is a superkey of R1:
# - AB -> C (given)
# - C -> E (given)
# - E -> A (given)
# Therefore, AB -> CDE, which means AB is a superkey of R1.

# Check if ADE is a superkey of R2:
# - No FD in F has ADE as its LHS, so ADE is not a superkey of R2.

# Thus, the first decomposition is lossy.

# Decomposition 2: R1(BCD), R2(ACE), R3(BD)

# Check if BCD is a superkey of R1:
# - B -> D (given)
# - B -> D (given)
# - C -> E (given)
# - E -> A (given)
# Therefore, BCD -> ADE, which means BCD is a superkey of R1.

# Check if ACE is a superkey of R2:
# - C -> E (given)
# - E -> A (given)
# Therefore, C -> EA, which means C is a superkey of R2.

# Thus, the second decomposition is lossless.
```
