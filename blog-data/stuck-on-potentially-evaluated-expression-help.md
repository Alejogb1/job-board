---
title: "\"Stuck on 'Potentially Evaluated Expression' - Help!\""
date: '2024-11-08'
id: 'stuck-on-potentially-evaluated-expression-help'
---

```c++
struct S {
    static float f; // declared but not defined
};

decltype(&S::f) p1; // since &S::f isn't potentially evaluated, this is well-formed

float* p2 = &S::f; // this is ill-formed
```
