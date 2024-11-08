---
title: "Pointers vs. References: Which Data Member Wins? 🤔"
date: '2024-11-08'
id: 'pointers-vs-references-which-data-member-wins'
---

```cpp
class A {};

class B
{
public:
    B(const A& a) : a(a) {}
    const A& a;
};

class C
{
public:
    C() : b(a) {} 
private:
    A a;
    B b; 
};
```
