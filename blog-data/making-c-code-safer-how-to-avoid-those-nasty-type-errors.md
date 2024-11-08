---
title: "Making C Code Safer: How to Avoid Those Nasty Type Errors"
date: '2024-11-08'
id: 'making-c-code-safer-how-to-avoid-those-nasty-type-errors'
---

```c
typedef struct { unsigned v; } cent_t;
typedef struct { unsigned v; } dollar_t;

#define DOLLAR_2_CENT(dollar) ((cent_t){ .v = 100 * (dollar).v })

void calc(cent_t amount) {
    // expecting 'amount' to semantically represents cents...
}

int main(int argc, char* argv[]) {
    dollar_t amount = { .v = 50 };
    calc(DOLLAR_2_CENT(amount));  // ok
    calc(amount);                 // raise warning
    return 0;
}
```
