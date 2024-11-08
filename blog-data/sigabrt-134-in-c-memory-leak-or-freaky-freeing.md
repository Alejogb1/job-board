---
title: "SIGABRT 134 in C: Memory Leak or Freaky Freeing?"
date: '2024-11-08'
id: 'sigabrt-134-in-c-memory-leak-or-freaky-freeing'
---

```c
Interval *newInterval(int b, int e, int m) {
    Interval *interval = malloc(sizeof(Interval));
    if (interval == NULL) {
        return NULL;
    }
    interval->b = b;
    interval->e = e;
    interval->m = m;
    return interval;
}

Signal *newSignal(int size) {
    Signal *signal = malloc(sizeof(Signal));
    if (signal == NULL) {
        return NULL;
    }
    signal->intervals = malloc(size * sizeof(Interval));
    if (signal->intervals == NULL) {
        free(signal);
        return NULL;
    }
    signal->top = 0;
    signal->size = size;
    return signal;
}

Stack *newStack(int size) {
    Stack *stack = malloc(sizeof(Stack));
    if (stack == NULL) {
        return NULL;
    }
    stack->signals = malloc(size * sizeof(Signal));
    if (stack->signals == NULL) {
        free(stack);
        return NULL;
    }
    stack->top = 0;
    stack->size = size;
    return stack;
}

void pop(Signal *s, int n) {
    if (n < s->top) {
        printf("[%d,%d)@%d ", s->intervals[n].b, s->intervals[n].e, s->intervals[n].m);
        pop(s, n + 1);
    } else {
        free(s->intervals);
    }
}

void printIntervals(Stack *st) {
    for (int i = 0; i < st->top; i++) {
        pop(&(st->signals[i]), 0);
        printf("\n");
    }
    for (int i = 0; i < st->top; i++) {
        free(st->signals[i].intervals);
        free(st->signals[i]);
    }
    free(st->signals);
    free(st);
}
```
