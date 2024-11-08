---
title: "Singleton Pattern:  The Quick & Easy Way to Get Started"
date: '2024-11-08'
id: 'singleton-pattern-the-quick-easy-way-to-get-started'
---

```cpp
class Singleton {
public:
    static Singleton& getInstance() {
        static Singleton instance;
        return instance;
    }

private:
    Singleton() {}
    Singleton(const Singleton&) = delete;
    Singleton& operator=(const Singleton&) = delete;
};
```
