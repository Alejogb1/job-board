---
title: "How can polymorphic vectors be implemented in C++ without object slicing?"
date: "2025-01-30"
id: "how-can-polymorphic-vectors-be-implemented-in-c"
---
Object slicing presents a fundamental challenge when attempting to store derived class objects in a vector of base class objects. The core issue stems from C++'s handling of object copies. When assigning a derived object to a base object, the derived portion is discarded, leading to data loss and the loss of polymorphic behavior. Avoiding this requires indirect storage and careful management of object lifetimes.

The crux of the matter lies in recognizing that vectors store objects directly within their memory. Polymorphism, on the other hand, relies on the dynamic binding of virtual functions, which is determined by the object's actual type, not the type of the variable pointing to it. Storing objects directly in a vector of base type forces all elements to have the size and characteristics of the base type, resulting in object slicing.

I've encountered this situation frequently while developing game engine components, particularly in managing entities with various derived behaviors under a common interface. My experience shows that we can circumvent object slicing by employing a vector of pointers or a vector of smart pointers instead of a vector of objects.

**1. Vector of Raw Pointers**

The most straightforward method is using a `std::vector` of raw base class pointers. This approach stores the addresses of dynamically allocated derived class objects. Polymorphism is preserved because the pointer points to the entire derived object, and virtual functions will resolve correctly during runtime. However, this method necessitates explicit memory management, including manual deallocation to avoid memory leaks.

Here’s an example:

```cpp
#include <iostream>
#include <vector>

class Base {
public:
  virtual void display() const {
    std::cout << "Base class" << std::endl;
  }
  virtual ~Base() = default;
};

class Derived : public Base {
public:
  void display() const override {
    std::cout << "Derived class" << std::endl;
  }
};

int main() {
    std::vector<Base*> entities;

    entities.push_back(new Derived()); // Dynamically allocate
    entities.push_back(new Base());

    for (Base* entity : entities) {
        entity->display(); // Calls derived class version, preserving polymorphism
    }

    // Manual deallocation
    for (Base* entity : entities) {
        delete entity;
    }

    return 0;
}
```

In this example, the `entities` vector stores `Base*`, pointers to `Base` objects. When we add `new Derived()` to the vector, we create a `Derived` object on the heap, and then add the *pointer* to this heap object to the vector. As we iterate through the vector and call `entity->display()`, the derived class’s implementation of the `display()` virtual function is executed correctly, demonstrating polymorphic behavior. However, a major caveat is the manual memory management. The `delete` operation must be explicitly called on each pointer to prevent memory leaks. Neglecting this detail can be a source of significant bugs.

**2. Vector of Smart Pointers**

To automate memory management, we can use smart pointers, specifically `std::unique_ptr`. `std::unique_ptr` offers exclusive ownership of the pointed-to object, automatically deleting the object when the `unique_ptr` goes out of scope.  This eliminates the need for manual `delete` calls, reducing the possibility of memory leaks.

Here’s an example of how to use it:

```cpp
#include <iostream>
#include <vector>
#include <memory>

class Base {
public:
    virtual void display() const {
        std::cout << "Base class" << std::endl;
    }
  virtual ~Base() = default;
};

class Derived : public Base {
public:
    void display() const override {
        std::cout << "Derived class" << std::endl;
    }
};

int main() {
  std::vector<std::unique_ptr<Base>> entities;

  entities.push_back(std::make_unique<Derived>());
  entities.push_back(std::make_unique<Base>());


    for (const auto& entity : entities) {
        entity->display();
    }
   // No explicit delete needed

   return 0;
}
```

In this improved version, the `entities` vector stores `std::unique_ptr<Base>`.  We use `std::make_unique` to create and initialize each `std::unique_ptr`, which also handles the underlying dynamic allocation.  The `for` loop iterates through each element, and `entity->display()` correctly calls the derived version where appropriate. When the vector goes out of scope, all `unique_ptr` elements are destroyed, in turn automatically deleting the allocated objects they pointed to and thus preventing memory leaks. I have found the `unique_ptr` method invaluable for projects with more complex object management requirements.

**3. Vector of Shared Pointers with Type Erasure**

For situations requiring shared ownership of objects, `std::shared_ptr` is a better fit. However, this comes with increased overhead due to the reference counting mechanism. Furthermore, a `shared_ptr` alone doesn’t completely eliminate the possibility of memory leaks if a circular dependency is created. To mitigate this, one approach is to use a combination of shared ownership and a form of type erasure to store different types of derived objects.

Consider the following example:

```cpp
#include <iostream>
#include <vector>
#include <memory>
#include <functional>

class Base {
public:
  virtual void display() const {
    std::cout << "Base class" << std::endl;
  }
  virtual ~Base() = default;
};

class Derived : public Base {
public:
  void display() const override {
    std::cout << "Derived class" << std::endl;
  }
};

class AnotherDerived : public Base{
public:
  void display() const override {
     std::cout << "Another Derived class" << std::endl;
  }
};


int main() {
   std::vector<std::function<void()>> actions;

   std::shared_ptr<Base> derivedPtr = std::make_shared<Derived>();
   std::shared_ptr<Base> anotherDerivedPtr = std::make_shared<AnotherDerived>();

    actions.emplace_back([derivedPtr]() { derivedPtr->display(); });
    actions.emplace_back([anotherDerivedPtr](){ anotherDerivedPtr->display(); });

    for(const auto& action : actions){
        action();
    }
  return 0;
}
```

This example uses a vector of `std::function<void()>`, employing lambda captures to implement type erasure. Each `std::function` stores an action that invokes the `display()` method on an object pointed to by a `std::shared_ptr`. Crucially, this technique allows us to combine objects with shared ownership and execute polymorphic calls while removing the need for a single storage type.  The lambdas encapsulate the specific derived type as a closure, enabling the `display()` function to behave polymorphically while storing an action rather than the object itself. This approach effectively decouples the storage of the action and the underlying object.  I find this type erasure strategy beneficial in systems requiring a higher level of abstraction in how actions are stored and executed.

**Resource Recommendations**

For a deeper understanding of these concepts, I strongly recommend consulting several resources.  First, study the core C++ language specifications, specifically the sections on virtual functions, polymorphism, and object lifetime. Second, research modern C++ coding standards regarding memory management; this will provide guidance on best practices for using raw pointers and smart pointers effectively. Finally, explore design patterns that utilize polymorphism, such as the Strategy pattern, which is directly relevant to situations requiring a vector of interchangeable behaviors. These resources will provide a broad context and practical advice for implementing effective polymorphic solutions.
