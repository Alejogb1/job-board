---
title: "How can I return a class instance from a trait?"
date: "2024-12-23"
id: "how-can-i-return-a-class-instance-from-a-trait"
---

Alright,  Instead of diving straight into code, let's first contextualize why this can be a bit of a nuanced area in programming. I remember back in my early days with Scala, specifically dealing with some rather intricate actor systems, I faced this exact conundrum. I needed to define some common behaviors using traits but needed those behaviors to ultimately produce a concrete instance of a specific class when invoked. The typical trait definition won't give you a direct path for returning a class instance; it’s more about providing functionalities than concrete objects. Now, the trick lies in how we strategically employ abstract types and type members within the trait.

The core issue stems from the nature of traits themselves. Traits, in languages like Scala, Kotlin, and even more indirectly in Python through mixin classes, are fundamentally blueprints for behavior. They aren't meant to dictate the exact *type* of the concrete object being created. We want that flexibility. If we were to force a trait to return a specific concrete class directly, we’d essentially lose much of the benefit of using traits in the first place. The solution involves making the type returned by a method an abstract type member or an associated type in the trait. This pushes the responsibility of defining the *concrete* class to the classes that implement or inherit from the trait.

Let’s get into some code examples, illustrating this point step-by-step.

**Example 1: Basic Trait with Abstract Type**

Let's start with a simple scenario. Suppose we have a trait called `Creatable` that should provide a mechanism to create some kind of object, but we don't know or want to specify the exact type in the trait itself. Here’s how you could define this using an abstract type member:

```scala
trait Creatable {
  type CreatedType
  def create(): CreatedType
}

class ConcreteCreator extends Creatable {
    type CreatedType = String // Defines the concrete type to be String
    def create(): String = "Created a String!"
}

object Example1 {
  def main(args: Array[String]): Unit = {
    val creator = new ConcreteCreator()
    val createdInstance: String = creator.create()
    println(createdInstance)
  }
}
```

In this snippet, `Creatable` has an abstract type member `CreatedType` and an abstract `create()` method that returns an instance of `CreatedType`. The class `ConcreteCreator` then provides the concrete implementation by setting `type CreatedType = String`, and then the `create()` method returns a string. This is the core concept: the trait only declares *what* can be returned, not *which concrete class* is returned.

**Example 2: Trait with Type Parameter and Higher-Kinded Types**

Now, let’s make it slightly more involved. Imagine a scenario where you want to create instances that might wrap other values, like with a generic `Container` type. We can make this trait even more flexible using type parameters and higher-kinded types. This is where the learning gets progressively more challenging but immensely powerful. Here we'll use a type parameter `T` which specifies the type we want to have in the container class:

```scala
trait ContainerCreatable[T] {
  type ContainerType[A]
  def create(value: T): ContainerType[T]
}

class ListContainerCreator[T] extends ContainerCreatable[T] {
  type ContainerType[A] = List[A]
  def create(value: T): List[T] = List(value)
}

class OptionContainerCreator[T] extends ContainerCreatable[T] {
  type ContainerType[A] = Option[A]
    def create(value: T): Option[T] = Some(value)
}


object Example2 {
  def main(args: Array[String]): Unit = {
    val listCreator = new ListContainerCreator[Int]
    val listInstance: List[Int] = listCreator.create(10)
    println(listInstance) // Output: List(10)

    val optionCreator = new OptionContainerCreator[String]
    val optionInstance: Option[String] = optionCreator.create("Hello")
    println(optionInstance) // Output: Some(Hello)
  }
}

```

Here, `ContainerCreatable` has a type parameter `T` representing the type of the value being held by the container. The trait has `ContainerType[A]` as an abstract type member, which takes a type parameter `A` and this allows flexibility, letting the concrete classes specify the particular container type (e.g., `List`, `Option`). Both `ListContainerCreator` and `OptionContainerCreator` define the container type, `List` and `Option` respectively, and implement their own respective version of `create`.

**Example 3: Self-Referential Types for Complex Scenarios**

For the final example, consider cases where your returned instance needs a self-reference to the class that is creating it. This is trickier, but also solvable by using self-types, sometimes referred to as "self-annotations". These aren't universally supported, but if you're using a language like Scala or some less common ones they can be very helpful.

```scala
trait SelfReferentialCreatable {
  type CreatedType <: SelfReferential
  def create(): CreatedType

  trait SelfReferential {
    this: CreatedType =>
    def getCreator(): SelfReferentialCreatable = SelfReferentialCreatable.this
  }

}

class MySelfReferentialCreator extends SelfReferentialCreatable {

    case class MyCreatedClass() extends SelfReferential {

    }
   type CreatedType = MyCreatedClass
    def create(): MyCreatedClass = MyCreatedClass()
}

object Example3 {
  def main(args: Array[String]): Unit = {
    val creator = new MySelfReferentialCreator()
    val instance: creator.CreatedType = creator.create()
    println(instance.getCreator() == creator)
  }
}
```

In this scenario, the trait has an inner trait named `SelfReferential` that has a method `getCreator()`, and the `CreatedType` within `SelfReferentialCreatable` is a subtype of `SelfReferential`, and in this specific case is `MyCreatedClass`. `MyCreatedClass` is defined as an inner class and extends `SelfReferential` , ensuring that instances created by this trait can get a reference back to the object creating it.

**Key Takeaways and Further Reading**

From a practical standpoint, the approach I’ve described has been remarkably effective in creating flexible systems without being tied to specific concrete classes. It gives you power when designing the architecture of complex software systems.

If you're looking to dive deeper, I'd strongly recommend the book “Programming in Scala” by Martin Odersky, Lex Spoon, and Bill Venners for an in-depth explanation of abstract types and type members, particularly in the context of traits. Furthermore, the paper "Scalable Component Abstractions" by Martin Odersky, Philip Wadler, and Matthias Felleisen is also valuable to understand the background on how these concepts were formulated and why they are the way they are. The concepts presented here, while demonstrated in Scala, are often transferable to other languages in some form. Understanding these core principles can dramatically improve the architecture of any software system, regardless of the specific language you are working with. It moves away from rigid inheritance hierarchies and embraces flexible, composable designs.
