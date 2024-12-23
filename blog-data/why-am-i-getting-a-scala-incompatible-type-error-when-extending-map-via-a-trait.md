---
title: "Why am I getting a Scala 'Incompatible type' error when extending Map via a trait?"
date: "2024-12-23"
id: "why-am-i-getting-a-scala-incompatible-type-error-when-extending-map-via-a-trait"
---

Let's dissect this issue, because it's something that's definitely tripped me up in the past, specifically when trying to introduce some common utility functions to a scala `map` implementation without resorting to implicits all over the place. The "incompatible type" error when extending `scala.collection.Map` using a trait, it usually boils down to how type variance and type parameters interact with traits, and specifically with immutable collections like scala's default `Map`. It's less about the compiler being capricious and more about it being incredibly precise and protective.

The heart of the matter lies in the fact that `scala.collection.Map` is defined in terms of invariant type parameters. For those not deep into type theory, it basically means that a `Map[String, Int]` is not considered a subtype of `Map[Any, Any]`, even though theoretically, you might expect it to be. Invariant type parameters mean the subtype relationship is only valid if the type parameters match exactly.

When you try to extend `Map` with a trait, you're essentially creating a new type that mixes in that behavior. However, since `Map` is invariant in its type parameters, the compiler doesn't automatically assume that your extended type is a valid substitution for `Map`. Furthermore, traits themselves, unlike classes, do not have constructors that can fully resolve the concrete instantiation of the type parameters, which is necessary to maintain the invariant nature of the Map's type definition. This causes the dreaded "incompatible types" error.

Think of it this way: scala's compiler treats `Map[A, B]` like a sealed container specific to `A` and `B`. Even if your trait *might* work with a different pair, the compiler cannot infer that safely. A simple solution of creating your own `MyMap` class which does implement `Map` will allow scala to understand the correct generic type parameters and how it interacts with the methods. Let's look at a practical example.

Imagine I had to create a custom map to hold configuration properties in one of my past projects. I thought, ", I need some extra utility functions, let me just extend `Map` with a trait". I started with something like this:

```scala
trait ConfigMapOps[K, V] extends Map[K, V] {
  def getOrElseThrow(key: K, msg: String): V =
    get(key).getOrElse(throw new IllegalArgumentException(msg))
}

class ConfigMap[K, V](private val underlying: Map[K, V]) extends Map[K, V] with ConfigMapOps[K, V] {
  override def +[V1 >: V](kv: (K, V1)): Map[K, V1] = new ConfigMap(underlying + kv)

  override def - (key: K): Map[K, V] = new ConfigMap(underlying - key)

  override def get(key: K): Option[V] = underlying.get(key)

  override def iterator: Iterator[(K, V)] = underlying.iterator
}

object ConfigMap{
  def empty[K,V] : ConfigMap[K, V] = new ConfigMap(Map.empty[K, V])
  def apply[K,V](items: (K, V)*): ConfigMap[K,V] = new ConfigMap(Map(items:_*))
}

object MyMapTest {
  def main(args: Array[String]): Unit = {
      val myMap = ConfigMap("db.host" -> "localhost", "db.port" -> 5432)
      println(myMap.getOrElseThrow("db.host", "Database host not found"))
      println(myMap.get("nonexistent"))
  }
}
```

This seems like it should work. We have defined the `getOrElseThrow` function, and we also create our custom map. However, this code, as defined, still doesn't work. There's a reason for this, and a key one: implementing the `Map` trait directly requires implementing *all* its methods, including those that change the collection's contents. This is necessary to abide by the invariant nature of Map, but its not useful in most use cases. It also forces the developer to re-implement immutable functionality that is already provided.

Let's take a step back and utilize scala's `Map` class instead. This can be done by utilizing type aliases:

```scala
trait ConfigMapOps[K, V] {
  type SelfMap <: Map[K,V]
  def self: SelfMap

  def getOrElseThrow(key: K, msg: String): V =
    self.get(key).getOrElse(throw new IllegalArgumentException(msg))
}

class ConfigMap[K, V](private val underlying: Map[K, V]) extends ConfigMapOps[K,V] {
  override type SelfMap = Map[K, V]
  override def self: Map[K, V] = underlying
}

object MyMapTest {
  def main(args: Array[String]): Unit = {
    val myMap = new ConfigMap(Map("db.host" -> "localhost", "db.port" -> 5432))
      println(myMap.getOrElseThrow("db.host", "Database host not found"))
      println(myMap.self.get("nonexistent"))
  }
}
```

This revised version is functionally equivalent but now leverages the existing implementation of Map, and also uses type aliases to enforce that our custom map still behaves as a standard `Map` while adding our custom method. If you need to perform operations that modify the original map, you can always utilize the provided methods on the `self` field directly. This is a crucial point to remember when working with immutable collections – transformations create new instances, and your custom map implementations will also require you to perform similar transformations. This approach allows you to augment the existing `Map` functionality without requiring implementing the whole thing.

This approach is far more flexible and follows the functional paradigm that scala champions. If you want to expand the functionality, you can also use the builder pattern or copy methods:

```scala
trait ConfigMapOps[K, V] {
  type SelfMap <: Map[K,V]
  def self: SelfMap

  def getOrElseThrow(key: K, msg: String): V =
    self.get(key).getOrElse(throw new IllegalArgumentException(msg))

  def addOrReplace(key: K, value: V): SelfMap
}


class ConfigMap[K, V](private val underlying: Map[K, V]) extends ConfigMapOps[K,V] {
  override type SelfMap = Map[K, V]
  override def self: Map[K, V] = underlying

  override def addOrReplace(key: K, value: V): SelfMap = {
    (underlying + (key -> value))
  }
}

object MyMapTest {
  def main(args: Array[String]): Unit = {
    val myMap = new ConfigMap(Map("db.host" -> "localhost", "db.port" -> 5432))
    val modifiedMap = myMap.addOrReplace("db.port", 8080)
      println(modifiedMap.getOrElseThrow("db.host", "Database host not found"))
      println(modifiedMap.get("db.port"))
  }
}
```

Here, `addOrReplace` returns a new instance of `SelfMap` with the modified value. You'll notice that `addOrReplace` in this case performs an addition operation, which means the object has to be transformed. The method is implemented using the existing `+` method provided on `Map` and returns a new instance.

The key takeaway is that extending `Map` directly in Scala with a trait will often lead to type compatibility issues due to type variance and the required methods. Instead, composition with type aliases and delegation as seen in the provided example is a more effective and flexible approach.

For a more in-depth understanding of type variance, I'd suggest looking at "Programming in Scala" by Martin Odersky, Lex Spoon, and Bill Venners. This book has excellent explanations and it will provide you with a strong understanding of the core concepts involved. Also, "Types and Programming Languages" by Benjamin C. Pierce is a good resource for a theoretical explanation on type theory, variance and programming language design if you prefer a more formal treatment.

Working with immutable collections and type systems requires an understanding of these subtle details, but with the right approach, you can create reusable and robust code. And hopefully, now you won't find yourself in that same trap I fell into all those years ago.
