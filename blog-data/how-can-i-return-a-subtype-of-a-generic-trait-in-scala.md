---
title: "How can I return a subtype of a generic trait in Scala?"
date: "2024-12-23"
id: "how-can-i-return-a-subtype-of-a-generic-trait-in-scala"
---

Alright,  It's a question that surfaces more often than you'd initially think, especially when you start building complex type hierarchies. I’ve personally spent a fair amount of time navigating this particular corner of Scala's type system, particularly when I was working on a system for processing various types of financial instruments. We had this generic `Instrument` trait, and, of course, different specialized subtypes like `Stock`, `Bond`, and `Option`. The challenge was returning these specific subtypes within a generic context, and doing it in a way that maintains type safety without relying on unsafe casts.

The core of the issue lies in the limitations of type erasure and covariance in Scala's generics. When you define a generic trait like `trait Parser[T] { def parse(): T }`, you're essentially saying: "This parser produces a *T*." The problem arises when you have specific implementations – like a `StockParser` or a `BondParser` – that each produce different subtypes of some common base type, say `Instrument`. Directly returning a `T` from `Parser[Instrument]` won’t cut it if your logic actually generates a `Stock`.

Here are three common patterns that address this situation, each with its own set of trade-offs.

**1. Type Parameter Specialization with Self-Types**

One approach is to refine the type parameter using self-types. Consider this:

```scala
trait Instrument { def identifier: String }
case class Stock(identifier: String, ticker: String) extends Instrument
case class Bond(identifier: String, isin: String) extends Instrument

trait Parser[T <: Instrument] {
  self: Parser[T] =>
  def parse(): T
}

class StockParser extends Parser[Stock] {
    override def parse(): Stock = Stock("S001", "APPL")
}

class BondParser extends Parser[Bond] {
  override def parse(): Bond = Bond("B001", "US00001")
}

object ParserClient {
  def main(args: Array[String]): Unit = {
    val stockParser: Parser[Stock] = new StockParser()
    val bondParser: Parser[Bond] = new BondParser()

    val stock: Stock = stockParser.parse()
    val bond: Bond = bondParser.parse()

    println(s"Parsed stock: ${stock.ticker}")
    println(s"Parsed bond: ${bond.isin}")
  }
}
```

In this setup, the `Parser[T]` trait is now constrained such that `T` must be a subtype of `Instrument`. Additionally, the self-type annotation `self: Parser[T] =>` ensures that implementations must be of the same generic type. This setup isn’t generic in the sense that it takes no external type information. Each specific parser provides its own return type without the need to upcast. This solution avoids any runtime casting or type erasure surprises. The downside is that you need explicit parser classes for every subtype. This approach is useful when you know at compile-time which kind of specific instrument you need to parse.

**2. Using Abstract Type Members**

A more flexible approach involves using abstract type members within the trait. This allows you to define the return type in a more abstract way:

```scala
trait Instrument { def identifier: String }
case class Stock(identifier: String, ticker: String) extends Instrument
case class Bond(identifier: String, isin: String) extends Instrument

trait Parser {
  type Result <: Instrument
  def parse(): Result
}

class StockParser extends Parser {
  type Result = Stock
  override def parse(): Stock = Stock("S002", "GOOG")
}

class BondParser extends Parser {
  type Result = Bond
  override def parse(): Bond = Bond("B002", "GB00002")
}

object ParserClient2 {
  def main(args: Array[String]): Unit = {
    val stockParser: Parser { type Result = Stock } = new StockParser()
    val bondParser: Parser { type Result = Bond } = new BondParser()

    val stock: Stock = stockParser.parse()
    val bond: Bond = bondParser.parse()

    println(s"Parsed stock: ${stock.ticker}")
    println(s"Parsed bond: ${bond.isin}")
  }
}
```

Here, `Parser` declares an abstract type member, `Result`, which is constrained to be a subtype of `Instrument`. Specific implementations, like `StockParser` and `BondParser`, specify the concrete type of `Result`. The key point is that the client code knows *exactly* what the `parse` method returns at compile time, through a refined type. The slightly awkward syntax `Parser { type Result = Stock }` helps to illustrate how the abstract type `Result` is defined for this particular parser. This technique provides greater flexibility than explicit parameterization as it decouples the generic type from the implementations. It works particularly well when you need polymorphism on parsers but the implementations differ on their output type.

**3. Using Path-Dependent Types**

Path-dependent types offer yet another way to achieve the desired result, often in combination with abstract type members. This pattern is valuable when a relationship exists between the parser and its specific return type instance.

```scala
trait Instrument { def identifier: String }
case class Stock(identifier: String, ticker: String) extends Instrument
case class Bond(identifier: String, isin: String) extends Instrument

trait ParserContainer {
  type Result <: Instrument
  trait Parser {
    def parse(): Result
  }
  def parser: Parser
}

class StockParserContainer extends ParserContainer {
  type Result = Stock
  class StockParser extends Parser {
    override def parse(): Stock = Stock("S003", "MSFT")
  }
  override val parser = new StockParser
}


class BondParserContainer extends ParserContainer {
  type Result = Bond
  class BondParser extends Parser {
     override def parse(): Bond = Bond("B003", "FR00003")
  }
  override val parser = new BondParser
}

object ParserClient3 {
  def main(args: Array[String]): Unit = {
      val stockParserContainer = new StockParserContainer()
      val bondParserContainer = new BondParserContainer()

      val stock: Stock = stockParserContainer.parser.parse()
      val bond: Bond = bondParserContainer.parser.parse()

      println(s"Parsed stock: ${stock.ticker}")
      println(s"Parsed bond: ${bond.isin}")

  }
}
```

In this example, `ParserContainer` holds a `Result` type, and `Parser` itself is defined as an inner trait dependent on this container. When an instance of `StockParserContainer` is created, its `parser` property returns a specific `StockParser` that returns a `Stock`. Likewise, the `BondParserContainer` provides a `BondParser` that returns a `Bond`. This allows you to achieve very specific type relations within instances of the classes.

**Considerations and Further Reading:**

Each of these approaches addresses the core challenge of returning a subtype within a generic context, but their suitability depends heavily on the specific problem. If you have a simple, static setup, type parameter specialization with self-types is often enough. When you need a bit more dynamism and flexibility, abstract type members, potentially in combination with path-dependent types, tend to fit the bill.

For a thorough treatment of Scala's type system, I recommend "Programming in Scala" by Martin Odersky, Lex Spoon, and Bill Venners, particularly the chapters covering generics, type parameters, and abstract types. Also, explore the official Scala documentation, which provides extensive explanations of these concepts. Specifically, pay close attention to the sections covering type variance and path-dependent types. Finally, a deep dive into the theory of type systems in works such as "Types and Programming Languages" by Benjamin C. Pierce can add substantial depth to your understanding.

These solutions reflect real-world challenges I’ve encountered and successfully navigated. Understanding how to utilize Scala's type system to achieve this level of type specificity is crucial when building robust and scalable systems. It avoids casts, maintains type safety and offers significant flexibility for handling situations when you're working with a hierarchy of related subtypes within generic contexts.
