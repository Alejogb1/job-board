---
title: "Why can a class call a Proc object created from a block?"
date: "2024-12-23"
id: "why-can-a-class-call-a-proc-object-created-from-a-block"
---

Let's unpack this. I’ve seen this tripping up a fair few folks over the years, and it really boils down to a fundamental understanding of closures and how Ruby handles them, specifically when dealing with blocks and `Proc` objects. It's not a magical leap, but rather a consequence of how Ruby's internals manage execution contexts. I remember once debugging a gnarly event system where this behavior was causing unexpected side effects—lesson learned, and it emphasized the need for clarity on this topic.

The short answer is this: a `Proc` object, instantiated from a block, encapsulates the code within the block and, crucially, the *lexical scope* in which that block was defined. This scope includes any variables accessible at the time of the block's creation. Consequently, when this `Proc` is called, even from within a different class or method, it can still access those variables, or indeed, execute code reliant on the environment present at its definition. It’s all about that closed-over environment, the very essence of a closure.

Now, let's get more granular.

When you use a block with methods like `each`, `map`, or define your own custom methods using `yield` or the `&block` parameter, you're essentially creating anonymous code snippets. These blocks are not directly executable as standalone entities. However, by wrapping them into a `Proc` object using `Proc.new { ... }` or utilizing the `&` syntax in method definitions (e.g., `def my_method(&block)`), you convert them into first-class objects. These objects can be passed around, stored in variables, and importantly, *called* using the `.call` method.

Crucially, when `Proc.new` is used to convert a block into a `Proc` object, it doesn't merely copy the code. It also captures the environment where the block was defined. This environment is known as a closure. The closure allows the `Proc` to access and modify the variables and contexts that were within scope when the block was created, even after that scope has technically ended. This mechanism enables the `Proc` to perform as the programmer intended, regardless of where it’s eventually invoked. Think of it as a sealed container, which stores not just the instructions but also the context in which they make sense.

Let's walk through a few examples that clearly demonstrate this behaviour:

**Example 1: Accessing Variables from the Outer Scope**

```ruby
class MyClass
  def initialize(multiplier)
    @multiplier = multiplier
  end

  def execute_proc(proc_obj)
    proc_obj.call(@multiplier)
  end
end

def create_proc(base_value)
  factor = 5  # This variable is part of the block's scope
  Proc.new { |mult| base_value * mult * factor }
end

initial_value = 10
my_proc = create_proc(initial_value)
my_object = MyClass.new(2)

result = my_object.execute_proc(my_proc)
puts "Result: #{result}" # Output: Result: 100 (10 * 2 * 5)

```

In this example, `create_proc` defines the `factor` variable. The block (and subsequently, the `Proc` object) has closed over this variable. The `MyClass` doesn't directly know or care about `factor`; it's just calling a Proc object that knows its context. When `my_proc` is called inside `execute_proc`, it uses the saved `factor` value (5) along with the passed `@multiplier` (2) from the `MyClass` instance, and also `initial_value` (10), which is in scope when the `Proc` was defined and not from the calling context inside the `MyClass`. This illustrates the closure concept.

**Example 2: Manipulating Variables from the Outer Scope**

```ruby
def counter_generator
  count = 0
  Proc.new { count += 1 }
end

counter1 = counter_generator
counter2 = counter_generator

puts counter1.call # Output: 1
puts counter1.call # Output: 2
puts counter2.call # Output: 1
puts counter1.call # Output: 3
puts counter2.call # Output: 2
```

Here, each call to `counter_generator` returns a *new* `Proc` object. Each of these `Proc` objects has its own independent `count` variable from the outer scope, which it manipulates. This shows that the `Proc` doesn’t just see the variable’s value but also can modify it within its captured scope. The key idea here is that each `Proc` has its *own* copy of the `count` variable in its closure.

**Example 3: Using a Block Passed to a Method**

```ruby
class Calculator
  def compute(value, &operation)
    operation.call(value)
  end
end

calculator = Calculator.new
multiplier = 3
result = calculator.compute(5) { |x| x * multiplier }

puts "Result from Calculator: #{result}" # Output: Result from Calculator: 15
```

Here, the block `{|x| x * multiplier}` is not explicitly converted to a `Proc` by the caller, but the ampersand in the definition of `compute` achieves this for us. It transforms the block into a `Proc` named `operation`. Again, note that the `multiplier` variable is captured by the block and its resulting `Proc` object, making it available when `operation` is called by the `Calculator` instance.

In all these cases, the crucial factor is that the `Proc` object acts as a closed environment, carrying with it not only the instructions of the block but also the state of the scope in which it was created. This allows it to interact with variables and contexts that might not be visible to the class where it’s eventually called.

For a deeper dive into this subject, I highly recommend examining the chapter on closures in "Eloquent Ruby" by Russ Olsen; it provides a very clear explanation with practical examples. For those interested in the more theoretical underpinnings, the works of Peter Landin on lambda calculus and the concept of closures are very insightful, although more abstract. Also, reading through the source code of Ruby itself, specifically parts related to block execution, can give you deep insight into how all this is accomplished at the interpreter level.

In short, when a class calls a `Proc` object created from a block, it’s not a mysterious act but a controlled interaction rooted in the way Ruby creates closures. Understanding this will save you significant debugging time and allow you to use blocks and procs much more effectively in your code. It’s not just syntax; it’s a design pattern deeply woven into the language.
