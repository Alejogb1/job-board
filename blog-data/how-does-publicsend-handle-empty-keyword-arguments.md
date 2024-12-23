---
title: "How does `public_send` handle empty keyword arguments?"
date: "2024-12-23"
id: "how-does-publicsend-handle-empty-keyword-arguments"
---

Alright, let's unpack how `public_send` behaves with empty keyword arguments. It's one of those seemingly simple areas that can trip folks up, and I’ve certainly been in situations where misinterpreting this led to some head-scratching debugging sessions. It's not always intuitive, and I’ve learned some useful lessons over the years. Specifically, my experience working on a high-throughput processing system involved frequent use of dynamic method dispatch, and this quirk with `public_send` and keyword arguments reared its head more than once.

The core of the issue revolves around how Ruby's method invocation system handles keywords, and how `public_send` interacts with that mechanism. `public_send`, as you likely know, is a powerful meta-programming tool allowing you to invoke methods on an object by their string or symbol representation, effectively enabling dynamic dispatch. However, when it comes to keyword arguments, the behavior becomes nuanced, particularly when no keyword arguments are actually specified.

When you invoke a method directly with empty keyword arguments (e.g., `my_method(**{})`), you’re explicitly telling the Ruby interpreter that this method *can* accept keyword arguments, even if none are provided at that specific moment. In contrast, when you invoke a method via `public_send` with no keyword arguments, Ruby doesn't interpret it as an intention to potentially pass keyword arguments. Instead, it sees it as simply invoking the method with whatever positional arguments are provided.

Let's delve into a specific situation that I ran into. In our system, we had a variety of data processors. Each processor had a `process` method, which could accept an optional set of parameters as keyword arguments. Some of these parameters were optional, while others were required depending on the specific processor. We used a configuration file to dynamically route incoming data to the appropriate processor and its configuration parameters. The parameters could be specified as keyword arguments when a processor is called dynamically, however sometimes, we were passing in parameters via `public_send`, we didn’t need any parameters and we made the error of thinking we could pass in empty keyword arguments.

Here's a code snippet that illustrates the problem:

```ruby
class DataProcessor
  def process(data, **options)
    puts "Processing data: #{data} with options: #{options}"
  end

  def process_without_options(data)
    puts "Processing data: #{data} without options"
  end
end

processor = DataProcessor.new

# Direct invocation with empty keyword args - works as expected
processor.process("data1", **{})

# Using public_send, if no explicit keyword arguments are provided, it does not act as an "empty" keywords argument hash.
processor.public_send(:process, "data2")

# Directly invoking, without kwargs, which also works fine.
processor.process_without_options("data3")

# This is how we want to invoke with public_send for an empty keywords hash.
processor.public_send(:process, "data4", {})

# The error occurs when passing a hash as if it were positional args.

begin
  processor.public_send(:process, "data5", {}, {})
rescue ArgumentError => e
    puts "Error message: #{e.message}"
end
```

In this example, calling `processor.process("data1", **{})` correctly invokes `process` with an empty hash for the `options` keyword argument. However, `processor.public_send(:process, "data2")` invokes `process` as if it’s a method with only one positional argument, which, whilst valid in this case, is not what would be expected if keyword parameters were needed. The third invocation `processor.process_without_options("data3")` is a case with no keywords, and this also works fine with direct invocation. What we must do to explicitly provide an empty keywords hash is `processor.public_send(:process, "data4", {})`, here we explicitly pass a hash, and Ruby understands that it's intended for keyword arguments even though it's empty.

The error `ArgumentError: wrong number of arguments (given 3, expected 1..2)` occurs because Ruby expects only one positional argument and 0..N keyword arguments when calling the method directly. The `public_send` call when we provide the hash as if it was positional arguments, `processor.public_send(:process, "data5", {}, {})`, Ruby interprets the `{}` as positional arguments which is invalid in this case.

This highlights a crucial point: if you want to explicitly pass empty keyword arguments via `public_send`, you *must* pass an empty hash as one of the arguments. Otherwise, Ruby simply treats the invocation as if there are no keyword arguments at all, and no potential for keyword arguments are implied.

To emphasize this further, let's look at a situation with more complex keyword argument usage.

```ruby
class ReportGenerator
  def generate_report(data, format: "pdf", detailed: false, header: nil)
    puts "Generating report with format: #{format}, detailed: #{detailed}, header: #{header}, data: #{data}"
  end
end

generator = ReportGenerator.new

# Direct invocation, with specified keyword arguments,
generator.generate_report("report data", format: "csv", detailed: true, header: "Custom Header")


# Direct invocation, with some keyword arguments.
generator.generate_report("report data 2", format: "csv")


#Direct Invocation with some empty keywords.
generator.generate_report("report data 3", **{})


# Using public_send, with specified keyword arguments.
generator.public_send(:generate_report, "report data 4", {format: "txt", detailed: false, header: "Another header"})

# Using public_send with no keyword arguments (note this is valid in this case because all the keywords have a default value)
generator.public_send(:generate_report, "report data 5")

# Using public_send with empty keyword arguments.
generator.public_send(:generate_report, "report data 6", {})
```

In this example, the `ReportGenerator` class has a `generate_report` method with multiple keyword arguments, some with default values. Again, we see that when we want to explicitly pass an empty keyword argument hash using `public_send`, we need to provide that `{}` as an argument, as demonstrated by the final `public_send` call. Without the {}, we are just providing a regular positional argument.

Finally, let's investigate when `public_send` fails when keyword arguments are mandatory.

```ruby
class MandatoryKeywordProcessor
  def process(data, mandatory_option:)
    puts "Processing with mandatory_option: #{mandatory_option}, data: #{data}"
  end
end

processor = MandatoryKeywordProcessor.new

#Direct invocation works.
processor.process("data", mandatory_option: "value")

begin
# public_send will error because we are not providing mandatory_option.
  processor.public_send(:process, "data")
rescue ArgumentError => e
  puts "Error message: #{e.message}"
end
begin
# public_send will error because we are providing a hash that is interpreted as a positional argument, and mandatory_option is not passed as keyword argument.
  processor.public_send(:process, "data", {})
rescue ArgumentError => e
  puts "Error message: #{e.message}"
end


# public_send must be passed an explicit keywords argument hash
processor.public_send(:process, "data", {mandatory_option: "value"})

```

Here, we can see that `public_send` must be used with a keyword argument hash in situations where keywords are mandatory, and without it, or when attempting to pass a hash as a positional argument, it raises an `ArgumentError`.

In essence, `public_send` treats keyword arguments differently compared to direct method invocation when dealing with empty or no arguments. It does not automatically assume that the absence of a hash implies the intention to pass an empty keyword argument hash. We need to explicitly provide it. This can be surprising, and it's an area of Ruby's metaprogramming that requires a careful approach to avoid subtle errors.

For a deeper understanding of method invocation in Ruby, I strongly recommend delving into "Metaprogramming Ruby 2" by Paolo Perrotta. This book offers comprehensive explanations of how Ruby handles arguments and method dispatch, including the nuances of keyword arguments and meta-programming techniques. Also, reviewing the source code of Ruby's method dispatcher can be exceptionally helpful, albeit complex. The Ruby implementation itself, specifically the `vm.c` file and method invocation logic within `ruby/ruby` repository will greatly help if you’re willing to dive into the lower-level details. These resources provide a thorough understanding of the underlying mechanisms, which can be quite beneficial in situations involving meta-programming. It’s a good exercise to start with the book before diving into the C source, though.

Through my past experiences, these kinds of subtle behaviors are crucial to understand when building complex, dynamically-driven systems. Always be explicit when using `public_send` and always have a thorough understanding of how Ruby handles method invocation.
