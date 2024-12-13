---
title: "ruby pp string printing formatting?"
date: "2024-12-13"
id: "ruby-pp-string-printing-formatting"
---

Alright so you want to print a string in Ruby and you need some fancy formatting using `pp` huh Been there done that more times than I care to admit Let's break this down because it's not always as straightforward as you might think especially if you're dealing with complex data or want something specific in your output.

First off `pp` is your friend when it comes to pretty printing Ruby objects It's way better than just `puts` or `print` for inspecting data structures because it actually formats things nicely making it readable That's its main purpose debugging and visualizing data. Now I've seen a lot of people misuse it. I had this one intern once who tried to use it in production logging…yeah that was a fun cleanup. Production isn’t for pretty printing is the lesson there folks.

Okay so you're asking about strings specifically. `pp` by default will print the string with quotes if it's a basic string. That's how it distinguishes it from other data types when displaying. If you have a string that contains escape sequences like newlines or tabs it'll show those escape sequences which can be handy but sometimes you might want the actual newlines and tabs to be rendered.

Let's start with some basic examples to illustrate this.

```ruby
require 'pp'

str1 = "Hello World"
pp str1  # Output: "Hello World"

str2 = "Hello\nWorld"
pp str2  # Output: "Hello\\nWorld"

str3 = "Hello\tWorld"
pp str3  # Output: "Hello\\tWorld"
```

See? The basic output is nothing too crazy just quotes around strings. The escape characters are visible as escaped sequences not as the new lines or tabs. This shows you the raw string content. Now if you want to print the actual rendered string with newlines and tabs there are a couple of ways to do it and `pp` is not your main tool here for the actual rendering. You'd use plain `puts` for that. It is one of the most basic but useful tools.

Now lets delve a bit deeper into cases where formatting needs a bit more attention. What if you have very long strings or strings with special characters that you want to highlight differently? The default `pp` output might be okay for short strings but it becomes a pain when you have something substantial. This is where understanding the format string functionality with Ruby comes in handy. It is not directly a feature of `pp` but it is how `pp` might present more complicated strings. It might not be the actual feature but it is the logic and behavior.

For example let’s say you want to print some structured data that contains strings inside. You still want to use `pp` but you want the strings to stand out a bit more. This isn’t directly related to the formatting of the string itself but the way `pp` shows you complex objects. I know I had to do this while debugging a JSON parser a while back. The JSON had all kinds of unicode characters that were just a mess until I found out how to highlight them in `pp`'s output by using a custom inspect method.

Here's an example of a slightly more complex case

```ruby
require 'pp'

data = {
  name: "John Doe",
  address: {
    street: "123 Main St",
    city: "Anytown",
    zip: "12345"
  },
  description: "A very long string that might need some formatting to make it more readable for debugging purposes this is a very very very long string.",
  notes: ["Note 1" "Note 2" "Another note", "A very long note that could be on another line if we had to format it manually with puts"]

}

pp data # Normal output
```

The standard `pp` output might not be enough here especially for very long strings. Now let’s say you want to highlight or change the way `pp` handles specific string. To do this you might modify the `inspect` method of the relevant objects if its an object and not just a normal string. It’s a bit advanced but if you’re dealing with complicated structures in your debugging workflows it’s something that will make you more efficient.

For example if you are developing a custom class or want to show how instances of a class should be displayed you could overwrite the `inspect` method. You could add a custom string formatting before `pp` gets to see your object. That means that `pp` uses that `inspect` to present the string.

Here's an example of how to override the `inspect` method

```ruby
require 'pp'

class MyString
  attr_reader :value
  def initialize(value)
    @value = value
  end

  def inspect
    "MyString: '#{value}'"
  end
end

string_instance = MyString.new("Custom formatted string")
pp string_instance
```

Here we are using a custom class `MyString` to simulate a data structure with some kind of formatting control. The `inspect` method is key to how `pp` displays objects. I did this once when I had a custom data structure I wanted to debug and it saved me a lot of time. Overwriting `inspect` is the closest you get to formatting with `pp` itself. Now a joke for you. Why did the Ruby developer break up with the data structure? Because she had too many issues with the formatting. I know it was a stretch but I had to put it in here.

Now if you need more control on how the strings themselves are printed in a generic manner you are no longer in the territory of `pp` and its domain. You would be better served with the use of `printf` or string interpolation with format specifiers and `puts` or just regular string manipulation.

For advanced reading I would recommend "The Ruby Programming Language" by David Flanagan and Yukihiro Matsumoto specifically the chapter on strings and input output that has all the details on string manipulation format specifiers and more. For those looking to dive deep into Ruby internals understanding the `inspect` method is something to explore further. Look for papers on "object reflection in dynamic languages" you will see similar concepts that will expand your knowledge on how to debug with `pp` in general. While not specifically focused on `pp` understanding how objects are inspected and how format specifiers works is crucial to more efficient debugging.

So to summarize `pp` is great for inspecting data structures showing the raw content of strings and highlighting the way objects are presented. For more fine-grained control over string formatting you would use other tools like `printf` `puts` or string interpolation and you can even customize how `pp` presents your objects by overriding `inspect`. Don’t be afraid to experiment with these features you’ll find that it makes your life a lot easier when debugging complex applications.
