---
title: "Why does `wrong constant name` get inferred by Module?"
date: "2024-12-16"
id: "why-does-wrong-constant-name-get-inferred-by-module"
---

Alright, let's tackle this. I've definitely seen this sort of thing crop up in various projects, and it’s a good question, one that highlights some important nuances in how modules and their constant resolution mechanics function in ruby. It’s a situation that might initially appear confusing but, with a bit of exploration, becomes quite logical.

The crux of the issue with "wrong constant name" being inferred by `Module` often revolves around the interplay of lexical scope, the constant lookup chain, and how ruby's module system operates. When we say "constant," we’re not talking strictly about something you’ve defined directly with an uppercase name. Instead, ruby treats anything that's not a variable or method call as a potential constant—and then goes searching for it in a defined path.

Let's break this down. Imagine I had a scenario years back, working on a large rails project. We had a complex service layer and, unfortunately, a somewhat inconsistent naming convention for certain modules. I inherited code where, inside a class `AnalyticsEngine`, there were calls like `Module.const_get(:AnalyticsData)` inside a method, intending to access `AnalyticsData` module. But then, in other parts of the codebase, there was a separate module called `EngineData` or even just plain `Data`, intended for related but subtly different data processing. The resulting confusion wasn't just a code smell—it actually produced runtime errors due to the dynamic behavior of `Module.const_get`. Specifically, ruby might not find what was intended.

The problem stems from how ruby searches for a constant. The general search order is lexical scoping, then up the module hierarchy. When you invoke `Module.const_get` inside a module, you're telling ruby to find a constant—by the supplied name—starting from that module's scope, then proceeding to parent modules, then the global scope. This is fairly straightforward but when the module hierarchy itself has similarly named entities, unexpected results occur. Furthermore, there’s a big distinction when compared to direct constant lookup. When you directly write `AnalyticsData`, the compiler can resolve the constant’s location at compile time. When you use `const_get` with a string, ruby cannot perform static resolution and instead relies on the runtime lookup mechanism. This is a powerful and flexible feature, but also a source of potential confusion.

Consider this example. It's a simplified demonstration, but illustrates the underlying issue:

```ruby
module Outer
  module Inner
    CONSTANT = 10

    def get_constant_using_symbol
       Inner.const_get(:CONSTANT)
    end

     def get_constant_using_string
       Module.const_get("CONSTANT")
     end

    def access_constant_directly
        CONSTANT
    end

  end
end
puts Outer::Inner.new.get_constant_using_symbol # Output: 10
puts Outer::Inner.new.get_constant_using_string # Output: 10
puts Outer::Inner.new.access_constant_directly # Output: 10
```

In the above example, all three methods resolve to the `CONSTANT` within `Inner`, precisely because that's the scope where ruby starts looking when using `const_get` on `Inner` and direct access within `Inner` respectively. So far so good, but let’s look at what happens if we add a second module at the top level:

```ruby
CONSTANT = "global_constant"

module Outer
  module Inner
    CONSTANT = 10
    def get_constant_using_symbol
       Inner.const_get(:CONSTANT)
    end
    def get_constant_using_string
      Module.const_get("CONSTANT")
    end
     def access_constant_directly
        CONSTANT
    end
  end
end

puts Outer::Inner.new.get_constant_using_symbol # Output: 10
puts Outer::Inner.new.get_constant_using_string # Output: "global_constant"
puts Outer::Inner.new.access_constant_directly # Output: 10

```

Observe carefully: while `access_constant_directly` correctly resolves to `Inner::CONSTANT`, and `const_get(:CONSTANT)` resolves as before, `Module.const_get("CONSTANT")` will resolve to the top-level constant because, in the context of `Module.const_get()`, ruby searches starting from Module’s scope which is the top level scope, and then searches the global scope, which is where it finds `CONSTANT = "global_constant"`.

And that’s a key detail: When using `Module.const_get()`, it bypasses the current module's scope in its search and begins the resolution from the top level scope or global scope. So, if a global constant or a constant in a different, unrelated module has the same name, you can end up accessing the wrong one. This can get particularly messy if you are, say, trying to access a constant within a nested module but happen to use `Module.const_get` instead of directly referencing the constant within that nested module. This is the "wrong constant name" being inferred - the wrong constant, from the perspective of the programmer, is being found.

Let’s illustrate this issue with a concrete example, adding an unrelated top level module with the same name as the constant:

```ruby
module GlobalModule
  CONSTANT = "Global Module Constant"
end

module Outer
  module Inner
    CONSTANT = 10
    def get_constant_using_string
        Module.const_get("CONSTANT")
    end
    def get_constant_using_symbol
       Inner.const_get(:CONSTANT)
    end
    def access_constant_directly
      CONSTANT
    end
  end
end

puts Outer::Inner.new.get_constant_using_string  # Output: "Global Module Constant"
puts Outer::Inner.new.get_constant_using_symbol # Output: 10
puts Outer::Inner.new.access_constant_directly # Output: 10
```
Again, `Module.const_get("CONSTANT")` now pulls in the constant from the *unrelated* `GlobalModule`, showing how the dynamic lookup can create these kinds of bugs. On the other hand, `Inner.const_get(:CONSTANT)` and direct access `CONSTANT`, both resolve to the `CONSTANT` inside `Inner`.

The main lesson here isn't to avoid `Module.const_get` entirely, but to understand its behavior and when to use it carefully. Typically, when you're dealing with constants *within* a module or class, directly accessing the constant or using `self.class.const_get(:CONSTANT)` or `<module_name>.const_get(:CONSTANT)` with a symbol is generally more robust. `Module.const_get` is best suited when you *intentionally* need a dynamic constant lookup, where the name might vary at runtime, or where you are working with the top level scope.

For further study, I'd recommend looking at the official ruby documentation on Modules and Constants; the 'Programming Ruby' book by Pragmatic Programmers provides an excellent, deep-dive into the mechanics of Ruby. Also, "Metaprogramming Ruby 2" by Paolo Perrotta delves deep into the topic of dynamic constant lookup. These resources will clarify not just the *how*, but the *why* behind these behaviors and empower you to design more reliable ruby applications. This is an area that I've spent quite some time investigating in my past, so I know these materials will prove invaluable. Ultimately, knowing the difference between compile time constant resolution versus dynamic constant lookup using `const_get` can dramatically improve your debugging abilities.
