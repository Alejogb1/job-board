---
title: "Why does Zeitwerk collapse namespacing?"
date: "2024-12-16"
id: "why-does-zeitwerk-collapse-namespacing"
---

Alright, let's tackle this one. It's something I've definitely encountered and, frankly, spent more than a few late nights debugging. Zeitwerk’s tendency to “collapse” namespacing, as you put it, isn't some capricious behavior; it's a direct consequence of its autoloading strategy, which operates under a specific set of assumptions about your directory structure and naming conventions. The problem arises not from a flaw in Zeitwerk itself but from mismatches between these assumptions and the actual organization of your codebase.

I recall working on a large Rails application a few years back. We had a deeply nested `lib` directory, something like `lib/etl/processors/data_normalization/string_helpers.rb`. We initially attempted to represent this namespace directly in our code using modules, which felt natural. However, we’d see occasional, seemingly random `NameError` exceptions when trying to reference classes defined within these modules. It looked for all the world like the module wasn’t loaded, even though it was there, seemingly in plain sight, in the `lib` directory, just like it was supposed to be. After quite a bit of investigation, we figured out what was happening – and what often happens with Zeitwerk.

The core of Zeitwerk's autoloading mechanism hinges on a one-to-one mapping between directory structure and namespace. Zeitwerk assumes that a file like `lib/etl/processors/data_normalization/string_helpers.rb` will, by default, define a constant (class or module) `Etl::Processors::DataNormalization::StringHelpers`. If you deviate from that, things can unravel quickly. Let's break that down.

Zeitwerk, when initializing, scans your configured load paths. For each file it finds, it derives the expected constant name by capitalizing each directory name and the filename (before `.rb`), separating each part by `::`. This is what you might call its “inference logic.” If the file content doesn’t define a constant with that exact inferred name, the autoloader, in essence, doesn’t know what it’s supposed to load, or even if that file is relevant.

This is especially confusing when you expect that nested modules will behave like traditional Ruby modules. If, inside `string_helpers.rb`, you attempt to define a class like this:

```ruby
# lib/etl/processors/data_normalization/string_helpers.rb
module StringHelpers
  class Normalizer
    def normalize(str)
       str.strip.downcase
    end
  end
end
```

Zeitwerk will not autoload `StringHelpers` to be accessible using `Etl::Processors::DataNormalization::StringHelpers`. It doesn't "collapse" namespaces in the sense of actively erasing them. Instead, it simply doesn't recognize that this file should define a class or module with that particular fully-qualified name. The inference logic fails because what's defined inside the file doesn't align with the expected namespace.

To clarify this, let's examine a scenario that works correctly with Zeitwerk's expectations:

```ruby
# lib/etl/processors/data_normalization/string_helpers.rb
module Etl
  module Processors
    module DataNormalization
      class StringHelpers
        class Normalizer
          def normalize(str)
            str.strip.downcase
          end
        end
      end
    end
  end
end
```

Here, because the module and class names mirror the file path and directory structure using nested modules, Zeitwerk loads everything as intended. You could then access `Normalizer` using `Etl::Processors::DataNormalization::StringHelpers::Normalizer`. This is what Zeitwerk expects and what leads to predictable loading behaviour.

Now, let’s consider a slightly more complex example with a parent and child class relationship. Say we have `lib/etl/processors/data_transformation/base.rb` and `lib/etl/processors/data_transformation/csv.rb`. Here’s what that code might look like to correctly work with Zeitwerk:

```ruby
# lib/etl/processors/data_transformation/base.rb
module Etl
  module Processors
    module DataTransformation
      class Base
        def transform(data)
         raise NotImplementedError
        end
      end
    end
  end
end

# lib/etl/processors/data_transformation/csv.rb
module Etl
  module Processors
    module DataTransformation
      class Csv < Base
         def transform(data)
          # CSV specific transformation logic here
          "transformed_from_csv"
         end
      end
    end
  end
end
```

In this setup, the `Csv` class will inherit from `Base` as anticipated, and Zeitwerk will handle both files correctly. The key here again is that the namespace structure in the code mirrors the file path.

To drive the point home, let's look at a practical situation where we might try to load files and unintentionally run into trouble due to namespace mismatches. Assume we have files named like our original example, but we want to keep the module name `StringHelper` in our first snippet. To make that code work, we would need to deviate from Zeirwerk’s conventional expectations and explicitly tell it how to find the classes:

```ruby
#config/application.rb
# Inside your Rails::Application class
config.autoload_paths << Rails.root.join("lib")
Zeitwerk::Loader.new.tap do |loader|
    loader.push_dir(Rails.root.join('lib'), namespace: Etl::Processors::DataNormalization)
    loader.ignore(Rails.root.join('lib/etl/*'))
    loader.ignore(Rails.root.join('lib/etl/processors/*'))
    loader.ignore(Rails.root.join('lib/etl/processors/data_normalization/*'))
  loader.setup
end

# lib/etl/processors/data_normalization/string_helpers.rb
module StringHelpers
  class Normalizer
    def normalize(str)
      str.strip.downcase
    end
  end
end
```
This snippet will require a deeper understanding of Zeitwerk, and explicitly tells it to only look for files in that specific directory and assign that namespace to the classes. The `ignore` statements make sure the standard loading convention is skipped for the other files under `lib/etl/`.

Now, when I say, “explicitly tells it” that needs to be done very carefully. If you’re not careful with how you set these up, you might find that Zeitwerk is still attempting to load them following its own conventions.

To really grasp all of this, I'd strongly recommend looking at the excellent documentation for Zeitwerk directly. It's available as part of the Ruby on Rails framework documentation, and it includes a lot more detail and examples. In addition to that, the "Understanding Ruby's Object Model" section from *Programming Ruby 1.9 & 2.0: The Pragmatic Programmers’ Guide* by Dave Thomas is an absolute must-read. That section explains the mechanisms underlying Ruby’s constant lookup and namespacing. Another helpful paper to review is “Autoloading with Ruby on Rails”, which provides a more in-depth analysis of different autoloading implementations and specifically explains the motivations behind using Zeitwerk in Rails.

In summary, Zeitwerk doesn’t inherently collapse namespaces. Rather, it enforces a strict convention between file structure and namespace that might seem like a collapse if your code doesn’t follow it. Understanding this mapping and carefully aligning your directory structure with the code is crucial for preventing these autoloader-related issues. Explicitly overriding the loading conventions should be done with care and only when necessary.
