---
title: "Why does Zeitwerk collapse namespacing to the root level?"
date: "2024-12-23"
id: "why-does-zeitwerk-collapse-namespacing-to-the-root-level"
---

, let's unpack this zeitwerk namespacing issue. I've definitely tangled with this particular quirk before, and it's often the source of a head-scratching moment for developers new to Rails’ autoloading mechanism. The short of it is that zeitwerk, by design, prefers a flat namespace structure relative to the configured autoload paths, which can *appear* as though it's collapsing namespaces to the root. It isn't strictly collapsing in the sense of ignoring the directory structure, but rather it maps directory paths to constants based on a predefined set of rules, and those rules don't necessarily create nested modules to reflect every directory level.

Let's break down the mechanics and, more importantly, why this is a considered choice and not a bug. Zeitwerk doesn't traverse deeply nested directory structures and automatically create modules mirroring that depth. Instead, it essentially establishes a mapping between the file path relative to your autoload directories and the constant names that Ruby will use. This mapping is based on naming conventions, such as camel casing and removing directory path separators. If these conventions result in identical or ambiguous constant names at the root level, then the last loaded file will be the one that is ultimately mapped to that name, effectively masking any previous occurrences.

Here’s where the 'collapse' illusion comes from. Imagine you have the following directory structure under your application's `app` directory, which by default is added to the autoload paths in a standard Rails setup:

```
app/
├── models/
│   ├── user/
│   │   └── details.rb
│   └── api/
│       └── user.rb
└── services/
    └── user.rb

```

In the code, you might reasonably expect to access these files as something like `Models::User::Details`, `Models::Api::User`, and `Services::User` respectively. However, zeitwerk will convert these file paths relative to the `app/` directory into a sequence of constant names using conventions. This is where the 'flattening' effect comes in.

The file `app/models/user/details.rb` would attempt to define the constant `Models::UserDetails`. The file `app/models/api/user.rb` would attempt to define `Models::ApiUser`, and the file `app/services/user.rb` would attempt to define `Services::User`. Notice that the subdirectories *user* and *api* are treated as naming components, not namespace components. If you tried to access something like `Models::User::Details`, you'd encounter a `NameError` because that inner `User` module won't be automatically defined. It's not that zeitwerk *collapsed* it, but rather it never *created* it. Instead, the code will often map to a file with the same root name, in this example, `Services::User` would often be called with simply `User`. If you defined both `app/models/user.rb` and `app/services/user.rb`, which is quite common, this would lead to conflicts.

This might seem limiting, and for developers accustomed to explicit module declarations, it certainly can be a surprise. However, this approach is intentionally designed for performance and simplicity. Zeitwerk avoids constant name resolution issues by preferring that root-level constant names map directly to class files within your configured load paths. Explicit module definitions, or explicitly loading files, are the solution if you do not want a flat namespace. It ensures the autoloading machinery can be efficient since there isn’t any recursive module-finding logic.

Let's look at some examples to solidify this.

**Example 1: Basic Namespacing Conflict**

Consider these three files in `app/models`:

```ruby
# app/models/user.rb
class User
  def greet
    "Hello from the model User"
  end
end

```

```ruby
# app/models/api/user.rb
class User
  def greet
    "Hello from the api User"
  end
end
```

```ruby
# app/models/legacy/user.rb
class User
  def greet
    "Hello from the legacy User"
  end
end
```

If you try to access `User.new.greet` after zeitwerk has done its autoloading magic, you won't get three different `User` classes. Instead, the last file loaded with a class `User` will replace any other defined class with that constant name. In this case, it'll be the last one the loader found, likely `app/models/legacy/user.rb`. You would see "Hello from the legacy User".

**Example 2: Using Explicit Modules**

To handle this situation correctly, you'd need to either use a different root-level name for each class or explicitly define module nesting, as zeitwerk doesn't.

```ruby
# app/models/user/base.rb
module User
  class Base
    def greet
      "Hello from the User::Base model"
    end
  end
end
```
```ruby
# app/models/api/user.rb
module Api
  class User
    def greet
      "Hello from the Api::User model"
    end
  end
end
```

Here, you would explicitly access `User::Base.new.greet` and `Api::User.new.greet`, respectively. Note that `Api` and `User` are explicit modules created to act as a namespace. This example reflects the correct practice when working with zeitwerk.
**Example 3: Using `require` to force manual loading**
```ruby
# app/models/user.rb
class User
  def greet
    "Hello from the main User"
  end
end
```
```ruby
# app/services/user_service.rb
require_relative "../models/user"

class UserService
  def use_user
    User.new.greet
  end
end

```

In this example, we explicitly load the User model in the `UserService`. Zeitwerk will not automatically load this file, but when you call `UserService.new.use_user`, the output will be "Hello from the main User". This allows you to load classes in the order you desire.

So, why does zeitwerk function this way? The core reason is performance and to avoid the complexity of a dynamic namespace resolution process. Zeitwerk's approach is designed for speed and simplicity. The mapping is deterministic: it directly relates file paths to constant names through a simple set of rules, without requiring traversal and resolution of implicit module structures. It leverages Ruby's constant lookup to be fast and predictable. This design choice results in a much faster startup time and avoids many common issues associated with complex autoloading mechanisms. When developers understand these principles, the seeming "collapse" is actually a deliberate decision for optimization.

If you want to delve deeper into autoloading patterns and the reasoning behind Rails’ choices, I’d highly recommend reading ‘Metaprogramming Ruby 2’ by Paolo Perrotta. While not exclusively focused on zeitwerk, it covers the underlying principles of metaprogramming in Ruby that drive these decisions. Also, the official Rails guides are invaluable; check the documentation on autoloading and eager loading within the context of zeitwerk. Finally, the Ruby language reference itself will explain how constants are resolved and how the `load` path works. Armed with this knowledge and practical understanding, the behavior of zeitwerk becomes much clearer and more manageable.
