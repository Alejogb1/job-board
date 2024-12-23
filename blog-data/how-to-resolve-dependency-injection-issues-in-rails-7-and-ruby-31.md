---
title: "How to resolve dependency injection issues in Rails 7 and Ruby 3.1?"
date: "2024-12-23"
id: "how-to-resolve-dependency-injection-issues-in-rails-7-and-ruby-31"
---

,  I've been through the trenches with dependency injection in Rails, particularly as it evolved through different versions, so I can definitely shed some light on the nuances in Rails 7 and Ruby 3.1. It's not always straightforward, especially when you're moving away from the more implicit mechanisms often relied on in older Rails applications.

Frankly, the shift toward explicit dependency injection is a healthy one, leading to more testable and maintainable code. However, it does require a deliberate change in how you structure your applications. One of the common pitfalls is clinging to global state or implicit dependencies. We've all been there, inheriting a codebase where seemingly magical things just "happen." The trouble is, it makes debugging and refactoring a nightmare. So, let's see how to fix it.

The crux of the issue generally lies in how we manage dependencies within services, components, or workers. Rails by default leans on `ActiveSupport::Dependencies`, which can make dependencies rather opaque. What we aim for instead is explicit injection, usually through constructors or setter methods. This makes the relationships within our code crystal clear and enables us to easily swap out implementations during testing or even at runtime.

Let's explore a scenario. Say we have a `UserService` that needs a `DatabaseClient` and a `Notifier`. In older Rails, you might just initialize these classes within the `UserService` itself. Here's how that can quickly turn problematic:

```ruby
# Bad Example: Implicit Dependencies
class UserService
  def initialize
    @db_client = DatabaseClient.new
    @notifier = Notifier.new
  end

  def create_user(params)
    # ... some logic ...
    @db_client.save(params)
    @notifier.send_notification(params[:email], "User created")
    # ... more logic ...
  end
end

class DatabaseClient
  def save(params)
    puts "Saving to database with #{params}"
    # database logic
  end
end

class Notifier
  def send_notification(email, message)
    puts "Sending email to #{email} with message #{message}"
    # notification logic
  end
end
```

The problem here is that `UserService` is tightly coupled to concrete implementations of `DatabaseClient` and `Notifier`. To test `UserService`, we'd either have to have the real database or a clumsy, global mock. This is not ideal. Instead, let's inject these dependencies. Here’s how a more robust approach looks:

```ruby
# Improved Example: Explicit Constructor Injection
class UserService
  def initialize(db_client, notifier)
    @db_client = db_client
    @notifier = notifier
  end

  def create_user(params)
    # ... some logic ...
    @db_client.save(params)
    @notifier.send_notification(params[:email], "User created")
    # ... more logic ...
  end
end

class DatabaseClient
  def save(params)
    puts "Saving to database with #{params}"
    # database logic
  end
end

class Notifier
  def send_notification(email, message)
    puts "Sending email to #{email} with message #{message}"
    # notification logic
  end
end


# Usage
db_client = DatabaseClient.new
notifier = Notifier.new
user_service = UserService.new(db_client, notifier)
user_service.create_user(email: "test@example.com", name: "Test User")

```

Now, `UserService` doesn't care *how* it stores data or sends notifications. It just knows it needs something that can `save` and something that can `send_notification`. This is a crucial step towards decoupling your code. During tests, we can pass in mock versions of `DatabaseClient` and `Notifier`.

But, we also need a way to *manage* these dependencies, especially as applications get larger. Manually creating and injecting everything can quickly become cumbersome. This is where dependency injection containers come into play. We can create a basic one ourselves. Here is a very simplified example that is suitable for demonstration:

```ruby
# Example Dependency Injection Container

class Container
  def initialize
    @dependencies = {}
  end

  def register(name, &block)
    @dependencies[name] = block
  end

  def resolve(name)
    dependency = @dependencies[name]
    raise "Dependency '#{name}' not found" unless dependency

    if dependency.is_a?(Proc)
        @dependencies[name] = dependency.call
    else
        dependency
    end
  end
end

# Setup the container
container = Container.new
container.register(:database_client) { DatabaseClient.new }
container.register(:notifier) { Notifier.new }
container.register(:user_service) { UserService.new(container.resolve(:database_client), container.resolve(:notifier)) }

# Resolve the UserService through the container
user_service = container.resolve(:user_service)
user_service.create_user(email: "container@example.com", name: "Container User")

```

The `Container` class acts as a simple registry and resolver. You register dependencies, specifying how they are created (in our case using a block or direct instance), and then you ask the container to resolve the dependency when you need it. This is a rudimentary version; real-world applications often require more sophisticated features like singleton scopes, per-request lifetimes, and perhaps support for interfaces (using modules or base classes).

Here's the gist:

1.  **Explicit Injection is Key:** Move away from implicit dependencies (like direct `new` calls in the class itself). Inject dependencies via constructors or setters.

2.  **Decouple:** Make your classes depend on *abstractions* (like `DatabaseClient`) rather than specific implementations. This lets you swap them out easily.

3.  **Containers Help:** A container manages the creation and injection of these dependencies, preventing "wiring" complexity from becoming unmanageable.

This approach is not unique to Rails but particularly relevant in its context because older Rails applications tend to use implicit dependencies heavily.

Now, while you *could* build your own container, it's often better to leverage existing libraries which have far more features. Two great options to explore for ruby and rails are:

*   **Dry-system (part of the Dry-rb ecosystem):** This library provides a robust dependency injection system, along with features like configurable dependency lifecycles and auto-registration. It requires some getting used to, but its power and flexibility are substantial. The documentation available at *dry-rb.org* is very comprehensive.
*   **`tsyringe`:** This is an up-and-coming dependency injection library for ruby that takes inspiration from the same named TypeScript library and offers a very easy to learn API to quickly create a functioning DI container. The source is available on github, and the documentation is still being written but offers a great starting point.

For more insights on design patterns and practices that support this approach, I'd recommend digging into these texts:

*   **"Patterns of Enterprise Application Architecture" by Martin Fowler:** This is a classic reference on architectural patterns and their trade-offs, offering an excellent grounding for understanding many dependency injection needs.
*   **"Growing Object-Oriented Software, Guided by Tests" by Steve Freeman and Nat Pryce:** This book explains how to design code for testability, which in turn is facilitated heavily by dependency injection.

I hope this breakdown helps you navigate dependency injection in your own Rails applications. Let me know if anything is unclear, or if there is a specific scenario you would like to dive deeper into. This field is dynamic, and I'm constantly refining my approach, so I'm happy to share whatever I've found useful so far.
