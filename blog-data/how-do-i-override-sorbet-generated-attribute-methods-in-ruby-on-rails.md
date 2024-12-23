---
title: "How do I override Sorbet-generated attribute methods in Ruby on Rails?"
date: "2024-12-23"
id: "how-do-i-override-sorbet-generated-attribute-methods-in-ruby-on-rails"
---

Okay, let's talk about overriding Sorbet-generated attribute methods in Rails. It’s something that’s come up a few times in my career, and it’s rarely straightforward. Sorbet, as a static type checker, brings immense value to large Ruby codebases, especially in a Rails context, but its interaction with the magic of Rails' attribute handling sometimes requires a bit of finesse.

The core of the issue stems from how Sorbet handles attributes declared via `T.prop` or `T.let` inside a class. When you use these constructs, Sorbet infers or expects specific accessors (getter/setter methods) to exist, which often conflict with the attribute management mechanisms Rails provides, like those established using `ActiveRecord::Base` and its columns, or those established using simple `attr_accessor` in Ruby.

In essence, you’re dealing with two systems vying for control over attribute manipulation: Sorbet's strict type-aware view and Rails' dynamic runtime approach. Directly overriding the generated methods is technically feasible but demands a careful understanding of both systems to avoid type errors and unexpected behavior.

My own baptism into this problem came at a previous position, working on a complex e-commerce platform. We adopted Sorbet midway through development, and that’s when we started encountering some rather frustrating issues regarding attribute access, particularly after some more nuanced data validations were introduced. Rails' validations interacted perfectly fine, but Sorbet’s type-checking threw tantrums when we tried certain data transformations *before* the validations, often within methods intending to set attributes. We had to find methods to override generated accessors while maintaining type safety.

Now, let's dissect this into a few common approaches, each with its own set of trade-offs:

**1. Explicit Redefinition with `def`**

The most basic way is to directly redefine the attribute setter or getter using a standard `def` block.

```ruby
class MyModel < ActiveRecord::Base
  extend T::Sig

  prop :price, Integer

  sig { params(value: Integer).void }
  def price=(value)
    @price = value * 100 # Example manipulation, storing in cents
  end

  sig { returns(Integer) }
  def price
     @price / 100  # Example manipulation, return value in dollars
  end

end
```

In this example, we've explicitly defined `price=` and `price` methods, overriding Sorbet's automatically generated versions. Sorbet still knows about the `prop :price, Integer` declaration, so its type checks will still be active. Crucially, notice we've also kept the correct type signatures (`sig`) so Sorbet can continue to provide its static type checking functionality, which is the entire reason we're using Sorbet! This lets us handle the internal representation of the price in a different manner (e.g., storing in cents but representing in dollars). While this is the most straightforward approach, it can quickly become repetitive if you need similar logic across several attributes, or if we are not careful we could introduce subtle inconsistencies. This might be the best approach if you have just one attribute to override, where you are trying to coerce the type before performing validations.

**2. Using Method Aliasing**

A slightly more elegant approach is method aliasing. This method allows you to wrap the original Sorbet-generated method with your own custom logic, providing a clean way to augment its behavior without entirely replacing it.

```ruby
class MyModel < ActiveRecord::Base
    extend T::Sig

  prop :name, String

  alias_method :original_name=, :name=

  sig { params(value: String).void }
  def name=(value)
    new_value = value.strip.downcase
    original_name=(new_value)
  end
end
```

In this snippet, we're aliasing the original setter method for `name` to `original_name=`. This allows us to execute additional code (trimming and lowercasing the input string) *before* the original method is invoked. This approach is great if you need to augment an existing setter with an added behavior that is general, and can be applied across attributes in a similar way. The readability is also generally better than the previous approach, as you can easily grasp the intention. Method aliasing can help avoid re-implementing the entire attribute access mechanism, and it is usually preferable to the explicit re-definition.

**3. Utilizing `before_type_cast` and ActiveRecord Callbacks**

Rails provides hooks, such as the `before_type_cast` method within ActiveRecord. While not a direct override of Sorbet methods, this technique allows manipulation *before* type conversion occurs, which can be useful if you want to adjust data before it gets assigned to your attribute and ultimately type-checked by Sorbet. Consider this example:

```ruby
class MyModel < ActiveRecord::Base
    extend T::Sig

  prop :age, Integer

  before_type_cast do
     self.age = params[:user_age].to_i if params[:user_age].present?
  end

end
```

Here, we're using `before_type_cast` to modify the `:age` attribute before it goes through ActiveRecord's type coercion mechanism *and* before Sorbet does its type checking. In this example, we imagine we are receiving a `:user_age` param from a request, which we are going to use to update the model's age. This approach isn’t an override in the strict sense of redefining method accessors, but it allows you to work with the raw values coming in and transform them before they become typed attributes. It is beneficial if your logic needs to apply *before* the type conversion, and this method allows us to centralize all of this work in the model itself. One drawback of this approach is that it tends to be a little less explicit, and the intention behind the modification might be harder to immediately grasp, especially if the `before_type_cast` block is long.

**Important Considerations:**

*   **Type Signatures:** Always use the correct type signatures (`sig`) when overriding methods. This ensures that Sorbet continues to provide accurate type checking, which is essential for maintainability in a Sorbet-enabled project. Without proper signatures, you are defeating the very purpose of having Sorbet in the first place.
*   **Testing:** Whenever you override Sorbet-generated methods, be extremely thorough with your testing. Check not only for the correctness of the override but also for the continued adherence to Sorbet's type system. Pay special attention to cases where type conversions might be unexpected.
*   **Understanding Rails:** A deep understanding of Rails’ internals, particularly the mechanisms behind attribute access and type coercion, is crucial. Reading the Rails source code related to `ActiveRecord::Base` is highly recommended and extremely beneficial.
*   **Refactoring:** When you have complex data transformations it is better to encapsulate it within service classes or domain objects rather than directly coupling that logic into the model.

**Further Reading:**

For a deeper dive into these concepts, I recommend consulting the following resources:

*   **The official Sorbet documentation:** The Sorbet website ([sorbet.org](http://sorbet.org)) is your primary source for understanding its type system and behavior. Their documentation on props and signatures, as well as methods and overrides, is essential.
*   **"Agile Web Development with Rails"**: This classic book by Sam Ruby, Dave Thomas, and David Heinemeier Hansson, though constantly updated in new editions, provides fundamental insights into Rails architecture and ActiveRecord, and can help you more clearly understand how Rails manipulates attributes. You can find the most recent edition on the Pragmatic Programmers website.
*   **Rails source code**: Don't be afraid to dive into the Rails source code itself. It's an invaluable way to understand how `ActiveRecord::Base` works. Start with files in the `activerecord` gem, specifically those relating to attributes and type casting. You can access the source code on the official Ruby on Rails GitHub page.

In conclusion, overriding Sorbet-generated attribute methods requires a balanced approach. You need to respect Sorbet's type system, be very careful not to break Rails conventions, and maintain clarity in your code. I hope this explanation, based on personal experience and the strategies I’ve developed, helps you manage these situations more effectively. Remember to always prioritize clear communication and maintainability in your codebase.
