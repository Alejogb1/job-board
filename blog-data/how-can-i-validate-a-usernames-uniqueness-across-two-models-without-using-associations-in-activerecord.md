---
title: "How can I validate a username's uniqueness across two models without using associations in ActiveRecord?"
date: "2024-12-23"
id: "how-can-i-validate-a-usernames-uniqueness-across-two-models-without-using-associations-in-activerecord"
---

,  It’s a problem I've encountered more often than I'd prefer, and it typically surfaces in systems that are evolving and have adopted certain design constraints early on. Let’s say we’ve got a classic scenario: we're dealing with a user model and, perhaps, a slightly less traditional "team" model, and both need usernames, but we don't want to go down the route of shared tables, polymorphic associations, or other typical ActiveRecord relation approaches. We need to ensure that, across *both* models, usernames are unique. I've seen this sort of thing in complex system architectures where different areas of the application are maintained by separate teams and are sometimes designed with different database schemas.

Now, why avoid associations? Well, sometimes, database schema restrictions, performance bottlenecks, or even just historical development decisions make it a less attractive option. Perhaps the models are completely distinct logical entities and shouldn’t be tangled up with AR associations for clarity and maintainability. In such cases, we must rely on other means to enforce this uniqueness constraint. We're going to approach this programmatically, and it will be robust if implemented correctly.

The core idea is straightforward: we’ll perform checks against each model when validating a new username. This essentially involves making sure the username doesn’t already exist within either the users or the teams table. Instead of delegating it to a database constraint—which would be ideal if we *could* do it—we will handle this in our application's code layer.

Here's how we can achieve this in Ruby, without ActiveRecord associations, focusing on code clarity and maintainability. I'll provide a few snippets showing different ways you could set this up based on how your specific app handles validation and data persistence. I'm going to assume you’re using Rails, given the ActiveRecord mention, but the general concept applies anywhere you’re dealing with models and need this validation.

**Snippet 1: Using a custom validator**

This approach encapsulates the uniqueness check logic into a dedicated validator, which is a clean way to separate concerns:

```ruby
# app/validators/unique_username_validator.rb
class UniqueUsernameValidator < ActiveModel::EachValidator
  def validate_each(record, attribute, value)
    return if value.blank?

    if User.exists?(username: value)
      record.errors.add(attribute, :taken, message: 'is already taken by a user')
    elsif Team.exists?(username: value)
      record.errors.add(attribute, :taken, message: 'is already taken by a team')
    end
  end
end
```

Then, in your models, you'd include this:

```ruby
# app/models/user.rb
class User < ApplicationRecord
  validates :username, presence: true, unique_username: true
end

# app/models/team.rb
class Team < ApplicationRecord
  validates :username, presence: true, unique_username: true
end
```

Here, we define a custom `UniqueUsernameValidator`, which when applied to a model, checks if a username exists in either the `User` or the `Team` table. The `validate_each` method is part of the `ActiveModel::EachValidator` interface, ensuring the logic applies to each attribute that uses it.

This method is neat as it’s reusable; if, later, you want to apply this logic to other models, you can. We can also customize the error message. The `.exists?` method is a quick and efficient way to check for the existence of a record based on a provided attribute, which is essential for this type of check.

**Snippet 2: Using a model-level validation method**

For simplicity, or if you prefer to handle validations directly in each model, this is an alternative. This is fine for simpler applications, but can lead to more duplication as the project grows:

```ruby
# app/models/user.rb
class User < ApplicationRecord
  validate :username_uniqueness

  private

  def username_uniqueness
    return if username.blank?

    if User.exists?(username: username) && (new_record? || username_changed?)
      errors.add(:username, 'is already taken by a user')
    elsif Team.exists?(username: username) && (new_record? || username_changed?)
        errors.add(:username, 'is already taken by a team')
    end
  end
end

# app/models/team.rb
class Team < ApplicationRecord
  validate :username_uniqueness

  private

  def username_uniqueness
    return if username.blank?

      if User.exists?(username: username) && (new_record? || username_changed?)
        errors.add(:username, 'is already taken by a user')
      elsif Team.exists?(username: username) && (new_record? || username_changed?)
          errors.add(:username, 'is already taken by a team')
      end
  end
end
```

Here, we define a private `username_uniqueness` method within each model. It performs the same basic check as our validator but does so directly within the model itself. I’ve also added a check here, using `new_record?` and `username_changed?` which ensures the uniqueness validation only happens during creation or when the username is being updated. This avoids unnecessary lookups. Notice the duplicated validation logic across both models, which highlights one disadvantage of this approach.

**Snippet 3: Using a service object (recommended for larger applications)**

Finally, for larger applications, we can abstract the uniqueness checking logic to a dedicated service object, following solid object-oriented design principles:

```ruby
# app/services/username_validator_service.rb
class UsernameValidatorService
  def self.username_available?(username)
    return false if username.blank?

    !User.exists?(username: username) && !Team.exists?(username: username)
  end
end
```

Then in the models, we apply this service and validate based on its response

```ruby
# app/models/user.rb
class User < ApplicationRecord
  validate :username_availability

  private

  def username_availability
    return if username.blank?

     unless UsernameValidatorService.username_available?(username)
       errors.add(:username, 'is already taken')
     end
  end
end


# app/models/team.rb
class Team < ApplicationRecord
  validate :username_availability

  private

  def username_availability
    return if username.blank?

     unless UsernameValidatorService.username_available?(username)
       errors.add(:username, 'is already taken')
     end
  end
end
```

This method has the advantage of centralizing the validation logic. The service object becomes a single point of truth for this logic. It also helps in keeping models slimmer by offloading cross-model validation logic. This approach improves maintainability and testability. Here, `UsernameValidatorService.username_available?` returns `true` only if username is not used in either model.

**Performance and Considerations**

When implementing these checks, performance is worth noting, especially when dealing with large datasets. It is generally recommended that you create database indexes on the `username` columns in both `users` and `teams` tables. This will drastically improve the lookup speeds. Another thing I’d note is the potential for race conditions if not handled carefully. Two concurrent requests might both find the username available and both attempt to save records using it. You can address this by utilizing database transaction mechanisms and potentially adding additional database constraints (even with programatic validation) like a unique index. Consider adding a retry mechanism for database update conflicts in production.

**Further Reading and Resources**

For anyone wanting to understand the details of custom validators within Rails, I highly suggest exploring the `ActiveModel` documentation, specifically the parts dealing with validations and custom validators. The book “Rails AntiPatterns: Best Practice Refactoring” by Chad Fowler is an extremely helpful resource for understanding best practices when dealing with model-level validations. To understand transaction management better, specifically in databases like Postgres, I suggest the documentation for your specific database system. A good book on database design principles will provide additional help here, such as “Database System Concepts” by Abraham Silberschatz et al.

In closing, you have a few good options for ensuring username uniqueness without using ActiveRecord associations. Choose the approach that best suits your system's size and complexity, but always consider maintainability, performance, and potential concurrency issues. My experience is that the service object pattern (snippet 3) gives the most scalable and testable solution for more complicated systems.
