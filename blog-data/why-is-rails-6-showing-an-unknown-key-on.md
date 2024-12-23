---
title: "Why is Rails 6 showing an unknown key 'on'?"
date: "2024-12-23"
id: "why-is-rails-6-showing-an-unknown-key-on"
---

Alright, let's unpack this 'unknown key on' error in Rails 6. It's a peculiar issue, and one I've encountered several times across projects – often in situations where configurations, especially those passed to active record queries or associated model setups, become a bit... *enthusiastic*. The core problem stems from a misunderstanding of how rails handles hash keys, particularly when those keys are strings versus symbols, and how certain configuration options are interpreted. Let's dissect this.

The error message, typically appearing as `ActiveRecord::UnknownAttributeError: unknown attribute 'on'`, often surfaces when you're attempting to pass a hash as an argument to methods like `where`, `joins`, or within the options of associations like `has_many` and `belongs_to`. The key component here is the word 'on'. It's not typically a standard column name, yet it's showing up as an unknown attribute. This suggests that Rails is interpreting 'on' not as a configuration directive, which it can be, but rather as an *actual* attribute name for the model it's interacting with.

Now, the heart of the matter. Rails, in its early versions, had a penchant for accepting strings and symbols interchangeably as hash keys in many contexts. This flexibility, while convenient, sometimes led to issues. Rails 6 introduced more rigorous parameter checking, resulting in a tighter distinction between strings and symbols, especially when dealing with Active Record. A key that might have previously been accepted as a symbol, often implicitly, is now interpreted strictly as a string – and potentially, as an attribute name.

The 'on' key, specifically, is frequently used within the context of association configuration in `has_many` or `belongs_to` declarations. It's meant to define the join condition when specifying a custom association or a through association. But, the key 'on', when passed as a simple string key in a hash, is now treated as if it’s an attribute name we’re looking up in the associated model. If that attribute doesn't exist, you’ll face the 'unknown key' error.

Let me offer some personal experience to clarify. I once had a rather hairy project where we were dealing with a legacy database structure. There was an unusual `has_many :through` relationship between `users`, `roles`, and a custom join table called `user_roles`. The association was defined as something akin to:

```ruby
class User < ApplicationRecord
  has_many :user_roles
  has_many :roles, through: :user_roles, source: :role
end

class UserRole < ApplicationRecord
  belongs_to :user
  belongs_to :role
end

class Role < ApplicationRecord
  has_many :user_roles
  has_many :users, through: :user_roles
end
```

Initially, everything seemed fine until we needed to apply some complex filtering criteria. I tried a naive approach that ended up throwing the dreaded error:

```ruby
# Naive approach (DOESN'T work)
User.joins(:roles).where("user_roles.role_id IN (?)", Role.where(name: ['admin', 'moderator']).select(:id).to_sql).where(active: true, 'on': Date.today.beginning_of_year)

```

This produced the "unknown attribute 'on'" error. The issue? The `where(active: true, 'on': Date.today.beginning_of_year)` part. Here, `'on'` was not being understood as a valid configuration key, but instead as an attribute of the `User` model. Rails thought I was trying to check for a column named 'on'.

The fix, and the crucial understanding, is that keys like `on`, when used within the context of join conditions or other active record methods that accept hash configurations, require a specific format to be correctly parsed. Typically, you would use a hash where the keys themselves are symbols, and often, the value is a string representing a sql fragment, which will be interpreted by active record as raw SQL for the join condition. In most cases, you would express the conditions as a `STRING` not a `DATE`, or you should use `ActiveRecord::Relation` methods which make the SQL building correctly. Here’s the corrected version, with multiple approaches:

```ruby
# Correct approach 1 (using symbol key and a string condition)
User.joins(:roles).where("user_roles.role_id IN (?)", Role.where(name: ['admin', 'moderator']).select(:id).to_sql).where(active: true).joins("INNER JOIN user_roles ON users.id = user_roles.user_id AND user_roles.created_at >= ?", Date.today.beginning_of_year)

# Correct approach 2 (using ActiveRecord::Relation methods instead of raw SQL)
User.joins(:roles).where("user_roles.role_id IN (?)", Role.where(name: ['admin', 'moderator']).select(:id).to_sql).where(active: true).joins(user_roles: :role).where(user_roles: { created_at: Date.today.beginning_of_year..Date.today.end_of_year })

# Correct approach 3 (using a simplified version, more common in Rails applications)
User.joins(:roles).where("user_roles.role_id IN (?)", Role.where(name: ['admin', 'moderator']).select(:id).to_sql).where(active: true, user_roles: { created_at: Date.today.beginning_of_year..Date.today.end_of_year })
```

Notice in the corrected version, we no longer pass an `'on'` key. In the first case, we use a standard `joins` method passing the raw SQL; in the second and third case, we are passing the desired condition as a hash key. We use more descriptive, ActiveRecord oriented methods. This is the key difference; by explicitly stating the join condition within the `joins` scope, Rails now correctly recognizes and parses the join condition without misinterpreting it as an attribute.

This also highlights the importance of understanding the underlying SQL being generated by Active Record. While Active Record simplifies database interactions, you need to be aware of how it's translating your ruby code into SQL. The 'on' key, in its string form, simply doesn't have a place in Active Record attribute lookups, and this is the change from older versions of Rails.

For a deeper understanding, I highly recommend looking into the *Active Record Query Interface* documentation in the official Rails Guides. Additionally, studying the source code for the `ActiveRecord::Relation` class, found in the Rails GitHub repository, will prove invaluable. Finally, reading a good book on database design and relational theory, such as "SQL and Relational Theory: How to Write Accurate SQL Code" by C.J. Date, will give you a stronger grasp of the underlying principles at play here.

In summary, the 'unknown key on' error in Rails 6 isn't some esoteric bug; it's a consequence of stricter type checking and the interpretation of configuration options as attributes. The solution involves using proper syntax for specifying custom join conditions, usually within the `joins` method and using active record relation method to filter the records instead of directly including sql code. When dealing with complex configurations, always verify the type of keys and values within your hashes to avoid these common errors. Learning to look at the actual SQL queries your code is generating will also aid in finding these issues quicker. Understanding this, you'll be better equipped to handle these kinds of errors in your Rails applications.
