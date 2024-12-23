---
title: "What caused the Active Record exception during the Rails 6.1.4.6 upgrade?"
date: "2024-12-23"
id: "what-caused-the-active-record-exception-during-the-rails-6146-upgrade"
---

Let's tackle this issue. I've seen my share of ActiveRecord hiccups, especially during significant Rails upgrades, and 6.1.4.6 certainly had its nuances. From my experience, what you're describing regarding an ActiveRecord exception, post-upgrade, often stems from a confluence of factors related to how Rails handles database interactions and model changes across versions. It’s rarely a single, glaring culprit but more commonly a subtle shift in behavior or underlying dependency.

Specifically, during that 6.1 upgrade, a few categories of issues were particularly prominent in my projects, and they often surfaced as perplexing ActiveRecord exceptions. Let's break these down:

**1. Changes in Default Association Handling and Foreign Key Management:**

One of the most common sources of post-upgrade headaches revolved around how ActiveRecord managed associations, particularly when dealing with foreign keys and join tables. Rails 6 introduced stricter enforcement around data integrity and foreign key constraints. For example, if you had models with `belongs_to` associations that were previously tolerating the absence of associated records (maybe due to legacy data), the 6.1.x updates often triggered validation errors or raised exceptions during create or update operations where these associations were mandatory according to the schema definition but lacked valid associated records.

Previously, these errors might've been silently ignored, or only flagged in logs; but the upgrade increased error visibility. In one project, we had a legacy 'order' model with a `belongs_to :customer` association. Data migrations over the years had created orphaned order records without a corresponding customer. Pre-6.1, this wasn't causing application crashes, but after the upgrade, the save operation on any such order would throw an `ActiveRecord::RecordInvalid` error. We had to implement a data repair script to assign a default "unknown" customer, or actively delete the invalid records, to mitigate the issue. The key was not just fixing the bug, but finding the root cause of the data inconsistency, something ActiveRecord 6.1 practically forced us to confront.

**Code Snippet Example 1:**

Here’s a simplified example of how such an issue can manifest. Suppose you have the following model definitions:

```ruby
# models/customer.rb
class Customer < ApplicationRecord
  has_many :orders, dependent: :destroy
end

# models/order.rb
class Order < ApplicationRecord
  belongs_to :customer
end
```

If there is an Order record in the database that has a `customer_id` that does not correspond to a `Customer` record. This code will not raise an error before 6.1:

```ruby
  order = Order.find_by(id: 5)
  puts order.customer # will likely return nil or return some cached object that does not exist.
  order.update(order_date: Date.today) #will probably succeed
```

However, after the update and an attempt to save it, you might encounter issues as Rails expects a valid customer_id and may try to load it with the update:

```ruby
  order = Order.find_by(id: 5)
  order.update(order_date: Date.today) # throws a ActiveRecord::RecordInvalid error because customer_id is invalid
```

The fix here involved adding a validation or handling the null association with a conditional statement, and then addressing the legacy data.

**2. Changes in Query Generation and Scope Behavior:**

Another common culprit was changes in how ActiveRecord generates sql queries, especially with regards to complex scopes or joins.  The query building process was refined in 6.1, and while this resulted in generally improved performance, some previously working scopes could behave differently, potentially resulting in exceptions.  For instance, we discovered an issue in a reporting system, where a complex chain of scopes used `includes` with multiple associations.  After upgrading, the generated sql was not including all required tables, resulting in null pointer exceptions (and in ActiveRecord contexts that often translates to an error) when trying to access attributes of those absent associated models in the views. The issue stemmed from an implicit join expectation that was not being met. The solution involved rewriting those scopes by explicitly specifying joins, and rewriting them to improve readability and maintainability.

**Code Snippet Example 2:**

Consider this set of related models:

```ruby
# models/user.rb
class User < ApplicationRecord
  has_many :posts
end

# models/post.rb
class Post < ApplicationRecord
  belongs_to :user
  has_many :comments
end

#models/comment.rb
class Comment < ApplicationRecord
  belongs_to :post
end
```

Let's assume you have a scope defined:

```ruby
# models/post.rb
scope :with_user_and_comments, -> { includes(user: [:comments]) }
```

In previous Rails versions, this might have "sort of worked" and not raised any exceptions. After the upgrade, depending on certain edge cases in your data, the generated SQL could be incorrect, resulting in problems when accessing attributes in your views:

```ruby
  posts = Post.with_user_and_comments
  posts.each do |post|
    puts post.user.name
    post.user.comments.each do |comment|
      puts comment.body
    end
   end
```

Depending on your dataset, `post.user.comments` might cause an error in some cases, because not all the records were included in the sql query, even though the model expected it. The proper solution here would be to use explicit joins and possibly split this into multiple scopes for better understanding of the query construction. The fix would look something like this:
```ruby
scope :with_user_and_comments, -> {
  joins(:user).
  joins('left join comments on comments.post_id = posts.id').
  select('posts.*, users.name AS user_name, comments.body as comment_body')
}
```

This ensures the joins are explicit and you are always pulling data when you expect it.

**3. Attribute Serialization and Type Casting:**

Finally, changes in how ActiveRecord handles attribute serialization and type casting could also introduce exceptions. Some custom attribute serialization logic or database-specific data types that worked without issue in prior versions might become problematic due to stricter type checks or changes in data handling. For example, if you have a text field that you use to store json objects directly, the serialization handling might have changed, which could result in an error when loading the data. During the upgrade from 5.x to 6.x, we encountered a situation where a timestamp field was not correctly being parsed because of subtle changes in the underlying time zone handling. Previously, it had been more forgiving, but the upgrade resulted in an error during model instantiation.

**Code Snippet Example 3:**

Imagine a model that stores JSON in a text column

```ruby
# models/config.rb
class Config < ApplicationRecord
  serialize :settings, JSON
end
```

If your database stores json strings in the `settings` attribute that don't comply with valid json, you might experience issues after the upgrade. For example, if the value is stored as `"[1,2,3]"`, prior versions might convert this into an array for ruby usage. After the upgrade, you might need to explicitly serialize/deserialize those values during migrations and model loading.

```ruby
 config = Config.find_by(id: 1)
 puts config.settings # throws an ActiveModel::SerializationError or similar
```

The fix here involved using a custom setter and getter to handle the serialization and ensure the values are always valid JSON.

To dive deeper, I would highly recommend you refer to the official Rails release notes, particularly the change logs for ActiveRecord for the 6.1.x series. They are invaluable for understanding the nuanced changes made in each release. Additionally, the book "Agile Web Development with Rails 6" by Sam Ruby et al. provides a thorough overview of ActiveRecord and can be an excellent resource, especially the chapters related to model associations and query building. Understanding the specific changes outlined in these resources will be instrumental in tracking down the exact cause of any particular ActiveRecord exception you might encounter. Also, digging through the specific commit logs for the ActiveRecord module can be beneficial as there is often a lot of context and justification for the code changes.

In my experience, these categories of issues were the major contributors to ActiveRecord related errors after the 6.1 upgrade. While each project's unique setup will have its specific challenges, hopefully, these areas provide a good starting point for your investigation. Remember, thorough testing and careful review of upgrade release notes are absolutely essential when upgrading Rails.
