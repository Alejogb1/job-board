---
title: "How do I combine active_record models and sort by shared properties?"
date: "2024-12-23"
id: "how-do-i-combine-activerecord-models-and-sort-by-shared-properties"
---

Okay, let's tackle this. Sorting ActiveRecord models based on shared properties—it's a classic challenge, and I've certainly banged my head against it a few times over the years. A common scenario is wanting to list users based on something like their most recent order date, which isn't directly stored on the `users` table, or perhaps ordering blog posts by the total number of comments. The trick isn't about hacking ActiveRecord, but about leveraging the database’s inherent capabilities and expressing your intent clearly.

The fundamental issue boils down to the fact that you want to sort based on data not immediately available within the model's table itself. Instead, this data usually exists in related tables, connected via associations. Now, ActiveRecord, being the ORM (Object-Relational Mapper) that it is, typically encourages you to operate with Ruby objects, leading some towards inefficient methods like loading everything and then sorting in memory. Don't do that, unless you *really* have to for a limited scope. The key is to perform that sorting operation at the database level, which is far more efficient.

My experience with a large e-commerce platform some years back solidified this. We initially tried loading all users and then sorting based on order date – it completely crippled the application. Refactoring to leverage SQL's capabilities turned the lights back on, so to speak.

Let's look at a few practical ways to do this, and I will provide specific working code examples, assuming a typical Rails setup.

**Scenario 1: Sorting Users by Latest Order Date**

Let’s say we have `User` and `Order` models with a typical `has_many` association. We aim to retrieve users sorted by the most recent order's `created_at` timestamp. Here’s a SQL-driven approach:

```ruby
class User < ApplicationRecord
  has_many :orders
end

class Order < ApplicationRecord
  belongs_to :user
end

# Example retrieval of sorted Users:
users = User.left_joins(:orders)
            .group('users.id')
            .order('MAX(orders.created_at) DESC NULLS LAST')
            .select('users.*, MAX(orders.created_at) AS last_order_at')

# You can now loop through 'users' and access last_order_at:
users.each do |user|
   puts "#{user.email}: #{user.last_order_at}"
end
```

Here's a breakdown of what's happening:

1.  **`left_joins(:orders)`:** We use a `LEFT JOIN`. This is essential because it includes users who don't have any orders (resulting in `NULL` values for associated fields).
2.  **`group('users.id')`:** We group the results by user id. This ensures that we're applying aggregate functions correctly to each user individually.
3.  **`order('MAX(orders.created_at) DESC NULLS LAST')`:** The core of the sorting. We use SQL's `MAX()` function to determine the latest order date per user. `DESC` ensures sorting from most recent to oldest. `NULLS LAST` ensures that users without orders are listed at the end.
4.  **`select('users.*, MAX(orders.created_at) AS last_order_at')`:** We select all user columns and calculate the most recent order's `created_at` date, which we alias as `last_order_at`. This creates a virtual attribute available on each user object, which you can use later.

This approach avoids loading all the data and performing any sorting in Ruby. Everything is handled by the database's query engine.

**Scenario 2: Sorting Blog Posts by Comment Count**

Next, let's imagine a blog scenario with `Post` and `Comment` models. We want to sort posts based on their total comment count.

```ruby
class Post < ApplicationRecord
    has_many :comments
end

class Comment < ApplicationRecord
  belongs_to :post
end

# Example of sorting posts by comment count:
posts = Post.left_joins(:comments)
           .group('posts.id')
           .order('COUNT(comments.id) DESC')
           .select('posts.*, COUNT(comments.id) AS comment_count')

# Example of how to use:
posts.each do |post|
  puts "#{post.title}: #{post.comment_count} comments"
end
```

The logic here is very similar to the previous example:

1.  **`left_joins(:comments)`:**  Again, we use a `LEFT JOIN` to include posts with no comments.
2.  **`group('posts.id')`:** We group by `post.id`.
3.  **`order('COUNT(comments.id) DESC')`:** We sort by the count of comments for each post in descending order. SQL’s `COUNT()` function efficiently performs this.
4.  **`select('posts.*, COUNT(comments.id) AS comment_count')`:**  We fetch all post attributes and calculate the total number of comments, assigning it to `comment_count`.

**Scenario 3: Complex Sorting with Multiple Joins**

Now, let’s consider something a bit more complex. Suppose we have a `User`, `Article`, and `View` model. We aim to order users based on how many articles they have written, sorted also by how many views those articles have received, in total.

```ruby
class User < ApplicationRecord
    has_many :articles
end

class Article < ApplicationRecord
  belongs_to :user
  has_many :views
end

class View < ApplicationRecord
  belongs_to :article
end

# Example of complex sorting:
users = User.left_joins(articles: :views)
             .group('users.id')
             .order('COUNT(articles.id) DESC, SUM(views.id) DESC')
             .select('users.*, COUNT(articles.id) AS article_count, SUM(views.id) AS total_views')


# Example of usage:
users.each do |user|
  puts "#{user.email}: #{user.article_count} articles, #{user.total_views} total views"
end
```

Key concepts here include:

1.  **`left_joins(articles: :views)`:** We use nested joins to fetch the related articles and views for each user.
2.  **`group('users.id')`:** Group by the user id.
3.  **`order('COUNT(articles.id) DESC, SUM(views.id) DESC')`:** We are now sorting by two things: First by number of articles and then secondly by total views of all user articles.
4.  **`select('users.*, COUNT(articles.id) AS article_count, SUM(views.id) AS total_views')`:**  Fetching the user attributes and the results of the counts, aliased appropriately.

**Further Reading and Resources**

For a deeper understanding, I recommend diving into a good database book that covers query optimization. "SQL and Relational Theory" by C.J. Date is excellent for understanding the underlying concepts, and for more specific query optimization techniques, consider “Database Internals” by Alex Petrov. For a more practical view regarding usage within the Rails framework, check out the “Agile Web Development with Rails” book. It has detailed sections about active record and its efficient use. Additionally, familiarizing yourself with the SQL dialect of your database will be immensely helpful.

In closing, the core to efficiently sorting ActiveRecord models based on shared properties lies in using SQL aggregation and joins effectively. Avoid loading all the records and then sorting; let your database do its job, which it does remarkably well. Focus on writing clear and concise queries to get the results you need. It may seem more complex initially, but in the long run, it will drastically improve your application performance.
