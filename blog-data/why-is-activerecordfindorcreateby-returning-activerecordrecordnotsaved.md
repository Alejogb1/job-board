---
title: "Why is ActiveRecord#find_or_create_by returning ActiveRecord::RecordNotSaved?"
date: "2024-12-23"
id: "why-is-activerecordfindorcreateby-returning-activerecordrecordnotsaved"
---

Let's unravel this particular ActiveRecord quirk; it's one I've certainly bumped into a few times in my years. ActiveRecord's `find_or_create_by` method, while wonderfully convenient, can indeed throw `ActiveRecord::RecordNotSaved` exceptions when you least expect it. The issue doesn't stem from inherent bugs in ActiveRecord itself, but rather from a fundamental misunderstanding of its transactional behavior and how validations fit into the picture.

Essentially, this exception indicates that while the `find_or_create_by` operation attempted to create a new record, that creation failed *after* the record was instantiated, but *before* it was fully committed to the database. The key phrase here is "after the record was instantiated." `find_or_create_by` isn't a single, atomic operation; it's more of a two-step process. First, it attempts to find a record matching the provided attributes. If that fails, *then* it instantiates a new record using those attributes. Finally, and crucially, it attempts to save that newly instantiated record to the database.

The `ActiveRecord::RecordNotSaved` error occurs *specifically* during this final save attempt, which means your record's data isn't making it into the database. This is where validations and transaction rollbacks enter the picture. Let me walk you through some scenarios I've personally encountered, and hopefully it’ll make this all a bit clearer.

First off, let’s consider validations. ActiveRecord validations are your defense mechanisms against bad data creeping into your database. For example, let’s say we're working with a `User` model which includes a validation that all emails must be unique:

```ruby
class User < ApplicationRecord
  validates :email, presence: true, uniqueness: true
end
```

Now, imagine you are attempting to do the following:

```ruby
begin
  user = User.find_or_create_by(email: "duplicate@example.com")
  puts "User created with ID: #{user.id}"
rescue ActiveRecord::RecordNotSaved => e
  puts "Failed to create user due to: #{e.message}"
end

# And separately, another attempt with a unique email
begin
  user_unique = User.find_or_create_by(email: "unique@example.com")
  puts "User created with ID: #{user_unique.id}"
rescue ActiveRecord::RecordNotSaved => e
  puts "Failed to create user due to: #{e.message}"
end

# In the case a record already exists, this shouldn't cause an error, assuming that the original record was valid when created.
begin
  user_exists = User.find_or_create_by(email: "unique@example.com")
  puts "User found with ID: #{user_exists.id}"
rescue ActiveRecord::RecordNotSaved => e
   puts "Failed to find or create user due to: #{e.message}"
end
```

If a user with the email "duplicate@example.com" already exists in the database, `find_or_create_by` will find it. However, if no such user exists, it will attempt to create a new one with that duplicate email address. Because the `email` attribute's uniqueness validation fails at the save stage, an `ActiveRecord::RecordNotSaved` exception is raised. The second example using "unique@example.com" will correctly create a new record and not trigger the exception, the third will correctly find the previously created record. The problem is not with `find_or_create_by` itself, but rather that validations can cause the create portion to fail.

Another situation I’ve seen is when working with nested associations that have validations. For instance, consider the following models, where a `Post` can have many `Comments`, which are validated:

```ruby
class Post < ApplicationRecord
  has_many :comments, dependent: :destroy
end

class Comment < ApplicationRecord
  belongs_to :post
  validates :body, presence: true
end
```

Let's assume you try to create a post and associated comments using `find_or_create_by`:

```ruby
begin
  post = Post.find_or_create_by(title: "My Blog Post") do |p|
    p.comments.build(body: "") # Intentionally invalid comment
  end
  puts "Post Created: #{post.id}"
rescue ActiveRecord::RecordNotSaved => e
  puts "Failed creating post and comments: #{e.message}"
end
```
Here, even though you use a block to set attributes, the comment with a blank `body` is still associated with the post object in memory. When the post is being saved, Rails will also attempt to save its associated objects. Because the comment is invalid, it will fail validation and trigger `ActiveRecord::RecordNotSaved`. This can be a confusing scenario to debug, particularly when you are dealing with lots of associations.

One additional scenario that caught me out early in my career involved database-level constraints. While they are similar to validations, they are enforced by the database and not by Rails' ActiveRecord layer. For example, consider a table with a unique index on two columns combined. Imagine our user model has a `name` and a `code`, and we have a unique constraint on `name` and `code` as a tuple in the database. Our validations might catch certain duplicate cases, but the database handles this case:

```ruby
class User < ApplicationRecord
  validates :name, presence: true
  validates :code, presence: true
  #No validation for unique name-code tuple.
end
```

Now, let's say you attempt to use find_or_create_by with a set of attributes which, though passing the validations within rails, will fail the unique constraint in the database.

```ruby
begin
  user = User.find_or_create_by(name: "test", code: "A123")
  puts "User created with ID: #{user.id}"
rescue ActiveRecord::RecordNotSaved => e
   puts "Failed to create user due to: #{e.message}"
end

# Assuming a user already exists with name: "test" and code: "A123", 
# and another user is created with the following
begin
  user_duplicate = User.find_or_create_by(name: "test", code: "A123")
  puts "User created with ID: #{user_duplicate.id}"
rescue ActiveRecord::RecordNotSaved => e
  puts "Failed to create user due to: #{e.message}"
end
```
In the first example, assuming no record exists, everything runs fine and a record is created. In the second example, assuming a user with a name "test" and code "A123" was already created, the validation within ActiveRecord will pass, but the database will raise an error because of its unique index constraint, which is translated to an `ActiveRecord::RecordNotSaved` exception on the ruby side.

To handle these cases effectively, you need to be explicit with your validations and aware of constraints at the database level. You also need to decide what the program should do when `find_or_create_by` fails. You might implement logic to retry creation with different data, notify an administrator, log the error for later analysis, or even decide to proceed with a different course of action altogether. I typically wrap such actions in rescue blocks and log failures.

A further refinement is to use `find_or_initialize_by` followed by a manual save operation, particularly when you need to set additional attributes *after* instantiation but *before* persisting. This offers finer grained control. You can check for invalid objects before saving them to avoid the error altogether.

For further reading and a more complete understanding of ActiveRecord and database interactions I'd recommend *Agile Web Development with Rails 7* by Sam Ruby et al. which provides excellent detail on these topics. Additionally, *Database Design and Relational Theory* by C.J. Date offers a comprehensive treatment of relational databases and constraints, which will provide valuable insight into why these things fail. Finally, diving into the official Ruby on Rails documentation for ActiveRecord is always a good step. The documentation will illuminate the inner workings of `find_or_create_by` and validations, providing a solid foundation for more robust coding. In my experience, it's always worthwhile to review these sources periodically, as your understanding of these systems will grow with experience.
