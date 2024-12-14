---
title: "How to handle Ruby On Rails: Database model Associations?"
date: "2024-12-14"
id: "how-to-handle-ruby-on-rails-database-model-associations"
---

alright, so you're asking about handling database model associations in ruby on rails, yeah? it's a core part of rails and honestly, something i've spent a fair chunk of my career grappling with. it seems straightforward at first, but the devil's always in the details when you start scaling or dealing with more complex relationships.

i’ve seen it all, from junior devs getting caught in endless loop queries because of eager loading issues, to projects where the database schema resembles a plate of spaghetti because the associations weren't thought out properly. trust me, been there. i even remember this one time back at 'code solutions inc.' where a badly defined association brought down the production server for a solid 40 minutes because of cascading deletes. learned that one the hard way. we were using a legacy database too, so the data migrations were a nightmare. let's just say it was a 'character-building' experience.

let's start with the basics. associations in rails are primarily about defining how different database tables are related. we’re talking about the usual suspects: `has_many`, `belongs_to`, `has_one`, and the polymorphic `has_many :through`. understanding the semantics of each is crucial.

`belongs_to` is pretty straightforward. imagine you have a `post` model and a `user` model. a `post` *belongs to* a `user`. so, in your `post.rb` file, you'd have something like this:

```ruby
class Post < ApplicationRecord
  belongs_to :user
end
```

and in your `user.rb`:

```ruby
class User < ApplicationRecord
    has_many :posts
end
```

this sets up the basic relationship where each post has a foreign key referencing a user. rails is smart enough to infer that this foreign key is named `user_id` in the `posts` table. easy peasy. accessing a post's user is just `post.user`, and accessing a user's posts is `user.posts`.

but what happens when you need more nuanced control? enter options. for example, you might want to specify a different foreign key name or a different class name:

```ruby
class Comment < ApplicationRecord
  belongs_to :commentable, polymorphic: true
  belongs_to :author, class_name: 'User', foreign_key: 'author_id'
end
```

in this example we're using a `polymorphic association`, making the `comment` belong to different types of records, a `post` or a `video` or whatever. we also explicitly specify that the association to user via the `author` is done by `author_id` and that the class that is being referenced is `User`. this kind of fine-tuning can save a lot of headaches later on.

now, let's talk about `has_many :through`, because this is where things often get a bit sticky. this association is your friend when you’re dealing with many-to-many relationships. for instance, let’s say you have `authors` and `books`, and an author can write multiple books, and a book can be written by multiple authors. you need a join table, which rails calls a 'join model', it’s a model that belongs to both entities.

```ruby
class Author < ApplicationRecord
  has_many :author_books
  has_many :books, through: :author_books
end

class AuthorBook < ApplicationRecord
  belongs_to :author
  belongs_to :book
end

class Book < ApplicationRecord
  has_many :author_books
  has_many :authors, through: :author_books
end

```

this sets up a many-to-many relationship using `author_books` as the join model. accessing all of an author's books is now `author.books`, and vice versa for `book.authors`. the magic is that you can extend the `author_book` model to store extra information if necessary such as the page number the author wrote and you can have an attribute like `contribution_percentage`, or what you might need.

my experience? i've learned that planning your database schema *before* you start coding is key. drawing out the relationships, thinking about the queries you’ll be running, can save you hours of refactoring later. i remember spending a whole weekend fixing a performance issue related to an unoptimized through association. the query was doing 1000s of lookups in the join table and was just terrible. i refactored the join model to add a `cache_column` using active record callbacks, which improved response time by milliseconds per user, times thousands of concurrent users equals major time saved. that was a satisfying fix. (and the caffeine intake was impressive that weekend).

about performance and eager loading, it's important to understand `includes`, `preload` and `eager_load` that active record provides. they solve the `n+1 query problem` which is where rails does many queries that are repetitive and that can be reduced to just one query, sometimes two. it's common, especially when working with nested associations, that you end up with one query to get the `posts`, then another query to get the user for every single post. this makes things slow. very slow. the solution? include it when you get posts:

```ruby
posts = Post.includes(:user).where(published: true)
posts.each { |post| puts post.user.name }
```

with `includes` active record will try to load the associated records in a way that will require less queries by using `LEFT OUTER JOIN`. this can be beneficial but depends on the database implementation of `LEFT OUTER JOIN` so be mindful of the data that you are querying or it can cause performance degradation instead of an improvement.

`preload` will instead issue 2 queries, the posts and the users separately and then in memory join both recordsets. this is beneficial for cases when the data that is being pulled is a lot and `JOIN` statements can be expensive.

`eager_load` is the most aggressive, it issues a single complex `JOIN` statement. this can be great but may cause performance issues. be careful when using it and make sure your database and table structure are well optimized.

when to use each is a matter of fine tuning and depends on the queries you are performing and how much data you have and the database engine you are using. each has its own use cases, and i learned the hard way that there is no single one size fits all approach.

another thing: validations. it’s crucial that you add validations to your models, especially when you have associations. making sure a `post` has a `user_id` or that it is a valid `user` can save you a ton of data corruption problems. this is a really easy trap to fall in because rails has a soft spot for letting you create records without validations. i spent a couple days chasing this kind of issue in another company once. it ended up being a forgotten `belongs_to` validation that we had to fix on a model that was 3 years old and hadn't been changed much since.

finally, think about callbacks, `before_save`, `after_save`, etc. you can use them when you need to perform actions on the associated records, such as updating a user’s `posts_count` when a post is created or destroyed. but careful, too many callbacks can turn your code into an incomprehensible mess, so only use when it makes sense. try to keep your model as thin as possible and move business logic into dedicated services or libraries, for instance using the `active_interaction` gem is good practice.

as for resources, skip the "rails tutorials for beginners" that most people point to, that is usually the most basic introduction and don't get into details. instead, i’d recommend looking into “the rails 5 way” by obie fernandez for a good deep dive and general patterns with rails. "database systems: the complete book" by hector garcia-molina and jeffrey ullman is amazing for the theoretical foundations, and a good read to understand the problems we are dealing with. and for those interested in performance optimizations "sql performance explained" by markus winand is a must, it goes into details how databases execute queries. these are not rails specific, but they give a strong foundation. i do not usually rely on blog posts and more on the theory that is behind the technology that i am using.

associations in rails are more than just shortcuts to the database. they represent relationships in your application's domain, and if you don't think them through, well, you might find yourself debugging strange issues at 3 in the morning. it happened to me. more than once, sadly. oh well, gotta love code huh?
