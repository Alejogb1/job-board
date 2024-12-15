---
title: "How to Load a related model in rails 7?"
date: "2024-12-15"
id: "how-to-load-a-related-model-in-rails-7"
---

alright, let's talk about loading related models in rails 7. it's a bread-and-butter kind of thing, but can get a little hairy if you don't approach it with the right tools. i've definitely been there, staring at a slow page load thinking, "there *has* to be a better way." and, yeah, there is. so, here's how i've tackled this in projects over the years, and some techniques that have saved my bacon.

first off, the core issue when loading related models is often the n+1 query problem. you query for a bunch of records, then for each of those records, you fire off another query to get the related data. this quickly spirals out of control when you have a decent sized data set. rails provides tools to avoid this, mostly centered around eager loading.

imagine this scenario: we have a `user` model and a `post` model, with a one-to-many relationship. a user has many posts. naive code might look like this:

```ruby
@users = User.all

@users.each do |user|
  puts "User: #{user.name}"
  user.posts.each do |post|
    puts "  - Post: #{post.title}"
  end
end
```

this is the classic n+1 offender. each time you call `user.posts`, it's firing off a new query to the database. on a small dataset, this is hardly noticable. but imagine you had 1000 users. that’s 1001 database queries in total, a single query to get all users and then 1000 to get their posts, ouch! this can really slow things down. i once had a dashboard that took almost 10 seconds to load because of this. i was so green at the time, i almost went for a different career, true story.

now, the solution is to tell rails to load all the posts at the same time as you load the users. we do this with `includes`. here's how you fix the code above:

```ruby
@users = User.includes(:posts).all

@users.each do |user|
  puts "User: #{user.name}"
  user.posts.each do |post|
    puts "  - Post: #{post.title}"
  end
end
```

with `.includes(:posts)`, rails loads all posts associated with the users in a single additional query. now it is two queries in total: one for all users and one for all posts in one go. you will see a huge performance boost here and your application will feel snappier. the time the page takes to load will drop to a fraction of what it was.

now, there are variations of `includes` that can be helpful. if you only want to load a small subset of the associated models you can be specific, using `where`.

let's say you only want the active posts, you can use a `where` clause within `includes`, like this:

```ruby
@users = User.includes(posts: :active_posts).all
```

assuming that `active_posts` is a scope you have defined in your `Post` model. this will eagerly load the active posts only. that is more efficient, you don’t want to load a ton of unecessary data. here's how to do that in your `Post` model:

```ruby
class Post < ApplicationRecord
  belongs_to :user
  scope :active_posts, -> { where(active: true) }
end
```

sometimes you have nested associations. for instance, a `post` could have many `comments`, and you want all of them too. rails handles this nicely too.

```ruby
@users = User.includes(posts: :comments).all
```

this tells rails to fetch all users, then their posts, and then the comments for those posts, all in an optimized set of queries. you can keep nesting like this, but you have to be very careful, i don’t recommend doing it more than 3 levels, there are better options if your use case requires that.

you also can load only what you need using `select`. maybe you only need specific columns from the related models.

```ruby
@users = User.includes(posts: :comments).select(:id, :name).all
```

here we’re selecting only `id` and `name` from users, reducing the amount of data transferred. this helps if your users table is very fat. if you want specific columns from `posts` and `comments`, you could do:

```ruby
@users = User.includes(posts: [
  { comments: [:id, :content] },
  :title
]).select(:id, :name).all
```

this loads users with only their id and name, posts with their title, and comments with only their id and content. this is particularly helpful when you have large text fields and blob fields, you can drastically reduce your query time this way, especially over a slow connection. i remember one time, a simple change like this reduced page load time from 4 seconds to just over half a second, the business manager asked me how i learned magic that day.

when using `includes` you have to be mindful of what you are actually loading, you might end up loading too much data if you are not careful.

now, another technique i find useful is using `preload` instead of `includes`. `preload` forces rails to use separate queries for the parent model and the associated models, it might not be the best for every situation but it can help if you need to be more control over when each query is executed, you can also use it if `includes` is not working as intended and you need a quick patch.

```ruby
@users = User.preload(:posts).all
```

`preload` can be more explicit sometimes, especially when you have a lot of complicated queries.

also, if your query requires you to sort and order data, you might want to look at `joins`. the general rule is that you use `joins` if you need filter or order the parent model based on a child model, and `includes` if you want to show the child models on the view.

but in general, and most often, `includes` will be your best friend when you need to access associated data. you may have to experiment which works best in your particular scenario, i know i have. in my early days, i was all over the place, trying all methods at the same time, i learned quickly that it is best to choose one method and stick to it.

if you're looking to really get down into the weeds with this kind of thing, i recommend reading "database internals: a deep dive into how distributed data systems work" by alex petrov. it isn't rails specific, but understanding the underpinnings of query optimization will make you a better rails developer. another good resource, specifically about rails is "rails anti-patterns: refactoring for elegance" by gregory t brown. they help you to avoid pitfalls and common mistakes. another helpful resource is the rails guides, they're pretty comprehensive too, very useful in edge cases.

i had another project, a big one, with lots of data and lots of relationships, i decided to do a full deep dive on performance. after a week of fine tuning the database queries, eager loading, and various caching techniques, my boss told me that the application was running so fast that he was starting to feel dizzy every time he clicked a button. it was quite the compliment, i'm not complaining.

in summary, loading related models in rails 7 effectively revolves around understanding and using eager loading techniques like `includes`. avoiding n+1 queries will drastically improve the performance of your applications, remember to select specific columns if you need to and use preload if you have an edge case. using joins in conjunction with these techniques can also enhance performance for specific situations, and remember to use a good database profiler and try to understand your queries. it's something i've been working on for years, and i'm still learning new things. there are always new things to learn, that is one of the best things about software, it is constantly evolving.
