---
title: "How to get an active record relation of polymorphic parents from a child object?"
date: "2024-12-14"
id: "how-to-get-an-active-record-relation-of-polymorphic-parents-from-a-child-object"
---

alright, so you're diving into the fun world of polymorphic associations with active record, i get it. it can feel a little tangled at first, especially when you want to hop directly from a child record up to its potentially various parent types. i've been there, spent more hours than i'd like to remember staring at similar problems. let me walk you through how i usually tackle this kind of situation, and share some code along the way.

basically, the challenge here is that active record's magic, while generally awesome, needs a little nudge when we're dealing with polymorphic relationships. it doesn't automatically know *which* parent table to look in, since a child can be related to different types of parent models.

let's say you have a `comment` model, which is designed to be associated with either a `post`, a `photo`, or maybe even a `video`. in your `comments` table you have `commentable_id` and `commentable_type` columns, these are your bread and butter for polymorphs.

so, given a `comment` instance, you want to get the corresponding `post` or `photo` or whatever it's related to. simple association methods won't cut it, at least not directly.

here's how i typically approach this. the key is to use the `commentable` association, which active record creates for you when you declare the polymorphic relationship in your model. this works if you access it from an instance of comment. this `commentable` method returns the parent.

for instance, if you're starting from a specific comment, say `@comment`, and you just want to get the parent model, the following code is going to give you the desired result:

```ruby
  parent_record = @comment.commentable
  puts parent_record.inspect
  # => #<Post id: 1, ...> or #<Photo id: 2, ...> or #<Video id: 3, ...>
```
pretty clean and straightforward. what happens if you try to get a list of all `commentable` objects? then it starts to be a problem.

but what if, instead of one comment, you have a collection of comments and you need their associated parents? the basic `each` loop will work fine, though it is a little verbose:

```ruby
  parents = []
  Comment.all.each do |comment|
    parents << comment.commentable
  end

  puts parents.inspect
  # => [#<Post id: 1, ...>, #<Photo id: 2, ...>, #<Video id: 3, ...>, ...]
```
this works, but it’s not very efficient, especially when you have a lot of comments. every time we loop to get a `commentable` object we are triggering a new sql query. this leads to n+1 issues.

a more optimized approach would be to use `includes`, which will try to load all records in a single query. you will save tons of sql queries this way.

```ruby
  comments = Comment.includes(:commentable).all
  parents = comments.map(&:commentable)

  puts parents.inspect
  # => [#<Post id: 1, ...>, #<Photo id: 2, ...>, #<Video id: 3, ...>, ...]

```
using `includes(:commentable)` tells active record to preload the associated parent records, minimizing database hits. the `map` operation then simply extracts the parent from each comment. if your parent record contains data that is relevant for the view or the application logic, then this is the recommended approach.

now, what if you needed to filter those comments based on specific parent types? for example, what if you only wanted to find all comments that belong to posts? we can achieve this by including a where clause that checks for a particular `commentable_type`.

```ruby
    post_comments = Comment.where(commentable_type: 'Post').includes(:commentable).all

    post_parents = post_comments.map(&:commentable)
    puts post_parents.inspect
    # => [#<Post id: 1, ...>, #<Post id: 3, ...>, ...]
```

this is a common use-case if you want to show only comments associated with a post, photo, or video. remember to use the string name of the model, like 'Post' and not `Post`.

i've also seen cases where people try to use joins directly on the polymorphic associations. while technically possible, it usually leads to more complex queries than what's really necessary. active record really does try to handle most of the heavy lifting behind the scenes so if you think about this from the point of view of which is the best api provided by active record then using the `commentable` attribute in combination with `.includes` and `.where` queries is usually a cleaner approach, rather than trying to build very complex sql. remember that these are all active record objects so you can use all the provided methods that active record gives you.

there was that one time, a long time ago, i was working on a social media app (before they were really cool), and i used a polymorphic relationship for likes across multiple content types. then i forgot the database was not an infinite resource, and i ended up writing a query to get all the parent types without `includes` and the app would just grind to a halt every time that query was called. a total disaster. i learned the hard way that preloading was crucial for performance in scenarios like this. it’s the kind of lesson you only need to learn once to never forget.

also, something that is worth mentioning, i have seen people try to override the methods of the model `Comment` like trying to define `def parent`. while that can work, it is best to stick to the default generated active record methods. sometimes doing magic like overriding methods can lead to unpredictable results.

i'd suggest diving deeper into the active record documentation for more insights. there are many books out there that cover active record in great detail, you should check the ruby on rails guides and maybe some books written by david heinemeier hansson himself, and for specific queries, maybe you should check books about database indexing. they are going to be more specific on how rails does queries.

so that’s how i usually deal with retrieving the parent records from a polymorphic association. it’s mostly about understanding the tools active record gives you and knowing when to use them. keep experimenting and happy coding!
