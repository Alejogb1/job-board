---
title: "How do factory_bot add two or more associations in the create method?"
date: "2024-12-15"
id: "how-do-factorybot-add-two-or-more-associations-in-the-create-method"
---

so, you're hitting that spot with factory_bot where you need to create a record with multiple associations at the same time, directly in the create call. yeah, been there, done that. it's a common scenario when you're setting up more complex tests and don't wanna write a ton of setup boilerplate.

i remember this one project, years back, we were building this internal tool for managing server deployments (it was way before docker took over, imagine that). we had models like `server`, `application`, and `deployment`. and obviously, a deployment had to be associated with a server *and* an application. our tests were getting messy with multiple `create` calls just to set up the necessary context before testing the deployment logic. that's when i started looking at more efficient ways to handle associations in factory_bot.

the key is understanding how factory_bot handles associations and how you can leverage its features to build what you need in a single `create` call. it basically comes down to using the attribute assignment with the association names. the association names are not just strings, you're not just naming a foreign key, or some sql column, they are the names of the factory bot objects, in this case we are going to use objects that represent a table or entity in the database.

let's break it down with some code examples. imagine we have these basic models:

```ruby
class User < ApplicationRecord
  has_many :posts
end

class Post < ApplicationRecord
  belongs_to :user
  has_many :comments
end

class Comment < ApplicationRecord
  belongs_to :post
end

```

and the factories,

```ruby
# factories/users.rb
FactoryBot.define do
  factory :user do
    name { Faker::Name.name }
    email { Faker::Internet.email }
  end
end

# factories/posts.rb
FactoryBot.define do
  factory :post do
    title { Faker::Lorem.sentence }
    body { Faker::Lorem.paragraph }
    user
  end
end

# factories/comments.rb
FactoryBot.define do
  factory :comment do
    body { Faker::Lorem.sentence }
    post
  end
end
```

now, let's say we want to create a post and associate it with a user and a comment all at once. you might be tempted to think you need to create each model separately and then do the associations, or use build and then save each entity after using the same `build`. something like `post.comments << comment; post.save` which is something that i did when starting, you know, before i knew this. but no, there is a more elegant way using `create`:

```ruby
post = FactoryBot.create(:post, user: FactoryBot.create(:user), comments: [FactoryBot.create(:comment)])
puts post.title
puts post.user.name
puts post.comments.first.body
# output:
# A new post title for this post
# A name here
# A comment
```

see how we’re passing the associations directly as parameters to the `create` method. we’re specifying that the `user` association should be a new user created using `FactoryBot.create(:user)`. for comments, we are passing an array of comments, we just used one for simplicity.

you can do this with multiple associations at the same time. suppose you need two different users and the post is associated to both for some specific scenario you want to test. you could do it as follows:

```ruby
post = FactoryBot.create(:post, user: FactoryBot.create(:user), second_user: FactoryBot.create(:user))
puts post.user.name
puts post.second_user.name
# output:
# Another new name
# Yet another different name
```

this creates a post, two user models and uses each for each association in the single create call, assuming you added `second_user` as an association in your `Post` model.

one important thing i learned the hard way during some frantic debug sessions in that old deployment app project: factory_bot evaluates those nested `create` calls from the inside out. it’s not just creating the post and then doing the associations later, no, it creates the associated records first. This is sometimes tricky if you have dependencies between associations, but that’s another topic. we’re not here to talk about complicated circular dependency issues… not yet at least.

also, sometimes, you don't need to create a whole new record for every association. for example, if you're testing something that doesn’t require a specific user for every post, you might not need to create a user on each single create call. you could use `association` instead to just reference a factory and get a new instance without creating the new entity until it is used.

```ruby
FactoryBot.define do
  factory :post do
    title { Faker::Lorem.sentence }
    body { Faker::Lorem.paragraph }
    association :user
  end
end
# in your test or other factory
post = FactoryBot.create(:post)
puts post.user.name
#output:
# A new user name.
```
the difference is very subtle, but very important. this `association :user` means that a user instance will be built using the user factory when the `post` object needs to access the `user` association. in this case it is done via the `puts post.user.name`. this saves db calls, which, well, is good.

a good resource to grasp better these underlying mechanisms in factory_bot is the book “test-driven development with rails” by david chelimsky and some other authors, this has a lot of theory and details in testing, but also the testing of rails apps and how the factory bot gem is used in those scenarios, how it is implemented and the rational behind the library, i also found the official gem documentation (you can find this one easily) to be useful, it might be dense with details but worth the read.

one last thing: don't overdo it with this nested association creation. sometimes, especially when you start having more complex objects, breaking up the creation logic into smaller factories or helper methods can improve readability and prevent factory definition spaghetti. keep it simple, keep it clear. its a good mantra.

and that's basically it. you can create records with multiple associations in a single `create` call, you can use nested `FactoryBot.create` or simply pass the association name, you can use `association` instead of `FactoryBot.create` and save db calls, and you can start building tests and factories that are way more readable and simple. using the correct tool for the correct problem makes a big difference. i once tried to fix a bug by typing the solution backwards, it didn't work, i guess i just wanted to see if it was a good idea, it wasn't.
