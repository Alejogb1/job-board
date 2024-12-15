---
title: "How to Rails filter an association based on a current_user class?"
date: "2024-12-15"
id: "how-to-rails-filter-an-association-based-on-a-currentuser-class"
---

alright, so you're looking at filtering an association in rails based on the current user, yeah? i've been down this road more times than i can count. it's one of those things that seems simple on the surface but can get tricky real fast, especially when you start throwing in more complex relationships or need to optimize for performance. let's break it down.

i remember this one project i had back in '13. we were building an internal tool for managing access to different datasets. we had users, we had datasets, and users could be granted access to specific datasets through a join table. standard stuff. the problem was, displaying only the datasets that the *current* user had access to in a dropdown was proving to be a headache, at first. we kept running into n+1 query issues, because we were naive and tried filtering on the view using ruby loops. not pretty. the database was groaning.

the core issue here is that you want to restrict the data being returned from a database query based on the attributes of the `current_user` object, but this filtering needs to happen at the database level, not in ruby after the fact. this avoids those nasty n+1 issues. there are a few ways you can do this, and the "best" approach really depends on your specific setup and preferences, but it always involves at least one association. let's start with a basic scenario and then we can talk about how you can use different approaches.

let's assume you have a `user` model and a `post` model, and that the relationship is that one user can create many posts. your models might look like this:

```ruby
# app/models/user.rb
class User < ApplicationRecord
  has_many :posts
end

# app/models/post.rb
class Post < ApplicationRecord
  belongs_to :user
end
```

and let's assume in your controller, you have `current_user`. if you wanted to get *all* posts by the `current_user` you could use this filter:

```ruby
#app/controllers/posts_controller.rb
def index
  @posts = current_user.posts
end
```

this is a very straightforward method, as long as all the posts are related to the current user, but what if you had a more complex many to many relationship, something like user have many groups and groups have many posts. in that case, you would use a `has_many through` relationship and your models might look something like this:

```ruby
# app/models/user.rb
class User < ApplicationRecord
  has_many :memberships
  has_many :groups, through: :memberships
  has_many :posts, through: :groups
end

# app/models/group.rb
class Group < ApplicationRecord
  has_many :memberships
  has_many :users, through: :memberships
  has_many :posts
end


# app/models/membership.rb
class Membership < ApplicationRecord
  belongs_to :user
  belongs_to :group
end

# app/models/post.rb
class Post < ApplicationRecord
  belongs_to :group
end
```

now this is where things get a little more involved. you can't just call `@posts = current_user.posts`, because that would return *all* posts related to the user, even if not part of a group the user is member of. instead, you need to filter through the association. one way to do this is by using a scope. it could look something like this:

```ruby
# app/models/post.rb
class Post < ApplicationRecord
  belongs_to :group

  scope :by_user, ->(user) {
    joins(group: :users).where(users: { id: user.id })
  }
end
```
and then in your controller:

```ruby
#app/controllers/posts_controller.rb
def index
   @posts = Post.by_user(current_user)
end
```

this approach is quite good because you are creating a named scope that is reusable and the filtering happens at the database level so, no n+1 queries. but what if you had to filter by multiple attributes in the `current_user`, i did once and it was terrible. for that scenario you might need to write a custom filter function in the model and that would look something like this:

```ruby
# app/models/post.rb
class Post < ApplicationRecord
  belongs_to :group

    def self.accessible_by(user)
        joins(group: :users).where(users: { id: user.id }).where(posts: {published: true})
    end
end
```
then in the controller you would call:

```ruby
#app/controllers/posts_controller.rb
def index
  @posts = Post.accessible_by(current_user)
end
```

in that approach, i was filtering the `posts` based on the user id and if the post was published, so it was more complex, but it showcases the ability to use multiple `where` filters inside a model. and i know what you are thinking, can't we just move the filters to the controller? and the answer is, you can, but it might create redundant code and make it harder to refactor later. personally, i like to keep the database logic inside the models as much as possible. you could say that's my *model* code of conduct.

now, the performance can be an issue too. if you have complex relationships, joins can slow things down. sometimes it's worth denormalizing a tiny bit if you can to reduce the amount of joins. or, if that isn't possible, consider using indices on the columns you're using to filter, it might be a life saver if you have a large database. also caching is your friend, if the `current_user` changes very little it might be better to get all records at once and filter them based on the user changes. it depends on the application. i've seen applications where we cached the whole result because the filters were small and the records were not that many, versus some other ones where we had thousands of `posts` and we had to be very careful on how we fetch them.

also consider using eager loading for optimizing those queries. if you want to load the associated records for your `posts` you could do this:

```ruby
@posts = Post.accessible_by(current_user).includes(:group, :other_related_model)
```

this would pre load the group and the `other_related_model` associated with each post, so you wouldn't have additional queries later when you want to access this information.

another common approach that we didn't cover is using a gem like `pundit` or `cancancan`. those gems take authorization to the next level, you can define complex rules, and it will handle all the authorization logic for you. but for most common scenarios just simple `scopes` or class methods with `joins` are enough.

for deeper knowledge, i'd really recommend checking out the "eloquent ruby" book by russ olsen. the chapter on active record is great. or the official rails guides, specifically the section about active record queries, or better even the active record source code, i've learned a lot from there. if you feel like reading something more academic, you could check some papers on relational algebra, since at the end of the day that is what database queries are all about.

filtering associations based on `current_user` in rails is a really common task. it's definitely a thing that takes some time to fully grasp the nuances of, but once you get the hang of it, it becomes second nature. the important thing is to understand that you must always, always filter as much as possible at the database level, avoid ruby loops after getting the records, and to consider performance and optimize your queries. remember to always validate your filters, test your code and use a good debugger. if you follow this guide you should be alright.
