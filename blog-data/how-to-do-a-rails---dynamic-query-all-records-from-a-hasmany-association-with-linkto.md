---
title: "How to do a Rails - Dynamic query all records from a has_many association with link_to?"
date: "2024-12-14"
id: "how-to-do-a-rails---dynamic-query-all-records-from-a-hasmany-association-with-linkto"
---

alright, so you're looking at dynamically querying records from a `has_many` association in rails, and you want to use `link_to` to drive that, i've been there, trust me. this is a classic problem, and there are several ways to approach it, each with its own trade-offs. let's break it down, and i'll try to keep it simple.

first off, the basic idea is that you need to generate links that, when clicked, will filter your associated records based on some criteria. this usually involves passing parameters through the url. the tricky part is making this dynamic, so the filter can change based on the user's choice.

i've seen this implemented poorly in so many legacy codebases. like, seriously bad. once, i had to deal with a project where they were constructing these queries by string concatenation. sql injection nightmare fuel! i spent weeks refactoring that mess, so believe me, i understand the importance of doing this securely and correctly. it also helps that it looks much cleaner and much more readable, and who doesn't love clean and readable code?.

let's assume we have a `user` model that `has_many :posts`. and the goal is to filter posts by, say, their status, like `published` or `draft`. here’s how we can implement this dynamically:

the basic starting point, in our model we have a classic assocation:

```ruby
# app/models/user.rb
class User < ApplicationRecord
  has_many :posts
end

# app/models/post.rb
class Post < ApplicationRecord
  belongs_to :user
  enum status: { draft: 0, published: 1 }
end
```

the first important piece is to generate the links in your view. this will include the filtering parameter we want to use:

```erb
# app/views/users/show.html.erb
<h1>user posts</h1>

<p>filter posts:</p>

<%= link_to 'all', user_path(@user) %> |
<%= link_to 'published', user_path(@user, status: 'published') %> |
<%= link_to 'draft', user_path(@user, status: 'draft') %>

<ul>
  <% @posts.each do |post| %>
    <li><%= post.title %> - <%= post.status %></li>
  <% end %>
</ul>
```

in this example, we’re using `link_to` to create three links. the first link goes back to the user’s show page without any filtering, showing all posts. the other two links include a `status` parameter, which will be used to filter the posts.

now, the second important piece is to update your controller. you need to read the `status` parameter and use it to filter the posts. here's how it might look:

```ruby
# app/controllers/users_controller.rb
class UsersController < ApplicationController
  def show
    @user = User.find(params[:id])
    @posts = if params[:status].present?
              @user.posts.where(status: params[:status])
            else
              @user.posts
            end
  end
end
```

in this controller action, if the `status` parameter is present, we use a `where` clause to fetch only the posts matching that status. if it's not present, we fetch all of the user's posts. simple and effective. we're using activerecord's nice `where` clause which avoids string concatenation and all of its problems.

now, there’s another approach, and i found it useful in a recent project where we needed to combine several filters together, this approach is more flexible because it can handle more advanced scenarios:

```ruby
# app/controllers/users_controller.rb
class UsersController < ApplicationController
  def show
      @user = User.find(params[:id])
      @posts = filter_posts(@user.posts, params)
  end

  private

  def filter_posts(posts, params)
      filtered_posts = posts
      if params[:status].present?
          filtered_posts = filtered_posts.where(status: params[:status])
      end
      if params[:title].present?
         filtered_posts = filtered_posts.where("title LIKE ?", "%#{params[:title]}%")
      end
      if params[:content].present?
         filtered_posts = filtered_posts.where("content LIKE ?", "%#{params[:content]}%")
      end
      filtered_posts
  end
end
```

in this version, we have a private method `filter_posts` that handles all the filtering logic. this makes the controller cleaner and allows you to add more filters by adding additional conditions. if your params are dynamically coming from user inputs then you can use things like `params.permit` to only allow certain parameters for security.

a good practice to add is handling the case when the user type something wrong in the params. because for now if you type a non existent status it will give a sql error. for example if you type `status=hello`, rails will convert that string into an integer since the `status` is an enum. here is how you can fix it:

```ruby
# app/controllers/users_controller.rb
class UsersController < ApplicationController
  def show
      @user = User.find(params[:id])
      @posts = filter_posts(@user.posts, params)
  end

  private

  def filter_posts(posts, params)
      filtered_posts = posts
      if params[:status].present?
        begin
          filtered_posts = filtered_posts.where(status: params[:status])
        rescue ArgumentError
          # Handle the case where the status parameter is invalid
          flash[:error] = "invalid status parameter."
          filtered_posts = posts
        end
      end
      if params[:title].present?
         filtered_posts = filtered_posts.where("title LIKE ?", "%#{params[:title]}%")
      end
      if params[:content].present?
         filtered_posts = filtered_posts.where("content LIKE ?", "%#{params[:content]}%")
      end
      filtered_posts
  end
end
```

this is just one example, it can be improved in many ways, like extracting the logic to a model class or using gems that help handle this kind of dynamic queries but for now this will be enough to handle this particular use case. the joke i want to add is: a sql query walks into a bar, joins two tables and says: "can i have a union?".

regarding resources, i would recommend looking into the "rails guides". the activerecord section of the rails guides is an excellent resource for understanding associations, queries, and scopes. also, "eloquent ruby" by russ olsen has some good insights on writing clean and effective ruby code and explains well how the code should be structured, specially the controllers. it is very useful when you need to handle a more advanced project. also make sure you look for resources on best coding practices, specially on rails like design patterns, that would improve code organization and reduce bugs. and as always, experiment in a test environment first. you know, break things, learn from it, fix it, and then do it in production, that's the way.
