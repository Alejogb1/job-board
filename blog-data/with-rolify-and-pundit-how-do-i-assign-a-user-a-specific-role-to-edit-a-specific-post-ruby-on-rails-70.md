---
title: "With Rolify and Pundit, how do I assign a user a specific role to edit a specific post Ruby on Rails 7.0?"
date: "2024-12-15"
id: "with-rolify-and-pundit-how-do-i-assign-a-user-a-specific-role-to-edit-a-specific-post-ruby-on-rails-70"
---

all good. so, you're trying to lock down post editing with rolify and pundit in rails 7.0, yeah? i've definitely been down that road a few times, and it can get a little tangled if you’re not careful. let me walk you through how i’d tackle it based on my past experiences with this exact setup.

first off, rolify gives you the role management framework, but it doesn't handle authorization. that's where pundit comes in. we need rolify to tag users with roles, and then pundit to define what those roles are allowed to do, specifically in the context of posts.

it sounds like you want a setup where a user has a specific role that lets them edit a *particular* post, not just any post. this means we need something a bit more granular than just a generic "editor" role. we'll use a "post_editor" role scoped to the post resource itself. think of it like a permission tag on a user, not on the user as a general user.

so, lets dive into what i usually do to get this working.

first, make sure you've got rolify and pundit installed, obviously. i’m assuming you do given the question, but sometimes, you never know. add the gems to your gemfile if you have not. then, run the installers. if you run into trouble, check the documentation, but if you get the gems into your gemfile and install correctly you will save a lot of headaches down the road.

now, let's look at how the rolify model should be set up. i usually have the `User` model and the `Post` model. assuming you have those, we need to link the two using rolify.

```ruby
# app/models/user.rb
class User < ApplicationRecord
  rolify
  # other model code
end

# app/models/post.rb
class Post < ApplicationRecord
  resourcify
  # other model code
end

```

here, `rolify` enables the role association on user. `resourcify` does the same for post so we can assign roles scoped to each specific post resource. the default rolify way is to have a generic role linked to a user, but `resourcify` lets you define a role scoped to that resource.

with that in place, we can move on to pundit. we create a policy file that manages how to verify user capabilities in posts. i usually put this inside the `app/policies` folder and name it `post_policy.rb`.

```ruby
# app/policies/post_policy.rb
class PostPolicy < ApplicationPolicy
  def edit?
    user.has_role?(:post_editor, record)
  end

  def update?
    edit?
  end

  class Scope < Scope
    # if needed for filtering indexes add scope functionality
  end
end

```
in this policy, we've created two methods: `edit?` and `update?`. both of these methods check if the user has the `post_editor` role scoped to that specific post (`record` refers to the current post instance). this means that just being an editor isn't enough to edit this post; they have to be a *post_editor* for *this specific* post.

now how do you add a role? we will add the role to a user in the controller using the `add_role` method. but we are not gonna add a generic role to user, we are gonna add it to a specific post record.

```ruby
# app/controllers/posts_controller.rb
class PostsController < ApplicationController
  before_action :set_post, only: [:edit, :update]
  # ... other controller actions

  def edit
    authorize @post, :edit?
  end

  def update
    authorize @post, :update?
    if @post.update(post_params)
      redirect_to @post, notice: 'post updated succesfully.'
    else
      render :edit, status: :unprocessable_entity
    end
  end

   def add_editor
    @post = Post.find(params[:id]) # assuming you get id from params
    @user = User.find(params[:user_id])
    @user.add_role(:post_editor, @post) #add role to user for this resource
     redirect_to @post, notice: 'user added as editor'
   end

  private

  def set_post
    @post = Post.find(params[:id])
  end
  def post_params
    params.require(:post).permit(:title, :content)
  end
end
```

you'll see that the `edit` and `update` actions now have `authorize @post, :edit?` and `authorize @post, :update?` to invoke our pundit policy. in `add_editor` we add the role to the user to the current post we are working with.

now, i’ve been in the exact situation you described. one time on a project i was building a community forum and we wanted to allow specific users to moderate certain discussions so, a user could moderate post A but have no permissions over post B or C. this solution provided a scalable structure where we could just add permissions directly to each post without having to maintain a crazy permission table, we just used the rolify tables.

if you want to get deep into rolify, i'd suggest you take a look at the source code. understanding how rolify works internally will save you a ton of time and troubleshooting. also, "patterns of enterprise application architecture" by martin fowler is a good resource to understand the concepts behind this type of problem, where you need to build scalable features based on permissions.

a quick note on naming, it seems like everyone is so happy with naming the policies `post_policy` i sometimes see projects with `user_policy`, `comment_policy` and so on. i prefer to create a more descriptive name so when i'm maintaining the project i know exactly what it does. for example i would call it `post_edition_policy` or `moderation_policy`, or whatever suits the need, it really makes a difference when dealing with bigger projects.

now let's tackle how you add a role to a user. this is the key aspect of making sure permissions are properly set on our app. let's add an action in the controller to add roles, you can use a form to select which users will get the `post_editor` role for the current post.

```ruby
  #in routes.rb
  resources :posts do
      member do
        post :add_editor
      end
    end
```

with this route, we have a new method inside our `posts` controller that allows to add a user as editor, which i added on the previous code snippet. that should give you a better idea of how it's hooked in.

regarding testing, it is crucial to have unit tests for your policies. i usually use rspec and have a dedicated folder for the policies tests. it helps avoid many potential errors in production.

oh, i almost forgot. one time, i spent a whole day debugging a problem related to rolify. i was tearing my hair out, and then i realized i had misspelled `resourcify`. that was the moment when i understood the importance of having test coverage, it saved me so much grief.

i think you should have a solid foundation now for your specific problem with rolify and pundit. it gives you the granularity that you require. if you run into any other problems, let me know, and ill be happy to provide assistance. remember that experience comes from a lot of reading and actually doing, and the best way to learn is to actually try, debug and learn from the experience.
