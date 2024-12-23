---
title: "Why is my admin's delete user link not working with Devise in Rails?"
date: "2024-12-23"
id: "why-is-my-admins-delete-user-link-not-working-with-devise-in-rails"
---

Okay, let's tackle this. It's a familiar situation, actually. I recall spending a good portion of a weekend a few years back debugging a very similar issue on a project using Rails and Devise. The delete user link, seemingly straightforward, refused to cooperate, and it turned out there were a few common culprits lurking beneath the surface.

Essentially, when a delete user link isn't working with Devise in Rails, the root cause often boils down to discrepancies in the intended route, missing or incorrect authorization, or issues with the delete method itself. Let's break it down, and I'll illustrate with examples based on common pitfalls i've seen.

Firstly, routing issues are frequently the culprit. Rails, being a highly structured framework, relies heavily on proper routing. Devise, in its default configuration, sets up a set of routes for user management. If your admin section is using a different scope, or has a nested path, then the intended delete action may be misrouted. It's crucial to inspect the output of `rails routes` in your terminal. You are looking for the route that should be associated with the user deletion. This route should correspond to your controller's `destroy` action. I've seen scenarios where the delete link was unknowingly pointing towards a `show` action, resulting in a 404, or some other unintended behavior.

For instance, let's assume you have an `Admin` namespace and a `Users` resource within that namespace, where the user model is called just 'User.' The relevant part of your `routes.rb` might look something like this:

```ruby
Rails.application.routes.draw do
  devise_for :users # standard devise routes

  namespace :admin do
    resources :users
  end
end
```

Now, in your `admin/users/index.html.erb` view, a naive attempt at a delete link might look like this:

```erb
<% @users.each do |user| %>
  <tr>
    <td><%= user.email %></td>
    <td><%= link_to "Delete", user, method: :delete, data: { confirm: "Are you sure?" } %></td>
  </tr>
<% end %>
```

This looks okay at first glance, but the generated link, in this scenario, would be something like `/users/:id`, not `/admin/users/:id`. This is because the `link_to user` is interpreting user to use the main user route not the admin scoped one.

Here's the corrected version with the necessary prefix:

```erb
<% @users.each do |user| %>
  <tr>
    <td><%= user.email %></td>
    <td><%= link_to "Delete", admin_user_path(user), method: :delete, data: { confirm: "Are you sure?" } %></td>
  </tr>
<% end %>
```

By using `admin_user_path(user)`, you're explicitly directing the link to use the admin scoped user route. This is a very common source of confusion.

The second major area to investigate is authorization. Even if you get the routing sorted, you need to ensure that the user initiating the delete action has the required permissions. This is especially relevant in administrative contexts. Devise itself doesn't handle authorization; it provides authentication. To manage permissions, you'll often use a separate gem like Pundit, CanCanCan, or Rolify. If you have implemented one of these, it's essential to verify that your policy or ability logic allows for user deletion under the specific conditions. I've previously spent several hours chasing an issue only to discover that the policy was incorrectly set to disallow deletion by any user type. It's crucial to ensure your authorisation layer is configured appropriately to allow deletion of the given user.

Let’s assume we are using Pundit and have a `UserPolicy` class that looks like:

```ruby
class UserPolicy < ApplicationPolicy
  def destroy?
    user.admin? # Only allow admins to delete users
  end
end
```

In your `admin/users_controller.rb`, you’d typically have something like:

```ruby
class Admin::UsersController < ApplicationController
  before_action :authenticate_user!
  before_action :set_user, only: [:destroy]
  def index
   @users = User.all
  end
 def destroy
    authorize @user # check policy for ability to delete
    @user.destroy
     redirect_to admin_users_path, notice: 'User deleted.'
 end

 private

  def set_user
    @user = User.find(params[:id])
  end
end
```

Here, the `authorize @user` line is where Pundit is doing its work. If the current user isn't an admin, the `destroy?` method in `UserPolicy` will return `false`, and Pundit will raise an exception, preventing the deletion. This is why inspecting your authorization layer is as critical as route definitions. When something fails seemingly without error this is a good area to check.

The third area to check is the actual delete method and associated callbacks in the model. While rare, there could be model level code interfering with the deletion process. For instance, a `before_destroy` callback that prevents the user record from being deleted or raises an error. This also relates to other database-related errors. If the deletion has been initiated with an unexpected id, the deletion action can fail quietly, with no visible error in the template or log. A good approach is to verify the user id being passed to the destroy method against the expected id of the user.
The `destroy` method, while straightforward for the simple cases, does call certain other mechanisms that can fail silently.

To illustrate this, let’s look at an example with an association that might cause issues:

```ruby
#User.rb
class User < ApplicationRecord
  has_many :posts, dependent: :destroy
  before_destroy :prevent_self_deletion

  def admin?
    self.role == "admin"
  end

  private
  def prevent_self_deletion
     throw(:abort) if self == Current.user # where Current.user is the currently logged in user.
    end
end

#Post.rb
class Post < ApplicationRecord
  belongs_to :user
end
```

In the above code example there is both a `dependent: :destroy` set against the posts association and also an added prevention of a user deleting themselves. The database will check that foreign keys are valid during the delete and if, for instance the posts belonging to the user are not properly deleted in the case of `dependent: :destroy` being missing, the database will raise an error. Moreover, the user could not delete them selves. If the current user was an admin and was trying to delete themselves the `prevent_self_deletion` method would throw an abort.

These types of model-level constraints, while essential for maintaining data integrity, can sometimes silently interfere with the deletion action if not properly understood and handled. Check the database logs to verify the user data is being referenced correctly when deleted.

In summary, when a delete user link isn't functioning correctly with Devise and Rails, focus on three key areas: routes, authorization, and the model's delete method and callbacks. Always start with `rails routes` and carefully inspect your authorization policies. If those appear correct, then review your model logic, specifically around the deletion behavior and any model hooks. For further in-depth reading on routing, the official Rails guides are incredibly thorough, and for authorization, I highly recommend “Secure by Design” by Dan Bergh Johnsson and Daniel Deogun. It will make a tremendous difference in your thinking around how authorization should be implemented. And for deep model work, “Agile Web Development with Rails 7” by Sam Ruby et al is a comprehensive resource. Remember, a methodical, step-by-step approach usually uncovers the underlying issue.
