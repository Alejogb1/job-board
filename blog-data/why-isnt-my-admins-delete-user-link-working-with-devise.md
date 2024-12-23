---
title: "Why isn't my admin's delete user link working with Devise?"
date: "2024-12-16"
id: "why-isnt-my-admins-delete-user-link-working-with-devise"
---

,  I’ve seen this particular puzzle crop up more times than I care to remember, and it’s almost always a subtle permissions issue tangled up with Devise's default behaviors and possibly a sprinkle of confusion around HTTP methods. So, let's break down why your admin's delete user link might be failing, even when it *looks* like it should work.

First off, Devise is remarkably good at handling authentication for user models, but it doesn't magically grant superpowers to your admin panel. What's often missed is that while Devise handles user login, signup, and password resets, it doesn't inherently provide any logic for different *roles* or *authorization*. You've got to layer that on yourself. My experience comes from building and maintaining a multi-tenant application where we had admins, moderators, regular users and some custom roles too. We battled through these sorts of things consistently.

The typical scenario is this: You’ve got a `User` model, Devise is happily authenticating users, and you've probably whipped up a basic admin interface that includes a list of users with a delete button beside each. The problem isn't usually with Devise itself, but with the way you're trying to send the request, and how the application is interpreting that request. Specifically, it likely comes down to a combination of:

1.  **Incorrect HTTP Method:** HTML forms, by default, often use the `GET` method, or, more commonly, the `POST` method, but `DELETE` requests for deleting records. Devise handles `DELETE` on the user model, so if your form is not sending a `DELETE` request, it'll fall down. HTML, on its own, doesn’t support `DELETE` natively, so you'd need a little workaround.

2.  **Authorization Issues:** Even if you are sending a `DELETE` request, your server-side code might not be checking if the *current* user (presumably the admin) is actually *authorized* to delete a user record. You need code explicitly allowing an admin user to perform the delete action but not a regular user. Devise only checks authentication, not authorization, which are two different things.

3.  **CSRF Protection:** Rails, by default, protects against Cross-Site Request Forgery (CSRF) attacks, and Devise plays nice with this. However, if your form is missing a CSRF token, the server will reject the request. This is most commonly missing from custom form solutions that don't use form helpers.

Let's go through some examples to see how this works.

**Example 1: Incorrect HTTP Method & No CSRF Protection:**

This is a common mistake. Consider this very basic, broken, example using plain HTML:

```html
<!-- Incorrect HTML Delete Form -->
<form action="/admin/users/1" method="post">
  <button type="submit">Delete User</button>
</form>
```

This HTML uses a `POST` method, which does not conform with Devise's `destroy` method that is expecting a `DELETE` request. Furthermore, this code doesn't include the CSRF token. So, let's say, you're admin with an ID of 1, logged in, and click the delete button alongside user with ID 1. Nothing will happen. Here is the error you will see in the rails logs:
```
ActionController::InvalidAuthenticityToken in UsersController#destroy
```
This is because a rails controller expects, by default, a CSRF token, and, in the case of this scenario, the correct http method.

**Example 2: Correcting the HTTP Method using Rails Helpers, but ignoring authorization:**

We need to use Rails form helpers to change the HTTP method to `DELETE`. This will include a CSRF token, by default, which is a great start, but not enough:

```erb
<!-- Correct HTML Delete Form with Rails helpers -->
<%= button_to "Delete User", admin_user_path(user), method: :delete %>
```

, this is much better. This uses Rails form helpers and will send a `DELETE` request with CSRF token (as long as you have `<%= csrf_meta_tags %>` in your layout file). However, if you did not implement authorization, *any* authenticated user could delete another user. This is clearly not good. We need to add authorization. Let’s assume, for the sake of example, that we can authorize admin users using the `admin` boolean column on the `users` model. Here’s the code:

```ruby
# app/controllers/admin/users_controller.rb
class Admin::UsersController < ApplicationController
  before_action :authenticate_user!
  before_action :authorize_admin, only: [:destroy] # Ensure only admin users can delete

  def destroy
    @user = User.find(params[:id])
    @user.destroy
    redirect_to admin_users_path, notice: "User deleted."
  end

  private

  def authorize_admin
    unless current_user.admin?
      redirect_to root_path, alert: "Not authorized."
    end
  end

end
```

With the above controller in place and, assuming you are logged in as an admin user with the boolean value set to true, and the view contains the correct helper, your delete button should work. In our multi-tenant application, we initially had problems like this, especially with inconsistent HTTP method usages, as a result, we made sure that `DELETE` was always handled correctly in our controller.

**Example 3: Authorization using Pundit:**

For more complex permission logic, I’d recommend a library like Pundit. It helps with organization and prevents scattered authorization checks. Imagine you have policies for each model. Here is how your user policy would look like, if using `pundit`:

```ruby
# app/policies/user_policy.rb
class UserPolicy < ApplicationPolicy
    def destroy?
        user.admin? # Only admins can delete users
      end
end
```

And, now, you can use `authorize` method inside the `UsersController` from example 2:

```ruby
# app/controllers/admin/users_controller.rb
class Admin::UsersController < ApplicationController
  before_action :authenticate_user!
    before_action :find_user, only: [:destroy]

  def destroy
    authorize @user # using pundit
    @user.destroy
    redirect_to admin_users_path, notice: "User deleted."
  end

  private

    def find_user
        @user = User.find(params[:id])
    end

end
```

In the above code, Pundit's `authorize` method will call our `UserPolicy` to see if the current user can delete a user. This is a much more scalable and maintainable approach as it moves the authorization logic to the correct location, and keeps the controller clear. We implemented Pundit to handle much more complicated authorization logic, after running into some of the basic challenges above, so we can recommend it from experience.

**Recommendations for Further Reading:**

For diving deeper into these topics, I highly recommend the following resources:

*   **"Agile Web Development with Rails 7"** by Sam Ruby, Dave Thomas, and David Heinemeier Hansson. It's a great practical guide to all things Rails, including a thorough treatment of forms, routing, and authentication, and it should provide the basics for all the topics touched upon.
*   **"Pundit documentation"** (online, of course). Their documentation is clear and complete. Pundit documentation goes into greater detail about how to write policies and how to correctly implement them within your application.
*   **“HTTP: The Definitive Guide”** by David Gourley and Brian Totty. This is a fairly dense book, but provides an excellent overview of the different HTTP methods and standards, which is essential for understanding why certain requests need particular verbs.
*   **The official Rails documentation:** The Rails guides are a constant companion for any Rails developer, and it's incredibly detailed with clear instructions. The official Rails documentation touches on all the aforementioned areas.

In conclusion, the issue is almost always a combination of an incorrect method (not a DELETE request), missed authorization checks, or CSRF token related issues. Check all of those, and your delete functionality should be in good working order. Remember, Devise provides the authentication piece, but not the authorization; that's on you to set up, so use something like the example controller or a library like Pundit. Let me know if you have any other questions; I’m happy to help further.
