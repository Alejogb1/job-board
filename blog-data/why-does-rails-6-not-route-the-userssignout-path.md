---
title: "Why does Rails 6 not route the '/users/sign_out' path?"
date: "2024-12-23"
id: "why-does-rails-6-not-route-the-userssignout-path"
---

Alright, let’s dive into this. I've seen this specific routing quirk trip up quite a few developers, and it's a great example of how framework defaults, while typically helpful, can occasionally lead us down a rabbit hole. The issue with Rails 6 seemingly ignoring `/users/sign_out` is not that it’s *not* routing, but rather how it's interpreting that request in the context of Devise, the authentication engine that’s usually behind such paths.

My personal experience traces back to a large project I worked on several years ago. We were migrating a very large Rails 4 application to Rails 6, and the upgrade process had its share of gotchas, including this exact routing problem with sign out. We expected `/users/sign_out` to behave like it did before, but it just wasn’t working. Debugging showed that the route wasn't actually missing, but Devise was intervening to handle the sign-out process. The problem wasn't that Rails 6 didn't understand the path, but that Devise had pre-emptively intercepted it, and that the route as it was conceived wasn’t the one it was looking for.

Fundamentally, the issue isn’t that Rails 6 *cannot* route `/users/sign_out`. It *can*, and it *does*. The problem lies in Devise’s default behavior. Devise, when configured to operate on a given resource (like `User`), generates a collection of routes including those associated with user registration, sign in and sign out. By default, Devise generates its sign-out route as a `DELETE` request to `/users/sign_out`. This default is important because HTTP verbs are crucial for RESTful routing, and `DELETE` verbs are semantically used for destroying a resource. Logging out is, in effect, terminating a user session and, hence, it fits that semantic model.

When you send a `GET` request to `/users/sign_out`, Rails' routing engine, even before Devise gets involved, is not going to find a matching route. Devise's router only has a route configured for the `DELETE` verb for this path.

Let's consider a typical `config/routes.rb` file with Devise configured for users:

```ruby
Rails.application.routes.draw do
  devise_for :users
  # other routes
end
```

This simple configuration automatically generates a suite of routes for user authentication. Specifically, the sign-out route, internally, is defined somewhat like this (and you can verify this with `rails routes`):

```ruby
   destroy_user_session DELETE /users/sign_out(.:format)  devise/sessions#destroy
```

Notice the `DELETE` verb. This is the critical piece. If you send a `GET` request to `/users/sign_out`, the routing engine will likely attempt to match a different route, one that perhaps doesn't exist.

How do we address this? Well, there are a few ways, each with their pros and cons. The most direct method, and the one usually recommended, is to make the sign-out request a `DELETE` request instead of a `GET`. This aligns with the way Devise expects the route to behave.

Here’s a quick example using a standard Rails form, where the link to sign out is a button that sends a `DELETE` request:

```erb
<%= button_to "Sign out", destroy_user_session_path, method: :delete %>
```

This will generate a form that sends a `DELETE` request to the correct path.

Alternatively, if for some reason, you're absolutely locked into using a `GET` request for a sign-out functionality, you *can* configure Devise to support it, but you really shouldn't as it goes against standard conventions and makes the behavior of the system less understandable for other developers. You need to modify how the routes are generated for Devise by specifying which HTTP verb they should use. Here’s how you might do it:

```ruby
Rails.application.routes.draw do
  devise_for :users, controllers: { sessions: 'users/sessions' } # Specify a custom controller

  # other routes
end
```

Then, you'd need a custom controller `app/controllers/users/sessions_controller.rb`:

```ruby
class Users::SessionsController < Devise::SessionsController

  def destroy
     signed_out = (Devise.sign_out_all_scopes ? sign_out : sign_out(resource_name))
     set_flash_message! :notice, :signed_out if signed_out
     yield if block_given?
    redirect_to after_sign_out_path_for(resource_name)
   end
end
```

And then modify the routes in `routes.rb` to support the `GET` request for `destroy_user_session` route.

```ruby
Rails.application.routes.draw do
    devise_for :users, controllers: { sessions: 'users/sessions' },  path: '', path_names: { sign_out: 'sign_out' }
    delete 'sign_out', to: 'devise/sessions#destroy', as: :destroy_user_session
  # other routes
end
```

In this case, we specifically declare a `DELETE` for the `sign_out` route while also having our custom controller be responsible for the `destroy` action. This is not ideal since we’re still supporting a route with two verbs. It's important to be consistent with RESTful design principles.

In essence, the most pragmatic and idiomatic approach is to follow Devise's conventions and use a `DELETE` request for sign out. It is technically feasible to make it work with a `GET` request, but it’s an unnecessary departure from convention. Understanding this behavior comes down to recognizing how Devise automatically generates routes and how these routes are matched by Rails’ routing engine. The best thing you can do is to familiarize yourself with the generated routes (using `rails routes`) and the behavior of your authentication engine.

For further understanding of RESTful API design principles, I recommend reading "RESTful Web Services" by Leonard Richardson and Sam Ruby. It's an excellent resource that delves into the rationale behind HTTP methods and RESTful design. On the Rails front, "Agile Web Development with Rails" by Sam Ruby and David Heinemeier Hansson remains an excellent resource for all levels of experience, especially its chapters on routing and authentication. You can also dive deep into the official rails guides for more targeted information on the specific routing mechanisms. Understanding these things allows you to handle scenarios like this without too much overhead. It is best to stick to conventions where possible.
