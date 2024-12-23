---
title: "How can I customize Devise's sign-in and sign-out redirect paths, and what purpose does `protect_from_forgery` serve?"
date: "2024-12-23"
id: "how-can-i-customize-devises-sign-in-and-sign-out-redirect-paths-and-what-purpose-does-protectfromforgery-serve"
---

Okay, let's tackle this. It's a common scenario, and honestly, I've spent more than a few late nights debugging these exact redirects. It always seems simple until the edge cases start surfacing. You're asking about customizing Devise's redirect behavior after sign-in and sign-out, and also about the role of `protect_from_forgery`. These are related, but let's break them down step-by-step, starting with Devise redirects.

Devise, the popular authentication gem for rails, provides a good out-of-the-box experience, but real-world applications nearly always require more granular control over redirects. The default behaviour usually sends users to a predefined root path or a user profile page after a successful sign-in, and typically just back to the home page after sign-out. However, consider an instance I encountered a few years back: I was working on an e-commerce platform where users often started a shopping cart as guests, and after creating an account (or signing in), they needed to be sent right back to that cart. Default redirects were not going to cut it; they'd lose their cart, leading to poor user experience and support calls.

The core customization with Devise lies in overriding specific controller methods. Instead of modifying the Devise gem itself, which is generally not advised, we'll customize our own controller. Specifically, we want to modify the `after_sign_in_path_for` and `after_sign_out_path_for` methods within our application controller, or even better, within a specific `sessions_controller` that inherits from `Devise::SessionsController`.

First, let's create our custom sessions controller:

```ruby
# app/controllers/users/sessions_controller.rb
class Users::SessionsController < Devise::SessionsController
  def after_sign_in_path_for(resource)
    stored_location = session[:previous_url]
    session[:previous_url] = nil
    stored_location || dashboard_path # default if no session location
  end

  def after_sign_out_path_for(resource)
    new_user_session_path
  end
end
```

Here, the `after_sign_in_path_for` method first checks for a stored location in the session under the key `previous_url`. This is a very common pattern and one I've found invaluable. We might set this prior to authentication, which would occur when a user tries to visit a restricted page while not logged in and is redirected to the login form. We store the originally-requested page and then redirect back after they log in, which helps a lot with user flow. If no stored location exists, we redirect to a default `dashboard_path`, or whatever path we designate as the home page for logged in users. The key is that we clear `session[:previous_url]` to avoid infinite redirect loops.

Conversely, the `after_sign_out_path_for` method redirects users to the `new_user_session_path`, which is our sign-in page. This is a common approach, preventing users from getting stuck on a logged-in view when they are no longer authenticated. It’s important to keep the flow clear.

To make sure Devise uses this new controller, we adjust our `routes.rb` file:

```ruby
# config/routes.rb
Rails.application.routes.draw do
  devise_for :users, controllers: { sessions: 'users/sessions' }
  # ... other routes
  get 'dashboard', to: 'dashboard#index'
  root 'home#index'
end
```

See that the `devise_for :users` line is now explicitly pointing to our custom controller. This simple change makes a huge impact.

Let's explore a second case. Say we want to allow users to be redirected to a specific path they requested before they were asked to log in, but only if it matches a particular scope. Here's what that would look like:

```ruby
# app/controllers/users/sessions_controller.rb
class Users::SessionsController < Devise::SessionsController
  def after_sign_in_path_for(resource)
      stored_location = session[:user_return_to]
      session.delete(:user_return_to)

      if stored_location && stored_location.start_with?('/profile')
          stored_location
      else
          dashboard_path
      end
  end

   def after_sign_out_path_for(resource)
    root_path
  end
end
```
In this iteration, we only redirect to a saved location if it begins with `/profile`, thereby allowing specific redirects based on the path the user requested initially. The variable name `user_return_to` reflects the specific context of the location we wish to store in the session.

Now, regarding `protect_from_forgery`, it's an essential rails security feature and completely separate from Devise's redirects, but often misunderstood. It’s aimed at preventing cross-site request forgery (csrf) attacks. These attacks involve malicious websites that trick a user's browser into performing unintended actions on a site they are authenticated with.

For example, imagine a malicious site has a form that makes a POST request to your application, something like this:

`<form action="https://your-site.com/users/1/delete" method="post"><input type="submit"></form>`

If a user is already logged into your site and then visits the malicious site, their browser would automatically send the credentials with the request. If you don't use `protect_from_forgery`, the request would look perfectly legitimate, leading to data manipulation you did not intend.

To prevent this, `protect_from_forgery` generates a unique token. This token is included in every form and Ajax request made from the application and sent as a parameter with the request. The server checks if the submitted token matches the generated token; if they don’t match, the request is rejected, thwarting the attack.

By default, all rails controllers inherit from `ActionController::Base`, which includes `protect_from_forgery with: :exception`. This configuration raises an exception if the token doesn't match. However, you might sometimes want to handle this exception differently. To understand this completely, I'd strongly suggest reading the documentation for `ActionController::RequestForgeryProtection`, especially around the `:exception` vs `:null_session` configurations. Additionally, the section on csrf in the OWASP website is crucial for understanding the nature of these attacks and how to properly defend against them.

Let's look at a final example. Suppose you want to customize the behaviour when a csrf token is invalid. To do this, you need to define an exception handler:

```ruby
# app/controllers/application_controller.rb
class ApplicationController < ActionController::Base
  protect_from_forgery with: :exception

  rescue_from ActionController::InvalidAuthenticityToken, with: :forgery_protection_error

  private

  def forgery_protection_error
    flash[:alert] = 'Security error, please try again.'
    redirect_to new_user_session_path # or whatever makes sense for you
  end
end
```

Here, we define `forgery_protection_error`, which is called when `InvalidAuthenticityToken` exception is raised, which usually implies that the token did not match. This could happen if the user session has timed out or if they manually navigate to a specific action (bypassing the form).

In my experience, understanding both the redirect mechanisms of Devise and the security implications of `protect_from_forgery` are vital for building secure and user-friendly web applications. This stuff is rarely plug-and-play. It’s crucial to tailor the functionality to each specific project requirement for the optimal experience, and having a solid grasp on these concepts allows for much more control and less frustration. I recommend diving into the Devise documentation for session management, as well as studying rails security guides, particularly the section on csrf protection, to develop a comprehensive understanding.
