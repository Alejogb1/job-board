---
title: "Why are generated Devise views loading from disk after deletion from the Rails app?"
date: "2024-12-23"
id: "why-are-generated-devise-views-loading-from-disk-after-deletion-from-the-rails-app"
---

, let's unpack this intriguing issue with Devise views. I recall a particularly frustrating incident back in my early days with Rails; I’d meticulously removed a specific view template, only to find it stubbornly reappearing. It’s a common head-scratcher, and the behavior, while counterintuitive at first glance, actually stems from Devise’s design and asset pipeline interactions, especially with regard to view caching and how Rails handles generated vs. application-defined resources.

The core of the problem lies in how Devise manages its views. When you use `rails generate devise:views`, Devise does indeed copy view templates into your `app/views/devise` directory. These copied views are meant to be a starting point for your customizations. However, these *aren't* the *only* views Devise is aware of. Crucially, Devise maintains an *internal* set of default view locations. Even after you remove the locally generated versions, Devise, by default, can fall back to these internal defaults.

Let me break down the primary mechanism at play here: the view lookup process. When a request comes in, and a controller action using Devise renders a view (like `new.html.erb` for signing in), Rails' view rendering engine searches for the corresponding template. It follows a specific lookup order. Generally, it first checks for a template within the application's specified view paths. If it doesn't find the template there, it can then look in paths registered by gems, including Devise. Therefore, even if you delete your local `app/views/devise/sessions/new.html.erb`, the rendering engine still finds Devise’s internally bundled copy of that view. The presence of your local copy simply *overrides* the gem's version when present. The absence of your file does not prevent rendering; instead, the framework uses its fallback mechanism.

Another layer to consider here is how Rails’ asset pipeline interacts with view files. While not directly implicated in *view* location, caching can create an illusion of the view files not being deleted correctly if you had previously loaded the page before the deletion. Cached versions, be it by the browser or proxy, could be displaying an outdated version, further complicating debugging.

To understand this better, let's look at some common scenarios and how to rectify them with code:

**Scenario 1: Removing a view entirely**

Let's say I want to prevent rendering of any default devise login forms; I want to show a custom error page instead. I decide I don't want *any* login forms to be renderable, so I remove `app/views/devise/sessions/new.html.erb`, and I create a stub error view: `app/views/errors/404.html.erb`.

```ruby
#config/routes.rb
Rails.application.routes.draw do
  devise_for :users

  # Catch all other routes and redirect to a custom 404 view
  match '*path', to: 'errors#not_found', via: :all
end

# app/controllers/errors_controller.rb
class ErrorsController < ApplicationController
  def not_found
      render 'errors/404', status: 404
    end
end
```
However, when I try to access the login page, the default devise login form still pops up! This happens because Devise, if not overridden, always attempts to load `app/views/devise/sessions/new.html.erb` from its default template location if it doesn’t exist locally. To *prevent* this, I must override the behavior of the `new` action in Devise.

```ruby
# app/controllers/users/sessions_controller.rb

class Users::SessionsController < Devise::SessionsController
  def new
    render 'errors/404', status: 404
  end
end
```

Now, you would want to reconfigure routes:
```ruby
#config/routes.rb
Rails.application.routes.draw do
  devise_for :users, controllers: { sessions: 'users/sessions' }
  # Catch all other routes and redirect to a custom 404 view
  match '*path', to: 'errors#not_found', via: :all
end
```
This approach, overriding the controller action, provides much more precise control over how devise renders pages.

**Scenario 2: Fully Customizing a View**

Let’s assume I want to create an entirely customized login form. I have already generated the devise views using `rails generate devise:views`. To change the login form, we would navigate to `app/views/devise/sessions/new.html.erb`, change its content with our custom markup, CSS, and logic, and it will work. However, I want it to use an entirely different path, say `/login`. Here is how I can make it work.
```ruby
# config/routes.rb
Rails.application.routes.draw do
  devise_for :users, path: '', path_names: { sign_in: 'login' }
   # Catch all other routes and redirect to a custom 404 view
  match '*path', to: 'errors#not_found', via: :all
end
```
In this approach, we do not need to modify the controller. We simply change the default path names and rely on our updated view to show in the new path.

**Scenario 3: Targeted Customization of Partial views**

Let’s say we want to customize the 'devise/shared/\_links' partial. We can do this by generating the partial views using `rails generate devise:views`, navigate to `app/views/devise/shared/_links.html.erb`, make the changes, and it will reflect in the devise forms.

```ruby
# app/views/devise/shared/_links.html.erb

<%- if controller_name != 'sessions' %>
  <%= link_to "Log in", new_session_path(resource_name) %><br />
<% end %>

<%- if devise_mapping.registerable? && controller_name != 'registrations' %>
  <%= link_to "Sign up", new_registration_path(resource_name) %><br />
<% end %>

<%- if devise_mapping.recoverable? && controller_name != 'passwords' && controller_name != 'registrations' %>
  <%= link_to "Forgot your password?", new_password_path(resource_name) %><br />
<% end %>

<%- if devise_mapping.confirmable? && controller_name != 'confirmations' %>
  <%= link_to "Didn't receive confirmation instructions?", new_confirmation_path(resource_name) %><br />
<% end %>

<%- if devise_mapping.lockable? && resource_class.unlock_strategy_enabled?(:email) && controller_name != 'unlocks' %>
  <%= link_to "Didn't receive unlock instructions?", new_unlock_path(resource_name) %><br />
<% end %>

<%- if devise_mapping.omniauthable? %>
  <%- resource_class.omniauth_providers.each do |provider| %>
    <%= link_to "Sign in with #{OmniAuth::Utils.camelize(provider)}", omniauth_authorize_path(resource_name, provider) %><br />
  <% end %>
<% end %>
```
Here, we make small changes to the links using conditional logic to exclude sign-in links in sign-in views, etc.
In this instance, we can make modifications to the local copy of the view and the changes will reflect. If we remove `app/views/devise/shared/_links.html.erb`, we will see the defaults rendered as Devise loads it from its gem location.

To deepen your understanding of these concepts, I'd strongly recommend delving into the following:

*   **"Rails 7: A Comprehensive Guide" by David Heinemeier Hansson:** Although this book is about rails, it contains crucial information about the view loading process, the asset pipeline, and how these mechanisms are intertwined. Focus on chapters discussing views, layouts, and assets.
*   **Devise documentation:** The official Devise wiki and documentation offer very specific explanations about how Devise manages views and customization. You should pay particular attention to the 'customizing views' section.
*   **Source code of ActionView and ActionController:** Examine the code that governs view lookups in Rails and understand the mechanics of inheritance within classes that are responsible for rendering. This approach provides a very hands-on understanding of the lookup process.

In summary, the ‘phantom’ views are a consequence of Devise having its own set of defaults. Deleting local copies doesn't remove the gem's template. Customizations require either directly editing generated local copies or overriding Devise controller actions, with full awareness of how view lookup paths and the asset pipeline function within Rails. Being mindful of these intricacies leads to better, more maintainable code.
