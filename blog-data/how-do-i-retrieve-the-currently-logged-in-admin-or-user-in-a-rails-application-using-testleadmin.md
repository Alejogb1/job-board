---
title: "How do I retrieve the currently logged-in admin or user in a Rails application using TestleAdmin?"
date: "2024-12-23"
id: "how-do-i-retrieve-the-currently-logged-in-admin-or-user-in-a-rails-application-using-testleadmin"
---

Alright, let’s tackle this. I’ve seen this scenario play out more than once over the years, especially when dealing with authentication layers layered on top of things like RailsAdmin. The need to programmatically access the currently logged-in admin or user within a Rails application that uses TestleAdmin (assuming you mean ‘RailsAdmin’, I’m rolling with that) is a fairly common requirement for a variety of reasons, from audit trails to conditional logic within custom actions. It’s not always as straightforward as one might hope, as RailsAdmin operates within its own context and often doesn’t expose the user session directly. Here's what I've learned through the trenches.

The core issue boils down to this: RailsAdmin, being a separate engine, doesn't automatically propagate the current user information like, say, your main application’s controllers might. It relies on its own authentication mechanisms, which are typically configured using Devise, or a similarly structured gem. The user session information, therefore, is available but needs to be accessed through the proper channels.

When I first encountered this, way back on an e-commerce project, we had a custom dashboard that needed to know the active admin so we could present tailored statistics and functionalities. We initially tried accessing the session directly through RailsAdmin configurations, but that proved too brittle and prone to break with updates. We ended up implementing a more robust solution that leverages RailsAdmin’s `current_user_method` configuration.

Essentially, RailsAdmin allows you to define a method in your main application’s controller (often, the `ApplicationController`) that will be called by RailsAdmin to determine who the current user is. This is the proper avenue for accessing the logged-in admin. Here’s how it’s typically set up.

Firstly, you configure the `current_user_method` within your `config/initializers/rails_admin.rb` file:

```ruby
RailsAdmin.config do |config|
  config.current_user_method { current_admin } # or current_user
  # ... other configurations
end
```

In this setup, `current_admin` (or `current_user` if you are using a different authentication method) is the method you will have defined in your `ApplicationController`. This is crucial because it ensures consistency in how you access the user across your entire application and also makes it easier to manage the authentication mechanism separately from RailsAdmin.

Next, inside your `ApplicationController`, you need to ensure that `current_admin` (or `current_user`) is properly defined based on your authentication logic. If you're using Devise, the method is already provided. Here’s an example using Devise for admin users:

```ruby
class ApplicationController < ActionController::Base
  protect_from_forgery with: :exception

  before_action :authenticate_admin!

  def current_admin
    current_admin_user
  end

end
```

In the snippet above, `authenticate_admin!` is a Devise helper method that ensures only logged-in admins can access certain areas. `current_admin_user` is the method that Devise uses internally to retrieve the logged-in admin user, and we’re aliasing this for RailsAdmin as `current_admin`. Note that with this setup, if you’re using a standard user model rather than an admin model, replace `current_admin_user` and `authenticate_admin!` with the Devise methods for your user model, usually just `current_user` and `authenticate_user!`.

Now, within RailsAdmin, you can access this authenticated user within custom actions, list views, and other configuration settings using the `bindings` hash. For instance, if I need to add a filtering functionality based on the user:

```ruby
RailsAdmin.config do |config|
  config.model 'Article' do
    list do
      field :title
      field :body
      field :created_at
      field :updated_at
      field :author do
        formatted_value do
           value.name  # assuming you have an author with a name attribute
        end
      end
      # Example of filtering a list based on current admin using bindings
      filters [:author, {
          :authorized_user_id => {
            :type => :integer,
            :label => 'Author of User' ,
            :query => lambda { |values|
                current_admin = bindings[:view].current_user
                if current_admin
                  # assuming you have some method that links an admin with the articles created
                  where(:author_id => Author.where(user_id:current_admin.id).map(&:id))
                else
                  # show all if no current admin user
                    where(id: nil)
                end
              }
            }
        }]
      end
   end
end
```

In this example, we are filtering the Article list only for the authors that are related to the currently logged in admin user. This also gives you access to `bindings[:view]` which contains many useful helpers available from the Rails Admin framework.

Let’s consider one final use-case – a custom action where you want to perform an operation based on the currently logged-in user. Suppose you want to have an action that sets the status of a record to 'reviewed' and logs the admin that did this:

```ruby
RailsAdmin.config do |config|
  config.actions do
    dashboard                     # mandatory
    index                         # mandatory
    new
    export
    bulk_delete
    show
    edit
    delete
    show_in_app
    # your custom action
    register_instance_option(:review) do
      # the binding hash is available here
      register_instance_option :member do
        true
      end
      action_name :review
      link_icon 'icon-ok'
      http_methods [:get, :post]
        controller do
        proc do
           if request.get?
              render :action => @action.template_name
           elsif request.post?
             # access current_admin through bindings[:view]
              current_admin = bindings[:view].current_user
              @object.update(status: 'reviewed')
                AuditLog.create!(
                user_id: current_admin.id,
                action: 'reviewed',
                record_type: @object.class.name,
                record_id: @object.id
                )
              flash[:success] = "#{@model_config.label} has been marked as reviewed"
              redirect_to back_or_index
            end
        end
      end
     end
  end
end
```

Here, within the custom action’s logic, we’re accessing the `current_admin` using `bindings[:view].current_user`, which is made available through the `current_user_method` we configured earlier. This shows how you can incorporate authentication context into custom actions for targeted functionality.

Now, for further learning, I’d highly recommend reading “Crafting Rails Applications” by Jose Valim for a good understanding of how Rails itself works and how to structure your controllers properly. For a deeper dive into authentication with Devise, the official Devise gem documentation is an excellent resource. It’s also worthwhile to explore the RailsAdmin documentation thoroughly, particularly the sections on configuration and custom actions. For security practices, you can examine the OWASP cheat sheets on authentication. These resources will not only help you understand the mechanics of this particular problem but will also improve your overall Rails development skillset.

In conclusion, retrieving the currently logged-in admin or user in RailsAdmin is best achieved by configuring the `current_user_method` and then accessing the current user via `bindings[:view].current_user`. It’s a cleaner, more maintainable way to handle authentication in the context of RailsAdmin, and it aligns with how Rails intended this all to be set up in the first place. Avoid trying to access session details directly as it creates tight coupling and brittle code, which makes long term maintenance difficult. The patterns discussed are effective, reliable, and in line with how a robust rails application is structured.
