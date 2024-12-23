---
title: "How can I use subdomains with Rails Admin?"
date: "2024-12-23"
id: "how-can-i-use-subdomains-with-rails-admin"
---

Alright, let’s tackle subdomains and Rails Admin. It’s a configuration area I’ve spent some time navigating, and while it initially seems straightforward, the devil, as always, is in the details. I remember a project a few years back where we were building a multi-tenant application. We had a need for each client to have their own subdomain, and naturally, we wanted a centralized administration interface. Rails Admin seemed perfect for that, but getting it to play nicely with the subdomain routing was a bit of a journey. The core challenge isn't with Rails Admin itself; it's about how Rails handles subdomains and then ensuring the admin routes remain accessible despite these subdomain-based requests.

Essentially, we're dealing with two layers: first, the routing mechanism that parses the incoming request and identifies the subdomain, and second, the configuration of Rails Admin that has to respect that routing context. It's important to understand that Rails Admin, by default, doesn't consider subdomain context. It’s designed for a primary, non-subdomain route (e.g., `/admin`). We need to explicitly tell it to work within the subdomain structure.

My approach, and the one I found most effective through various iterations, involves strategically using constraints in Rails routing along with some careful configuration of the Rails Admin initializer. Let's look at how this typically plays out.

The first crucial piece is modifying `config/routes.rb`. This is where we define how incoming requests are handled. Here's a simplified example of how to set up a subdomain constraint:

```ruby
Rails.application.routes.draw do
  constraints subdomain: /.+/ do
    scope module: 'tenant', as: 'tenant' do
        # Custom routes specific to a tenant, if any
        get '/', to: 'dashboards#show' # Example for tenant dashboard
      end
  end

  # Primary domain (no subdomain): Rails Admin
  mount RailsAdmin::Engine => '/admin', as: 'rails_admin'
  root 'home#index'
end
```

In this example, `constraints subdomain: /.+/` ensures that any request with a subdomain is directed within the scope. The `scope module: 'tenant', as: 'tenant'` sets up a module called 'tenant' for these requests (which may be used to organize controllers and models) and also creates a name space for route helper functions using the `tenant_` prefix. The line `get '/', to: 'dashboards#show'` is just a stand-in for any route defined within the tenant subdomain, and you'll likely have much more complex application routing in that scope. Notice how the `mount RailsAdmin::Engine => '/admin', as: 'rails_admin'` is *outside* of the subdomain constraint. That's intentional. Rails Admin should, generally speaking, be hosted on the primary domain, not the subdomain. This keeps administrative tasks separated from the tenant application. In real-world scenarios, you might also want to use a specific subdomain for the administration panel, such as `admin.example.com` instead of placing it on the root. This is achievable by adjusting the routing constraints accordingly.

Now, let’s consider a setup where you *do* want a specific subdomain to host your admin panel. Suppose that you want your admin panel to be located at `admin.example.com`. The setup looks slightly different, and will use more explicit constraints:

```ruby
Rails.application.routes.draw do
    # Admin subdomain
    constraints subdomain: 'admin' do
      mount RailsAdmin::Engine => '/', as: 'rails_admin'
    end

    # Tenant subdomain routes
   constraints subdomain: /.+/ do
     scope module: 'tenant', as: 'tenant' do
       # Custom routes specific to a tenant
       get '/', to: 'dashboards#show'
     end
   end

    # Primary domain (no subdomain) routes for marketing content or any global information
  root 'home#index'
end
```

Here we specifically constrain the mount point of RailsAdmin to the `'admin'` subdomain. This configuration requires that the admin user navigate to `admin.example.com` to access the RailsAdmin dashboard. Tenant based routes use the wildcard character (`.+`) which specifies a route of one or more characters excluding none, for any other subdomain requests. The primary domain routes exist in the absence of a defined subdomain, and serve primarily for marketing/landing page purposes.

Lastly, and this is a common pitfall, you should ensure that your `rails_admin.rb` initializer (usually located in `config/initializers/`) is correctly configured to handle the subdomain. Most of the time, the default configuration doesn't require any change. However, depending on your environment, you might need to tweak authentication or access control to be aware of the current subdomain. I haven’t generally needed much customization on the rails admin side in this context, as the routing usually handles the separation nicely. If a more advanced use-case were present, and the models being worked with are dependant on the subdomain, then some level of modification may be needed in order to dynamically fetch the data needed.

Here is an example where a model may need dynamic fetching due to subdomain context:

```ruby
# config/initializers/rails_admin.rb
RailsAdmin.config do |config|
  config.model 'User' do
    list do
      # Custom list of records for a specific subdomain
      # Example: only show the users that belong to the current subdomain.
      field :name
      field :email
      field :created_at
      field :updated_at
      # Using a proxy to access the current tenant
      # Note, this assumes a method is set to obtain the current subdomain/tenant id.
      # A middleware should be implemented to set the current tenant.
       configure :tenant_id do
         pretty_value do
           # Assuming current_tenant method available through a context setter middleware.
           # The actual implementation may vary considerably.
           bindings[:view].current_tenant_id
         end
       end
     end

     # Example of access controls
     configure :tenant_id do
        hide
     end

     create do
        exclude_fields :tenant_id
     end

     edit do
        exclude_fields :tenant_id
     end

     show do
        exclude_fields :tenant_id
     end
    end
end

```

In the above snippet, the `User` model is configured within Rails Admin. Specifically, a `tenant_id` column which is used to identify the tenant a user belongs to is added to the fields list, and configured to fetch the id dynamically. The tenant_id has been excluded from the `create`, `edit`, and `show` routes in order to avoid allowing user to set this manually within the Rails Admin dashboard. A custom middleware would generally be used to set the tenant (subdomain identifier) on each request. This is illustrative of how you may need to modify the model level configuration of Rails Admin in order to reflect the routing being done at the `config/routes.rb` level.

These examples demonstrate the key aspects of configuring subdomains and Rails Admin together. While this seems complex at first, the logic is primarily based around proper route constraints and a correct understanding of how each component will behave. In practice, you’ll find that carefully defining the constraints and taking care not to have any conflicting routes tends to be enough to address most challenges. The key is to be aware that Rails Admin will respect the route context, so configure the routing first, and Rails Admin second. If your models require knowledge of the current request subdomain (or the identifier of your tenant), ensure that you're using a middleware to set the current tenant and expose that information in your model configurations, as shown in the last snippet.

For further, in-depth understanding on routing, I strongly recommend reading “Crafting Rails Applications” by José Valim. It provides an extensive view of routing mechanics in Rails, which will give you a strong foundational understanding of this topic. Additionally, the official Rails documentation (guides.rubyonrails.org) is always an excellent resource. Specifically, the documentation around "Rails Routing from the Outside In" provides a practical overview of the subject. As for middleware, if you need to research that area to set context for your Rails Admin routes, “Metaprogramming Ruby 2” by Paolo Perrotta does an excellent job at covering metaprogramming and techniques for handling request context. I hope this provides some practical guidance that you can utilize.
