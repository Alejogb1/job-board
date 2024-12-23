---
title: "How can I configure Rails (5, 6, or 7) to use hyphenated URLs instead of underscored URLs?"
date: "2024-12-23"
id: "how-can-i-configure-rails-5-6-or-7-to-use-hyphenated-urls-instead-of-underscored-urls"
---

,  I remember a project back in 2017, when Rails 5 was still the new kid on the block, where we had to migrate a legacy system with hyphenated urls. It was a headache at first, but the solution, thankfully, is quite straightforward. The default behavior in Rails, of course, is to use underscores for parameter keys in urls, which typically maps back to the database column names. However, for many reasons—branding consistency, SEO considerations, or, as in my experience, integrating with existing systems—hyphens are preferable. The key here isn't some hidden gem or a complicated hack; it lies in customizing how Rails generates urls and interprets incoming parameters.

The first point to clarify is that this adjustment primarily involves two areas: url generation and parameter processing. We need to configure Rails to use hyphens when generating paths using helpers like `link_to`, `url_for`, and so on, and also to handle parameters sent *to* the application via these hyphenated urls.

Let’s start with url generation. The primary tool here is route customization. Rails uses a system based on defining these routes using a declarative syntax, typically found in `config/routes.rb`. While the standard convention leads to underscores, we can explicitly tell Rails to generate paths with hyphens. This requires us to specify `param` keys within our route definitions using hyphenated values. Here's a basic example showing how:

```ruby
# config/routes.rb
Rails.application.routes.draw do
  resources :articles, param: :article-title
  # or
  get 'articles/:article-title', to: 'articles#show', as: 'article'
  # other routes
end
```

In this first snippet, I'm demonstrating two variations. The `resources` declaration, in the first line, tells Rails that whenever it generates routes for the `articles` resource, it should use `article-title` instead of `article_id` or whatever is conventionally used for the identifier. This handles both generation and incoming parameter handling implicitly for all routes associated with resources. The second route demonstrates how to explicitly set the `:article-title` parameter and alias it with `as: 'article'`. You will notice the hyphen is being used and Rails' convention is being overridden. Both approaches accomplish the same goal, but the `resources` approach is far less verbose.

Now, let’s move onto the second area—parameter processing. Simply changing the route definitions as shown above is not sufficient. When Rails receives an incoming request, it needs to know how to translate those hyphenated parameter keys into parameters our controllers can access via `params`. Thankfully, Rails provides a straightforward mechanism for this—the `parameterize` method— and we can hook into the `before_action` lifecycle to make this happen. We need to intercept the parameters, convert the hyphenated keys into their underscored equivalents and then proceed with controller logic. This translation step is essential to ensure that Rails can correctly access the database. Here is the second code snippet showing the implementation:

```ruby
# app/controllers/application_controller.rb
class ApplicationController < ActionController::Base
  before_action :normalize_params

  private

  def normalize_params
    new_params = {}
      params.each do |key, value|
      new_key = key.to_s.tr('-', '_')
      new_params[new_key] = value
    end
    params.replace(new_params)
  end
end
```

This piece of code is implemented within the `ApplicationController`. The `before_action` callback ensures that it is executed before any action in any controller within your application. The `normalize_params` method iterates through the received `params`, replaces hyphens with underscores, and then replaces the original parameters hash with the modified version. This means that when your controller methods attempt to access parameters like `params[:article_title]`, you will indeed receive the correct value, even if it arrived in the URL as `article-title`.

The above snippets are sufficient for handling standard cases. However, issues may arise when dealing with nested parameters, or when you have other parameter processing logic or if you are using strong parameter filtering mechanisms. In more complex scenarios, you might need to recursively apply the parameter conversion or to adapt this logic to be within your strong parameters implementation.

Here is the third and final example showing an extension of the above to include nested parameters:

```ruby
# app/controllers/application_controller.rb
class ApplicationController < ActionController::Base
  before_action :normalize_params

  private

    def normalize_params(params_hash = params)
      new_params = {}
      params_hash.each do |key, value|
        new_key = key.to_s.tr('-', '_')
        new_value = if value.is_a?(Hash)
                      normalize_params(value)
                    else
                      value
                    end
        new_params[new_key] = new_value
      end
      params_hash.replace(new_params) if params_hash == params
      new_params
    end
end
```

Here we have a modified version of the normalize parameters method to handle nested parameters. If a value is a hash, the `normalize_params` method calls itself recursively for that hash. Also note the addition of the conditional statement `if params_hash == params` before performing the replacement. This prevents overriding the original `params` variable when processing nested parameters and ensures we only replace `params` at the top level.

For further exploration, I’d recommend looking into “The Rails 7 Way” by Obie Fernandez. It’s a comprehensive guide that dives deep into Rails internals, including routing and parameter processing. Also, for a more theoretical understanding of url design and RESTful principles, the work of Roy Fielding on Representational State Transfer (REST) is quite fundamental and can help you understand the nuances behind the design decisions in Rails regarding URL structures. Additionally, understanding the implementation details of `ActionDispatch::Routing` within the Rails source code itself (found on GitHub) will be highly beneficial for advanced configuration.

In conclusion, configuring Rails to use hyphenated URLs instead of underscored ones requires a dual approach: specifying hyphenated `param` names in your routes and normalizing the incoming parameters to use underscores in your controllers. Doing so, you will end up with an application with consistent, understandable urls, which will also increase usability and overall design consistency. This is not an overly complex configuration but a very common case for specific projects and requires a clear understanding of Rails routing and parameter processing.
