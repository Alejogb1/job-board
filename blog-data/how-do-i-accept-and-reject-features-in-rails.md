---
title: "How do I accept and reject features in Rails?"
date: "2024-12-23"
id: "how-do-i-accept-and-reject-features-in-rails"
---

Alright, let's tackle feature flagging in Rails. It's a topic I’ve navigated many times, from early-stage startups pushing out rapid iterations to larger enterprises needing to control releases across diverse user segments. I've seen the good, the bad, and the downright ugly when it comes to managing features. So, rather than jump straight into a canned approach, let’s explore the landscape and then delve into some practical implementation details.

The core issue, as I’m sure you’re aware, boils down to this: how do you deploy new functionality without exposing it to everyone at once, and how do you roll back when things inevitably go sideways? It’s not merely about switching things on or off. It's about the granular control, the ability to target specific user groups, and the inherent need to monitor the impact of these changes. In my experience, haphazard feature implementation leads to chaotic deployments, frustrated users, and debugging nightmares. The solution isn't just about the tooling; it's about establishing a process and sticking to it.

When I started at Acme Corp some years ago, we had a rather… *interesting* way of enabling features. Think conditional blocks buried deep within the codebase, enabled by environment variables that were mostly guesswork and frequently forgotten. It was messy. Real messy. Imagine trying to debug an issue that only appeared for a small percentage of users who just happened to have a particular environment variable set in their testing account. Nightmare material. It's precisely this experience that made me a firm believer in the importance of a robust and well-architected feature flagging system.

Let's break down the process conceptually before diving into code. A feature flag essentially acts as a gatekeeper. We define flags (typically as strings or symbols) and associate them with specific pieces of functionality. Then, within our application logic, we query the current state of the flag, effectively deciding whether to execute the new code or to stick with the established path. This allows you to deploy new code into production and only enable the features for a controlled subset of users, or sometimes, for no users at all, initially.

Now, in Rails, we have a few approaches available to us. We could implement a basic system ourselves – which might be fine for very small projects or simple features – but generally, I'd recommend using a dedicated feature flagging gem. These gems provide a layer of abstraction, offering more powerful functionality like user targeting, gradual rollouts, and A/B testing support. Two gems that I've found particularly useful are `flipper` and `rollout`. For this explanation, let’s consider `flipper` as our primary example due to its straightforward implementation and clean api.

Here’s a basic illustration of how to use `flipper` in your rails application, showing a simple on/off scenario. We would first, after installation and configuration (refer to the official `flipper` documentation for set-up specifics), define our feature:

```ruby
# config/initializers/flipper.rb

require 'flipper'
require 'flipper/adapters/memory'

Flipper.configure do |config|
  config.default = Flipper::Adapters::Memory.new
end
```

Then, in our application code, we would check if the feature is enabled before executing the relevant code:

```ruby
# app/controllers/products_controller.rb
class ProductsController < ApplicationController

  def index
    if Flipper.enabled?(:new_product_listing)
      @products = Product.featured
    else
      @products = Product.all
    end
  end
end

```

This snippet demonstrates a very basic approach: if the `:new_product_listing` feature is enabled, we'll retrieve featured products; otherwise, we'll fetch all products. Note that by itself, this might be useful but isn't particularly flexible.

Here’s a slightly more sophisticated example, illustrating how we might enable a feature for a specific user:

```ruby
# app/controllers/application_controller.rb

class ApplicationController < ActionController::Base

  def current_user
    @current_user ||= User.find_by(id: session[:user_id])
  end
end

# app/controllers/products_controller.rb
class ProductsController < ApplicationController
  def index
    if Flipper.enabled?(:new_product_listing, current_user)
      @products = Product.featured
    else
      @products = Product.all
    end
  end
end

#Somewhere else - enable a feature for a user
Flipper[:new_product_listing].enable(User.find(1))

```

Here, the feature is checked against the `current_user`. `flipper` uses the provided object to determine if the feature should be enabled for this particular user. We are enabling the feature for the user with id 1. This approach allows for user-based targeting, useful for beta testers, or a percentage rollout based on some attributes of the current user.

Now, let's move onto another scenario that combines feature flags with active support concern:

```ruby
# app/concerns/feature_flagable.rb
module FeatureFlagable
  extend ActiveSupport::Concern

  def feature_enabled?(feature_name)
    Flipper.enabled?(feature_name, current_user)
  end

  def with_feature(feature_name, &block)
    if feature_enabled?(feature_name)
      yield
    end
  end
end

# app/controllers/application_controller.rb
class ApplicationController < ActionController::Base
  include FeatureFlagable

  def current_user
    @current_user ||= User.find_by(id: session[:user_id])
  end

end

# app/controllers/products_controller.rb
class ProductsController < ApplicationController
  def index
      with_feature(:new_product_listing) do
        @products = Product.featured
      end
      @products ||= Product.all #fallback if block not yielded
  end

    def show
       with_feature(:product_details_v2) do
         render 'show_v2'
       end
       render 'show' unless performed?
    end
end
```

In this example, we’ve created a `FeatureFlagable` concern, which provides convenience methods for checking features, and conditionally executing blocks based on the state of those features. Notice how `show` has an alternative rendering path dependant on if the feature is enabled. This approach keeps the controller code cleaner and less repetitive. Also notice the use of `performed?` to ensure only one render occurs. This is very important when considering how blocks of code will execute based on feature flag status.

Regarding resources for a more in-depth understanding of feature flagging, I would highly recommend reading "Continuous Delivery: Reliable Software Releases through Build, Test, and Deployment Automation" by Jez Humble and David Farley. While not solely focused on feature flags, it thoroughly discusses the importance of controlled releases and the practices that underpin feature management. You might also find "Feature Flagging: The Best Practices for Managing Feature Rollouts" by Pete Hodgson incredibly useful, it’s a great practical guide. Additionally, many online blogs from companies actively using feature flags (e.g., Airbnb, Netflix) provide excellent real-world insights. Search for case studies on engineering blogs of large tech companies; that’s where you will find the cutting edge practices that you can adapt to your own needs.

To conclude, feature flagging isn’t just about enabling or disabling features. It’s about embracing a workflow that prioritizes control, monitoring, and gradual rollout. This way you can mitigate risks and ensure a smoother experience for both your users and your development team. Choose tools that align with your requirements, establish clear naming conventions for your flags and always remember to clean up flags once they are no longer needed. Remember, messy feature flags are just as bad as messy code. Good luck!
