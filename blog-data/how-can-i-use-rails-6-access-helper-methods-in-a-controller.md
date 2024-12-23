---
title: "How can I use Rails 6 access helper methods in a controller?"
date: "2024-12-23"
id: "how-can-i-use-rails-6-access-helper-methods-in-a-controller"
---

Alright,  I've spent a fair bit of time in Rails land, and the interplay between controllers and view helpers, particularly those introduced in Rails 6, is something I've grappled with directly on several projects. Specifically, I recall needing to leverage those helpers for a complex internationalization feature, so this topic hits close to home. It's crucial to understand that Rails controllers aren't inherently designed to directly utilize view helper methods, and attempting to do so without proper care can lead to architectural issues and even subtle bugs. However, there *are* legitimate ways to achieve this functionality while maintaining a clean, well-structured application.

The core challenge lies in the separation of concerns. Controllers, traditionally, manage the flow of requests and application logic, while view helpers, as their name suggests, focus on formatting and presenting data in the view layer. Directly invoking view helpers from a controller would blur these lines and create tightly coupled components, making your codebase less maintainable and testable in the long run. Think of it this way: your controller should be the orchestrator, and view helpers are more akin to the instrumentalists, each with their specific role.

However, there are situations where you might need to leverage some functionality normally relegated to helpers within your controller logic. Perhaps you have a piece of utility logic, initially expressed in a helper, that you now need to execute as part of the controller’s actions. This is where we introduce the concept of a “service object” or some form of separation between the helper logic and the controller.

One approach, and the one I've personally found most effective, is to encapsulate the logic residing in your helper into a separate class or module, often located under the `/app/services/` directory. This class can then be instantiated and called from both your controller and view helpers, effectively sharing logic without violating the separation of concerns. Let's look at a basic example:

**Example 1: Encapsulating String Formatting Logic**

Assume you have a helper that capitalizes strings:

```ruby
# app/helpers/string_helper.rb
module StringHelper
  def capitalize_string(str)
    str.to_s.capitalize
  end
end
```

Instead of directly calling `capitalize_string` from our controller, we will move the core logic to a service object:

```ruby
# app/services/string_formatter.rb
class StringFormatter
  def self.capitalize(str)
    str.to_s.capitalize
  end
end
```

Now we can update our helper:

```ruby
# app/helpers/string_helper.rb
module StringHelper
  def capitalize_string(str)
    StringFormatter.capitalize(str)
  end
end
```

And in our controller, we call the same method of the service object:

```ruby
# app/controllers/my_controller.rb
class MyController < ApplicationController
  def create
    @formatted_name = StringFormatter.capitalize(params[:name])
    # ... rest of controller logic
  end
end
```

Notice how both the controller and the helper now utilize the `StringFormatter` service, keeping the logic centralized and easily accessible from both sides. This pattern is highly recommended and has served me well in numerous situations.

A second common scenario arises when you need to leverage URL generation within a controller. Often, view helpers like `link_to` or `polymorphic_url` are essential for creating correct URLs based on your route definitions. While you wouldn't use `link_to` directly in a controller, you might need to generate URLs for redirects or other purposes. In these cases, you can leverage the `rails_helpers` module in your service object as well. Let's look at an example:

**Example 2: Generating URLs**

Let’s say you need to generate a URL within your controller to redirect after a user creates a new record.

```ruby
# app/services/url_generator.rb
class UrlGenerator
  include Rails.application.routes.url_helpers

  def self.redirect_url(model)
    Rails.application.routes.url_for(action: 'show', id: model.id)
  end
end
```

Then in your controller you can make use of this:

```ruby
# app/controllers/users_controller.rb
class UsersController < ApplicationController
    def create
        @user = User.new(user_params)
        if @user.save
          redirect_to UrlGenerator.redirect_url(@user)
        else
          render :new
        end
    end

  private
  def user_params
    params.require(:user).permit(:name, :email)
  end

end
```

In this case, `Rails.application.routes.url_for` is a valid method in controller context so this approach works. However, including the `Rails.application.routes.url_helpers` module within the service object can be useful to use helper methods such as `user_path` or `edit_user_path` if needed.

Finally, the concept of `ApplicationController.helpers` allows you to access helper methods specifically from within an instance of your controller without the need to instantiate anything, however this is still not ideal for business logic, rather you should treat it as helper-like functionality that is needed.

**Example 3: `ApplicationController.helpers` Example (Use sparingly)**

Let's demonstrate a very simple use case where you need access to a form helper within your controller.
```ruby
# app/controllers/my_controller.rb
class MyController < ApplicationController
  def show
    formatted_name = helpers.capitalize_string("example")
    @name = formatted_name
  end
end
```

This demonstrates how you can access the `capitalize_string` method within your controller class, by using `helpers` directly. However, as stated before, avoid using this pattern extensively as this couples your controller to view helper logic and should be used when no better alternative exists.

Remember that the examples above are rather simplified to demonstrate the core idea. In real-world scenarios, the service objects could be more complex, incorporating error handling, logging, and other necessary functionalities. Also, take some time to investigate the design pattern known as "Presenter" or "Decorator" patterns. These patterns, while being more suitable for the view layer, may influence how you structure your logic and could be used as an alternative to helpers within specific use cases.

For a deeper understanding of these concepts, I would suggest looking at *“Patterns of Enterprise Application Architecture”* by Martin Fowler, to gain a solid grasp of architecture in complex applications. Understanding the principles laid out in that book will help you make informed decisions regarding the layering of your application logic. In the specific context of Rails, *“Crafting Rails Applications”* by Jose Valim provides excellent insight into best practices for building maintainable and scalable applications. Additionally, explore the official Ruby on Rails guides, which are an invaluable resource for learning about the framework's capabilities and recommendations, specifically on the topic of `ActiveSupport::Concern` and module inclusion strategies. By combining these foundational resources, you’ll be well-equipped to tackle real-world scenarios when you need to leverage Rails 6 access helpers in your controller. Remember that the key is to maintain clear boundaries and ensure that your controllers are primarily orchestrators of requests, not direct manipulators of view-specific logic.
