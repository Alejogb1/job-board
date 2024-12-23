---
title: "How can Rails handle non-persistent models and query strings?"
date: "2024-12-23"
id: "how-can-rails-handle-non-persistent-models-and-query-strings"
---

Okay, let's tackle this. Been there, done that a few times in my career, especially when building those complex dashboards that needed a lot of temporary data handling. It's interesting how Rails, a framework known for its strong emphasis on database-backed models, manages the ephemeral world of non-persistent data and query strings. It doesn't inherently *persist* such things, but it provides a robust set of tools to handle them effectively.

Essentially, when we talk about non-persistent models, we're referring to ruby objects that behave like active record models but are not backed by a database table. They exist only in memory, typically within the lifespan of a request or a user session. This is crucial for situations where you're working with temporary forms, data transformations, or external api results before you decide to persist any of that data. Query strings, of course, are the ubiquitous key-value pairs appended to urls, used to transmit parameters between client and server. Rails provides mechanisms to parse, access, and work with these parameters conveniently.

The key to handling non-persistent models is to avoid assuming that every model needs a database. Instead, we embrace plain ruby objects and perhaps, if they represent complex data structures, employ `ActiveModel::Model` or a `Virtus` inspired solution, which provides a foundation for mimicking active record behaviors, such as validations and form handling, without the database persistence piece. This gives you the benefits of rails form helpers and conventions, without forcing data into a database prematurely.

Consider this scenario, for instance: I was once building a filtering interface where users could search through a large dataset based on several optional criteria. These criteria were not always used, and storing every possible permutation in the database would have been insane. We didn't want to bog down the database and create a ton of garbage records. So, we implemented a `SearchCriteria` non-persistent model. Here's a simplified example:

```ruby
require 'active_model'

class SearchCriteria
  include ActiveModel::Model

  attr_accessor :keywords, :category, :min_price, :max_price

  validates :min_price, numericality: { only_integer: true, greater_than_or_equal_to: 0 }, allow_nil: true
  validates :max_price, numericality: { only_integer: true, greater_than_or_equal_to: :min_price }, allow_nil: true

  def initialize(attributes = {})
    attributes.each { |name, value| send("#{name}=", value) }
  end
end
```

In this example, `SearchCriteria` acts like a regular model, allowing us to use it in forms through Rails' form helpers, perform validations, and generally treat it like data that's tied to the request but without committing anything to a persistent datastore. The `attr_accessor` defines what attributes this object will accept and the `ActiveModel::Model` inclusion provides the necessary validations and form interaction functionality. The initialize method allows us to instantiate the object with the params from the query string or submitted form.

Now, let's move to the query string aspect. Rails parses query strings automatically and makes them accessible through the `params` hash in controllers. These parameters come in two primary forms: those attached to the url via GET requests and those included in POST, PUT, or PATCH requests. The handling is mostly transparent, allowing for easy access by their keys. For our `SearchCriteria`, you could imagine a controller like this:

```ruby
class SearchController < ApplicationController
  def index
    @search_criteria = SearchCriteria.new(params.permit(:keywords, :category, :min_price, :max_price))
    if @search_criteria.valid?
      @results = perform_search(@search_criteria) # A separate method for database querying
    else
      render :search_form # Render with errors
    end
  end
end
```

Here, we utilize `params.permit` to whitelist the allowed parameters, ensuring that we don't inadvertently expose our application to security issues by accepting un-sanitized user input. We then use that to instantiate a `SearchCriteria` object, which can be validated. The validation ensures the user supplied data are of the expected types and satisfies business requirements. If the parameters are valid, the query can be executed, if not the form is re-rendered with appropriate error messages. Note how the controller doesn't directly interact with the database based on the query parameters; instead, it uses the non-persistent `SearchCriteria` model to encapsulate the query logic. This improves code organization and testing capabilities.

Another common scenario where this approach is handy involves multiple steps within a form process. The intermediate data often doesn't need to persist until the final step. Consider a simplified signup flow where we temporarily hold user details before final confirmation. A simplified version would look like this using `session`:

```ruby
class UsersController < ApplicationController

  def new_step_one
    @user_details = session[:user_details] || UserDetails.new
  end

  def create_step_one
    @user_details = UserDetails.new(params.permit(:first_name, :last_name, :email))
    if @user_details.valid?
      session[:user_details] = @user_details
      redirect_to new_step_two_users_path
    else
      render :new_step_one
    end
  end

  def new_step_two
    @user_details = session[:user_details] || UserDetails.new
    if @user_details.first_name.blank?
      redirect_to new_step_one_users_path
      return
    end
  end

  def create_step_two
    @user_details = UserDetails.new(session[:user_details].attributes.merge(params.permit(:password, :password_confirmation)))
    if @user_details.valid?
      @user = User.create(@user_details.to_h)
      session.delete(:user_details)
      redirect_to dashboard_path, notice: "Account created!"
    else
      render :new_step_two
    end
  end

  private

  class UserDetails
     include ActiveModel::Model
     attr_accessor :first_name, :last_name, :email, :password, :password_confirmation
     validates :first_name, :last_name, :email, presence: true
     validates :email, format: { with: URI::MailTo::EMAIL_REGEXP }
     validates :password, confirmation: true, length: { minimum: 6 }, if: -> { password.present? }
  end
end
```

Here, a `UserDetails` model is created using the ActiveModel approach. It is then populated and validated. The validated attributes are stored in the `session` between requests. This prevents unnecessary saving in the database and avoids having to create a temporary table just to manage multi step flows.

For further study on these topics, I strongly recommend delving into the "Active Model" guide in the official rails documentation to understand non-persistent models in greater depth. For a broader understanding of working with query strings and request/response cycles, look into the "Rails Routing" guide and how the params hash is populated. Furthermore, the "Programming Ruby 3.2" by Dave Thomas et al. offers an excellent fundamental overview of the underpinnings of ruby objects and classes, as well as how to define your own objects. Finally, "Refactoring: Improving the Design of Existing Code" by Martin Fowler will be very useful for understanding how to structure the controller code and avoid code bloat while managing non-persistent data in rails applications. These resources are more than enough to get you acquainted with these concepts and techniques.

In summary, Rails' design allows for a graceful handling of both non-persistent models and query strings. We can treat query string parameters as input to these models, which gives us the ability to validate and transform the request before making calls to the database. This approach facilitates building more flexible and efficient web applications.
