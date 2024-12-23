---
title: "How can a form field be captured in a Rails model from a product view?"
date: "2024-12-23"
id: "how-can-a-form-field-be-captured-in-a-rails-model-from-a-product-view"
---

Alright, let's tackle this one. It's a common scenario, and I’ve certainly navigated this particular terrain more times than I care to count – often in projects where the data model wasn't quite as straightforward as we'd hoped. The essence of your question hinges on the interaction between your user interface (the product view, presumably rendered via erb or similar) and your application’s data layer (the rails model). Capturing form data involves a multi-step process, and I'll break it down with examples that should clarify each piece of the puzzle.

Fundamentally, the process starts with the form itself in your product view, typically using rails form helpers, then it progresses to the controller, where data is received and processed, and finally lands within your model. It’s important to understand that these layers shouldn't directly interact without mediation.

Here's the framework as I generally approach it:

1.  **The View: Creating the Form:**
    Your view contains the form that users interact with. This is the front line for gathering user inputs. Think of it as the initial data capture point. You'll use the rails form helpers to define input fields and associate them with model attributes or parameters, but there are choices here that can dramatically impact things down the line. I've seen projects where people went with raw html form elements, and while functional, the long term maintenance usually introduces more complexities. You generally want to stick with the Rails built in helpers for a variety of reasons, but primarily security and consistency.

2.  **The Controller: Handling the Submission:**
    The controller acts as the intermediary. Upon form submission, it receives the parameters sent by the browser, then filters and sanitizes them before handing them off to the model. Proper parameter whitelisting here is crucial for security – allowing only expected attributes to be set on the model prevents malicious users from manipulating the model in ways not intended. I can not emphasize that enough: security breaches usually come from weak parameter handling.

3.  **The Model: Saving the Data:**
    Finally, the model, the bedrock of your data, receives the data from the controller and persists it to the database. This involves attribute assignment based on the whitelisted parameters and initiating any relevant validations before saving the data.

Let's illustrate this with some code examples. Let’s say we’re building a product catalog, and we want to allow users to add reviews to products.

**Example 1: Basic Product Review Capture**

Assume we have a `Product` model and we are adding the ability to create a `Review` model associated with the product. Here's how our product view might look:

```erb
# app/views/products/show.html.erb
<h1><%= @product.name %></h1>
<p><%= @product.description %></p>

<h2>Add a Review</h2>
<%= form_with(model: @review, url: product_reviews_path(@product), method: :post) do |form| %>
  <div>
    <%= form.label :rating, 'Rating:' %>
    <%= form.select :rating, (1..5).to_a %>
  </div>
  <div>
    <%= form.label :comment, 'Comment:' %>
    <%= form.text_area :comment %>
  </div>
  <div>
    <%= form.submit "Submit Review" %>
  </div>
<% end %>
```

Key things to notice here:
  *   We're using `form_with`. This is the new standard and encourages best practices regarding security.
  *   `model: @review` indicates the form is associated with a `Review` object.
  *   `url: product_reviews_path(@product)` routes the form submission to a specific action in the `ReviewsController`.
  *   The form fields, like `:rating` and `:comment`, correspond to attributes in the `Review` model.

And here's our controller action:

```ruby
# app/controllers/reviews_controller.rb
class ReviewsController < ApplicationController
  before_action :set_product, only: [:create]

  def create
    @review = @product.reviews.build(review_params) # builds a review in memory that will automatically get the product_id assigned.
    if @review.save
      redirect_to @product, notice: 'Review was successfully created.'
    else
      # Handle validation errors – for simplicity, we re-render the product page with the errors.
      @product.reload #Reload the product in case of changes.
      render 'products/show', status: :unprocessable_entity
    end
  end

  private

  def set_product
    @product = Product.find(params[:product_id])
  end

  def review_params
    params.require(:review).permit(:rating, :comment) # strong params is key here.
  end
end
```

Here, the `create` action handles the form submission. The `review_params` method ensures only the permitted attributes are processed. The controller then associates the review with the given product and saves it to the database. This is important because you almost never want a user to control the product id of a new review ( or any other nested resource association.)

**Example 2: Handling Nested Attributes**

Let's expand a little more and assume that the `Product` model has a field `manufacturer_details` that takes a json representation of the manufacturers details. We need to update that field from the product edit page:

```erb
# app/views/products/edit.html.erb
<h2>Edit Product</h2>

<%= form_with(model: @product, url: product_path(@product), method: :patch) do |form| %>
  <div>
    <%= form.label :name, 'Name:' %>
    <%= form.text_field :name %>
  </div>
    <div>
      <%= form.label :description, 'Description:' %>
      <%= form.text_area :description %>
  </div>
  <div>
    <%= form.label :manufacturer_details, 'Manufacturer Details:' %>
    <%= form.text_area :manufacturer_details %>
  </div>
  <div>
    <%= form.submit "Update Product" %>
  </div>
<% end %>
```

The corresponding controller will look like:

```ruby
# app/controllers/products_controller.rb
class ProductsController < ApplicationController
   before_action :set_product, only: [:edit, :update]


  def edit
     # @product is set via before_action
  end

  def update
    if @product.update(product_params)
      redirect_to @product, notice: 'Product was successfully updated.'
    else
      render :edit, status: :unprocessable_entity
    end
  end

  private

  def set_product
    @product = Product.find(params[:id])
  end

  def product_params
    params.require(:product).permit(:name, :description, :manufacturer_details) # this will serialize into the jsonb field if needed
  end
end
```

Here, the `product_params` permit function will allow us to update the `manufacturer_details` field with data we provide in the form as long as it is a serialized type in the database such as json, or jsonb, or as a string.

**Example 3: Form Validations and User Experience**

Let’s introduce a few validations into the review model. If for example the review comment was required and needed a minimum length, then our model validation would look like this:

```ruby
# app/models/review.rb
class Review < ApplicationRecord
    belongs_to :product
    validates :comment, presence: true, length: { minimum: 10 }
    validates :rating, presence: true, inclusion: { in: 1..5 }
end
```

In the controller example above, if the validation fails, `if @review.save`  will return false, and `render 'products/show', status: :unprocessable_entity` will be executed. The key here is that we are re-rendering the show page for the product, but passing the `@review` variable from the controller. This allows us to show errors back to the user using the form helpers:

```erb
# app/views/products/show.html.erb

<h1><%= @product.name %></h1>
<p><%= @product.description %></p>

<h2>Add a Review</h2>

<%= form_with(model: @review, url: product_reviews_path(@product), method: :post) do |form| %>

  <% if @review.errors.any? %>
    <div id="error_explanation">
      <h2><%= pluralize(@review.errors.count, "error") %> prohibited this review from being saved:</h2>

      <ul>
        <% @review.errors.full_messages.each do |message| %>
          <li><%= message %></li>
        <% end %>
      </ul>
    </div>
  <% end %>

  <div>
    <%= form.label :rating, 'Rating:' %>
    <%= form.select :rating, (1..5).to_a %>
  </div>
  <div>
    <%= form.label :comment, 'Comment:' %>
    <%= form.text_area :comment %>
  </div>
  <div>
    <%= form.submit "Submit Review" %>
  </div>
<% end %>

```

Here, the errors for the review will be displayed above the form on each re-render.

**Relevant Resources**

For a solid foundation in rails forms and model handling, I'd recommend "Agile Web Development with Rails" by Sam Ruby et al., it provides a comprehensive look at the framework’s inner workings. For more on database handling, including topics such as storing json and jsonb, consider reading “PostgreSQL Documentation on JSON and JSONB types” it has all you need to understand how rails treats and stores these special data types. Lastly, for general design patterns in rails, I would recommend “Practical Object-Oriented Design in Ruby” by Sandi Metz which should help you understand how to design your models.

In closing, capturing form data is a layered process in rails. By understanding the flow from view to controller to model, and handling each piece correctly, you can build robust and secure applications. Focus on form helpers, strong parameters, and meaningful validations and you'll be on the right track.
