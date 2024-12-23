---
title: "How can I apply or adapt filters from one ActiveAdmin model to another?"
date: "2024-12-23"
id: "how-can-i-apply-or-adapt-filters-from-one-activeadmin-model-to-another"
---

Okay, let's tackle this. I've bumped into this exact scenario more times than i care to remember, usually when dealing with interconnected data structures across different administrative interfaces. The need to reuse filter logic from one ActiveAdmin model to another is surprisingly common, especially as projects grow and the relationships between your models become more complex. We're not just talking about simple copy-pasting here; we're aiming for a maintainable and efficient solution.

The core problem stems from ActiveAdmin's model-centric nature. Each `ActiveAdmin.register` block tends to be quite self-contained, creating a barrier to easy filter reuse. However, there are several effective patterns we can employ to overcome this. What’s key here is avoiding repetitive code and enforcing a single source of truth for our filtering logic.

First, a common approach, and the one I often find myself gravitating towards, is to define reusable filter configurations using helper methods. This keeps our admin dashboard clean and consistent. Let's assume we have two models, `Order` and `LineItem`, and we want to filter both by similar customer attributes (like first_name, last_name, etc., which are stored on the related `Customer` model). Instead of writing out the same `filter` calls in both `Order` and `LineItem` ActiveAdmin configurations, we abstract this logic into a helper.

Here's how I’d set it up. Inside `app/admin/helpers/filter_helper.rb`:

```ruby
module FilterHelper
  def customer_filters
    [
      :customer_first_name,
      :customer_last_name,
      :customer_email,
      { label: 'Customer ID', attribute: :customer_id, as: :numeric },
    ]
  end
end

```

Then, we include this helper in our ActiveAdmin configuration files. Here's the `Order` registration:

```ruby
# app/admin/orders.rb
ActiveAdmin.register Order do
  include FilterHelper

  filter *customer_filters
  # Other order configurations...
end

```

And here's the `LineItem` registration:

```ruby
# app/admin/line_items.rb
ActiveAdmin.register LineItem do
  include FilterHelper

  filter :order_id, as: :numeric
  filter *customer_filters # Reusing customer filters
  # Other line_item configurations...
end
```

This method encapsulates the customer filter definitions, making them readily reusable. The asterisk (`*`) in `filter *customer_filters` is crucial here – it spreads the array of symbols and hashes into individual arguments for the `filter` method. This keeps our code DRY and reduces the chance of inconsistencies.

Another strategy, particularly helpful when dealing with more complex filter logic or when you need to conditionally apply filters, involves using class methods within our models or custom classes that encapsulate filter specifications. For example, if you needed to base filters on the order’s current status or some custom logic that involves multiple attributes on multiple related models, a model-level method is a more robust approach than the helper method above.

Let's say that for our `Order` model, we need to provide filters that can be used based on their shipping status. Here's a demonstration of what this approach would look like. First we define a class method in our `Order` model that returns the available filters:

```ruby
# app/models/order.rb
class Order < ApplicationRecord
  belongs_to :customer
  has_many :line_items

  def self.shipping_status_filters
    [
      { attribute: :shipping_status, as: :select, collection: ['Pending', 'Shipped', 'Delivered'] },
      { attribute: :created_at, as: :date_range },
    ]
  end
end

```

In your `ActiveAdmin` configuration, you then simply call this class method and apply the filters:

```ruby
# app/admin/orders.rb
ActiveAdmin.register Order do
    filter *Order.shipping_status_filters
    # other configurations
end
```
This way, we can move more complex logic into the model, allowing for conditional filter application based on the model itself. This makes your filters not just reusable, but more adaptable and closer to the data where they belong, which aids in maintainability and testability.

Finally, consider creating a specialized class to encapsulate filter specifications if your filtering logic becomes very complex or involves multiple distinct sets of filters. This offers a high level of control and can improve the clarity of your code. Imagine we need multiple sets of filters that handle order date filtering, and also customer contact information, while the former could be useful to the `Order` model, the latter could also be useful to the `Customer` model as well. Let’s set up a class that takes care of this for us:

```ruby
# app/admin/filters/admin_filters.rb
class AdminFilters

  def self.order_date_filters
    [
        { attribute: :created_at, as: :date_range, label: "Order Date" },
        { attribute: :updated_at, as: :date_range, label: "Order Update Date" }
    ]
  end

  def self.customer_contact_filters
      [
          :customer_phone,
          :customer_email
      ]
    end
end
```
Now, in our `ActiveAdmin` resource registration, we can apply the needed filters as such:

```ruby
# app/admin/orders.rb
ActiveAdmin.register Order do
    filter *AdminFilters.order_date_filters
    filter *AdminFilters.customer_contact_filters
   # other configurations
end
```
```ruby
# app/admin/customers.rb
ActiveAdmin.register Customer do
    filter *AdminFilters.customer_contact_filters
   # other configurations
end
```
This pattern offers a robust, object-oriented way to define and manage filters for multiple ActiveAdmin resources.

For further exploration, i recommend looking into the "Rails AntiPatterns" book by Chad Pytel, which provides general software engineering practices that align with good design patterns. To dig deeper into ActiveAdmin’s internal workings and filter implementation you can review the [ActiveAdmin’s official documentation](https://activeadmin.info/), but for general programming design patterns the "Design Patterns: Elements of Reusable Object-Oriented Software" book by Erich Gamma et al. provides great foundations that are often forgotten these days.

In conclusion, the key to reusing filters across ActiveAdmin models lies in abstracting that logic into reusable components. Helper methods, class methods on your models, or specialized classes can all serve as effective means of achieving this, each offering different strengths depending on the complexity and needs of your project. Remember that consistent structure and a single source of truth for filter logic greatly simplifies maintenance and improves overall code quality.
