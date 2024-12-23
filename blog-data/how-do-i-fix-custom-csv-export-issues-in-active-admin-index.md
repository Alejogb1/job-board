---
title: "How do I fix custom CSV export issues in Active Admin Index?"
date: "2024-12-23"
id: "how-do-i-fix-custom-csv-export-issues-in-active-admin-index"
---

Okay, so, let's talk about custom csv exports in Active Admin. I've definitely been down that rabbit hole a few times, especially back when I was working on that large e-commerce platform. The default csv export is, well, *default*, and often needs some serious tweaking to handle real-world data complexities. We’re not just dealing with simple strings and numbers, are we? We’ve got related models, calculated attributes, complex data formats… it all gets interesting.

The core issue is that Active Admin’s built-in CSV generation is geared toward a simple table export. It assumes a one-to-one mapping between database columns and CSV columns. When you need more control – like exporting related model data, formatting timestamps, or creating custom columns – you need to dive into custom configurations.

My typical workflow for this begins by overriding the `csv_builder` configuration. The key here isn't just adding columns, but also understanding the context and how to efficiently pull data. The worst approach is to do a ton of individual queries within the csv builder itself. That will *absolutely* kill performance on anything but a small dataset. Instead, it's better to preload associations if you need data from related models. Let's look at an example. Imagine an `Order` model that has an associated `Customer` and several `OrderItems`.

Here's a first step, showing how to use the `csv_builder` to include basic data, and also how to preload:

```ruby
ActiveAdmin.register Order do
  index do
    #... other index configurations
    actions
    csv do
      # Preload customers to avoid N+1 queries
      preload_associations = [:customer, :order_items]

      column :id
      column(:customer_name) { |order| order.customer.name if order.customer }
      column(:total_amount) { |order| order.total }
      column :created_at
       # you can also include data from your order items
       column(:item_names) { |order| order.order_items.map(&:name).join(', ') }


      before_build_csv do |csv|
        # This will ensure our associations are loaded properly
        resource_collection.includes(preload_associations).to_a if resource_collection.respond_to?(:includes)
      end
    end
  end
  #... other configurations
end
```

Notice the `before_build_csv` block. This is critical. If you are working with a large dataset, and you need to include data from associated models, preload them by using `.includes()` *before* the csv builder starts iterating. Without this, the csv builder will call each of your `order.customer` or `order.order_items` methods hundreds or thousands of times depending on your data set which will generate huge performance issues. This code snippet demonstrates how to pre-load associations and how to access the related data for your output. This addresses the basic requirement of accessing related data without hammering your database with multiple queries.

But we're not always dealing with straightforward data formats, are we? Sometimes, you need to transform data into more human-readable formats or perform calculations. Let’s assume our `Order` model has a `status` field that's an integer and maps to a status enum, and we want it displayed as a string in our csv output. Let’s add that to our code above:

```ruby
ActiveAdmin.register Order do
  index do
    #... other index configurations
    actions
    csv do
      # Preload customers to avoid N+1 queries
      preload_associations = [:customer, :order_items]

      column :id
      column(:customer_name) { |order| order.customer.name if order.customer }
      column(:total_amount) { |order| order.total }
      column :created_at
      column(:item_names) { |order| order.order_items.map(&:name).join(', ') }
      column(:status) do |order|
        case order.status
        when 0 then "Pending"
        when 1 then "Processing"
        when 2 then "Shipped"
        when 3 then "Delivered"
        else "Unknown"
        end
      end

      before_build_csv do |csv|
        # This will ensure our associations are loaded properly
        resource_collection.includes(preload_associations).to_a if resource_collection.respond_to?(:includes)
      end
    end
  end
  #... other configurations
end
```

Here, the `column(:status)` block uses a case statement to translate the integer status into a human-readable string. This shows how you can format output directly within the `csv` block. It’s important to keep these transformation simple; if you have complex logic, consider putting that logic into a method in your model or creating a presenter object to abstract that complexity away.

Finally, sometimes you need something even more custom, like a column with aggregated values, or data that needs significant processing. In that situation, you can define a method on the model itself. Imagine each order has a `total_weight` which is calculated from the weight of each of its order items. We would want to compute that value in our model and then present that as an additional column in our csv.
Let's update our code one more time, also assume that we want to sort the CSV output by order id in ascending order:

```ruby
ActiveAdmin.register Order do
  index do
    #... other index configurations
    actions
    csv do
      # Preload customers to avoid N+1 queries
      preload_associations = [:customer, :order_items]

      column :id
      column(:customer_name) { |order| order.customer.name if order.customer }
      column(:total_amount) { |order| order.total }
      column :created_at
      column(:item_names) { |order| order.order_items.map(&:name).join(', ') }
      column(:status) do |order|
        case order.status
        when 0 then "Pending"
        when 1 then "Processing"
        when 2 then "Shipped"
        when 3 then "Delivered"
        else "Unknown"
        end
      end
      column(:total_weight) { |order| order.total_weight } # Assuming this is now a method on the Order model

      before_build_csv do |csv|
        # This will ensure our associations are loaded properly
        resource_collection.includes(preload_associations).order(id: :asc).to_a if resource_collection.respond_to?(:includes)
      end
    end
  end

  #... other configurations
end

class Order < ApplicationRecord
  #... other model definitions
  has_many :order_items
  def total_weight
   order_items.sum(&:weight)
  end
end

```

In this last example, I've shown the inclusion of the total weight column, which calls a method in the `Order` model to sum item weights. This demonstrates that sometimes, it's best to offload complicated calculation to your models. I also changed the `before_build_csv` method to now also sort by id. In many of the instances I worked on in the past, sorting by id is often a desired feature.

A few more points to keep in mind:

*   **Error Handling:** The code snippets I’ve shown are fairly basic for clarity. In your real application, you’ll want to add some error handling—especially when dealing with associations. Check for nil values before attempting to call methods on related objects; otherwise your export might throw errors and crash.
*   **Large Datasets:** Preloading associations is critical, but also consider pagination and background processing if you are dealing with a large number of records for exporting. Exporting thousands or millions of rows will take some time and it’s best to do this in the background using something like Sidekiq or Resque.
*   **Testing:** Make sure to write integration tests around your csv generation to verify the correct data is being exported. You would not want your client to discover a bug in your CSV output.

For further exploration, I highly recommend reading "Agile Web Development with Rails 7" by Sam Ruby, David Thomas, and David Heinemeier Hansson. While not directly about Active Admin, it provides a deep understanding of the underlying Rails framework and good practices. Also, the Active Admin documentation is quite comprehensive; pay close attention to the `csv` configuration section. In addition, you should also review the Ruby on Rails guides, specifically, the section on active record query interface because preloading is a key part of creating effective CSV exporters.

Ultimately, customizing CSV exports is all about controlling the data pipeline, ensuring you fetch what you need efficiently, transform it into the proper format, and avoid database bottlenecks.
