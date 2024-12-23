---
title: "How can I add a custom sortable column to Active Admin?"
date: "2024-12-23"
id: "how-can-i-add-a-custom-sortable-column-to-active-admin"
---

Alright, let's unpack this. Custom sortable columns in Active Admin. I’ve certainly spent some time on this, especially back when we were migrating a legacy e-commerce platform. We had this incredibly intricate order management system where sorting by calculated shipping costs or some equally convoluted customer segmentation metric was crucial. The default sort behaviors just didn’t cut it; we needed something more. It’s a common scenario. Let's walk through how you can approach this, and I’ll throw in some code examples to make it concrete.

The core challenge with custom sortable columns lies in understanding that Active Admin, by default, relies on the underlying database model's columns for its sorting capabilities. When you introduce calculated values or values derived from associations that aren't directly stored, you have to intervene and provide the sorting logic yourself. The key here is using `ransack`, which Active Admin uses behind the scenes for searching and filtering. Essentially, we’ll extend `ransack` to understand how to sort on our custom field.

Let's start with a simplified example. Imagine a `Product` model with a method, `average_rating`, which calculates the average rating based on associated reviews. We want to sort the product list in our admin panel by this average rating.

Here's how we'd approach it using Active Admin and Ransack:

```ruby
# app/admin/products.rb
ActiveAdmin.register Product do
  index do
    selectable_column
    id_column
    column :name
    column :average_rating do |product|
      product.average_rating
    end
    actions
  end


  # Configure sorting for the average rating column
  config.sort_order = 'average_rating_desc'  # Default sorting
  controller do
    def scoped_collection
        super.includes(reviews: :rating) # Preload reviews to avoid N+1 queries
    end
    def apply_sorting(chain)
      if params[:order] == 'average_rating_asc'
         chain.sort_by { |product| product.average_rating }
      elsif params[:order] == 'average_rating_desc'
        chain.sort_by { |product| product.average_rating }.reverse
      else
         super # Use default ransack sort if not the average rating
      end
   end
  end
end
```

In this first example, we’ve done a couple of key things. First, we've displayed the average rating in the index view via a block, and importantly, we’ve also defined `scoped_collection` to eager load the `reviews` association, which prevents N+1 issues. Then, within the controller block, we implement `apply_sorting`, which checks if the incoming sort parameter is related to our `average_rating`. If it is, we sort the collection in-memory using Ruby's `sort_by` method. Notice the usage of `includes` to ensure we avoid N+1 queries, which are a real performance headache in this context. It's important to understand that this method of sorting sorts *after* the database query, and thus may be unsuitable for very large tables. This example works well, however, when your collection is relatively small and you need to sort on calculated properties.

Let’s move onto a scenario where we want to sort on a calculated field that is more complex and involves multiple associations. Say, for instance, you have a ‘Customer’ model, and you want to sort customers based on their ‘total_spent’. This requires querying and summing across associated orders. We could take the following approach:

```ruby
# app/admin/customers.rb
ActiveAdmin.register Customer do
  index do
    selectable_column
    id_column
    column :name
    column :email
    column :total_spent do |customer|
      customer.total_spent
    end
    actions
  end


  config.sort_order = 'total_spent_desc'

  controller do
     def scoped_collection
         super.includes(orders: :order_items)
     end
    def apply_sorting(chain)
      if params[:order] == 'total_spent_asc'
          chain.sort_by { |customer| customer.total_spent }
      elsif params[:order] == 'total_spent_desc'
          chain.sort_by { |customer| customer.total_spent }.reverse
      else
          super
      end
    end

  end
  # Define the total_spent method within the Customer model
  # (This could also be a database view for more performance)
  Customer.class_eval do
    def total_spent
       orders.joins(:order_items).sum("order_items.price * order_items.quantity")
    end
  end
end
```

Here, the approach is fundamentally the same as the first example – we calculate our sort value in a model method and sort the returned collection using Ruby's `sort_by`, after eagerly loading all of the associated models to avoid N+1 queries. This is more computationally expensive, since it requires us to iterate through every customer record after querying the database.

Now, if you find yourself in a situation where sorting in memory after the query is proving to be a performance bottleneck (as it can be on large datasets), you'll need to lean into the database for the sorting. For that, you need to extend `Ransack`. Let’s consider the earlier example with customers, but this time use a database-backed method for sorting:

```ruby
# app/admin/customers.rb
ActiveAdmin.register Customer do
  index do
    selectable_column
    id_column
    column :name
    column :email
     column :total_spent do |customer|
      customer.total_spent_db
    end
    actions
  end

  config.sort_order = 'total_spent_db_desc'

  controller do
      def scoped_collection
          super.includes(:orders)
      end

  end

  Customer.class_eval do
    def self.ransackable_attributes(auth_object = nil)
        super + %w[total_spent_db]
    end
     def self.ransackable_scopes(auth_object = nil)
       super
     end
     def self.ransacker :total_spent_db do
      Arel.sql("(SELECT SUM(order_items.price * order_items.quantity) FROM orders INNER JOIN order_items ON orders.id = order_items.order_id WHERE orders.customer_id = customers.id)")
    end
    def total_spent_db
        # Returns the total spent using the method defined in the ransacker, which makes use of the database
        # This could be any implementation which is equivalent to the ransacker, but is not required
        Customer.connection.select_value(Customer.ransackers[:total_spent_db].sql)
    end
   end
end

```

This is perhaps the most complex example. We define a `total_spent_db` method, and then we define `ransackable_attributes` to tell ransack it's ok to filter on this column. Importantly, we have also used `ransacker :total_spent_db` to define the actual SQL to perform the calculation of total spending. This is the real workhorse here. It ensures that when sorting is triggered using the `total_spent_db` column, the database executes this SQL, performing the aggregation at the database level, which is far more efficient for larger datasets. The `total_spent_db` method which is used for display in the `index` view is only called *after* the records have been retrieved using the SQL defined in the `ransacker`.

For further reading on Active Admin and its intricacies, I highly recommend delving into the Active Admin documentation itself, it's incredibly comprehensive. Additionally, for a deeper dive into Ransack and its querying capabilities, consulting the Ransack gem's documentation will be invaluable. Lastly, understanding Arel (Active Record’s SQL abstraction layer) as used in the ransacker example can be improved with a deeper understanding of database concepts in general. Reading through books like "SQL for Dummies," or more technical materials that describe SQL fundamentals are essential.

This should give you a solid foundation for implementing custom sortable columns in your Active Admin interface. The trick is choosing the right method of implementation for your specific use case: in-memory sorting is simpler for small datasets, but leveraging the database, as shown in our last example, is essential when you start dealing with larger amounts of data. Let me know if you have any other specific scenarios, and I will gladly elaborate.
