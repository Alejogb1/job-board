---
title: "How can I detect ActiveRecord changes for either creation or updates?"
date: "2024-12-23"
id: "how-can-i-detect-activerecord-changes-for-either-creation-or-updates"
---

,  From my experience building a fairly complex inventory management system several years back, tracking model changes, especially for auditing purposes, was crucial. The core issue, as I recall, was knowing precisely when an ActiveRecord record was either created or updated, and specifically what changed. ActiveRecord provides a decent set of tools, but getting the granularity you often need requires a slightly deeper understanding of its lifecycle.

The fundamental approach revolves around leveraging ActiveRecord’s callbacks. These are essentially hooks that ActiveRecord executes at various points in a record's lifecycle, like before validation, after saving, and so on. For our purposes, `after_create` and `after_update` are prime candidates for detecting when records are either created or updated, respectively. However, the challenge isn’t just about *knowing* something happened; it's often about *knowing what* happened. That’s where ActiveRecord’s built-in `changes` method becomes incredibly useful.

Let's consider the detection of changes during an update. When an update is performed on a model instance, ActiveRecord maintains a hash called `changes` that stores the original value of an attribute and the new value being assigned, *before* the changes are actually committed to the database. Crucially, this `changes` hash is only available *after* the model has been through the update process. It is populated in memory but doesn't persist unless you programmatically log it somewhere, such as in an auditing table.

Here’s a conceptual example to illustrate: Suppose you have a `Product` model with attributes like `name`, `price`, and `quantity`. Let's assume we need to log every update to a `ProductChange` model:

```ruby
class Product < ApplicationRecord
  has_many :product_changes

  after_update :log_changes

  def log_changes
    if saved_change_to_name?
       product_changes.create(
         change_type: 'name_change',
         from: changes[:name].first,
         to: changes[:name].last
       )
    end
    if saved_change_to_price?
       product_changes.create(
         change_type: 'price_change',
         from: changes[:price].first,
         to: changes[:price].last
       )
    end
    if saved_change_to_quantity?
       product_changes.create(
         change_type: 'quantity_change',
         from: changes[:quantity].first,
         to: changes[:quantity].last
       )
    end
  end
end

class ProductChange < ApplicationRecord
  belongs_to :product
  attribute :change_type, :string
  attribute :from, :string
  attribute :to, :string
end
```
In this example, the `log_changes` method leverages `saved_change_to_*?` helper methods to conditionally create entries in the `ProductChange` table. We access the old value and the new value using the `changes` hash where the key is the attribute name. The first element of the value array is the old value and the last is the new value.

The `saved_change_to_*?` predicate methods are powerful because they check not just if the attribute *was modified* but also specifically if it *was actually saved* to the database as a result of the update operation. If, for instance, a user inputs a new price value, but it's not distinct from the existing price, then the `saved_change_to_price?` would return false and the logging action for price would not occur.

Now, let’s shift focus to detecting creation. When a new ActiveRecord record is created, the `changes` hash will be empty before the creation. However, by the time `after_create` callback runs the `previous_changes` hash will contain all attribute's values.

Here is a code example:
```ruby
class Product < ApplicationRecord
  has_many :product_changes

  after_create :log_creation

  def log_creation
    previous_changes.each do |attribute, values|
        product_changes.create(
            change_type: "creation_#{attribute}",
            to: values.last,
        )
      end
    end
end
```
In this `log_creation` method, we iterate over each key-value pair in `previous_changes` and create a log entry with details regarding the created attributes. This will allow for auditing of all initial values created during an insert to the database.

Now, a critical note. While `after_create` and `after_update` work well for basic logging, there can be concurrency issues, especially if other operations are happening on your database concurrently. Consider using asynchronous processing (like ActiveJob) for the logging, or move the processing logic to a dedicated service, so you don't inadvertently slow down your user facing request. We can see an example of ActiveJob using the after_commit callback:

```ruby
class Product < ApplicationRecord
  has_many :product_changes

  after_commit :enqueue_audit_job, on: [:create, :update]

  private

  def enqueue_audit_job
    ProductAuditJob.perform_later(id, saved_changes, created_at)
  end

end

class ProductAuditJob < ApplicationJob
  queue_as :default

  def perform(product_id, changes, created_at)
    product = Product.find(product_id)
      if product
        changes.each do |key, value_array|
          if value_array.present?
              if key == "created_at"
                 product.product_changes.create!(change_type: "creation_time", to: created_at)
              elsif value_array.length == 1
                 product.product_changes.create!(change_type: "creation_#{key}", to: value_array.last)
              else
                 product.product_changes.create!(change_type: "#{key}_change", from: value_array.first, to: value_array.last)
               end
          end
       end
   end
  end
end
```
In this implementation we've moved our logging to a background job, preventing any additional latencies during regular request operations. `after_commit` is used here, as it will trigger only after the database transaction completes; hence why it's necessary to pass in the `saved_changes` and `created_at` attribute to the job as the model will not be reloaded for changes. Additionally, we must handle both creation and updates.

For further reading, I'd suggest looking into Martin Fowler's work on *Patterns of Enterprise Application Architecture* for insights on event-driven architectures, which can be beneficial when dealing with a more complex event systems. Also, *Rails AntiPatterns* by Chad Pytel and Tammer Saleh is a great resource for understanding potential pitfalls and best practices with ActiveRecord. You can deepen your knowledge of ActiveRecord internals in *Agile Web Development with Rails 7* by Sam Ruby, David Bryant Copeland, and Dave Thomas; this work dives deep into the ActiveRecord life-cycle, including callbacks and the `changes` method. These resources, combined with practical experimentation and a solid grasp of the fundamentals, should equip you to effectively detect and handle ActiveRecord changes in your applications.
