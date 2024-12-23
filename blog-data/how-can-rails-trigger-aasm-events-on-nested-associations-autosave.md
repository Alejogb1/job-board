---
title: "How can Rails trigger AASM events on nested associations' autosave?"
date: "2024-12-23"
id: "how-can-rails-trigger-aasm-events-on-nested-associations-autosave"
---

Alright,  It's a fairly nuanced challenge you’ve stumbled upon, and it's something I recall spending a fair bit of time on back when I was building a fairly complex order management system. We had a situation where order line items needed to trigger events in the order itself based on their status changes, and the autosave functionality of rails with nested attributes made it... interesting.

The crux of the problem lies in the fact that while rails happily saves the nested associations through `accepts_nested_attributes_for`, it doesn’t intrinsically trigger any lifecycle callbacks or custom events on the parent based on changes within those nested attributes *during the autosave process*. This means we need to manually orchestrate event triggers based on changes to the nested models. Let's break down a reasonable strategy and then look at some implementation options.

The challenge, essentially, is that rails' automatic persistence mechanism acts on the nested model directly, bypassing any defined event mechanisms unless explicitly invoked. When you modify an `order_item` and rails autosaves it along with its parent `order`, the automatic saving doesn't go through the `order`'s defined event mechanism managed by aasm (or any other similar state machine implementation). We need to ensure that the parent object is *aware* of significant changes in its children and can act on them.

Here are a few techniques that I've found to be effective, each with trade-offs:

**1. Using a Callback on the Nested Model with Manual Event Trigger**

One of the first solutions i explored involved adding a `after_save` (or `after_update`) callback on the nested model itself. In this callback, we would then explicitly examine changes to its attributes and call the appropriate event on its parent. This is direct and relatively easy to implement, but it places logic related to the parent on the nested model, which, generally, isn't an ideal separation of concerns.

Here's a snippet to illustrate this technique:

```ruby
# app/models/order_item.rb
class OrderItem < ApplicationRecord
  include AASM
  belongs_to :order, inverse_of: :order_items

  aasm column: :status do
    state :pending, initial: true
    state :processed
    event :process do
      transitions from: :pending, to: :processed
    end
  end

  after_update :trigger_order_update, if: :status_changed?

  private

  def trigger_order_update
    return unless saved_change_to_status? #only proceed if status has truly changed
    if self.processed?
      order.process_item_completed if order.may_process_item_completed?
      order.save
    end
  end
end

# app/models/order.rb
class Order < ApplicationRecord
  include AASM
  has_many :order_items, inverse_of: :order, dependent: :destroy
  accepts_nested_attributes_for :order_items, allow_destroy: true

  aasm column: :status do
     state :pending, initial: true
     state :processing
     state :completed
     event :process_item_completed do
       transitions from: :pending, to: :processing, if: :all_items_processed?
     end
    event :complete do
      transitions from: :processing, to: :completed, if: :all_items_processed?
    end
  end

  def all_items_processed?
    order_items.all?(&:processed?)
  end
end
```
Here, in the `OrderItem` model's `after_update` callback, we check if the status changed, and specifically, to 'processed'. If it did, we then try to trigger the `process_item_completed` event on the parent `Order`. Crucially, we have an explicit check in our event guard on the order, `all_items_processed?` to allow the order to move forward, and we include `order.save` to ensure changes are persisted. Notice the `saved_change_to_status?` method. This ensures that we only trigger the order update if the status *actually* changed to avoid unnecessary processing. This avoids potential infinite loops. This pattern is workable for simple scenarios, but can quickly get unruly as logic grows.

**2. Using a Service Layer and explicitly Saving Child Associations**

Another pattern i've successfully used is to avoid auto-saving nested attributes on the parent *entirely* and handle association updates via a dedicated service layer. Instead of using `accepts_nested_attributes_for` extensively for modifications, we would use it to handle the initial creation and then move further processing to a dedicated class. This allows granular control over the process.

```ruby
# app/services/order_management_service.rb
class OrderManagementService
  def self.update_order(order, order_params)
      ActiveRecord::Base.transaction do
        order.update!(order_params.except(:order_items_attributes))
        if order_params[:order_items_attributes]
         order_params[:order_items_attributes].each do |_, item_attrs|
            item = order.order_items.find(item_attrs[:id]) if item_attrs[:id].present?
            if item
              item.assign_attributes(item_attrs.except(:id))
              if item.status_changed? && item.may_process?
                item.process!
                order.process_item_completed if order.may_process_item_completed? && order.all_items_processed?
              end
              item.save!
            else
              order.order_items.create!(item_attrs) #For new item creation
            end
          end
         order.save!
        end
    end
  end
end

# app/controllers/orders_controller.rb
class OrdersController < ApplicationController
  def update
    @order = Order.find(params[:id])
    OrderManagementService.update_order(@order, order_params)
    redirect_to @order, notice: 'Order updated successfully'
  end

 private

  def order_params
    params.require(:order).permit(:some_order_attribute,
      order_items_attributes: [:id, :some_item_attribute, :status])
  end
end
```

In this approach, our controller calls the `OrderManagementService` to handle the complex updates. The service iterates through the provided `order_items_attributes`, identifies the existing child item or creates a new one if an ID isn't present, applies the new attributes, then triggers state changes as needed, all while being aware of the surrounding order. The key here is explicit, targeted saving of associations and triggering of events, not merely relying on the automatic save behavior. We still need to save `order` after the fact as the nested item updates have not triggered events that the order is monitoring, if applicable. This gives more flexibility but increases code complexity.

**3. Utilizing a Background Job or Delayed Processing**

If updating and triggering events on an association change is resource-intensive or can be done asynchronously, then we might move the event trigger logic to a background job. This is beneficial when order processing can afford some latency and does not need immediate feedback. This also helps if state transitions are complex.

```ruby
# app/jobs/order_item_processing_job.rb
class OrderItemProcessingJob < ApplicationJob
  queue_as :default

  def perform(order_item_id)
    item = OrderItem.find(order_item_id)
    if item.processed?
        order = item.order
        order.process_item_completed if order.may_process_item_completed? && order.all_items_processed?
        order.save!
    end
  end
end

# app/models/order_item.rb
class OrderItem < ApplicationRecord
 include AASM
 belongs_to :order, inverse_of: :order_items

 aasm column: :status do
   state :pending, initial: true
   state :processed
   event :process do
     transitions from: :pending, to: :processed
   end
 end
 after_update :enqueue_order_update, if: :saved_change_to_status?

 private

  def enqueue_order_update
    OrderItemProcessingJob.perform_later(self.id) if self.processed?
  end
end

#app/models/order.rb
class Order < ApplicationRecord
    include AASM
    has_many :order_items, inverse_of: :order, dependent: :destroy
    accepts_nested_attributes_for :order_items, allow_destroy: true

    aasm column: :status do
       state :pending, initial: true
       state :processing
       state :completed
       event :process_item_completed do
         transitions from: :pending, to: :processing, if: :all_items_processed?
       end
      event :complete do
         transitions from: :processing, to: :completed, if: :all_items_processed?
       end
    end

    def all_items_processed?
      order_items.all?(&:processed?)
    end
end
```

In this version, after an `OrderItem` is updated and its status changes to `processed`, a background job is enqueued. This job is responsible for loading the related `OrderItem`, and then calling appropriate event triggers on the parent `Order` if the item is marked as processed. This shifts the heavy logic to a background worker and prevents blocking the user interface or other process. For a deep dive, I would highly recommend checking out the book *Patterns of Enterprise Application Architecture* by Martin Fowler for advanced service-layer techniques, and *Working with Rails 7* by Stefan Wintermeyer for more in-depth look at lifecycle callbacks. These references will help you in choosing the most robust solution for your needs. Remember to consider the performance implications and trade-offs of each approach. The chosen solution depends highly on the complexity and performance needs of your application. The key takeaway: Rails autosave is great but has limitations, and sometimes, you need to take a more hands-on approach to ensure your events are properly triggered.
