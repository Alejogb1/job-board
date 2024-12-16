---
title: "How can I use Pagy with multiple tables on a single page in Rails?"
date: "2024-12-16"
id: "how-can-i-use-pagy-with-multiple-tables-on-a-single-page-in-rails"
---

Alright, let's tackle this. I remember a project a few years back, a complex reporting dashboard, where we faced exactly this challenge: paginating multiple tables simultaneously on a single page using Rails. It's a common scenario when you're dealing with aggregated data or distinct sets of information that belong together visually, but come from separate database queries. The core problem, as I see it, isn't about paginating *one* collection, but orchestrating multiple pagination workflows and their UI components gracefully. We can absolutely make Pagy play nice with this.

The immediate, and probably the most obvious, approach that often gets suggested is to try and merge all your data into a single collection, paginate that, and then try and slice and dice that merged collection for display. While in theory, that might be *possible*, it’s generally a terrible idea in terms of performance and maintainability. If your tables come from distinct models or queries, attempting to homogenize them introduces considerable overhead. Instead, we need to treat each table's dataset as an independent paginatable entity. Pagy facilitates this fairly easily by providing you with individual Pagy instances.

The fundamental technique I’ve consistently found success with is to create separate pagy instances within the controller for each table that needs pagination. Each pagy object corresponds directly to the specific query for a particular table. Then, you simply pass these separate pagy instances to the view for rendering the appropriate UI components alongside the data for each respective table. The key here is that *each* pagy object will maintain its own state, its own page numbers, and control its data for one specific data set. This maintains separation of concerns, and prevents the single pagination issue.

Let’s explore a practical example. Imagine we have a simple page that needs to display both “Recent Orders” and “Pending Shipments” in two distinct tables. Each table represents a different query.

Here's how we might achieve this in the controller:

```ruby
class DashboardController < ApplicationController
  def index
    @orders = Order.order(created_at: :desc)
    @shipments = Shipment.where(status: 'pending').order(created_at: :desc)

    @pagy_orders, @orders = pagy(@orders, items: 5, page_param: :orders_page)
    @pagy_shipments, @shipments = pagy(@shipments, items: 7, page_param: :shipments_page)

    render 'index' # or whatever your view is called
  end
end
```

In this controller code:
1. We retrieve the data sets for Orders and Shipments. Each is ordered by `created_at` for simplicity.
2. We invoke the `pagy` method *twice*, once for `@orders`, and again for `@shipments`. Importantly, we're passing in the `page_param` to distinguish each pagination object, `orders_page` and `shipments_page` respectively. This avoids naming collisions in the URL, preventing issues where navigating one paginated table causes the other to shift.
3. We’re also controlling how many items appear on each paginated table using the `items:` parameter. This is specific to each pagy instance.

Next, let’s consider the view rendering. We will use Pagy’s helpful view methods that come with the helper module.

```erb
<h1>Dashboard</h1>

<section>
  <h2>Recent Orders</h2>
  <table>
    <thead>
      <tr><th>Order ID</th><th>Created At</th></tr>
    </thead>
    <tbody>
      <% @orders.each do |order| %>
        <tr><td><%= order.id %></td><td><%= order.created_at %></td></tr>
      <% end %>
    </tbody>
  </table>

  <div class="pagination">
    <%== pagy_nav(@pagy_orders) %>
  </div>
</section>

<section>
  <h2>Pending Shipments</h2>
  <table>
    <thead>
      <tr><th>Shipment ID</th><th>Created At</th></tr>
    </thead>
    <tbody>
      <% @shipments.each do |shipment| %>
        <tr><td><%= shipment.id %></td><td><%= shipment.created_at %></td></tr>
      <% end %>
    </tbody>
  </table>

    <div class="pagination">
      <%== pagy_nav(@pagy_shipments) %>
    </div>
</section>
```

Key takeaways from the view code:
1. We render two distinct tables, one for `@orders` and the other for `@shipments`. Each table displays its own set of data.
2. Most crucially, each table has its own `<div class="pagination">` container where we invoke `pagy_nav()` with the correct Pagy object. This is what generates the actual pagination controls. Using separate Pagy objects ensures that clicking on the “Next” button on the order table only affects the order table's data, and so on.
3. The `<%==`  tag is important to use, it is the raw HTML output from `pagy_nav`, and without it, the HTML for pagination won't render.

Now, for a more complex scenario, you may want to customize the styling or structure of the pagination controls. Perhaps we have custom styling elements or a different kind of UI.  Pagy offers that flexibility too.

Here's a slightly more intricate pagination rendering example within the view, using `pagy_bootstrap_nav` (assuming you have a Bootstrap-compatible style):

```erb
<section>
  <h2>Recent Orders</h2>
  <table>
    <thead>
      <tr><th>Order ID</th><th>Created At</th></tr>
    </thead>
    <tbody>
      <% @orders.each do |order| %>
        <tr><td><%= order.id %></td><td><%= order.created_at %></td></tr>
      <% end %>
    </tbody>
  </table>
  <div class="pagination-container">
  <%= pagy_bootstrap_nav(@pagy_orders) if @pagy_orders.pages > 1 %>
  </div>
</section>


<section>
  <h2>Pending Shipments</h2>
  <table>
    <thead>
      <tr><th>Shipment ID</th><th>Created At</th></tr>
    </thead>
    <tbody>
      <% @shipments.each do |shipment| %>
        <tr><td><%= shipment.id %></td><td><%= shipment.created_at %></td></tr>
      <% end %>
    </tbody>
  </table>
  <div class="pagination-container">
    <%= pagy_bootstrap_nav(@pagy_shipments) if @pagy_shipments.pages > 1 %>
  </div>
</section>
```

In this rendition:
1. We've replaced `pagy_nav` with `pagy_bootstrap_nav`. This generates pagination controls styled for Bootstrap. Ensure you have the Pagy Bootstrap gem installed and configured for this to render as intended.
2. Crucially, we've added a conditional: `if @pagy_orders.pages > 1`. This renders the pagination only if there is more than one page worth of data, preventing the pagination bar from appearing if it’s not needed.
3. You can replace `pagy_bootstrap_nav` with other options that Pagy provides, such as `pagy_bulma_nav` or a completely custom implementation should you need to. Consult the Pagy documentation for more details.

Remember, in real-world applications, you might also want to consider adding parameters to your routes to explicitly specify which table is being navigated if needed, particularly if you are using query parameters rather than a path-based approach.

For further reading and a more robust understanding of these types of issues, I would recommend delving into "Refactoring Databases: Evolutionary Database Design" by Scott W. Ambler and Pramod J. Sadalage, especially if you're dealing with data-heavy applications requiring efficient pagination strategies. Additionally, examining the source code and associated articles of Pagy itself is invaluable for fully leveraging its capabilities. Check out the official Pagy GitHub repository. The author, @ddnexus, has done great work explaining the internals. Also, explore the Rails Guides on Active Record Querying for further information on optimizing database interactions. Understanding how different query structures affect page load speed can be immensely beneficial for any application that handles data pagination.

In conclusion, effectively paginating multiple tables on a single page with Pagy involves creating separate pagy objects, passing the objects to the view, and rendering the navigation controls appropriately. This is a fairly clean and maintainable way to handle complex pagination scenarios. Don't hesitate to delve deeper into the Pagy documentation, explore Active Record's querying capabilities, and understand how database pagination works at the level of SQL. You will then have a solid grasp of efficient pagination implementation in Rails applications.
