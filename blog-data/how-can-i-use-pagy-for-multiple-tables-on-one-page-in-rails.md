---
title: "How can I use Pagy for multiple tables on one page in Rails?"
date: "2024-12-23"
id: "how-can-i-use-pagy-for-multiple-tables-on-one-page-in-rails"
---

Let's tackle this pagination challenge with multiple tables on a single rails page. I’ve seen this crop up quite a few times, especially when dealing with dashboards or reports that aggregate data from various sources. It's definitely not a one-size-fits-all solution, and it does require a bit of planning to avoid performance bottlenecks. The key is to understand how Pagy works and then tailor it to our multi-table scenario.

The common approach that often leads to trouble is attempting to apply a single Pagy instance to multiple ActiveRecord relations simultaneously. Pagy is designed around paginating a single collection efficiently. Trying to force it to handle several at once results in conflicts and unexpected behavior. We need separate Pagy instances for each table. It's crucial to remember that behind the scenes, Pagy is primarily calculating offset and limit clauses for your database query.

Over my years, I recall an application we built for managing different types of inventory across multiple warehouses. We had several tables displaying these different inventory categories on the same overview page. Initially, we tried using a single global pagination instance. That quickly became a mess of confused logic and conflicting page navigation. After that experience, we shifted to individual Pagy instances per table and that solved most of our issues.

The first thing to focus on is how we will structure our controller. Rather than loading all the data into the view and paginating there, the optimal approach is to handle the pagination logic within the controller action, creating separate Pagy instances, and passing paginated datasets to the view. Here is a basic example controller action where I'm pulling data from `Products`, `Orders`, and `Customers` tables:

```ruby
def index
  @pagy_products, @products = pagy(Product.all, items: 10, page_param: :products_page)
  @pagy_orders, @orders = pagy(Order.all, items: 5, page_param: :orders_page)
  @pagy_customers, @customers = pagy(Customer.all, items: 8, page_param: :customers_page)
end
```

Notice the `page_param` option. This is crucial. Without it, each Pagy instance would use the default 'page' query parameter, leading to navigation conflicts. By using `products_page`, `orders_page`, and `customers_page`, we ensure that each table’s pagination is independent.

Now, let’s translate that into the view. We’ll need to use multiple `pagy_nav` helpers, referencing the correct Pagy instance for each table. Here’s an example using erb syntax:

```erb
<h2>Products</h2>
<table>
  <thead>
    <tr><th>Name</th><th>Price</th></tr>
  </thead>
  <tbody>
  <% @products.each do |product| %>
    <tr><td><%= product.name %></td><td><%= product.price %></td></tr>
  <% end %>
  </tbody>
</table>
<%= pagy_nav(@pagy_products) %>

<hr>

<h2>Orders</h2>
<table>
  <thead>
    <tr><th>Order ID</th><th>Date</th></tr>
  </thead>
  <tbody>
    <% @orders.each do |order| %>
      <tr><td><%= order.id %></td><td><%= order.created_at %></td></tr>
    <% end %>
  </tbody>
</table>
<%= pagy_nav(@pagy_orders) %>

<hr>

<h2>Customers</h2>
<table>
  <thead>
    <tr><th>Name</th><th>Email</th></tr>
  </thead>
  <tbody>
    <% @customers.each do |customer| %>
      <tr><td><%= customer.name %></td><td><%= customer.email %></td></tr>
    <% end %>
  </tbody>
</table>
<%= pagy_nav(@pagy_customers) %>
```

This view snippet renders three tables, each with its own pagination controls, all working independently, and correctly linked to its corresponding paginated data set. This ensures a clear and organized presentation of multiple data sources on the same page.

Finally, let's delve into a slightly more complex example, using a more real-world setup. Suppose we want to paginate through user posts, comments, and user follows. This involves potentially more complex queries and joins. The goal remains the same: separate Pagy instances, correct parameter handling, and distinct pagination controls in the view:

```ruby
def dashboard
    user = User.find(params[:id])

    @pagy_posts, @posts = pagy(user.posts.order(created_at: :desc), items: 5, page_param: :posts_page)
    @pagy_comments, @comments = pagy(user.comments.order(created_at: :desc), items: 8, page_param: :comments_page)
    @pagy_followers, @followers = pagy(user.followers, items: 10, page_param: :followers_page)
  end
```

And the view may look like:

```erb
<h2>Posts</h2>
<table>
    <thead>
        <tr><th>Title</th><th>Created At</th></tr>
    </thead>
    <tbody>
        <% @posts.each do |post| %>
            <tr>
                <td><%= post.title %></td>
                <td><%= post.created_at %></td>
            </tr>
        <% end %>
    </tbody>
</table>
<%= pagy_nav(@pagy_posts) %>
<hr>

<h2>Comments</h2>
<table>
  <thead>
      <tr><th>Comment</th><th>Created At</th></tr>
  </thead>
  <tbody>
    <% @comments.each do |comment| %>
        <tr>
            <td><%= comment.body %></td>
            <td><%= comment.created_at %></td>
        </tr>
      <% end %>
  </tbody>
</table>
<%= pagy_nav(@pagy_comments) %>

<hr>

<h2>Followers</h2>
<table>
  <thead>
        <tr><th>Name</th></tr>
    </thead>
    <tbody>
      <% @followers.each do |follower| %>
          <tr><td><%= follower.name %></td></tr>
      <% end %>
  </tbody>
</table>
<%= pagy_nav(@pagy_followers) %>
```

Notice in each case how the approach remains consistent. Each table is handled with an individual pagy instance and the corresponding pagination controls. This structure is extremely robust and flexible.

In closing, the key to effectively paginating multiple tables on a single page with Pagy involves careful management of separate Pagy instances and distinct `page_param` values. Don’t try to force a single Pagy object to paginate multiple sets of data; this approach is brittle and will quickly lead to problems. Handle pagination logic in the controller and make sure the view displays the correct navigational links for each table. Finally, for a deeper dive into the complexities and best practices surrounding database pagination, I recommend reading "Database Design and Implementation" by Michael J. Hernandez and Thomas Bell, and "SQL and Relational Theory" by C.J. Date. They both provide very thorough technical context that will prove invaluable. I’ve leaned on those resources countless times in my own work.
