---
title: "How to use Pagy for multiple tables on one page in Rails?"
date: "2024-12-16"
id: "how-to-use-pagy-for-multiple-tables-on-one-page-in-rails"
---

Okay, let's tackle this. I've seen variations of this requirement quite a few times in my years of working with rails applications. The scenario is common: you've got a dashboard, a complex view, or maybe a reporting page that needs to display paginated data from multiple database tables simultaneously. And yes, while rails' default pagination mechanisms are good for simple cases, they fall short when you need that more sophisticated multi-table presentation. Pagy, a gem I've come to appreciate for its performance and flexibility, offers an elegant solution.

The key here isn’t to try to force pagy to magically handle multiple tables in a single, monolithic call. Instead, we'll treat each table’s data and pagination separately and combine the presentation in the view. This gives us fine-grained control and keeps the logic clean. It's akin to modular design principles - each pagination concern is isolated.

First, understand that pagy works with an array, an active record relation, or other enumerable objects, and each instantiation of pagy is independent. So, to paginate multiple tables, you'll create multiple instances of `Pagy`. Let’s assume I’ve been working on a project where we needed to display both `users` and `orders` in a dashboard, and this is where I've implemented this strategy before.

Here's how you can implement it in your controller. Let's say you have a dashboard controller:

```ruby
# app/controllers/dashboard_controller.rb
class DashboardController < ApplicationController
  include Pagy::Backend

  def index
    @users_pagy, @users = pagy(User.all, items: 5)
    @orders_pagy, @orders = pagy(Order.all, items: 7)
  end
end
```

In this snippet, `Pagy::Backend` is included to make the `pagy` method available. We make two calls to the `pagy` function, one for the `User` model and one for the `Order` model, providing the records to paginate and, optionally, the number of items per page. The returned values are assigned to `@users_pagy`, `@users`, `@orders_pagy`, and `@orders` instance variables, which will be used in the view. Notice that I’ve specified different `items` counts. This is common, as different tables may require different levels of granularity in their display.

Now, the view. You will need to iterate over the respective collections and display pagy controls for each of them. I would structure it like this, using erb:

```erb
<%# app/views/dashboard/index.html.erb %>

<h2>Users</h2>
<table>
  <thead>
    <tr>
      <th>ID</th>
      <th>Name</th>
    </tr>
  </thead>
  <tbody>
    <% @users.each do |user| %>
      <tr>
        <td><%= user.id %></td>
        <td><%= user.name %></td>
      </tr>
    <% end %>
  </tbody>
</table>
<%= pagy_nav(@users_pagy) %>

<h2>Orders</h2>
<table>
  <thead>
    <tr>
      <th>ID</th>
      <th>Order Date</th>
    </tr>
  </thead>
  <tbody>
    <% @orders.each do |order| %>
      <tr>
        <td><%= order.id %></td>
        <td><%= order.order_date %></td>
      </tr>
    <% end %>
  </tbody>
</table>
<%= pagy_nav(@orders_pagy) %>
```

Here, the `pagy_nav` helper is called for each `pagy` instance (`@users_pagy` and `@orders_pagy`). This renders the pagination controls for each table independently, and each table's data is displayed within its specific HTML table. It keeps the display context clear, meaning pagination for each data set does not interfere with the others.

Now, let’s consider a slightly more complex scenario. What if I needed to paginate data that comes from a more elaborate query, potentially involving joins or aggregations? In a previous application, I needed to show a list of authors and their total number of books, paginated. This required a more intricate data preparation in the controller.

Here's an example of how that controller might look:

```ruby
# app/controllers/reports_controller.rb
class ReportsController < ApplicationController
  include Pagy::Backend

  def authors_with_books
    authors_with_book_counts = Author.select("authors.*, COUNT(books.id) as book_count")
                                  .left_outer_joins(:books)
                                  .group("authors.id")

    @authors_pagy, @authors_with_counts = pagy(authors_with_book_counts, items: 10)

  end
end
```

In this controller method, `authors_with_book_counts` performs a join and groups the authors by ID, along with calculating the book count. This prepared query is then passed into `pagy()`, resulting in `@authors_pagy` and `@authors_with_counts`. You would, then, use these instance variables similarly in the corresponding view, using `pagy_nav(@authors_pagy)` to display the pagination controls. The crucial point is that the `pagy` method works seamlessly with the custom active record relation that results from the query.

To round this out, I'll mention that Pagy doesn’t just offer basic pagination. It can be configured with many options, such as different styles for pagination controls, support for localization, and other performance tweaks. Refer to the official gem documentation on GitHub for an exhaustive list of configurations. For a deeper understanding of efficient database querying techniques used in the examples, I would strongly recommend reading "SQL and Relational Theory" by C.J. Date and "Database Internals: A Deep Dive into How Query Engines Work" by Alex Petrov. While not specific to rails or pagy, understanding database query optimization is paramount to getting good performance, particularly when paginating data from complex queries, or large tables which are something I’ve had to deal with numerous times when working with large databases. Additionally, for those looking to further explore advanced rails techniques, "Agile Web Development with Rails 7" by Sam Ruby et al, offers a great overview of the ecosystem, including best practices for working with data in a rails application.

In conclusion, handling multiple tables with pagy on a single page isn't about some single function call. It’s about approaching the problem from a modular point of view. It’s about preparing each data set appropriately in your controller, treating each pagination requirement as an independent entity. Then, you present this data and its respective pagination control in your view. This structured approach not only makes your code more manageable but also more maintainable and significantly improves overall performance. This technique is one I’ve consistently relied on for projects where data display requirements get a little more intricate, and it has always served me well.
