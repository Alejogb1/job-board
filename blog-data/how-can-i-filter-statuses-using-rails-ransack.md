---
title: "How can I filter statuses using Rails Ransack?"
date: "2024-12-23"
id: "how-can-i-filter-statuses-using-rails-ransack"
---

,  It's something I’ve certainly found myself grappling with more than once, particularly back when I was working on that large social media aggregation project a few years ago. We needed extremely fine-grained control over status filtering, and simply relying on basic database queries wasn't going to cut it. Ransack became our go-to solution, and with good reason. It provides a beautiful way to turn user-facing inputs into complex database queries, all without writing a ton of custom SQL. Let’s break down how to effectively filter statuses with Ransack in Rails.

The core concept with Ransack revolves around ‘search attributes’. These attributes are defined on your model and tell Ransack which fields and associations can be used for filtering. For instance, let's imagine we have a `Status` model with attributes like `content`, `user_id`, `created_at`, and a relation to `User` model via `belongs_to :user`.

First, you must ensure you’ve correctly installed and configured the Ransack gem. If not, add `gem 'ransack'` to your Gemfile and run `bundle install`. Once that's sorted, we can start setting up our model. Inside your `app/models/status.rb` file, we'll define the search attributes using a bit of Ransack's magic. While Ransack will, by default, try to handle column names as searchable attributes, specifying them explicitly ensures you have more control.

Here’s a simplified version of our `Status` model with some defined search attributes:

```ruby
class Status < ApplicationRecord
  belongs_to :user

  ransacker :created_at_date, formatter: ->(date) { Date.parse(date) } do |parent|
    parent.table[:created_at]
  end
  ransackable_attributes %w[content user_id created_at_date]
  ransackable_associations %w[user]
end
```

Notice the `ransacker` method there. I've used this to create a search attribute specifically for date comparisons while filtering by `created_at`. Without this, Ransack would try to search against datetime, which might not be what the user expects when filtering by date alone. I've also added `ransackable_attributes` and `ransackable_associations` which is good practice, allowing us to explicitly limit the search parameters. This can be crucial in terms of security. We don't want users manipulating unexpected queries.

Now, how does this translate into practical filtering in our controller? Let’s consider a `StatusesController`. Inside our `index` action, we'll initialize a Ransack search object, and pass in the incoming parameters.

```ruby
class StatusesController < ApplicationController
  def index
    @q = Status.ransack(params[:q])
    @statuses = @q.result.includes(:user).order(created_at: :desc).page(params[:page])
  end
end
```

The `@q` object represents our Ransack search. We initialize it with `params[:q]`, which is where Ransack expects its search parameters to come from. Then we call `result` on it, which executes the query, returning an `ActiveRecord::Relation`. Here I’ve also shown how to eager load the user association using `includes` to avoid N+1 query issues and how to include basic pagination using the `kaminari` gem's `page` method, assuming you have it installed.

The beauty of this approach is that you can build complex queries from simple form elements. In your view, you might have a form like this using Rails form helpers and Ransack's `search_form_for`:

```erb
<%= search_form_for @q, url: statuses_path do |f| %>
    <%= f.label :content_cont, 'Content Contains' %>
    <%= f.search_field :content_cont %>

    <%= f.label :user_id_eq, 'User ID Equals' %>
    <%= f.search_field :user_id_eq %>

    <%= f.label :created_at_date_eq, 'Created on date' %>
    <%= f.date_field :created_at_date_eq %>

    <%= f.label :user_name_cont, 'User name contains' %>
    <%= f.search_field :user_name_cont %>

  <%= f.submit 'Search' %>
<% end %>
```

Here, the `content_cont`, `user_id_eq`, `created_at_date_eq`, and `user_name_cont` are called "predicates".  Ransack offers a range of these for performing different types of comparison such as `eq` (equals), `cont` (contains), `lt` (less than), and many others. This means you don’t have to write custom SQL to handle these varied conditions. It also allows you to search through association attributes such as the `user_name` if you have set up the association correctly in the model. For instance, to search for all statuses from users with a name that contains “john”, Ransack automatically translates `user_name_cont` in the url params to join the status table with the users table on the user_id and applies the like clause.

Let's say I wanted to demonstrate a more complex scenario. What if I wanted to filter status content by a phrase that's case-insensitive and also wanted to filter by a range of created dates? With a few adjustments in our form and in the `ransacker` in the model, we can achieve this with ease. Let's add another search attribute to the `Status` model:

```ruby
class Status < ApplicationRecord
  belongs_to :user

  ransacker :created_at_date, formatter: ->(date) { Date.parse(date) } do |parent|
    parent.table[:created_at]
  end

   ransacker :content_i_cont do |parent|
      Arel::Nodes::NamedFunction.new('LOWER', [parent.table[:content]])
    end

  ransackable_attributes %w[content_i_cont user_id created_at_date]
  ransackable_associations %w[user]
end

```
Here, we use a different approach with `Arel::Nodes::NamedFunction` to achieve case-insenstive search. This ensures that regardless of capitalization in the user input or the stored value, they match during the search. The `i_cont` part means 'case insensitive contains'.

Now we will update the form to include this new filter, alongside an additional date range field:

```erb
<%= search_form_for @q, url: statuses_path do |f| %>
  <%= f.label :content_i_cont, 'Content Contains (Case Insensitive)' %>
  <%= f.search_field :content_i_cont %>

  <%= f.label :created_at_date_gteq, 'Created After Date' %>
  <%= f.date_field :created_at_date_gteq %>

  <%= f.label :created_at_date_lteq, 'Created Before Date' %>
  <%= f.date_field :created_at_date_lteq %>
  
   <%= f.label :user_name_cont, 'User name contains' %>
    <%= f.search_field :user_name_cont %>

  <%= f.submit 'Search' %>
<% end %>
```

We use `created_at_date_gteq` and `created_at_date_lteq` to filter between two dates, which are the Ransack equivalents for 'greater than or equal to' and 'less than or equal to' respectively. This will filter statuses created within a specific date range. The `params[:q]` will now include these new search criteria.

In summary, Ransack is a powerful and versatile tool. While it's very helpful in most scenarios, there are cases where you might want to revert to building custom queries. When you're encountering issues with more advanced queries or performance, I would recommend looking at publications on optimal database indexing for your specific database and application. Also consider exploring "Active Record Query Interface" documentation for a deep dive into ActiveRecord's query language. For a solid understanding of advanced query optimization, "SQL Performance Explained" by Markus Winand, offers a practical perspective on writing efficient SQL. This combined knowledge will help you understand when to leverage Ransack and when a custom query might be necessary.
