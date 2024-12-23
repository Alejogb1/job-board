---
title: "How do I use Pagy with multiple tables on one page in Rails?"
date: "2024-12-23"
id: "how-do-i-use-pagy-with-multiple-tables-on-one-page-in-rails"
---

, so you’re looking at the challenge of paginating data across multiple tables displayed on a single page in your Rails application using Pagy. Been there, tackled that. It's more common than you might think, especially when you're dealing with dashboards or composite views where different types of data are aggregated for the user. My experience comes from building a resource management tool where we had to display projects, users, and their recent activity all on the same overview page, each with their own pagination. Let's break down how to approach this effectively, along with the code snippets that should clarify it all.

The core issue stems from Pagy, by default, being designed to work with a single set of records. To integrate it with multiple tables, we need to create distinct Pagy instances for each set of data and handle them separately within the view. It's not about bending Pagy to do something it's not designed for, but rather about leveraging its flexibility to accommodate multiple datasets.

First, in your controller, you'll need to instantiate a separate Pagy object for each data set you intend to paginate. For instance, imagine you have three models: `Project`, `User`, and `Activity`. Your controller might look like this:

```ruby
# app/controllers/dashboard_controller.rb
class DashboardController < ApplicationController
  include Pagy::Backend

  def index
    @projects_pagy, @projects = pagy(Project.order(created_at: :desc), items: 5)
    @users_pagy, @users = pagy(User.order(name: :asc), items: 10)
    @activities_pagy, @activities = pagy(Activity.order(created_at: :desc), items: 7)
  end
end
```

Here, we’re creating three different pagy objects and associated result sets. The `items:` argument is key; it allows us to define different page sizes for each table. This approach avoids trying to shoehorn everything into a single pagy object and keeps it clean and understandable. This is fundamental; you avoid a single pagy object trying to handle disjoint data.

Next, in your view, you'll need to access these different pagy objects and render their respective pagination links. Assuming you're using ERB, your view might look something like this:

```erb
<!-- app/views/dashboard/index.html.erb -->
<h2>Projects</h2>
<table>
  <thead>
    <tr>
      <th>Name</th>
      <th>Created At</th>
    </tr>
  </thead>
  <tbody>
    <% @projects.each do |project| %>
      <tr>
        <td><%= project.name %></td>
        <td><%= project.created_at %></td>
      </tr>
    <% end %>
  </tbody>
</table>
<%= pagy_nav(@projects_pagy) %>

<h2>Users</h2>
<table>
  <thead>
    <tr>
      <th>Name</th>
      <th>Email</th>
    </tr>
  </thead>
  <tbody>
    <% @users.each do |user| %>
      <tr>
        <td><%= user.name %></td>
        <td><%= user.email %></td>
      </tr>
    <% end %>
  </tbody>
</table>
<%= pagy_nav(@users_pagy) %>

<h2>Recent Activity</h2>
<table>
  <thead>
    <tr>
      <th>Description</th>
      <th>Timestamp</th>
    </tr>
  </thead>
  <tbody>
    <% @activities.each do |activity| %>
      <tr>
        <td><%= activity.description %></td>
        <td><%= activity.created_at %></td>
      </tr>
    <% end %>
  </tbody>
</table>
<%= pagy_nav(@activities_pagy) %>
```

Notice how we use `@projects_pagy`, `@users_pagy`, and `@activities_pagy` respectively with the `pagy_nav` helper, ensuring each table has its own navigation control. It’s crucial here that you correctly associate the pagy instance to its corresponding data set. The `pagy_nav` helper is smart enough to look for the specific pagy instance.

Now, a common point where things could potentially go south is when you start introducing filters or search parameters. The crucial thing here is to remember that these parameters will impact your data sets. You can apply these parameters on your models before they are passed to `pagy`. Here’s a modified controller example to illustrate this:

```ruby
# app/controllers/dashboard_controller.rb
class DashboardController < ApplicationController
  include Pagy::Backend

  def index
    project_scope = Project.order(created_at: :desc)
    user_scope = User.order(name: :asc)
    activity_scope = Activity.order(created_at: :desc)


    if params[:project_search].present?
      project_scope = project_scope.where("name LIKE ?", "%#{params[:project_search]}%")
    end

     if params[:user_search].present?
      user_scope = user_scope.where("name LIKE ?", "%#{params[:user_search]}%")
    end

      if params[:activity_search].present?
      activity_scope = activity_scope.where("description LIKE ?", "%#{params[:activity_search]}%")
    end

    @projects_pagy, @projects = pagy(project_scope, items: 5, params: params.permit(:project_search))
    @users_pagy, @users = pagy(user_scope, items: 10, params: params.permit(:user_search))
    @activities_pagy, @activities = pagy(activity_scope, items: 7, params: params.permit(:activity_search))
  end
end

```
Notice how we define `project_scope`, `user_scope` and `activity_scope` and then chain a where clause if we receive a search parameter. Most importantly, when we call `pagy`, we must pass the `params:` argument specifying which parameters will affect the pagination itself. Failure to do so will lead to the pagination links not maintaining those search parameters.

In your view, you will need the corresponding forms to collect these search parameters.

```erb
<!-- app/views/dashboard/index.html.erb -->

<h2>Projects</h2>
<%= form_with(url: dashboard_path, method: :get, local: true) do |f| %>
  <%= f.text_field :project_search, value: params[:project_search] %>
  <%= f.submit 'Search Projects' %>
<% end %>
<table>
  <!-- Projects table content -->
</table>
<%= pagy_nav(@projects_pagy) %>

<h2>Users</h2>
<%= form_with(url: dashboard_path, method: :get, local: true) do |f| %>
  <%= f.text_field :user_search, value: params[:user_search] %>
  <%= f.submit 'Search Users' %>
<% end %>
<table>
  <!-- Users table content -->
</table>
<%= pagy_nav(@users_pagy) %>

<h2>Recent Activity</h2>
<%= form_with(url: dashboard_path, method: :get, local: true) do |f| %>
  <%= f.text_field :activity_search, value: params[:activity_search] %>
  <%= f.submit 'Search Activities' %>
<% end %>
<table>
  <!-- Activities table content -->
</table>
<%= pagy_nav(@activities_pagy) %>

```

The forms ensure that when a user performs a search, the search parameter is passed back to the controller, and used to filter the relevant data.

This approach provides a clear and maintainable structure for handling multiple paginated tables on the same page. Remember that each pagy instance operates independently, so you will want to carefully keep track of the pagy objects and corresponding data.

For more in-depth understanding of pagination and its efficiency implications, I'd strongly recommend exploring "Database Systems: The Complete Book" by Hector Garcia-Molina, Jeff Ullman, and Jennifer Widom. Although it's not Rails-specific, its principles are crucial in understanding the data layer. Also, while it's not a technical book, Eric Evans' "Domain-Driven Design" is incredibly valuable when architecting complex applications that display data from multiple sources. Understanding domain concepts makes it easier to separate data in a clean and logical manner. Also look into the Pagy documentation as it covers all the various options. Finally, the Rail's guide on Active Record queries can help optimise the database calls before it reaches Pagy.

Essentially, the key to handling multiple paginated tables with Pagy is to treat each table as a distinct pagination problem, rather than trying to make a single pagy object handle all data. This not only simplifies your code, but it also makes the system easier to maintain and debug.
