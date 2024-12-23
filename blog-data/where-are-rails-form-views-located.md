---
title: "Where are Rails form views located?"
date: "2024-12-23"
id: "where-are-rails-form-views-located"
---

, let’s tackle this one. It’s not as straightforward as just pointing at a single directory, because the location of Rails form views depends heavily on the context within your application's architecture. Over the years, I’ve encountered numerous scenarios, from straightforward projects to complex multi-tenant systems, and the way forms are handled evolves with the project's demands. Let's break it down from a practical standpoint.

Fundamentally, Rails form views reside within the `app/views` directory, but their *specific* location is determined by the controller and the action being executed. If you’re using the standard conventions, you’ll find them neatly organized into subfolders that mirror your controllers. For instance, if you have a `UsersController` managing user-related actions, then the form views related to that controller will typically live under `app/views/users/`. Specifically, if you’re looking for the form view for the `new` action (which commonly renders a new user creation form) you’ll often find it at `app/views/users/new.html.erb` (or whatever your templating engine is configured for).

The first thing to grasp is the concept of *implicit rendering*. Rails, by default, will look for a view file that matches the name of the controller action. That means if your controller action is `def create` in a `UsersController` class, and you don't explicitly specify what view to render, Rails looks for `create.html.erb`. That's often not a form, but the output *after* the form is submitted, but you'll find the forms often follow this pattern. So, for the *display* of a form, you'll almost certainly be looking at the 'new' or 'edit' actions.

Let's assume we’re discussing a typical model-backed form. In practice, this often involves creating a new resource (like a user) or editing an existing one. In those scenarios, I often saw form views at `app/views/<controller_name>/new.html.erb` for creating new entries and `app/views/<controller_name>/edit.html.erb` for updating them. This is just a standard convention, of course, but it’s where most projects *start*, and that’s a useful reference point.

Now, it's critical to understand that Rails uses a concept of *layout*. While your views hold the individual form elements, you typically wrap that form content inside a general layout which controls the overall page structure. Layouts usually live in `app/views/layouts/` and frequently named something like `application.html.erb`. Inside your layout, you'll generally see a `<%= yield %>` statement, and this is where the specific form views content are inserted. This ensures consistency across pages, and you can have per-controller layouts if the project requires a different structure.

Let’s move into some practical coding examples.

**Example 1: Simple User Creation Form**

Here’s how a simple new user form might look, residing in `app/views/users/new.html.erb`:

```erb
<h1>New User</h1>

<%= form_with(model: @user, url: users_path, method: :post) do |form| %>

  <div>
    <%= form.label :username, "Username:" %>
    <%= form.text_field :username %>
  </div>

  <div>
    <%= form.label :email, "Email:" %>
    <%= form.email_field :email %>
  </div>

  <div>
    <%= form.label :password, "Password:" %>
    <%= form.password_field :password %>
  </div>

  <div>
    <%= form.submit "Create User" %>
  </div>

<% end %>
```

In this snippet, `@user` is an instance variable passed from the `new` action in the `UsersController`. The `form_with` helper handles the creation of the form tag, pointing to `users_path` which typically is associated with `create` action. You’ll notice that each input is tied to attributes of `@user` via method calls like `:username`. This is where the model interaction in rails becomes very clear.

**Example 2: Edit User Form**

Building off the previous example, an edit form, typically found at `app/views/users/edit.html.erb`, would look very similar, but with differences in how the form is configured and pre-filled:

```erb
<h1>Edit User</h1>

<%= form_with(model: @user, url: user_path(@user), method: :patch) do |form| %>

  <div>
    <%= form.label :username, "Username:" %>
    <%= form.text_field :username %>
  </div>

  <div>
    <%= form.label :email, "Email:" %>
    <%= form.email_field :email %>
  </div>

  <div>
    <%= form.label :password, "Password:" %>
    <%= form.password_field :password, value: "" %>
   <small>Leave blank to keep current password.</small>
  </div>

  <div>
    <%= form.submit "Update User" %>
  </div>

<% end %>
```

Here the key differences are the form's url. Instead of `users_path`, we're using `user_path(@user)` passing the existing `@user` object to the edit form. Also, since we are editing an existing password we should *not* be pre-populating the password, as that would be a security risk. And finally, the method is `:patch`, reflecting the convention of using HTTP PATCH to update resources. The rest is very similar, with `@user` being passed into the controller and accessible inside the view.

**Example 3: Forms with Partials**

In more complex applications, you might want to break down your form views into reusable components. This is often achieved using partials. A partial is just another view fragment that lives under its corresponding controller's folder, but instead of naming them `<action_name>.html.erb` they usually start with an `_`. For example, you could have a partial called `_form_fields.html.erb` in `app/views/users/`.

Then you can render it in both `new.html.erb` and `edit.html.erb` like this:

```erb
<!-- app/views/users/new.html.erb -->
<h1>New User</h1>

<%= form_with(model: @user, url: users_path, method: :post) do |form| %>
    <%= render 'form_fields', form: form %>
    <div><%= form.submit "Create User" %></div>
<% end %>

```

```erb
<!-- app/views/users/edit.html.erb -->
<h1>Edit User</h1>

<%= form_with(model: @user, url: user_path(@user), method: :patch) do |form| %>
    <%= render 'form_fields', form: form %>
    <div><%= form.submit "Update User" %></div>
<% end %>
```

And in the `_form_fields.html.erb` we would place the common form fields:

```erb
<div>
    <%= form.label :username, "Username:" %>
    <%= form.text_field :username %>
  </div>

  <div>
    <%= form.label :email, "Email:" %>
    <%= form.email_field :email %>
  </div>

  <div>
    <%= form.label :password, "Password:" %>
    <%= form.password_field :password, value: "" %>
   <small>Leave blank to keep current password.</small>
  </div>
```

This approach promotes code reuse and maintainability. I’ve seen large, complex applications really benefit from such granular views, as they reduced duplication and complexity significantly.

As for further reading, consider exploring “Agile Web Development with Rails” by Sam Ruby, Dave Thomas, and David Heinemeier Hansson. The book provides a thorough grounding in Rails architecture and best practices, which greatly helps in understanding view locations, controllers and routing. Additionally, the Rails Guides are a fantastic resource. You'll find detailed explanations of how form helpers and partials function. Specifically, the guides on layouts, forms, and the asset pipeline will be useful.

In summary, Rails form views are typically located within the `app/views` directory, organized into subdirectories based on controller names, usually named `new.html.erb` for creating forms and `edit.html.erb` for editing existing records. Understanding the implicit rendering conventions, partials, and the use of layouts is crucial for navigating view locations. And the above three examples hopefully clear up the practical application of those concepts. It's a system based on convention over configuration, which makes maintaining medium and large-sized Rails projects a lot easier to scale.
