---
title: "How can Active Record's acts_as_nested_set be used in forms?"
date: "2024-12-23"
id: "how-can-active-records-actsasnestedset-be-used-in-forms"
---

Let's tackle this interesting challenge of integrating `acts_as_nested_set` into forms. It's a scenario I've encountered numerous times in my career, particularly when dealing with hierarchical data like categories, organizational charts, or complex menu structures. We often think of model data as being represented linearly in forms, but nested sets require a different approach. The core issue revolves around the fact that nested sets store hierarchy implicitly using `lft` and `rgt` attributes, rather than through explicit parent-child relationships that are more intuitive for direct manipulation via forms. This means that a standard select box, directly populated with categories, doesn't give the control needed to create the right nesting structure. So, we need to leverage existing relationships to build an interface that feels natural to the user.

I recall a project for a content management system years ago where we had to manage thousands of articles organized into a dynamic category tree. It became clear very quickly that simply allowing users to directly manipulate `lft` and `rgt` was a recipe for disaster. Instead, we focused on presenting an interface that allowed them to define *where* in the hierarchy they wanted a category to sit, and then letting the `acts_as_nested_set` gem handle the intricate database updates.

The key to this integration lies in understanding how we can leverage the provided `acts_as_nested_set` methods in the model to maintain the integrity of the tree and using those methods appropriately to build a functional form. Primarily, we are going to be concerned with creating, editing, and moving nodes in a meaningful way.

Let's break this down into a few key scenarios and provide concrete examples. I'm assuming you are already using `acts_as_nested_set` in your model. Let’s consider a Category model:

```ruby
class Category < ApplicationRecord
  acts_as_nested_set
end
```

**Scenario 1: Creating a New Category**

Creating a new category usually involves allowing the user to select a *parent* category. We present this as a select element of some kind within the form. When submitted, we use the `append_child` method to insert the new category correctly.

Here's the view component:

```erb
<%= form_with(model: @category, url: categories_path, method: :post) do |form| %>
  <div>
    <%= form.label :name %>
    <%= form.text_field :name %>
  </div>

  <div>
    <%= form.label :parent_id, "Parent Category" %>
    <%= form.select :parent_id, Category.all.collect {|c| [c.name, c.id]}, include_blank: true %>
  </div>

  <div>
    <%= form.submit "Create Category" %>
  </div>
<% end %>
```

And here's the corresponding controller action:

```ruby
def create
  @category = Category.new(category_params)
  if params[:category][:parent_id].present?
    parent = Category.find(params[:category][:parent_id])
    parent.append_child(@category)
  else
    @category.save
  end

  if @category.valid?
    redirect_to categories_path, notice: 'Category created successfully.'
  else
    render :new
  end
end

private

def category_params
  params.require(:category).permit(:name)
end
```

In this example, we are taking the `parent_id` from the submitted form and utilizing `append_child`. If no parent is selected, the new category simply becomes a root node. Notice there’s no direct manipulation of `lft` or `rgt` happening in our controller; we’re letting the `acts_as_nested_set` gem handle that.

**Scenario 2: Editing an Existing Category and Moving It**

When editing, we not only want to be able to change a category’s name but also reposition it within the hierarchy. We can accomplish this again with the help of a select element, with the additional consideration that it cannot be moved under itself or any of its descendants.

Here is the edited view with an additional `move_to` functionality:

```erb
<%= form_with(model: @category, url: category_path(@category), method: :patch) do |form| %>
  <div>
    <%= form.label :name %>
    <%= form.text_field :name %>
  </div>

  <div>
    <%= form.label :move_to_id, "Move To" %>
    <%= form.select :move_to_id, Category.all.reject {|c| c == @category || @category.is_descendant_of?(c) }.collect {|c| [c.name, c.id] }, include_blank: true  %>
  </div>

  <div>
      <%= form.submit "Update Category" %>
  </div>
<% end %>
```

And the updated controller action to include moving logic:

```ruby
def update
  if params[:category][:move_to_id].present?
     move_to_category = Category.find(params[:category][:move_to_id])
     @category.move_to(move_to_category)
  end

  if @category.update(category_params)
     redirect_to categories_path, notice: 'Category updated successfully.'
   else
    render :edit
   end
end
```

We’re using `move_to` to change the category's position in the tree. The `reject` clause in the `select` options ensures that we’re not offering a move target that would lead to circular or invalid nesting using `is_descendant_of?` method which is readily provided by the gem. We've separated updating name, and moving it within the tree to be more explicit.

**Scenario 3: Rendering a Nested Select Box**

Finally, it’s useful to be able to render a nested select box with appropriate indentation to show the hierarchical structure. While this is not for *manipulation*, it's often useful for *displaying* the structure in other parts of the application.

Here is the helper function in your controller or helper:

```ruby
def nested_options(categories, level = 0)
  options = []
  categories.each do |category|
      options << [("– " * level + category.name), category.id]
    if category.children.any?
       options += nested_options(category.children, level + 1)
    end
  end
  options
end
```

Then you can use it in the view:

```erb
<%= select_tag "category_dropdown", options_for_select(nested_options(Category.roots)), prompt: "Select a Category" %>
```

This function recursively traverses the tree, adding leading spaces based on the depth. The `options_for_select` view helper can render those options as a dropdown element. This function, which might have appeared deceptively simple at first glance, utilizes a recursive algorithm, an important concept to become comfortable with when you're dealing with tree structures.

These examples should provide you with a foundational understanding of integrating `acts_as_nested_set` with your forms. It’s crucial to remember that the gem is abstracting away the complexities of nested set management. The key is to focus on providing a meaningful user interface that translates to the underlying tree structure without directly exposing the `lft` and `rgt` attributes.

For further in-depth understanding, I strongly recommend reading “SQL for Smarties: Advanced SQL Programming” by Joe Celko, particularly for its extensive treatment of tree structures and their implementation in relational databases. Also, the documentation for the `acts_as_nested_set` gem itself is extremely comprehensive and provides further examples and explanations of its methods. Additionally, the classic book, "Refactoring Databases," by Scott Ambler and Pramod Sadalage, discusses in great detail how to handle data model modifications and migrations when incorporating structures like these in a live application and deserves consideration when implementing changes like the ones above to your production database.
