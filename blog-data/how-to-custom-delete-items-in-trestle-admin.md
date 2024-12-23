---
title: "How to custom delete items in Trestle Admin?"
date: "2024-12-23"
id: "how-to-custom-delete-items-in-trestle-admin"
---

Alright, let's talk about custom deletion in Trestle Admin. I've encountered this challenge quite a few times throughout my career, especially when dealing with complex relational data models where a standard delete action simply wouldn't cut it. It’s not always straightforward, and relying on Trestle’s defaults can lead to orphaned records and data integrity issues. I recall one particular project, a content management system for a large media house, where the standard delete functionality would have wreaked havoc on their publishing pipeline. We absolutely had to implement customized deletion logic.

Fundamentally, Trestle Admin, being built on top of ActiveAdmin, inherits a good portion of its underlying mechanics. However, it doesn't expose the full breadth of customization options immediately in the UI. Therefore, we typically need to dive into the underlying controller logic. The key here is understanding that Trestle gives you mechanisms to override its default actions. You’re not just stuck with what it gives you out of the box.

The standard `destroy` action, which Trestle wraps, usually performs a straightforward deletion from the database. However, a custom delete involves more nuanced behavior; think archiving instead of deletion, or handling related records in a specific way. We need to go beyond simply calling `destroy` on a model instance. To do this effectively, we often override the `destroy` method within the model itself, or, more powerfully, override Trestle’s delete action within the admin configuration.

Let's explore this with some examples, starting with the basic concept and escalating in complexity. Assume we have a `BlogPost` model and associated `Comment` models.

**Example 1: Soft Deletes with Model Overrides**

The simplest form of customization is implementing soft deletes using a gem like `paranoia`, or just a simple `deleted_at` timestamp. This approach alters the behavior of `destroy` at the model level, so it's applicable across the application, not just within Trestle. Let's look at an example using `deleted_at`.

```ruby
# app/models/blog_post.rb
class BlogPost < ApplicationRecord
  has_many :comments, dependent: :destroy # Keep for dependent deletions
  default_scope { where(deleted_at: nil) }

  def destroy
    update(deleted_at: Time.current)
  end

  def self.with_deleted
      unscoped
  end
end

# app/models/comment.rb
class Comment < ApplicationRecord
  belongs_to :blog_post
  default_scope { where(deleted_at: nil) }

  def destroy
    update(deleted_at: Time.current)
  end

    def self.with_deleted
        unscoped
  end
end
```

In this snippet, we redefine the `destroy` method in both `BlogPost` and `Comment`. Instead of direct deletion, we update the `deleted_at` column with the current timestamp. We also add a `default_scope` to filter out deleted records, ensuring that standard queries won’t return them, and a `with_deleted` scope for getting all results if needed.  The `dependent: :destroy` on the association still operates, but now it will trigger the modified destroy method on the `Comment` models.

**Example 2: Trestle Admin Action Overrides**

Now, let's say we want the ability to completely remove a blog post and its comments in specific administrative scenarios while still keeping soft-deletes the standard behavior. We can do this by explicitly overriding the `destroy` action within the Trestle configuration.

```ruby
# app/admin/blog_posts_admin.rb
class BlogPostsAdmin < Trestle::Resource
  # other configurations ...

  def destroy(instance)
    if params[:force_delete] == 'true'
        instance.comments.with_deleted.each(&:destroy!)
        instance.destroy!
    else
      instance.destroy
    end
    flash[:message] = 'Record destroyed.'
    redirect_to admin.path
  end

  controller do
    def destroy
      super
    end
  end

  form do |blog_post|
    # ... form fields
    actions do
        delete_button  label: 'Delete', confirm: 'Are you sure?', method: :delete
        link_to 'Force Delete', admin.path(force_delete: true, id: blog_post.id), method: :delete, class: 'btn btn-danger', data: {confirm: 'Are you absolutely sure you want to hard delete this and all associated comments? This cannot be undone.'}
    end
  end
end
```

Here, we've overridden the `destroy` method within the `BlogPostsAdmin` class. The action checks if a `force_delete` parameter is set. If it is, it removes the `blog_post` and all associated comments from the database using `destroy!`. This bypasses the soft-delete logic from Example 1, deleting the records permanently and we added a confirmation to the hard-delete link, since that operation cannot be undone. If `force_delete` isn't set, it executes the default `destroy` method on the model, which performs the soft delete. We added both actions to the form so they can be triggered by the user. We also use the controller to pass the request down from the rails controller to our overriden function.

**Example 3: Conditional Deletion and Auditing**

For our final example, let's introduce another layer of complexity: we want to prevent the deletion of posts that have been published and add audit logging.

```ruby
# app/admin/blog_posts_admin.rb
class BlogPostsAdmin < Trestle::Resource
    # other configurations ...

    def destroy(instance)
      if instance.published_at.present?
          flash[:error] = 'Published posts cannot be deleted. Archive instead.'
          redirect_to admin.path
          return
      end

      AuditLog.create(action: "delete", resource_type: "BlogPost", resource_id: instance.id, user: current_user)

        instance.comments.each(&:destroy)
        instance.destroy


        flash[:message] = 'Record deleted.'
        redirect_to admin.path
    end

    controller do
        def destroy
            super
        end
    end

    form do |blog_post|
        # ... form fields
        actions do
            delete_button  label: 'Delete', confirm: 'Are you sure?', method: :delete
        end
    end
  end

# app/models/audit_log.rb
class AuditLog < ApplicationRecord
    belongs_to :user, optional: true
end

# add `user_id` column to audit_logs table, optionally a username
# add a string column called action
# add a string column called resource_type
# add an integer column called resource_id
```

In this case, we've added some restrictions within the `destroy` method: If a blog post has been published, we won’t allow deletion, displaying an error message instead. If it's not a published post, we’ll audit that action, logging the event along with the user making the change to our `audit_log` table, before continuing with deletion. We again ensure to call `super` in our controller action in order to allow the request to pass to our overwritten method, without that the action would never be triggered.

These three examples illustrate a progressive approach to customizing deletion within Trestle. We started with model overrides for soft deletes, moved to Trestle-specific action overrides, and finished with conditional checks and audit logging. The core takeaway is this: Trestle provides the hooks, and we must provide the custom logic needed for the particulars of our application.

For further reading, I recommend diving into “Agile Web Development with Rails” by Sam Ruby, Dave Thomas, and David Heinemeier Hansson, to solidify your fundamental understanding of Rails models and controllers. The official ActiveAdmin documentation, while not a direct match, provides a good foundation for Trestle’s inner workings; explore the sections on resource customization and controller actions. For more advanced topics, explore patterns related to Command Query Responsibility Segregation (CQRS), which can drastically improve the clarity and maintainability of custom controller actions in complex admin systems. Finally, if you find yourself working with audit logs more often, there are many gems that can help streamline the process. These resources, combined with hands-on experimentation, should empower you to effectively manage custom deletion logic in Trestle Admin.
