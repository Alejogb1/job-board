---
title: "How to purge a file on attaching a new file in Rails 7?"
date: "2024-12-23"
id: "how-to-purge-a-file-on-attaching-a-new-file-in-rails-7"
---

,  The scenario of needing to purge an existing file when attaching a new one in Rails 7 using Active Storage is a surprisingly common hiccup, and I've certainly encountered it more than once in production environments. It's not something that's handled automatically by Rails, so you do have to be explicit about managing this lifecycle.

First, let's be clear that "purging" here, in the context of Active Storage, means completely removing the stored file from your chosen service (be it AWS S3, Google Cloud Storage, or a local disk). It’s more than just deleting the database association; we need the file gone from storage, otherwise, you'll be accumulating old, potentially large files, consuming unnecessary storage space and potentially impacting costs.

From my own experience, I recall building a rather ambitious image processing pipeline a few years back. Users could upload new versions of images, but we initially overlooked the crucial step of purging the old ones. We ended up with a mountain of obsolete images, and fixing it retrospectively was… less than pleasant. I learned a valuable lesson about handling object lifecycles thoroughly. So, let me share the key techniques and considerations based on what I've learned.

The fundamental principle revolves around managing the `ActiveStorage::Attached::One` or `ActiveStorage::Attached::Many` association that you’ve declared on your models. We need to tap into the lifecycle hooks provided by ActiveRecord. Here's how we typically go about it:

**The `before_save` Callback Approach**

This approach is probably the most straightforward and most often recommended. Essentially, you use an ActiveRecord callback to identify if the attachment has changed. If it has, you purge the previous attachment **before** the new one is saved.

Here's an example demonstrating this for a single attachment named `:avatar` on a `User` model:

```ruby
class User < ApplicationRecord
  has_one_attached :avatar

  before_save :purge_previous_avatar, if: :avatar_changed?

  private

  def purge_previous_avatar
    return unless avatar.attached?
    avatar.purge
  end

end
```

In this example:

1.  `has_one_attached :avatar`: declares that the model has one file associated through Active Storage.
2.  `before_save :purge_previous_avatar, if: :avatar_changed?`:  this line registers a `before_save` callback. `avatar_changed?` checks if the attached file has been modified. If it has, it invokes the `purge_previous_avatar` method.
3.  `purge_previous_avatar`: if an attachment exists (checked via `avatar.attached?`), the method calls the `purge` method, effectively removing the file from storage.

This method is effective for basic replacement scenarios.

**Using a Custom Setter Method**

For more control, especially when you need additional actions during the replacement, you can create a custom setter method for the attachment. This approach also avoids the potential for race conditions that could arise during concurrent operations, where the file might have been changed in between the check and purge. Here is how it might look like:

```ruby
class User < ApplicationRecord
  has_one_attached :avatar

  def avatar=(new_avatar)
    if avatar.attached?
      avatar.purge
    end
    super(new_avatar)
  end
end

```

Here, `super(new_avatar)` calls the default Active Storage setter, which processes the actual attachment. Crucially, the old attachment is purged _before_ the new one is processed, giving you more certainty about its state.

**Handling Multiple Attachments**

When dealing with multiple attachments (using `has_many_attached`), the pattern remains similar, but you will need to iterate over the collection:

```ruby
class Document < ApplicationRecord
  has_many_attached :files

  before_save :purge_previous_files, if: :files_changed?


  private

  def purge_previous_files
    return unless files.attached?

    # If all the files are being replaced (i.e. files = [newfile1, newfile2], we need to purge all)
    if files_changed?
        files_was.each { |file| file.purge}
      return
    end

    # Else we need to identify the removed files and purge them
    removed_files = files_was - files
      removed_files.each { |file| file.purge}

  end
end

```

In this case:

1.  `has_many_attached :files` sets up the model to have multiple file attachments.
2.  `before_save :purge_previous_files, if: :files_changed?` registers the callback that will run when the files have changed
3. `purge_previous_files` will either purge all files if the whole collection has been replaced, or it will iterate through a `removed_files` array to determine which files need to be purged. The `files_was` will contain previous values of the attached files.

**Key Considerations**

*   **Error Handling:** I've omitted detailed error handling in these examples for brevity, but in real-world scenarios, you'll want to gracefully handle cases where the purge fails (e.g., network errors when connecting to cloud storage). You might consider logging failed purges for review.
*   **Transaction Safety:** If purging is critical to your application’s integrity, ensure this process is done inside a database transaction. If the purge fails and the database operation succeeds, your database and storage will be out of sync.
*   **Performance:** If you have very large files, purging can be time-consuming, so you might consider offloading the purging operation to a background job to avoid blocking the main application thread.
*   **`files_was` and File Tracking:** The methods shown that rely on checking if the file has changed depend on Rails tracking changes on the association. This is usually reliable, but can sometimes fail.  The `files_was` method is incredibly valuable for identifying specifically which files have been removed or replaced in `has_many_attached` associations.
*   **Alternatives for bulk replacements:** For use cases where all the attachments are being replaced, it's beneficial to be able to use `.purge` method on the `has_many_attached` association, as it is usually a more performant way of purging all.

**Recommended Resources**

To further your understanding, I would recommend the following:

*   **"Agile Web Development with Rails 7"**: This is a comprehensive guide that will provide you with a deeper understanding of Active Storage and ActiveRecord callbacks.
*   **The official Rails documentation on Active Storage:** Always check the official documentation for the latest details and best practices. You will find details on the lifecycle methods, performance, and the configuration of Active Storage.
*   **"Patterns of Enterprise Application Architecture" by Martin Fowler**: Understanding application architecture is crucial for creating scalable and maintainable applications.
*   **Active Record Callbacks Guide**: Review the official ActiveRecord documentation, particularly on callbacks.  It is crucial that you fully understand the order in which these callbacks execute.

In summary, managing file purging when attaching new files is a critical task for resource management. Using `before_save` callbacks, custom setters, or variations of these techniques with `files_was` alongside a robust error handling strategy will help you build applications that are both functional and efficient. Remember that diligent management of your storage layer is essential for long-term application health.
