---
title: "How can I retain existing files in Rails Active Storage?"
date: "2024-12-23"
id: "how-can-i-retain-existing-files-in-rails-active-storage"
---

Alright, let’s talk about preserving those precious files when using Rails Active Storage – something I’ve had to navigate more times than I care to count, particularly during those large application migrations. It’s not always as straightforward as it appears. Essentially, the challenge boils down to ensuring your data isn't inadvertently purged during updates, association changes, or object deletions. It’s a crucial piece for reliable data management.

The default behavior of Active Storage, while practical for many scenarios, isn't inherently designed to hold onto data you've explicitly attached if the parent record's association is modified or if the record itself is destroyed. So, when you're dealing with existing files that shouldn't disappear, you need to be proactive. Here are a few techniques I've found to be particularly effective.

First and foremost, let's consider the case where you're updating an existing record and its attachments. Often, the issue isn't creating the initial attachment, but rather *modifying* an association. When a user (or process) changes an attached file, the default action is to delete the old one. This is usually desirable, but sometimes it’s not. What I often did in previous roles was to leverage `has_one_attached` or `has_many_attached` with specific callbacks. Let's look at `has_one_attached`, as it's typically a more common use case. Here’s an example:

```ruby
class User < ApplicationRecord
  has_one_attached :avatar, dependent: :purge_later

  before_destroy :purge_later_avatar

  def purge_later_avatar
    avatar.purge_later
  end
end
```

Here, we’ve introduced `dependent: :purge_later`. What this does is instruct Active Storage to schedule the deletion of the file asynchronously, rather than immediately upon association change or record deletion. Crucially, this will still delete the file, just not right away. The `before_destroy` callback and its method does the same thing, ensuring that even if the user record is deleted the `avatar` is purged later and the associated job is processed.  This approach provides a slight safety net by delaying the deletion, and if the deletion job fails for some reason, the file is still there in your storage until the jobs are retried or handled accordingly. It's a good starting point for non-critical files. For a comprehensive understanding of asynchronous operations and job processing in Rails, I recommend reviewing the Ruby on Rails Guides documentation, specifically on Active Job. It's a core concept in making Active Storage reliable, especially when dealing with large files.

However, this approach doesn't prevent deletion during updates or association changes. What if you want to retain the original file even when a new one is uploaded? You need a bit more control for that. This often happened in legacy systems where the history of uploaded data was required to be maintained.

For that, you can utilize a custom approach where you store the old file data and re-attach it if required. Here’s an example demonstrating this:

```ruby
class Document < ApplicationRecord
  has_one_attached :file

  before_update :preserve_previous_file

  def preserve_previous_file
    return unless file.attached? && file_changed?

    old_blob = file.blob
    new_file = self.file
    new_file.attach(old_blob)
    self.file = new_file
    
  end
  
  def file_changed?
     changes.key?('file_blob_id')
  end
end

```

In this snippet, before an update operation occurs, the `preserve_previous_file` method checks if a file is already attached *and* if it has been changed, by inspecting if the `file_blob_id` is present in the changes hash. If it's not, then we just move on, and this way, no file change means the previous file data is preserved, and the same file object is returned. If the file was changed, we essentially re-attach the old blob and reassign to self.file to trigger a new attachment and preserving the old file.

While this example directly copies the blob data (which works for the first iteration), you might want to move toward a versioning strategy when dealing with a more demanding scenario. For that you would want to implement a dedicated model to track the file changes.

Here's a more complex solution using a separate `AttachmentHistory` model for versioning:

```ruby
class Document < ApplicationRecord
  has_one_attached :file
  has_many :attachment_histories, dependent: :destroy

  before_update :create_file_version

  def create_file_version
    return unless file.attached? && file_changed?

    attachment_histories.create(file_blob: file.blob)
    self.file = file

  end

  def file_changed?
     changes.key?('file_blob_id')
  end

end

class AttachmentHistory < ApplicationRecord
  belongs_to :document
  belongs_to :file_blob, class_name: 'ActiveStorage::Blob'
end
```

Here, whenever the file attachment changes on an updated `Document`, we now create a new `AttachmentHistory` record, associating it with the old file blob using the `ActiveStorage::Blob` model. This creates a historical record of all versions. To restore a specific version, you'd need a separate function to re-attach the historical blob. This is especially helpful in cases where a history of file updates is crucial for audit logs or reverting to a previous state, which I faced during an application migration project where data integrity was paramount. For a deeper dive into data modeling techniques, and especially on versioning strategies, "Data Modeling Essentials" by Graeme Simsion is a great resource that provides a more formal approach to data structures and historical data management.

These are three strategies that I’ve found effective in handling file retention. While using `purge_later` for delayed deletion is useful in some scenarios, it isn't designed to preserve the data when you need to. By capturing the original blob and re-attaching, or storing version history, you achieve much greater control over Active Storage data. The key takeaway is that while Active Storage is powerful, default behavior sometimes needs a bit of manual steering to fit your exact needs, and I’ve found this flexibility is critical in robust real-world application development. Always remember to test your solutions thoroughly, particularly around deletion and updates in various scenarios, to avoid data loss. This is a lesson I had to learn the hard way during some rather challenging debugging sessions.
