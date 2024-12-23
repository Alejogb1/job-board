---
title: "How do I delete records from Active Storage blobs?"
date: "2024-12-23"
id: "how-do-i-delete-records-from-active-storage-blobs"
---

Okay, let's tackle the tricky, and surprisingly common, issue of deleting records from active storage blobs. It's not as straightforward as you might initially think, especially when considering the various potential failure points in a production environment. I’ve certainly had my share of late-night debugging sessions over this very topic.

The core issue arises from the relationship between your application's database and the actual storage of files, be that on a local disk, amazon s3, google cloud storage, or any other provider. Active storage provides a convenient layer, but we must be mindful of the underlying processes. Simply deleting a database record might leave orphaned blobs, consuming storage space and potentially creating a security risk. Conversely, deleting blobs without correctly updating or removing database associations can lead to application errors.

Fundamentally, active storage blobs are not automatically deleted when associated records are destroyed. This behavior is intentional to prevent accidental data loss. Think of it: a user might casually delete a record while perhaps intending to keep the associated image. We, therefore, need to be explicit and methodical in our blob deletion strategy.

Let's break down the necessary steps, accompanied by some code examples illustrating common use cases:

**The Core Mechanism: Direct Blob Deletion**

The most basic approach involves directly deleting the blob. This can be done using the `purge` method on the active storage attachment or its underlying blob. When you call `.purge` it both removes the blob from storage *and* removes the associated database record from the `active_storage_blobs` table. It's crucial to use this when you absolutely want the blob and it's data *gone*.

Here's a simple example, let's say we have a `User` model that has an attached avatar:

```ruby
class User < ApplicationRecord
  has_one_attached :avatar
end

# Example Usage:
user = User.find(1)

# To remove the blob directly and the record
user.avatar.purge
# At this point, the image and related database entries are gone
```

In this code, after running `user.avatar.purge`, the actual file stored in s3 or whatever backing storage you use is removed, and the relevant row in the `active_storage_blobs` table is deleted. Importantly, the `active_storage_attachments` row is also removed as this records the link between a record and a blob.

**Background Deletion: Ensuring Scalability**

In practice, deleting a large number of blobs synchronously can be detrimental to your application's performance. Especially when dealing with external storage providers, the time taken to remove the blobs can cause significant bottlenecks. Thus, it's often better to offload these deletion tasks to a background job.

Here's how you can implement that, utilizing the powerful `ActiveJob` framework:

```ruby
# app/jobs/delete_blob_job.rb
class DeleteBlobJob < ApplicationJob
  queue_as :default

  def perform(blob_id)
    blob = ActiveStorage::Blob.find_by(id: blob_id)
    blob&.purge
  end
end


# Example usage within your application logic
user = User.find(1)

if user.avatar.attached?
  DeleteBlobJob.perform_later(user.avatar.blob.id)
  user.avatar.detach # detach attachment from the record
end

user.destroy # Delete the record after queuing blob deletion.
```

In the example, `DeleteBlobJob.perform_later(user.avatar.blob.id)` pushes the deletion task onto the background queue, allowing your application to proceed without delay. The `.detach` call removes the association. If you were going to delete the `user` immediately, and only needed to delete the blob because the `user` was deleted, you could include the `purge` call directly within the callback for `before_destroy` if you are careful about only deleting in the context of a fully deleted model. I've omitted this to focus on demonstrating asynchronous deletion.

**Batch Deletion for Efficiency**

When dealing with many records and their associated blobs, you'll find that looping through each one and scheduling individual jobs can be inefficient. To address this, implement batch deletions. This involves identifying all the relevant blobs and deleting them in bulk, preferably in the background.

Let's consider the scenario where we want to delete all users older than a certain age *along* with their avatars:

```ruby
# In your service or relevant module:
def batch_purge_old_users(cutoff_age)
    User.where('created_at < ?', cutoff_age.years.ago).find_each do |user|
        if user.avatar.attached?
          DeleteBlobJob.perform_later(user.avatar.blob.id)
          user.avatar.detach
        end
        user.destroy
    end
end

# Example Usage:
cutoff = 60
batch_purge_old_users(cutoff)

```

Here, `.find_each` helps process records in chunks, avoiding excessive memory usage, especially when dealing with large datasets. We then queue the asynchronous deletion and detach. This approach is highly scalable and ensures that your application remains responsive.

**Important Considerations and Recommendations**

Several crucial points are worth considering:

1.  **Error Handling**: Background jobs are subject to failure. You need robust error handling to retry failed deletions or to alert administrators to manual cleanup if necessary. Monitoring systems, like prometheus or others, are vital to identify and address any issues arising in these processes. Look into `ActiveJob::Base#retry_on` for examples on how to gracefully handle errors.
2.  **Storage Provider Consistency**: Always verify that the storage provider is consistent after a delete operation, especially with external systems like s3, or GCP Cloud Storage. Occasionally, there might be a delay, and relying on assumptions can be problematic.
3. **Cascading Deletes:** Be careful with database relationships. If your database is set up to cascade deletes, deleting a parent record will delete child records (like the attachment). However this does not delete the blob, so you might still end up with orphaned blobs if you do not carefully handle this process via a callback, or directly delete the record.
4. **Database Schema:** It's essential to review your database schema, specifically regarding `active_storage_attachments` and `active_storage_blobs`, to understand the precise relationships between your records and stored files.

**Recommended Resources**

For more detailed and comprehensive knowledge, I would recommend the following resources:

*   **"Agile Web Development with Rails 7"** by Sam Ruby, David Bryant Copeland, and Dave Thomas (or the latest equivalent version). This book provides an excellent foundational understanding of active storage and its interaction with the rails framework.
*   **The official Ruby on Rails documentation** itself should be your primary source of truth. Refer to the active storage documentation for deep-dives on specific areas, including deletion processes.
*   **"Database Internals"** by Alex Petrov for a more thorough understanding of how databases function, this can be especially useful when understanding the nuances of cascading deletes and the associated risks.
* **"Designing Data-Intensive Applications"** by Martin Kleppmann, this book provides excellent insights into the challenges associated with data management and distributed system design that will inform how you write code that works in a production environment.

In summary, correctly deleting blobs in active storage is a combination of understanding its mechanisms, using background jobs for scale, and carefully planning data deletion operations. It is not overly complex, but attention to detail and a clear methodology will always be required. Always test thoroughly, and you'll avoid issues down the line.
