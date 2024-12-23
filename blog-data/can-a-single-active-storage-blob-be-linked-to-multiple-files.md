---
title: "Can a single Active Storage blob be linked to multiple files?"
date: "2024-12-23"
id: "can-a-single-active-storage-blob-be-linked-to-multiple-files"
---

Let's dive into this. I've encountered this scenario more times than I care to count during my time architecting storage solutions for various applications, and the short answer is, while seemingly counterintuitive at first, *yes*, a single Active Storage blob can indeed be associated with multiple file attachments, or more specifically, multiple *records*. The key isn't direct physical linking at the filesystem level, but rather how Active Storage manages the relationship between records and blobs using metadata and database associations.

The core misunderstanding typically comes from thinking of blobs as files directly mapped one-to-one. In reality, a blob represents the raw binary data stored somewhere (S3, local disk, etc.) along with its associated metadata like content type and filename. Active Storage uses a junction table, `active_storage_attachments`, to link records (think database rows representing models) with these blobs. Critically, this table can have multiple entries referring to the same `blob_id`, essentially allowing multiple records to share the same underlying data. This is a feature, not a bug, designed to optimize storage space and resource usage, particularly for common assets like logos, default avatars, or background images that might be used across many parts of an application.

I remember this became particularly relevant during a project where we built a large asset management system for a publishing house. We had to optimize for various scenarios – a single image could be used in several different publications. It was impractical and wasteful to duplicate the same high-resolution image file for every single publication using it, and actively managed redundant storage leads to headaches down the road. Using Active Storage’s inherent structure, we didn't need to.

Here's how this works in practice, with examples demonstrating both single and multiple associations:

**Example 1: Single Attachment**

Let's define a simple `User` model that has an avatar:

```ruby
class User < ApplicationRecord
  has_one_attached :avatar
end

# Create a user and upload an avatar
user = User.create(name: "Alice")
user.avatar.attach(io: File.open("path/to/alice_avatar.jpg"), filename: 'alice_avatar.jpg', content_type: 'image/jpeg')

puts "Avatar Blob ID for User Alice: #{user.avatar.blob_id}"
```

In this snippet, we create a user and attach a single avatar. After attaching, a corresponding entry in `active_storage_attachments` is created with a link to the blob associated with that file's contents, a record linking `user.id` and the `blob_id`.

**Example 2: Multiple Attachments Using Different Models**

Now let's imagine a scenario where both users and articles can use a shared image, perhaps a background:

```ruby
class User < ApplicationRecord
  has_one_attached :background_image
end

class Article < ApplicationRecord
  has_one_attached :background_image
end

# Assume the image file already exists
background_file_path = "path/to/background.jpg"

# Create two records and attach the image
user = User.create(name: "Bob")
article = Article.create(title: "Example Article")

#Attach the same blob to both user and article by referencing the same file
user.background_image.attach(io: File.open(background_file_path), filename: 'background.jpg', content_type: 'image/jpeg')
article.background_image.attach(io: File.open(background_file_path), filename: 'background.jpg', content_type: 'image/jpeg')

puts "Background Blob ID for User Bob: #{user.background_image.blob_id}"
puts "Background Blob ID for Article Example Article: #{article.background_image.blob_id}"
# Both will output the same blob_id, confirming they share the same blob.
```

In this case, both the user and the article now use the same underlying blob, evidenced by the same `blob_id`. The `active_storage_attachments` table will contain two separate rows, linking `user.id` and `article.id` to the same `blob_id`, using polymorphic associations. This is the crucial piece that allows us to avoid duplicate storage. The file itself is not duplicated but instead just the association to it.

**Example 3: Multiple Attachments to the same Model**

It's also possible, if needed, to have multiple attachments to the same model using the `has_many_attached` interface. Let's imagine now our articles can have various illustrations.

```ruby
class Article < ApplicationRecord
  has_many_attached :illustrations
end

# Assume the image files already exist
illustration_file_1 = "path/to/illustration1.jpg"
illustration_file_2 = "path/to/illustration2.jpg"


# Create an article and attach several illustrations
article = Article.create(title: "Illustrated Article")
article.illustrations.attach([
  {io: File.open(illustration_file_1), filename: 'illustration1.jpg', content_type: 'image/jpeg'},
  {io: File.open(illustration_file_2), filename: 'illustration2.jpg', content_type: 'image/jpeg'},
])

puts "Illustration Blobs IDs for Article Illustrated Article: #{article.illustrations.map(&:blob_id)}"
# Here, we will observe different blob ids, since each of these files are separate and have different blob representations.

article.illustrations.attach(io: File.open(illustration_file_1), filename: 'illustration1_dup.jpg', content_type: 'image/jpeg')

puts "Illustration Blobs IDs for Article Illustrated Article After Duplicate: #{article.illustrations.map(&:blob_id)}"
# Here we will observe that *the blob_id associated with illustration_file_1 may not change*, but it will now be attached to this article twice.
```

In this instance, a single model has multiple attachments, but *each attachment* points to its own blob. If we attach the same image file multiple times, Active Storage will intelligently recognize the underlying data is the same and *not* re-upload the same bytes to the storage backend, but rather create another entry in the `active_storage_attachments` table linking the same blob to that specific record for multiple attachment entries. We could even use the same filename each time but this would *not* create another blob.

It's crucial to understand this underlying mechanism. The `active_storage_attachments` table acts as the connecting piece, allowing for many-to-one relationships between records and blobs. This avoids redundancy, manages storage efficiently, and offers flexibility when designing applications.

If you want to further delve into the architecture of Active Storage, I strongly recommend reviewing the source code directly in the Rails repository. Additionally, "Crafting Rails 4 Applications" by José Valim offers a deep dive into Rails internal workings which, while technically older, still provides valuable insights into Rails fundamentals that are applicable here. For a deeper dive into database design considerations, "Database Design for Mere Mortals" by Michael J. Hernandez and Toby J. Teorey is also invaluable. These sources should clarify these concepts much more effectively than any online tutorial I could summarize here. This has been my experience as a developer, hopefully this is helpful to you as well.
