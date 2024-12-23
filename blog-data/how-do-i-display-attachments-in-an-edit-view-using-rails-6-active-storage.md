---
title: "How do I display attachments in an edit view using Rails 6 Active Storage?"
date: "2024-12-23"
id: "how-do-i-display-attachments-in-an-edit-view-using-rails-6-active-storage"
---

,  From my experience, handling attachments in edit views with Rails’ Active Storage can sometimes feel like navigating a labyrinth if you're not careful, especially when juggling existing files and new uploads simultaneously. I've seen it trip up even experienced developers, often leading to awkward user experiences, so I want to break down a robust and logical approach to this problem.

The core challenge is ensuring that your users can view existing attachments, remove them if desired, and also add new ones during the editing process, all while maintaining data integrity and a clean user interface. Let’s consider a scenario. A few years ago, I was working on a content management system for a small publishing house, and a big hurdle was precisely this: handling images and documents associated with articles during the editing phase. It involved multiple image uploads and pdf attachments that needed to be easily managed during revisions. We learned a few crucial things that I'll be sharing.

First, it’s essential to remember that Active Storage stores file information in a `has_many_attached` or `has_one_attached` relationship with your model. This means we need to iterate through these attachments when rendering the edit form. The `form_with` helper in Rails simplifies this process, and we can use it to our advantage.

Here’s a basic breakdown of how to structure your view:

1.  **Display Existing Attachments:** Loop through existing attachments associated with your record and display them, allowing the user to remove them.
2.  **Upload New Attachments:** Provide the necessary form fields for users to upload new files.
3.  **Handle Updates:** Ensure your controller handles both existing attachment removals and the addition of new ones during model updates.

Let's start with a view snippet. Assume we have a `Post` model with `has_many_attached :documents` and `has_one_attached :cover_image`. We’ll craft the edit form to handle these:

```erb
<%= form_with model: @post, local: true do |form| %>

  <div>
    <%= form.label :title, "Post Title" %>
    <%= form.text_field :title %>
  </div>

  <div>
    <%= form.label :cover_image, "Cover Image" %>
    <% if @post.cover_image.attached? %>
      <%= image_tag @post.cover_image, size: "200x150" %>
      <%= form.check_box :remove_cover_image, id: "remove_cover_image" %>
      <%= form.label :remove_cover_image, "Remove Cover Image" %>
    <% else %>
       <%= form.file_field :cover_image %>
    <% end %>
  </div>


  <div>
    <%= form.label :documents, "Documents" %>
    <% @post.documents.each do |document| %>
       <p>
         <%= link_to document.filename, rails_blob_path(document, disposition: "attachment") %>
        <%= form.check_box "remove_documents[]", { multiple: true }, document.id, false %> Remove
       </p>
    <% end %>
    <%= form.file_field :documents, multiple: true %>
  </div>

  <%= form.submit "Update Post" %>
<% end %>
```

This snippet displays existing images using the `image_tag` helper and documents using `link_to`. It also creates checkboxes to allow removing specific attachments, making sure to name them using the array syntax for easier processing in the controller. The file field is in place to accept new uploads. The removal logic is handled within the controller.

Now, for the controller action:

```ruby
def update
  @post = Post.find(params[:id])

  if params[:remove_cover_image] == '1'
    @post.cover_image.purge
  end

  if params[:remove_documents].present?
    params[:remove_documents].each do |document_id|
      @post.documents.find(document_id).purge
    end
  end
  
  if @post.update(post_params)
    redirect_to @post, notice: 'Post updated successfully.'
  else
    render :edit
  end
end

private

def post_params
  params.require(:post).permit(:title, :cover_image, documents: [])
end
```

In the controller's update method, we first handle the purging of any requested files. We check for the existence of `remove_cover_image` to handle single attachment removal and iterate through `remove_documents` when present. We then update the rest of the model’s attributes. Critically, for the params, we use `documents: []` to allow multiple file uploads at once via the file input field.

Let's add a little detail for cases where we want to display specific previews or have custom rendering needs for different document types, let's say we want to render image previews and not for pdfs. Here's the updated view snippet:

```erb
  <div>
    <%= form.label :documents, "Documents" %>
    <% @post.documents.each do |document| %>
       <p>
        <% if document.content_type.start_with?('image/') %>
          <%= image_tag document, size: "100x75" %>
        <% else %>
          <%= link_to document.filename, rails_blob_path(document, disposition: "attachment") %>
        <% end %>
        <%= form.check_box "remove_documents[]", { multiple: true }, document.id, false %> Remove
       </p>
    <% end %>
    <%= form.file_field :documents, multiple: true %>
  </div>
```

Now, based on the content type, we conditionally display images as previews and other documents via direct links.

Key takeaways:

*   **Use `form_with`:** It simplifies form building and data handling.
*   **Check for `attached?`:** Before rendering images or other attached assets to prevent errors.
*   **`Purge` method:** Utilize Active Storage’s `purge` method to delete attachments from the storage service.
*   **Parameter handling:** Use array syntax (`remove_documents[]`) and allow multiple file uploads through params sanitization (`documents: []`).
*   **Content Type checking:** Implement content type checks for differentiated rendering based on the attachment’s mime type.

Now, to deepen your understanding beyond these code snippets, I strongly recommend delving into the official Rails documentation for Active Storage, particularly the sections on file uploads and handling existing files. It's the most thorough guide and always a go-to resource. Additionally, “Agile Web Development with Rails 6” by Sam Ruby, Dave Thomas, and David Heinemeier Hansson is another valuable resource. It gives a great context on how forms work within the Rails ecosystem. I also advise reading the research papers that detail the underlying principles of cloud storage. Amazon’s S3 documentation can also provide deeper insight if your storage is S3-based. These references offer practical advice and in-depth insights beyond the surface-level implementation.

I've found that this approach has been quite reliable and flexible across different projects, and it simplifies a lot of the potential issues. It's a good starting point for anyone seeking to implement Active Storage within their Rails applications effectively. Handling file uploads well is critical for a polished user experience, and with these steps, you should be well-equipped to tackle this challenge in Rails.
