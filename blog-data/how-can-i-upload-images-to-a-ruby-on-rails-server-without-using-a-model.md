---
title: "How can I upload images to a Ruby on Rails server without using a model?"
date: "2024-12-23"
id: "how-can-i-upload-images-to-a-ruby-on-rails-server-without-using-a-model"
---

Alright,  I've seen this particular challenge crop up more times than I care to count, especially when you're dealing with scenarios that don't quite fit the usual model-centric rails paradigm. The key here is understanding that while rails often nudges you towards database-backed models, it doesn't force you. Uploading images without a model is perfectly achievable, and quite frankly, can be more efficient in certain cases, particularly with smaller projects or specific API endpoints that aren't directly linked to persistent data entities.

Essentially, we're bypassing the active record layer and working directly with the file upload functionality provided by rack, the underlying server interface of rails. This implies we'll be handling the multipart/form-data content ourselves. My experience working with a legacy system that had a custom file management layer highlighted the importance of understanding the underlying mechanisms— it saved us a lot of refactoring headaches.

The first thing to grasp is how rails receives file uploads. When a user sends a file, it arrives as part of the params hash, with the uploaded file details contained within a `ActionDispatch::Http::UploadedFile` object. This object encapsulates information about the file, like its name, original name, type, and a temporary location on the server.

Here's a basic example within a rails controller action:

```ruby
  # app/controllers/uploads_controller.rb
  class UploadsController < ApplicationController
    skip_before_action :verify_authenticity_token # remove this in prod unless you implement CSRF protection another way

    def create
      uploaded_file = params[:file]

      if uploaded_file.present?
        file_path = Rails.root.join('public', 'uploads', uploaded_file.original_filename)
        File.open(file_path, 'wb') do |file|
          file.write(uploaded_file.read)
        end
        render json: { message: 'File uploaded successfully', filename: uploaded_file.original_filename }, status: :created
      else
        render json: { error: 'No file uploaded' }, status: :bad_request
      end
    end
  end
```

This snippet does a straightforward file upload. It retrieves the file from the params, constructs a path within the public/uploads directory, opens a file handle, and writes the content to it. Critically, this requires us to skip csrf token verification which you absolutely should address if planning on using this method within a publicly facing app. The skip before action is for demonstration purposes only.

Now, this approach is quite basic. In a real-world scenario, we’d need more sophistication. For instance, we might want to rename files to avoid name collisions, check file types and sizes, and possibly move them to cloud storage. Here’s a more advanced example incorporating filename sanitization and a basic check on the content-type:

```ruby
  # app/controllers/uploads_controller.rb
  class UploadsController < ApplicationController
    skip_before_action :verify_authenticity_token

    def create
      uploaded_file = params[:file]

      if uploaded_file.present?
        if !uploaded_file.content_type.starts_with?('image/')
            return render json: { error: 'Invalid file type, only images are allowed'}, status: :unprocessable_entity
        end

        sanitized_name = SecureRandom.uuid + File.extname(uploaded_file.original_filename).downcase
        file_path = Rails.root.join('public', 'uploads', sanitized_name)
        File.open(file_path, 'wb') do |file|
          file.write(uploaded_file.read)
        end
         render json: { message: 'File uploaded successfully', filename: sanitized_name }, status: :created
      else
         render json: { error: 'No file uploaded' }, status: :bad_request
      end
    end
  end
```

In this adjusted code, we've added a content-type check to ensure that only image files are accepted. We’ve also replaced the original filename with a uuid, ensuring that no two files have the same name, to prevent overwriting already uploaded files. The filename sanitization, while basic, is an important first step in securing your application. Again, skipping the CSRF token check is only advised in this case for demonstration.

For very large files or asynchronous processing of uploads, directly writing to disk might not be the best approach. In that case, consider using a background job processing system like sidekiq or delayed_job to handle the actual file processing. The rails controller can receive the upload and delegate the handling to the background process. Here’s a conceptual snippet that incorporates background jobs:

```ruby
  # app/controllers/uploads_controller.rb
  class UploadsController < ApplicationController
    skip_before_action :verify_authenticity_token

    def create
        uploaded_file = params[:file]

        if uploaded_file.present?
           UploadFileJob.perform_later(uploaded_file.tempfile.path, uploaded_file.original_filename)
           render json: { message: 'File upload started' }, status: :accepted
        else
          render json: { error: 'No file uploaded' }, status: :bad_request
        end
    end
  end
```

```ruby
# app/jobs/upload_file_job.rb
class UploadFileJob < ApplicationJob
  queue_as :default

  def perform(temp_file_path, original_filename)
    # add any checks needed here

    sanitized_name = SecureRandom.uuid + File.extname(original_filename).downcase
    file_path = Rails.root.join('public', 'uploads', sanitized_name)
    File.open(file_path, 'wb') do |file|
      file.write(File.read(temp_file_path))
    end

    # potentially trigger additional actions, like image resizing, notification, etc.
  end
end
```

In this scenario, `UploadFileJob` handles the actual processing, allowing your controller to respond almost immediately, and importantly, the large upload won’t tie up your web server process. The job runs asynchronously and uses the temporary file’s path. It's a good practice to use temp files so that we are dealing with a path instead of potentially very large upload in memory. `ActionDispatch::Http::UploadedFile` objects contain a temp file path you can use.

For deeper understanding of the rack internals and how file uploads work, I’d highly recommend reading “The Rack Interface” section of the “The Architecture of Open Source Applications” book available online. Also, check out the rails guides regarding ActionDispatch, specifically the section concerning file uploads, which are very detailed and informative. Finally, the source code itself is incredibly informative; digging into the `ActionDispatch::Http::UploadedFile` class itself in the rails source will provide an in-depth look at how it operates.

In summary, uploading images without models is about directly interacting with the request parameters and the file system or background jobs. It gives you more control but at the cost of having to manage more details manually. Choose this method when your specific use case warrants it, and always remember to include security considerations. It’s definitely an approach I’ve employed multiple times, and when done correctly, it can be quite effective.
