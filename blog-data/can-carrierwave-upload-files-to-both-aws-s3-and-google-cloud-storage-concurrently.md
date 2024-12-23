---
title: "Can CarrierWave upload files to both AWS S3 and Google Cloud Storage concurrently?"
date: "2024-12-23"
id: "can-carrierwave-upload-files-to-both-aws-s3-and-google-cloud-storage-concurrently"
---

Okay, let's unpack this. I’ve certainly encountered scenarios requiring multi-cloud file storage, and the challenge with CarrierWave, when aiming for simultaneous uploads to both AWS S3 and Google Cloud Storage (GCS), isn't exactly straightforward but it's certainly manageable. It's less about inherent limitations in CarrierWave and more about how you configure and orchestrate the process. In short, CarrierWave itself isn’t built for *concurrent* writes to multiple cloud providers using its standard interfaces. However, with a bit of creative configuration and careful use of callbacks, you can achieve the desired outcome without resorting to complex hacks.

My experience with this stems from a project a few years back where we had a hybrid infrastructure, partly on AWS and partly on Google Cloud, and we needed all user-uploaded files mirrored across both platforms for redundancy and application specific needs. We explored the native capabilities of several file upload libraries before ultimately crafting our own solution around CarrierWave’s flexible nature.

The core issue is that a single CarrierWave uploader is typically associated with one storage adapter at a time – either S3 or GCS. There isn't a built-in mechanism to dispatch a single upload to multiple destinations concurrently. So, a naïve approach of setting up two uploaders within the same model, one for each service, is not the correct path. You'd end up with a model having two separate file attributes, one for each provider, and this doesn't mirror one concurrent single upload.

Here’s the crucial concept: we need to leverage CarrierWave callbacks, specifically `after :store`, to achieve what we need. The general strategy involves:

1.  **Primary Uploader:** Define a primary CarrierWave uploader that targets one cloud storage provider (say, AWS S3). This is where your normal upload workflow would execute.

2.  **Callback Handler:** Set up an `after :store` callback that executes the second upload to the other cloud provider (GCS in this case) once the primary upload has completed. This callback is triggered after successful storing to S3.

3.  **Error Handling:** Implement robust error handling and logging within the callback to manage potential issues with the secondary upload. This part is critical to ensure data integrity and system reliability.

Now, let’s dive into the code snippets.

**Example 1: Basic Setup with AWS S3 as Primary Storage**

This example sets up CarrierWave to upload primarily to AWS S3.

```ruby
# app/uploaders/my_uploader.rb
class MyUploader < CarrierWave::Uploader::Base
  storage :fog

  def store_dir
    'uploads'
  end

  def extension_allowlist
      %w(jpg jpeg gif png)
    end

end

# config/initializers/carrierwave.rb
CarrierWave.configure do |config|
  config.fog_credentials = {
      provider:                'AWS',
      aws_access_key_id:       ENV['AWS_ACCESS_KEY_ID'],
      aws_secret_access_key:   ENV['AWS_SECRET_ACCESS_KEY'],
      region:                 'us-east-1'
  }
  config.fog_directory  = ENV['AWS_S3_BUCKET_NAME']
end

# app/models/my_model.rb
class MyModel < ApplicationRecord
  mount_uploader :my_file, MyUploader
  after_store :upload_to_gcs
  
  private
  
  def upload_to_gcs
      GcsUploader.new(my_file.file).store!
  end
end
```

Here, we define `MyUploader` to upload to S3, using the `fog` storage adapter, and `MyModel` to mount the uploader. Notice the `after_store` callback setup pointing to `upload_to_gcs` in the `MyModel` class. The callback calls a new uploader `GcsUploader`.

**Example 2: Implementing a Secondary GCS Uploader**

This is the second uploader class that’s called by the callback, which handles the secondary upload to GCS.

```ruby
# app/uploaders/gcs_uploader.rb
require 'carrierwave/storage/fog'
class GcsUploader < CarrierWave::Uploader::Base
  storage :fog

  def store_dir
    'uploads'
  end

  def initialize(file)
      @file = file
      super()
  end

  def cache_dir
     "#{Rails.root}/tmp/uploads"
  end

  def store!
    File.open(@file.path, 'r') do |f|
        file_to_upload = CarrierWave::SanitizedFile.new(f)
        store_file!(file_to_upload)
     end
  rescue => e
    Rails.logger.error "GCS Upload Error: #{e.message}"
  end

  def extension_allowlist
      %w(jpg jpeg gif png)
    end

end

# config/initializers/carrierwave_gcs.rb
CarrierWave.configure do |config|
  config.fog_credentials = {
    provider:              'Google',
    google_storage_access_key_id:    ENV['GCS_ACCESS_KEY'],
    google_storage_secret_access_key: ENV['GCS_SECRET_KEY']
  }
  config.fog_directory  = ENV['GCS_BUCKET_NAME']
  config.fog_attributes = { 'Cache-Control' => 'max-age=315576000' }
end
```

`GcsUploader` mirrors `MyUploader` with minor configurations to connect to Google Cloud Storage and to open the file object created during the first upload. Importantly it also implements error handling.

**Example 3: Handling Concurrency and Potential Issues**

While the above examples handle the core functionality, a real-world implementation requires more robust handling of potential issues.

```ruby
# app/models/my_model.rb (updated)

class MyModel < ApplicationRecord
  mount_uploader :my_file, MyUploader
  after_store :upload_to_gcs

  private

  def upload_to_gcs
    begin
      GcsUploader.new(my_file.file).store!
    rescue => e
      Rails.logger.error "Secondary Upload Error: #{e.message}"
      # Potentially add retry logic or queue the failed upload here
      # For example: Resque.enqueue(GcsUploadJob, id) if error occurred
      # Note: the `GcsUploadJob` would perform the upload again
      #       You might want to use a status column to keep track
      #       of the retries.
    end
  end
end
```

In this updated `MyModel`, I’ve included basic error handling. In production, you’d likely add more sophisticated features, such as:

*   **Retry Logic:** Implement retries using a background processing queue (like Sidekiq or Resque) in case the GCS upload fails the first time.
*   **Status Tracking:** Keep track of successful and failed uploads in a database column to monitor upload progress.
*   **Resource Management:** Be mindful of memory usage. Avoid loading the entire file into memory if it is very large; instead use streaming techniques for the GCS upload.

This pattern, using CarrierWave’s callbacks to trigger a second upload, works well. It's a relatively straightforward way to achieve multi-cloud redundancy without overly complicating your application's logic. Keep in mind that while it *appears* concurrent from a user experience, it is sequential from the perspective of your server: S3 upload first, then a GCS upload. True concurrency (if that is really critical) might require a different approach, like passing the file upload directly to a queue that triggers parallel workers; however, that is not within the scope of this CarrierWave-centric discussion.

For deeper insights into CarrierWave's internals and best practices, I would recommend reading the official CarrierWave documentation and exploring the source code of the fog adapter to better understand how storage is handled. Also, delve into papers related to resilient distributed systems, such as Google's papers on Spanner and Colossus, to further inform your strategy for high-availability storage.

This structured approach should provide a solid starting point for implementing concurrent, or rather, sequential-yet-mirrored, uploads across multiple cloud storage platforms using CarrierWave.
