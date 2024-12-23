---
title: "Why is Carrierwave's S3 upload in Rails 7 so slow only on localhost?"
date: "2024-12-23"
id: "why-is-carrierwaves-s3-upload-in-rails-7-so-slow-only-on-localhost"
---

Alright, let’s tackle this. I’ve definitely seen this one pop up a few times over the years, often catching developers off guard. It’s the frustrating scenario where your Carrierwave-powered S3 uploads in a Rails 7 application are lightning fast in production but grind to a halt on your localhost. The problem isn’t inherently with Carrierwave itself, or necessarily even with your code directly. Instead, it usually boils down to a confluence of factors involving how your development environment simulates cloud storage interactions.

The primary culprit, in my experience, is the way your local setup handles S3 requests. When you’re developing locally, you’re generally not *actually* talking to Amazon S3. You’re typically relying on a service like `fog-aws`, `shrine`, or `mini_magick` (if you’re processing images), or some other mocking solution, to intercept those S3 calls and simulate responses. These mocks are great for isolating your application during development and avoiding unnecessary AWS costs, but they're often the source of slowdowns when dealing with significant data payloads.

Consider what has to happen: your upload process on localhost is likely being intercepted, the file (image, document, etc.) is being read from the temp folder, likely processed through some transformation layer (image resizing, for example), then ‘stored’ in memory or on your local disk in a simulated S3 bucket. That storage emulation itself can introduce latency, and the repeated read/write operations add up, especially when the mock is poorly optimized or if the file is particularly large. This is distinctly different from a streamlined pipeline to AWS S3, which is optimized for rapid handling of binary blobs over the network using HTTP/2, and advanced mechanisms for handling large uploads.

One issue I encountered a couple years back was a case where the developer had inadvertently configured their S3 credentials (used by Fog) to use the AWS "us-east-1" region when the AWS CLI on their localhost was defaulting to "us-west-2". This caused Fog to try resolving endpoints against the wrong region and would silently error, but it slowed down every request to a crawl while it tried and retried. It wasn't a *Carrierwave* issue exactly, but the result was slow uploads on localhost. We could pinpoint it by enabling detailed logging from the fog gem and then comparing the request URIs to the AWS region configured.

Another frequent problem, especially with large files, is an overreliance on synchronous processing in the upload pipeline. On your development machine, you might be running a single Puma or WEBrick process, effectively funneling all S3 interactions through this single thread. In contrast, a typical production setup uses multiple processes and worker threads to handle concurrent requests. Locally, that single thread can get bogged down by multiple slow synchronous operations while waiting for file manipulation. If you're resizing images using MiniMagick and doing it within your main thread, that’s guaranteed to slow down your uploads in your local server.

Here are a couple of examples illustrating these points, and some techniques for mitigating the slowness:

**Example 1: Poorly Optimized Mock**

Let's imagine that you're using `fog-aws` for S3 emulation, and a naive implementation. This is more representative of an inefficient mock implementation than a common issue for fog. But I have seen equivalent performance issues when a mock library is implemented incorrectly:

```ruby
# config/initializers/carrierwave.rb (example)
if Rails.env.development?
  CarrierWave.configure do |config|
    config.storage = :file # using local filesystem
    config.enable_processing = false # to bypass image processing for this demo
  end
else
  CarrierWave.configure do |config|
    config.storage = :fog
    config.fog_provider = 'fog/aws'
    config.fog_credentials = {
      provider:              'AWS',
      aws_access_key_id:     ENV['AWS_ACCESS_KEY_ID'],
      aws_secret_access_key: ENV['AWS_SECRET_ACCESS_KEY'],
      region:                ENV['AWS_REGION'] # Assuming this is set correctly, or use a specific one like 'us-east-1'
    }
    config.fog_directory = ENV['AWS_BUCKET_NAME']
    config.fog_public     = false
    config.fog_attributes = { cache_control: 'public, max-age=31557600' }
  end
end
```

If your application is configured with `:file` storage locally, each upload will save to disk, but this example shows the issue with a naive `fog` mock implementation. Let's pretend that the local fog instance is set up to only write one byte at a time. In production, the transfer rate will be at the network's maximum speed, but locally it’ll be horrifically slow:

```ruby
# pretend fog is doing something like this for each file (this is just for demonstration and is incorrect for fog)
def put_object(file_path, target_path)
  File.open(file_path, "r") do |file|
    while byte = file.read(1)
       # simulating poor performance of a mock
       File.write("local_disk/" + target_path, byte, mode: 'a')
       sleep(0.001)
    end
  end
end
```

This code shows what happens with a single file. It's only simulating one operation per byte and it's still much slower than what S3 would do. Imagine how this code would perform if each of the files is several megabytes. The mock implementation, if not optimized for efficiency, can dramatically affect the upload speed.

**Example 2: Synchronous Image Processing**

Let's consider a common scenario where we're resizing images directly in our uploader class. This will block the application thread. Note that `mini_magick` should *not* be used in the main thread, especially for production usage. This example shows how its use can slow down localhost uploads.

```ruby
# app/uploaders/image_uploader.rb (example)
class ImageUploader < CarrierWave::Uploader::Base
  include CarrierWave::MiniMagick

  version :thumb do
    process resize_to_fit: [100, 100]
  end

  def store_dir
    "uploads/#{model.class.to_s.underscore}/#{mounted_as}/#{model.id}"
  end
end
```

If you upload a large image, the `process resize_to_fit` operation will happen synchronously in the request/response cycle. On localhost, this operation may take a longer time to execute, causing your upload to appear very slow while your application waits for the resizing operation to complete and store to disk or the mock S3 bucket.

**Example 3: Asynchronous Processing with Background Jobs**

The solution to the previous example is to process those images in the background. Here's how we can do that with ActiveJob and a gem such as sidekiq or resque:

```ruby
# app/uploaders/image_uploader.rb (modified)
class ImageUploader < CarrierWave::Uploader::Base
  include CarrierWave::MiniMagick

  def store_dir
    "uploads/#{model.class.to_s.underscore}/#{mounted_as}/#{model.id}"
  end

  def process_version!(new_version)
    # this is just an example - real code should be more careful about version availability and data.
    ImageProcessorJob.perform_later(model.class.to_s, model.id, mounted_as, new_version)
  end
end

# app/jobs/image_processor_job.rb
class ImageProcessorJob < ApplicationJob
  queue_as :default

  def perform(model_class, model_id, mounted_as, version_name)
    model = model_class.constantize.find(model_id)
    uploader = model.send(mounted_as)
    uploader.send(:process_version!, version_name) # this will trigger re-upload if any transformations need to happen

  end
end
```

Here, the `process_version!` is delegated to an ActiveJob. This means the upload request will finish quicker, allowing your application to move on to the next request. The actual image processing will occur asynchronously through Sidekiq/Resque or another background processing system.

The key takeaways from this are to:

*   **Review your S3 mocking strategy:** Ensure your mocking solution is as lightweight as possible, and if you use a file system solution, see if using ramdisk can help. Sometimes simpler file upload mocks can perform better than complex s3 mocks in development.
*   **Defer heavy processing to background jobs:** Image processing, file transformations, and other time-consuming operations should be handled in the background to keep your main request/response cycle fast.
*   **Profile your application:** Use tools like the Rails Profiler or `rack-mini-profiler` to identify performance bottlenecks and address them accordingly.
*   **Verify your environment settings**: Ensure that the S3 credentials and region specified in your Rails application configuration are actually correct when using the fog gem.

For a more in-depth understanding of background processing in Rails, I highly recommend the official Rails guides on Active Job. For Carrierwave, the official gem documentation is thorough and useful. Also, reading up on network performance on the S3 storage layer itself can lead to a deeper understanding of what you are mocking in your application. A good resource for this would be "High Performance Browser Networking" by Ilya Grigorik. These three resources, the official guides, gem documentation and networking books, have been the most helpful to me in the past.

Finally, remember that your local environment is not a perfect representation of production, but it is important to get it as close as possible to avoid unexpected surprises. By tackling these common problems with well-defined mock setups and asynchronous processing, you'll find that your local development can be much smoother and less frustrating.
