---
title: "Why is Shrine returning 'undefined method `set_derivatives''?"
date: "2024-12-23"
id: "why-is-shrine-returning-undefined-method-setderivatives"
---

,  The "undefined method `set_derivatives'" error with Shrine, it’s a classic, and something I've definitely seen a few times – most recently back in 2019, if memory serves, when we were migrating a fairly large image processing pipeline. It usually points to a particular kind of configuration issue or an incompatibility in your Shrine setup, specifically around derivatives processing. It's not a bug *per se*, but a clear indicator that something isn't wired up correctly.

Essentially, the `set_derivatives` method is part of Shrine's derivatives functionality, which allows you to create different versions or 'derivatives' of an uploaded file. Think thumbnails, resized images, or even different video encodings. If Shrine can't find that method, it’s because it hasn't been properly extended with the necessary processing module. It’s most often related to the attachment and its processing definitions, which are crucial for the overall behavior.

To be more precise, the error usually crops up when you try to use the `derivatives` plugin, specifically when trying to assign a processed version to a derivative key (like `:thumbnail` or `:medium`). This isn't an issue with the core Shrine library; instead it's almost always a sign that you haven't explicitly included the *derivatives* module with your Shrine attachment. It’s quite simple to miss, especially when first getting to grips with the library.

Here's how to break it down and where things are likely going wrong:

1.  **Missing Plugin Inclusion:** The most common reason is simply that you haven't loaded the `derivatives` plugin. This plugin provides the functionality required for the `set_derivatives` method, allowing Shrine to process and store multiple versions of the same attachment. If this step is missed, then the underlying code won't have the proper functions or setup.

2.  **Incorrect Attachment Definition:** Another potential point of failure is the definition of your Shrine attachment itself. Are you loading the derivatives plugin within the attachment's definition? It’s easy to declare your attachment but forget about the required processing instructions.

3.  **Mismatched Shrine Version or Plugins:** Although less common, there's also a chance that there’s an incompatibility issue with Shrine itself or with the versions of plugins you’re using. This is less probable but still worth investigating if you've verified all other configuration steps.

Let me illustrate with some code examples, which I hope will clarify things.

**Example 1: Basic Scenario - Missing Plugin**

Let's assume you have an uploader setup:

```ruby
require "shrine"
require "shrine/storage/memory"

Shrine.storages = {
    cache: Shrine::Storage::Memory.new,
    store: Shrine::Storage::Memory.new,
}

class ImageUploader < Shrine
  # missing plugin!
end

class Article
  include Shrine::Attachment(:image)
end

article = Article.new
article.image = File.open("path/to/image.jpg") # Placeholder, could be any image file
# The following would raise the error
# article.image.derivatives[:thumbnail]
```

In this case, the `ImageUploader` is missing `plugin :derivatives`. When you call the `derivatives` method on the attachment, it attempts to access functionality that was never added. The `set_derivatives` method that causes the error is not defined on the class.

**Example 2: Correct Implementation**

Here’s the fix, explicitly including the required plugin:

```ruby
require "shrine"
require "shrine/storage/memory"
require "image_processing/vips" # Or other suitable processor


Shrine.storages = {
    cache: Shrine::Storage::Memory.new,
    store: Shrine::Storage::Memory.new,
}

class ImageUploader < Shrine
  plugin :derivatives, create_on_promote: true
  plugin :processing # Need this if you will be using Image Processing

  process(:store) do |io, context|
    {
      original: io,
      thumbnail: ImageProcessing::Vips
                     .source(io)
                     .resize_to_limit!(100, 100)
    }
  end

end


class Article
  include Shrine::Attachment(:image)
end

article = Article.new
article.image = File.open("path/to/image.jpg") # Placeholder, could be any image file
thumbnail = article.image.derivatives[:thumbnail]
# This will work correctly now
```

Notice how I've added `plugin :derivatives` to `ImageUploader`. Also, I’ve included an example using `ImageProcessing::Vips` to show a typical image processing workflow and make use of the `process` block. In the `process(:store)` method I’m telling Shrine that after the file is uploaded to the store it should produce a `:thumbnail` version with the result from the Vips image processor.  This version is now correctly defined and will not result in the `undefined method` error when requested.

**Example 3: An Incorrect Attempt at Derivatives Processing**

This final example highlights a subtle but common error – trying to use a derivative directly in the promote function instead of the `process` function, or not using a processor at all.

```ruby
require "shrine"
require "shrine/storage/memory"


Shrine.storages = {
    cache: Shrine::Storage::Memory.new,
    store: Shrine::Storage::Memory.new,
}

class ImageUploader < Shrine
  plugin :derivatives, create_on_promote: true


  def generate_derivatives(original)
      { thumbnail: original }
  end

end


class Article
  include Shrine::Attachment(:image)
end

article = Article.new
article.image = File.open("path/to/image.jpg") # Placeholder, could be any image file
# This will raise an error during the upload process.
```

Here, I defined `generate_derivatives` instead of the `process` block. The `generate_derivatives` method is only meant to modify file data once it's already been uploaded to the store. We should instead define processing instructions in the process block to have access to the original file and create processed derivatives at upload time. The derivative in this case is not processed in any way, and while that wouldn't cause an error directly, the absence of a processor is usually the cause of this error in production environments as the required `set_derivatives` call is not made.

For deeper understanding, I recommend reading “The Well-Grounded Rubyist” by David A. Black, which covers the intricacies of Ruby metaprogramming, which can be very helpful when debugging such dynamic issues as this. For more specific details about Shrine, the library's official documentation, particularly the section on plugins and derivatives, is invaluable. Also, explore the ImageProcessing gem documentation if you're dealing with image or video processing, as these libraries often intertwine in the context of Shrine.

In summary, the "undefined method `set_derivatives`" error almost always points to a misconfiguration regarding Shrine's derivatives plugin. Ensure you have included it in your uploader definition and are using the `process` block in your upload class to define the required operations. With careful attention to these details, you should be able to get derivatives processing up and running seamlessly.
