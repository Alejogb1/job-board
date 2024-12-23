---
title: "How can I improve DRYness in Rails 7 image attachments?"
date: "2024-12-23"
id: "how-can-i-improve-dryness-in-rails-7-image-attachments"
---

Alright, let’s tackle this. I've certainly spent my share of time in the trenches dealing with image attachments in Rails projects, and it’s one area where repeating yourself can become surprisingly easy if you’re not vigilant. I’ve found that Rails 7, with its Active Storage improvements, gives us a lot of solid tools for this, but they need to be applied thoughtfully to maintain that crucial "don't repeat yourself" principle, or DRYness. My experiences, particularly with a past e-commerce platform, ingrained in me the necessity of this. We had numerous product categories, each with slightly different image requirements, and the mess that resulted before we refactored is a vivid reminder why we need to strive for DRY code.

The core problem, as I've observed, often lies in how attachments are declared and processed within models. If you're defining image variants, validations, or even presentation logic directly in each model, you'll very quickly notice the same code appearing over and over. The solution to this problem revolves around three central areas: abstracting attachment configurations, encapsulating common processing steps, and leveraging inheritance or composition.

Let's start with abstracting attachment configurations. Instead of declaring variations, validations, and custom processors directly in each model, it's more maintainable to define these parameters in reusable methods or configuration files. I typically prefer using modules for this purpose as they’re quite flexible. So, consider a module named `ImageAttachmentConfiguration`.

```ruby
# lib/image_attachment_configuration.rb
module ImageAttachmentConfiguration
  extend ActiveSupport::Concern

  included do
    def self.has_image_attachment(name, variations: {}, validations: {}, processors: {})
      has_one_attached name do |attachable|
        variations.each do |variant_name, variant_options|
          attachable.variant variant_name, variant_options
        end

        attachable.validate validations
      end

      define_method("#{name}_url") do |variant = nil|
        if self.send(name).attached?
          if variant
            Rails.application.routes.url_helpers.url_for(self.send(name).variant(variant).processed)
          else
             Rails.application.routes.url_helpers.url_for(self.send(name))
          end
        else
           nil
        end
      end

      # Apply any custom processors specified
      processors.each do |processor_name, processor_method|
          define_method("#{name}_#{processor_name}") do |*args|
             self.send(name).then(&processor_method).processed
          end
      end
    end

  end
end
```

This module introduces a `has_image_attachment` method which centralizes how you define attached images. It accepts the attachment's name (`name`), variations (`variations`), validations (`validations`), and processors (`processors`). Variations are applied via the `attachable.variant` method, validations are added using `attachable.validate`, and I've included a helper method to generate the url. Custom processors, if specified, are applied after the attachment is processed. Now, instead of repeating this configuration in every model, we can include this module and use `has_image_attachment`.

Next, consider a model for a product that uses these configurations.

```ruby
# app/models/product.rb
class Product < ApplicationRecord
  include ImageAttachmentConfiguration

  has_image_attachment :main_image,
    variations: {
      thumb: { resize_to_limit: [100, 100] },
      medium: { resize_to_limit: [300, 300] },
      large: { resize_to_limit: [600, 600] }
    },
    validations: {
        content_type: ['image/png', 'image/jpeg'],
        size: { less_than: 5.megabytes }
    },
    processors: {
      watermark: ->(image) {  WatermarkProcessor.call(image) }
    }

  has_image_attachment :secondary_image,
      variations: {
          thumb: { resize_to_limit: [50, 50]}
      },
      validations: {
          content_type: ['image/png', 'image/jpeg'],
          size: { less_than: 2.megabytes }
      }
end

# app/services/watermark_processor.rb
class WatermarkProcessor
  def self.call(image)
    #Logic to apply a watermark to the image. This is a placeholder.
    image
  end
end
```

Here, `Product` uses our `ImageAttachmentConfiguration` module. We’ve defined two attachments: `main_image` and `secondary_image`, each with specific variations, validations, and the `main_image` even uses a custom processor `watermark` which calls a WatermarkProcessor class. The important point is that all of this is now encapsulated, and easily applied to other models without code duplication. Each attachment now also has the methods `#main_image_url` and `#secondary_image_url`, and an additional `main_image_watermark` method that will return the image after applying the watermark.

The final technique I want to emphasize is inheritance. You can build abstract classes that predefine common attachments and then inherit from them in your actual models. This is particularly handy when you have a hierarchy of models.

For example, let's create a base class for media-heavy models:

```ruby
# app/models/concerns/media_base.rb
module MediaBase
  extend ActiveSupport::Concern

  included do
    include ImageAttachmentConfiguration
    has_image_attachment :cover_image,
        variations: {
            thumb: { resize_to_limit: [150, 150] },
            medium: { resize_to_limit: [400, 400] }
        },
        validations: {
          content_type: ['image/png', 'image/jpeg'],
          size: { less_than: 3.megabytes }
        }
  end
end

# app/models/article.rb
class Article < ApplicationRecord
  include MediaBase
  # ... additional properties or relationships
end

# app/models/blog_post.rb
class BlogPost < ApplicationRecord
  include MediaBase
  # ... additional properties or relationships
end
```

Here, both `Article` and `BlogPost` inherit the `cover_image` attachment, along with its predefined variations and validations, simply by including the `MediaBase` module. This avoids the repetition of that specific configuration, and the methods are automatically generated for them.

This approach ensures that if, for example, you need to modify the thumbnail size for all cover images, you'd only make that change in the `MediaBase` module instead of tracking down the individual declarations in every model that has a `cover_image`.

To solidify the technical understanding behind these methods, I recommend delving into the following resources: 'Agile Web Development with Rails 7' by Sam Ruby, Dave Thomas, and David Heinemeier Hansson is excellent for understanding Rails conventions and best practices. For the deeper dive into Active Storage, the official Rails documentation is essential. And, although this is often overlooked, the source code of Rails itself is a rich source of knowledge, specifically inspecting the Active Storage modules will reveal a lot about how the framework is designed to be extended.

These techniques, drawn from my own experiences building and maintaining scalable systems, are about maintaining long-term maintainability, readability, and of course, keeping your codebase nice and DRY. It's not just about writing code that works; it's about crafting code that's easy to work with, now and in the future. Remember, a little planning can save a lot of headaches later on.
