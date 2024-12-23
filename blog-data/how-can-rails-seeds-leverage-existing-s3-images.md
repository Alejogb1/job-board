---
title: "How can Rails seeds leverage existing S3 images?"
date: "2024-12-23"
id: "how-can-rails-seeds-leverage-existing-s3-images"
---

Let's dive straight in, shall we? The idea of efficiently seeding a rails application with existing s3 images isn't just a hypothetical exercise; it's a practical challenge I've encountered multiple times during development and maintenance phases. Back in the day, during a large e-commerce platform migration, we faced the exact scenario – thousands of product images already residing happily in our s3 bucket. Re-uploading them during each database reset was, frankly, ludicrous.

The core issue revolves around preventing needless data transfers. We've got our images stored remotely; re-downloading them only to re-upload them via active storage, or similar, during seeding operations is a wasteful process, both in terms of time and bandwidth. We need to essentially 'point' our database to the existing s3 image urls, rather than creating completely new uploads. Let's break down how to achieve this using a combination of active storage and ruby's powerful scripting capabilities.

First, we’ll need to ensure our models are properly configured to use active storage. I’m assuming, for this example, a typical setup where a `Product` model has a `thumbnail` attachment. Here's a quick reminder of what that might look like in your model:

```ruby
class Product < ApplicationRecord
  has_one_attached :thumbnail
end
```

The magic really starts in the `seeds.rb` file. We're going to avoid using `attach` directly in this scenario because that’s intended to perform the file upload process, which we are trying to avoid. Instead, we’ll leverage the ability of `ActiveStorage::Blob` to create blobs using existing data, which, in our case, happens to be available as s3 urls. Let's look at the first approach which is appropriate if your s3 paths follow a predictable pattern. Suppose all your image urls follow a consistent naming convention: `https://your-bucket.s3.your-region.amazonaws.com/product_images/{product_id}.jpg`. Here’s how you can effectively seed the database:

```ruby
# seeds.rb
require 'open-uri'

100.times do |i| # Generate 100 example products
  product = Product.create!(name: "Product #{i+1}", description: "This is product number #{i+1}")
  s3_url = "https://your-bucket.s3.your-region.amazonaws.com/product_images/#{product.id}.jpg"

  begin
    uri = URI.parse(s3_url)
    http = Net::HTTP.new(uri.host, uri.port)
    http.use_ssl = true if uri.scheme == 'https' # Enable ssl for https
    response = http.request_head(uri.path)

    if response.code.to_i == 200
       blob = ActiveStorage::Blob.create_before_direct_upload!(
          filename: File.basename(uri.path),
          content_type: "image/jpeg", # Or other correct type
        )
       product.thumbnail.attach(blob.signed_id)
    else
      puts "Image at #{s3_url} not found. Skipping."
    end
  rescue StandardError => e
     puts "Error processing #{s3_url}: #{e.message}"
  end

  puts "Product #{product.id} seeded with S3 image"
end
```

This script iterates through a number of products, builds the s3 url based on the created product's id, then it validates if the image exists using an http head request, creates a `blob` using `ActiveStorage::Blob.create_before_direct_upload!`, and attaches the `blob` to the product’s `thumbnail` using the signed blob id. Crucially, no actual file download is performed unless it's a head request to validate if the resource is available.

However, in practice, the above example’s consistent naming convention is not always the reality. Your images might not follow such a strict pattern, or they might be stored at locations controlled by a third party that don't align directly with your models' ids. A slightly more complex approach is required if you have a dataset of products with corresponding s3 image urls. Let’s say you have a json or yaml file storing this data. Here's a demonstration of reading a hypothetical `products.json` file, which holds product information along with their respective s3 image urls:

```json
[
  { "name": "Product 1", "description": "A great product.", "s3_url": "https://your-bucket.s3.your-region.amazonaws.com/images/image1.jpg" },
  { "name": "Product 2", "description": "Another excellent product", "s3_url": "https://your-bucket.s3.your-region.amazonaws.com/assets/img2.png" },
    { "name": "Product 3", "description": "The best product", "s3_url": "https://your-bucket.s3.your-region.amazonaws.com/public/image3.jpeg" }
]
```

Here's how you would adjust the `seeds.rb` file to handle it:

```ruby
# seeds.rb
require 'json'
require 'open-uri'

file_path = File.join(Rails.root, 'db', 'seeds', 'products.json')
products_data = JSON.parse(File.read(file_path))

products_data.each do |product_data|
  product = Product.create!(name: product_data["name"], description: product_data["description"])
  s3_url = product_data["s3_url"]

  begin
    uri = URI.parse(s3_url)
    http = Net::HTTP.new(uri.host, uri.port)
    http.use_ssl = true if uri.scheme == 'https'
    response = http.request_head(uri.path)

    if response.code.to_i == 200
      blob = ActiveStorage::Blob.create_before_direct_upload!(
        filename: File.basename(uri.path),
        content_type:  "image/jpeg", # Or dynamically determine based on file extension.
      )
      product.thumbnail.attach(blob.signed_id)
     else
       puts "Image at #{s3_url} not found. Skipping."
    end
  rescue StandardError => e
    puts "Error processing #{s3_url}: #{e.message}"
  end

  puts "Product #{product.id} seeded with s3 image"
end
```
This script does a similar job of making a head request, but this time, the s3 urls are pulled directly from the json file. This way you can easily control the seed data and the associated s3 image. If your json or yaml data includes information about content type, then you can use that instead of making a hard assumption as I did here for `image/jpeg`.

Finally, there’s a third scenario where you need to generate the s3 urls from a function or external service. This is more complex and often encountered with system migrations or when the bucket is not directly controlled by the application. Let's say we have a hypothetical service generating a map of `product_id` to an `s3_url`:

```ruby
# some_service.rb, assuming this is in your lib directory or similar.

class ImageService
  def self.fetch_image_url_for_product_id(product_id)
    # Imagine some complex logic or api call here.
    # Return s3 url or nil if there's no corresponding image.
    "https://your-bucket.s3.your-region.amazonaws.com/migrated_images/image_#{product_id}.jpg"
  end
end

```

The seeds.rb file in this scenario would look something like this:
```ruby
# seeds.rb
require 'open-uri'
require './lib/image_service' # Or wherever your service class resides

10.times do |i| # Let's do a smaller set this time
  product = Product.create!(name: "Product #{i+1}", description: "Another Example product #{i+1}")
  s3_url = ImageService.fetch_image_url_for_product_id(product.id)
    next if s3_url.nil? # Skip if image service provides nothing.

  begin
    uri = URI.parse(s3_url)
    http = Net::HTTP.new(uri.host, uri.port)
    http.use_ssl = true if uri.scheme == 'https'
    response = http.request_head(uri.path)

    if response.code.to_i == 200
     blob = ActiveStorage::Blob.create_before_direct_upload!(
       filename: File.basename(uri.path),
        content_type: "image/jpeg",  #Or again, set according to file extension or content type logic.
      )
     product.thumbnail.attach(blob.signed_id)
    else
      puts "Image at #{s3_url} not found. Skipping."
    end

   rescue StandardError => e
     puts "Error processing #{s3_url}: #{e.message}"
  end

  puts "Product #{product.id} seeded with s3 image"
end
```
In this case, you call a method of your hypothetical service, and the logic and code is encapsulated in the `ImageService` class instead of being directly hardcoded in the `seeds.rb` file.

In all these examples, the core principle is to use `ActiveStorage::Blob.create_before_direct_upload!` to construct the blob based on existing s3 resources and then to attach this blob to your active storage attachment. This avoids unnecessary file downloads and uploads, making your seed process far more efficient. Remember to always perform the necessary error handling and validations that the script needs, so it can handle various situations, such as the s3 image not existing or any other error during the execution of your code.

For a deeper understanding of active storage, I recommend reviewing the official Rails guides for active storage. To understand how `ActiveStorage::Blob` functions in more detail, referring to the Rails source code on github can be extremely beneficial. For network operations, "TCP/IP Guide" by Charles Kozierok provides a robust foundation on http protocol and networking concepts. Finally, if you are dealing with complex data formats like json or yaml, I’d also suggest looking into "Working with JSON" by Eric Goebelbecker or similar resources focused on data parsing. These resources, paired with practical hands-on experience, will make you proficient with seeding your database using existing s3 images.
