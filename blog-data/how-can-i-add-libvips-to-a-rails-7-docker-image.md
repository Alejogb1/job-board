---
title: "How can I add libvips to a Rails 7 Docker image?"
date: "2024-12-23"
id: "how-can-i-add-libvips-to-a-rails-7-docker-image"
---

Alright, let's tackle this. Getting `libvips` working smoothly in a Rails 7 Docker environment can definitely feel like a bit of a puzzle sometimes, but it's a common enough issue. I've had my fair share of encounters with image processing dependencies and containerization, so let me walk you through the process based on what I've found works well. We're going to focus on a solid, repeatable approach rather than quick hacks.

The core challenge, as you might have guessed, revolves around ensuring that `libvips` and its development headers are present during the image build phase, and then that the shared libraries are also available at runtime. Rails, with its reliance on gems like `ruby-vips`, assumes these prerequisites are in place. Let’s break this down into a few key areas.

Firstly, we need a Dockerfile that sets up the correct environment. I’ve seen many variations, and the one I’ve consistently returned to relies on a multi-stage build. This allows us to keep our final application image relatively small and uncluttered by build-time dependencies. This separation is vital for security and for reducing image sizes, which consequently speeds up deployments.

Here’s how that looks:

```dockerfile
# Stage 1: Builder image for dependencies
FROM ruby:3.2-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    libvips-dev

# Set the working directory for the next commands
WORKDIR /app

# Copy over Gemfile and Gemfile.lock to cache bundle dependencies
COPY Gemfile Gemfile.lock ./

# Bundle install, this will populate the vendor directory with dependencies.
RUN bundle install --jobs=4 --retry=3


# Stage 2: Final application image
FROM ruby:3.2-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libvips

# Set the application's working directory again
WORKDIR /app

# Copy built gems from the builder image
COPY --from=builder /app/vendor ./vendor

# Copy over our application code from the root of the local build context into the app directory
COPY . .

# Expose the default Rails port
EXPOSE 3000

# Set the user to rails for better container security
USER rails

# Set the command to execute when a container is created
CMD ["bundle", "exec", "rails", "server", "-b", "0.0.0.0"]
```

In the builder stage (the first `FROM` clause), we install `libvips-dev` and its dependencies. This provides us with the necessary tools to build the gems locally within the container, including native extensions that link to `libvips`. In the subsequent stage, we switch to a stripped-down image with only the essential runtime libraries, specifically `libvips` itself, not the development headers. Then, crucially, we copy over the `vendor` folder containing compiled dependencies from the builder stage. This process helps keep the final image streamlined, reduces attack surface, and contributes to a faster deploy pipeline.

Now, let's move on to the `Gemfile`. You'll likely already have `ruby-vips` included, but it's worth double-checking:

```ruby
gem 'ruby-vips'
```

The specific version of `ruby-vips` doesn’t usually matter too much, but staying relatively up-to-date ensures compatibility with newer versions of `libvips` and often incorporates performance improvements and bug fixes.

After these steps, it’s usually worth checking the configuration within your Rails environment to ensure things are as expected. You can write a small test to verify that `ruby-vips` can initialize and process images correctly. Here’s an example snippet that I use fairly often when debugging these kinds of problems:

```ruby
require 'vips'

begin
  # Load a test image (this would need to exist somewhere in your app)
  image = Vips::Image.new_from_file("test_image.jpg")

  # Perform a simple operation
  resized_image = image.resize(0.5)

  # Output the dimensions of the resized image
  puts "Resized image width: #{resized_image.width}"
  puts "Resized image height: #{resized_image.height}"
  
  # You might save the resized image here to disk and inspect it
  # resized_image.write_to_file("resized_test_image.jpg")

  puts "libvips is working correctly"

rescue Vips::Error => e
    puts "Error with libvips: #{e.message}"
rescue LoadError => e
    puts "Could not load 'vips': #{e.message} - check installation"
end
```

This script will attempt to load an image, resize it, and then output its new dimensions. If you run this within your Rails console (`rails console`), you should see the output confirming that `libvips` is installed correctly. Any errors here usually point to issues with the gem installation or with `libvips` itself not being available at runtime, usually due to the issues with the dockerfile setup I mentioned before. A `LoadError` generally points to the `ruby-vips` gem not finding a matching `libvips` library.

Also, remember the importance of proper error handling, especially with file processing. The `rescue` clauses shown in the code snippet are not optional when dealing with images that can have various formats or be corrupted. When an image processing fails due to format issues, it’s always good to have proper error logging in place.

When things don’t go as planned (and they occasionally won't), it’s crucial to go back to basics, to ensure you have covered each of these points. Double-check the Dockerfile, make sure the gem is correctly included in your `Gemfile` and that the version is not too old. And finally, always test within your container environment. Testing within a local Rails development environment is sometimes misleading because the dependency resolution is done very differently on different systems.

For further, more advanced information, I highly recommend diving into "Digital Image Processing" by Rafael C. Gonzalez and Richard E. Woods. It’s a solid foundation for understanding image processing principles. Additionally, reviewing the documentation for `libvips` itself on its official website is incredibly beneficial. For more gem specific details, check the `ruby-vips` project repository on Github. It's well-maintained and detailed, so any issues there are usually well documented. These resources should provide a more comprehensive view of the technology and its application within the Rails and Docker landscape.

Implementing this approach, I've managed to solve similar dependency issues countless times in production, and I've found this to be reliable and highly performant. It also makes maintenance significantly simpler, which is a plus. Keep those dependencies tight and your containers lean.
