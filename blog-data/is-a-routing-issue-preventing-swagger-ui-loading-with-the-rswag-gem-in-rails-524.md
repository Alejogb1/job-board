---
title: "Is a routing issue preventing swagger-ui loading with the rswag gem in Rails 5.2.4?"
date: "2024-12-23"
id: "is-a-routing-issue-preventing-swagger-ui-loading-with-the-rswag-gem-in-rails-524"
---

Alright, let's unpack this. I’ve seen this particular headache, or variations of it, crop up more than a few times over the years, especially with the older rails versions and rswag integration. The core issue often appears as a swagger-ui that stubbornly refuses to load, typically presenting a blank screen or an error, while seemingly all the configuration seems in place. While “routing” might seem like a direct culprit, it's usually a symptom of more intricate configuration discrepancies. Let me walk you through how I've approached this in the past and what to consider when troubleshooting.

The initial gut feeling that it’s purely a routing issue, while not incorrect, is generally an oversimplification. Yes, if the routes aren't configured to map requests to the rswag-generated swagger ui assets, you'll absolutely face a loading failure. However, the root cause often stems from several interlinked factors, including asset serving, configuration within `rswag.rb`, and even subtle version mismatches between related gems. For rails 5.2.4 in particular, remember we're dealing with a slightly older asset pipeline and subtle nuances compared to more recent versions.

First, let's address the core routing concept. Rswag, when configured correctly, typically mounts the swagger-ui assets under a configurable path, often `/api-docs`. A common mistake is failing to add or correctly configure this path within your `config/routes.rb`. Here's what a basic configuration might look like. This is the first area to verify:

```ruby
# config/routes.rb

Rails.application.routes.draw do
  mount Rswag::Ui::Engine => '/api-docs'
  # Your other routes
end
```

If the route is missing or is commented out, you will definitely experience a broken swagger UI. So, double-check that the mount instruction is present and correct. The path segment after the `=>` symbol, in this example `/api-docs`, will be the location where the UI tries to load all its resources. You should be able to navigate to `http://your_app_domain/api-docs` in your web browser (assuming your development environment runs on port 3000, you should try `http://localhost:3000/api-docs`). If you receive a ‘404 Not Found’ error, then routing is definitively an immediate suspect, but again, not always the only source.

Assuming your routes are in place, the next critical component lies within the rswag configuration itself, usually residing in an `rswag.rb` initializer file (typically within `config/initializers`). I've had situations where the configuration was seemingly present, but had incorrect paths specified, or some other subtle error that prevented the UI from loading.

For example, here’s a simplified version of what your `rswag.rb` could look like (some of the options might differ depending on your project):

```ruby
# config/initializers/rswag.rb

Rswag.configure do |config|
  config.swagger_root = Rails.root.join('swagger').to_s # Path to swagger files
  config.swagger_docs = {
    'v1/swagger.json' => {
      swagger: '2.0',
      info: {
        title: 'API Documentation',
        version: 'v1'
      }
    }
  }
end
```

Specifically, pay close attention to `config.swagger_root` and the file path(s) indicated within `swagger_docs`. If `config.swagger_root` points to the wrong directory, or the files specified within `swagger_docs` don't exist or are inaccessible, the swagger UI will fail to load because the resources are simply not there. It will not resolve or parse. It's also vital that the `swagger` version specified matches with your version, although most often it defaults to `2.0` in older Rails and Rswag versions. When debugging, I would always explicitly define this.

Now, this brings us to the issue of asset serving. Rails 5.2.4 utilizes the asset pipeline, and sometimes, the generated swagger-ui assets don't get compiled or served properly. This can manifest as a blank screen, console errors indicating problems with JavaScript or CSS resources, or other generic loading issues.

A good test to confirm this is to manually try to access the underlying javascript files. For example, try and access the swagger-ui bundle directly, for example, if you're using swagger-ui 3, you may try the url `http://localhost:3000/assets/swagger-ui-bundle.js`. If you get a 404 or a generic error, it indicates an asset serving issue. To confirm this, run `rake assets:precompile`. This will ensure that your assets are compiled and ready to be served.

Another potential culprit, and one I've encountered frequently enough to be vigilant about it, is gem version conflicts. Older versions of rswag, when paired with certain versions of dependencies like `swagger-ui-rails` (which rswag leverages under the hood), can exhibit subtle, difficult-to-debug behavior. Specifically, version mismatches between `rswag` itself, `swagger-ui-rails` gem, and the `rails` gem, can lead to incompatibility issues. If you've tried the routing and asset serving steps and you are still having problems, then try explicitly stating the dependency you are relying on in your Gemfile (especially the version of the `swagger-ui-rails` gem) and running `bundle update` again. You may try the latest stable version of all the related gems to ensure compatibility.

```ruby
# Gemfile
gem 'rswag-ui'
gem 'swagger-ui-rails', '~> 3.25'  #explicitly state the version
# your other gems
```

After updating your `Gemfile`, make sure to run `bundle install`. And, of course, if you are testing in development, ensure your rails development server is restarted.

Troubleshooting such issues is a process of elimination. First verify the route, then verify `rswag.rb`, then verify asset serving. If all of these have been verified, then consider the possibility of gem version conflicts. And remember to always clear your browser cache when testing.

For further reading, I strongly recommend "Agile Web Development with Rails 5.1" by Sam Ruby, Dave Thomas and David Heinemeier Hansson, and the specific chapter on the asset pipeline and custom rails setups. While it may seem to be a little older (since you are using Rails 5.2.4), the core principles behind asset management and routes haven't fundamentally shifted. Also, the rswag gem's official documentation on Github is always a good reference point. And finally, for more information on the swagger specification itself and usage guidelines, consult the official Swagger/OpenAPI specification documents on their website (OpenAPi Initiative). They are extremely detailed and have a wealth of information. These resources provide much more detail on the concepts I've outlined, providing a more thorough background understanding. The specific chapters on routing within the Rails guide (and any Rails related books) would provide a more robust base to start your development on.
