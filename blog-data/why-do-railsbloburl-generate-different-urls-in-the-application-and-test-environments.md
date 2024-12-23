---
title: "Why do `rails_blob_url` generate different URLs in the application and test environments?"
date: "2024-12-23"
id: "why-do-railsbloburl-generate-different-urls-in-the-application-and-test-environments"
---

Let's unpack this. I recall dealing with a particularly frustrating bug early in my career that directly stemmed from this very issue—the inconsistent behavior of `rails_blob_url` between application and test environments. It's not uncommon, and often arises from a subtle interplay of configuration and context. It's not that the method itself is faulty; rather, it's how those environments configure their underlying storage services that leads to divergent output. The core problem lies in understanding that `rails_blob_url` doesn't directly produce URLs out of thin air. Instead, it generates a URL based on the currently configured *active storage service* and its associated settings.

In a nutshell, different environments often utilize different active storage configurations. You might, for example, use a locally stored disk service for development and testing and then switch to an s3 bucket in production. This discrepancy is precisely what causes those URLs to change—the underlying storage mechanism is actually different. The `rails_blob_url` method simply asks the current active storage service to provide the correct URL for the blob. If the service changes, naturally the URL generated will also change.

Let's dive a bit deeper into how these configurations work. Rails leverages environment-specific configurations to define its active storage service. This is primarily set up in `config/storage.yml` and loaded into the application during initialization. When you generate a new Rails application with active storage included, this file will generally contain separate definitions for `local`, and `test` environments at minimum. These services are then chosen at runtime based on the current environment set via `ENV['RAILS_ENV']`, typically.

In a production environment, for example, you’ll likely be pointing to a cloud storage bucket (like s3, google cloud storage, etc.), with each of these options having its own specifics in terms of URL generation and security configurations. The test environment, by default, often defaults to a disk service – writing files to a temporary folder and therefore generating local URLs. This is crucial for isolated testing and not polluting production-like storage during your testing process. The problem is, the urls generated against a disk service won't match against an s3-bucket service.

To visualize this, let's look at three example configurations:

**Example 1: `config/storage.yml` – A Typical Setup**

```yaml
local:
  service: Disk
  root: <%= Rails.root.join("storage") %>

test:
  service: Disk
  root: <%= Rails.root.join("tmp/storage") %>

aws:
  service: S3
  access_key_id: <%= ENV['AWS_ACCESS_KEY_ID'] %>
  secret_access_key: <%= ENV['AWS_SECRET_ACCESS_KEY'] %>
  region: <%= ENV['AWS_REGION'] %>
  bucket: my-production-bucket
```

In this example, the `local` and `test` environments will save files to disk, but in different locations. The `aws` environment uses the `S3` service. `rails_blob_url` will then return file system urls for both local and test and s3 urls for `aws`.

Now let’s illustrate how this impacts code in a model using `rails_blob_url`.

**Example 2: Generating a URL in the Model**

```ruby
class User < ApplicationRecord
  has_one_attached :avatar

  def avatar_url
    if avatar.attached?
      Rails.application.routes.url_helpers.rails_blob_url(avatar, only_path: true)
    else
      nil
    end
  end
end
```

This method simply returns the URL of the user's avatar if attached. Notice the use of `only_path: true`. This ensures that we get a relative path when in a development setting rather than a full url, it's not actually related to the original problem though, just good practice. Here's the rub: in tests, given the `test` configuration above, the url would be something akin to `/rails/active_storage/blobs/...`. However, if you switched to the production environment, it would turn into something like `https://my-production-bucket.s3.amazonaws.com/...`. These are entirely different urls, pointing to different locations and different storage types.

To further illustrate how this is handled differently by different services, let’s take a look at a custom disk service.

**Example 3: Custom Disk Service with a Specific URL Generator**

Let's imagine you've created your own active storage disk-based service, and that in that service, you want all urls for uploaded images to point to a particular domain name or use a proxy.

```ruby
# config/initializers/active_storage_custom_service.rb

require 'active_storage/service'

module ActiveStorage
  class Service::CustomDiskService < Service::DiskService
    def url(key, **options)
      Rails.application.routes.url_for(
        controller: 'assets',
        action: 'show',
        id: key,
        host: 'custom-domain.com',
        only_path: false
      )
    end
  end
end


# config/storage.yml
custom_local:
  service: CustomDisk
  root: <%= Rails.root.join("storage") %>

#config/environments/test.rb
Rails.application.configure do
  config.active_storage.service = :custom_local
end
```

Here, we override the `url` method to generate URLs using our custom routing system and the provided id. This shows that each service is fundamentally responsible for the way it generates urls. This service could be anything from cloud storage, to a custom solution like above. That’s really the root of the issue, the method for url generation is not a function of rails’ `rails_blob_url`, but of the `ActiveStorage::Service` it's delegating to.

The key takeaway here is that the `rails_blob_url` does not directly create the url itself. It is not a magical function. Instead, it calls the appropriate url generation method of the currently configured storage service. So, each environment behaves differently not because `rails_blob_url` is doing anything different, but because the *storage service* itself is configured differently.

**Practical Steps to Avoid Confusion**

1.  **Understand your storage configuration:** Carefully examine `config/storage.yml` and ensure you know which services are used in each environment. Be absolutely clear on which services are being used and their specific configurations.

2.  **Test all active storage integrations:** It’s a good idea to explicitly test interactions with your active storage setup in your test suite. Using a `Disk` service for tests works great, but make sure your tests actually interact with the service, not mock it. There are valid scenarios where mocking is ok, however in these kinds of tests you will specifically want to see the impact of your service, and as such, avoid the use of mocking.

3.  **Use consistent environment variables:** If you are using cloud-based services like AWS S3, ensure that your environment variables are correctly set in each environment. You might have a staging environment with a different bucket compared to production, or a test environment with a mocked service that returns specific values. It's important to know how your services are configured in each environment.

4.  **Consider using a dedicated test storage service:** If your testing involves uploading, manipulating, and downloading large files, a memory-backed or a fast disk-based storage service for testing might be beneficial. I've found in practice that using an in-memory storage service is great for smaller tests, but I frequently encounter situations where this introduces inconsistencies in my tests due to the service being fundamentally different than what's being used in production. I prefer a disk-based approach for the vast majority of active storage related tests.

**Recommended Resources**

*   **The Official Rails Guides:** Specifically, the Active Storage guide is an essential read. It covers configuration, different service options, and testing.

*   **"Crafting Rails Applications" by Jose Valim:** This book, while not exclusively on active storage, does have a section on file uploads and demonstrates how to approach these types of problems in an idiomatic way.

*   **"The Rails 7 Way" by Obie Fernandez:** This book offers a more generalized perspective on rails, including how settings are applied in different contexts, and it's an excellent resource for better understanding rails.

In conclusion, the variance in URLs produced by `rails_blob_url` across environments isn't arbitrary. It's a direct consequence of using different storage configurations. Understanding this relationship is essential for a solid debugging foundation and can prevent frustrating issues. Remember, the `rails_blob_url` function is a reflection of the currently configured service and does not perform any internal url processing itself; it's just a helper method that delegates to the appropriate service provider, and those providers are configured within your `storage.yml` file. Always verify your configurations and ensure all services are setup and working as intended in all environments.
