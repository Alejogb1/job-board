---
title: "How do I add a permissions policy header in Rails 6?"
date: "2024-12-23"
id: "how-do-i-add-a-permissions-policy-header-in-rails-6"
---

Alright, let's get into the specifics of setting up a permissions policy header in a Rails 6 application. It’s a vital step in bolstering the security of your web applications, and I’ve certainly dealt with its complexities in various projects over the years. The concept itself isn't exceptionally difficult, but the specifics of implementation, especially with the subtle nuances of policy directives, deserve some close attention.

The permissions policy, often called the feature policy, is a response header that controls which browser features are available in your web application. It gives you fine-grained control over what functionalities your site can use, helping to prevent malicious scripts from exploiting sensitive browser APIs. Think of it as an access control list specifically for web browser features. I vividly recall a project where we had to implement a very strict permissions policy due to client security requirements. We had to disable numerous features, and ensuring compatibility across different browsers was a key consideration. This early experience really drove home the importance of granular control in this area.

In Rails 6, there are several ways to add this header. The most straightforward and, in my experience, most maintainable is to use middleware. This allows us to apply the policy to every request without cluttering our controllers. We’re essentially intercepting each request and adding the necessary header before the response is sent.

Here's a snippet showing how you'd create this middleware. First, we'll create a file, say `app/middleware/permissions_policy_middleware.rb`:

```ruby
class PermissionsPolicyMiddleware
  def initialize(app)
    @app = app
  end

  def call(env)
    status, headers, body = @app.call(env)

    headers['Permissions-Policy'] = "accelerometer=(), camera=(), geolocation=(), gyroscope=(), magnetometer=(), microphone=(), payment=(), usb=(), interest-cohort=()"

    [status, headers, body]
  end
end
```

This simple middleware intercepts the request, adds a `Permissions-Policy` header and sets the value. In this example, we are denying access to several potentially sensitive browser features. Notice the use of `=()` which specifies that no origins are allowed to use the named feature. This is generally a safe default. It's a good starting point, and then you can selectively allow features and origins based on specific application needs.

Next, we need to register this middleware within our Rails application. This is done within `config/application.rb` or your environment-specific configuration files like `config/environments/production.rb`:

```ruby
config.middleware.use PermissionsPolicyMiddleware
```

By using `config.middleware.use`, we effectively tell Rails to include our custom middleware in the request/response processing pipeline. This ensures that every request going through your application will now include the `Permissions-Policy` header.

Let’s expand on that with a slightly more intricate example to illustrate how to permit specific features for particular origins. Imagine, for instance, that you want to enable geolocation for specific subdomains on your site:

```ruby
class AdvancedPermissionsPolicyMiddleware
    def initialize(app)
        @app = app
    end

    def call(env)
        status, headers, body = @app.call(env)
        policy =  "accelerometer=(), camera=(), gyroscope=(), magnetometer=(), microphone=(), payment=(), usb=(), interest-cohort=(), "
        policy += "geolocation=('self' 'https://location.example.com' 'https://maps.example.net')"

        headers['Permissions-Policy'] = policy
        [status, headers, body]
    end
end
```

In this extended example, you’re now seeing the inclusion of `geolocation=()`. We've modified this directive to include `('self' 'https://location.example.com' 'https://maps.example.net')`. This means that:

*   `'self'`: Allows geolocation for the application's origin itself
*   `'https://location.example.com'`: Explicitly allows geolocation for scripts or iframes from `https://location.example.com`
*   `'https://maps.example.net'`: Similarly, allows the use of geolocation for scripts or iframes from `https://maps.example.net`

This nuanced control is critical in many scenarios. It provides the capability to restrict browser features to only trusted sources. You might have an internal mapping service, for example, that you want to allow access to geolocation but strictly deny it from other third-party content.

One crucial aspect to remember is that the `Permissions-Policy` header can become complex. There are numerous features, and each can have its own specific configuration. In more advanced situations, you might prefer to use a configuration file or a service to construct this header rather than hard-coding everything within a middleware. This approach promotes better maintainability. Consider this configuration-driven approach:

```ruby
# config/permissions_policy.yml
permissions:
  accelerometer: []
  camera: []
  geolocation: ['self', 'https://location.example.com']
  gyroscope: []
  magnetometer: []
  microphone: []
  payment: []
  usb: []
  interest-cohort: []

# app/middleware/configurable_permissions_middleware.rb
require 'yaml'

class ConfigurablePermissionsMiddleware
    def initialize(app)
      @app = app
      @config = YAML.load_file(Rails.root.join('config', 'permissions_policy.yml'))['permissions']
    end

    def call(env)
        status, headers, body = @app.call(env)

        policy_string = @config.map do |feature, origins|
          if origins.empty?
             "#{feature}=()"
          else
            "#{feature}=(#{origins.map { |origin| "'#{origin}'"}.join(' ')})"
          end
        end.join(", ")

        headers['Permissions-Policy'] = policy_string
        [status, headers, body]
    end
end
```

We now have a `permissions_policy.yml` file that defines the policy directives. The middleware then loads this file, iterates through each policy and creates the corresponding `Permissions-Policy` string. This approach allows for far greater flexibility and manageability. You can change the allowed origins or enabled features without modifying the code.

A good resource to delve deeper into the specifics of different directives and their impact is the W3C specifications for Feature Policy. While it has evolved into the Permissions Policy specification, understanding both is essential for a comprehensive grasp of the subject. You can also look at the content security policy (CSP) specifications – which often overlap with feature control discussions – they contain a wealth of information about securing web applications through headers. Also, I would recommend researching any of the various browser-specific documentation on the permission headers to help you deal with the idiosyncrasies between various browsers. Books on web security, like "Web Application Hacker's Handbook", are also useful as they help to contextualize feature policy in the broader picture of web security. Be aware that the permissions policy is a continuously evolving area, so staying current with the latest drafts and browser updates is always a good practice.

In summary, setting up the permissions policy in Rails 6 involves creating middleware, crafting the policy string, and incorporating it into your application. The key is to understand how each directive works and to apply the appropriate restrictions based on your application's needs. And remember to always start with a restrictive policy before selectively enabling the features you require, adhering to the principle of least privilege in web security. This, in my experience, ensures a more robust and secure application.
