---
title: "How can I test if a Rails controller enforces HTTPS based on the environment using minitest?"
date: "2024-12-23"
id: "how-can-i-test-if-a-rails-controller-enforces-https-based-on-the-environment-using-minitest"
---

Alright,  I've been down this road myself, several times in fact, and it's a crucial aspect of application security, especially when dealing with sensitive data in production environments. Ensuring that your rails controllers enforce https based on the environment is not just a best practice, it's essential, and testing it properly via minitest is a worthwhile endeavor.

Initially, in my early days of development, we had a close call with an application where http was accidentally enabled in production. We got away with it, thankfully, but the wake-up call was clear. Ever since, rigorous testing of https enforcement has been a non-negotiable part of my workflow. What we’re aiming for is to ensure that our controller tests mirror real-world scenarios where different configurations apply based on the environment.

To achieve this, we need a good understanding of how Rails typically handles https redirection and environment configurations. Rails generally uses middleware for this purpose, often configured in `application.rb` or environment-specific files (e.g., `environments/production.rb`). The `config.force_ssl` setting is the primary driver here. In testing, we need to simulate this behavior, specifically targeting whether that setting and its associated middleware have been correctly applied. Minitest provides all the necessary tools to check this, and I'll demonstrate how we can accomplish this effectively with environment-aware tests.

The fundamental technique relies on setting up different test cases that accurately mimic different environments, specifically, the conditions under which https enforcement should be in play. We'll use mocking to avoid actually hitting https endpoints during testing which would lead to a large amount of slow, external requests. Instead, we will target and check for the presence of the middleware layer. Here's the breakdown in practice:

First, let’s consider a scenario where your application enforces https in production but not in the development or test environment. This is quite common, as it prevents certificate issues during local development and testing.

**Example 1: Testing Production HTTPS Enforcement**

Here's how to structure your test to verify this particular behavior:

```ruby
require 'test_helper'

class MyControllerTest < ActionDispatch::IntegrationTest

  def setup
    @original_force_ssl = Rails.application.config.force_ssl
  end

  def teardown
    Rails.application.config.force_ssl = @original_force_ssl
  end

  test "https is enforced in production" do
    # Mock the environment as 'production'
    Rails.env = 'production'
    Rails.application.config.force_ssl = true


    get '/some_action' # Replace with a valid route in your app
    assert_redirected_to 'https://www.example.com/some_action' # Replace with your url

     # If you expect a redirect, assert it's an https redirect
     assert_response :redirect
     assert @response.redirect_url.start_with?("https://")
  end


    test "https is not enforced in development" do
      # Mock the environment as 'development'
      Rails.env = 'development'
      Rails.application.config.force_ssl = false

      get '/some_action'
      assert_response :success # Expecting a successful http request

    end

end
```

In this example, within each test case, we are not mocking the web server itself, but rather we are modifying the rails application config to mimic the intended environment's behavior. Specifically, by setting the `Rails.env` variable, we force the configuration system to load the settings specific to that environment. Then we explicitly set the `force_ssl` setting as per the desired scenario, then the test will behave accordingly. Note the `setup` and `teardown` methods: these are vital for resetting the config so it doesn't leak between tests. Without this reset, you'd have all your tests trying to run in `production` which would be very problematic. This approach provides a focused and reliable way to test your https enforcement strategy, without involving complex web server mocks.

However, if you need to test the actual middleware presence, the following could be useful:

**Example 2: Testing the presence of the https middleware**

```ruby
require 'test_helper'

class HttpsMiddlewareTest < ActionDispatch::IntegrationTest
  def setup
    @original_force_ssl = Rails.application.config.force_ssl
  end

  def teardown
    Rails.application.config.force_ssl = @original_force_ssl
  end

  test "ensure ssl middleware is present in production" do
    Rails.env = 'production'
    Rails.application.config.force_ssl = true
    app = Rails.application # get the application object

    middleware = app.middleware.find { |m| m.klass == ActionDispatch::SSL }
    assert_not_nil middleware, "Expected ActionDispatch::SSL middleware to be present in production."
  end

  test "ensure ssl middleware is not present in development" do
    Rails.env = 'development'
    Rails.application.config.force_ssl = false
    app = Rails.application

    middleware = app.middleware.find { |m| m.klass == ActionDispatch::SSL }
     assert_nil middleware, "Expected ActionDispatch::SSL middleware not to be present in development."
  end
end
```

In the above test cases, we are directly inspecting the middleware stack of the rails application to ensure that when `force_ssl` is active the `ActionDispatch::SSL` middleware has been added, and conversely, that when it is not set the middleware does not exist. This test is more explicit in ensuring that the middleware is actually in place as opposed to solely asserting on the end behavior of redirects.

Finally, sometimes you might want to use a custom logic to implement the https enforcement. Here's another example of this:

**Example 3: Custom HTTPS Enforcement**

This particular method simulates a scenario where there's custom logic controlling redirection, for instance, based on a config parameter rather than `force_ssl`.

```ruby
require 'test_helper'

class CustomHttpsEnforcementTest < ActionDispatch::IntegrationTest

  def setup
      @original_custom_ssl_enforce = Rails.application.config.custom_ssl_enforce
    end

    def teardown
      Rails.application.config.custom_ssl_enforce = @original_custom_ssl_enforce
    end

  test "https is enforced when custom flag is set" do
    Rails.application.config.custom_ssl_enforce = true # Set your custom flag for https
    Rails.env = 'production'

    get '/some_action'
    assert_redirected_to 'https://www.example.com/some_action' # Replace with your url
    assert @response.redirect_url.start_with?("https://")

  end

  test "https is not enforced when custom flag is not set" do
     Rails.application.config.custom_ssl_enforce = false
    Rails.env = 'production'

    get '/some_action'
      assert_response :success # Expecting a successful http request

    end
end
```

This example shows how we could use a custom flag set in the `config`, which may be useful in more complex scenarios where you have more fine grained control over the behavior of the redirect. The key remains the same: manipulate the configuration in a controlled manner to simulate different scenarios, then write test assertions against the expected outcome.

When writing these kinds of tests, I’d recommend reviewing resources like "Agile Web Development with Rails 7," which goes into depth on Rails testing practices, and the official Rails documentation around middleware and configurations. Specifically the section in Rails Guides that goes over `ActionDispatch::SSL` is useful for deeper understanding of what that middleware layer does. Also, "Refactoring: Improving the Design of Existing Code," by Martin Fowler is invaluable for building good, readable test suites. Another helpful resource is the "Testing Rails" by Noah Gibbs, this book provides great details on different testing strategies. These resources provide a more complete picture of testing and building secure Rails applications, which goes beyond the scope of this answer. Remember, comprehensive testing is crucial for the security and stability of any production application. Thorough testing ensures the behavior you expect and helps catch errors before they reach your users. Don't neglect these, or you'll find yourself in trouble.
