---
title: "How do I render custom error pages with layouts in Rails 5 and later?"
date: "2024-12-23"
id: "how-do-i-render-custom-error-pages-with-layouts-in-rails-5-and-later"
---

Okay, let's tackle this. I remember a particularly challenging project a few years back where we had incredibly granular error handling requirements, way beyond the default Rails setup. We needed customized error pages, with layouts, but also with conditional content and logging. It wasn't straightforward, but it underscored the importance of mastering Rails' error handling. Here's how you can approach rendering custom error pages with layouts in Rails 5 and later, avoiding the pitfalls I encountered, and focusing on practical solutions.

The default Rails error handling often falls short when you require branding consistency and detailed error information. Simply displaying a raw stack trace isn't user-friendly, nor is it particularly useful from a debugging standpoint in production environments. What you need is a controlled, consistent experience that provides the necessary information while maintaining user interface coherence.

Fundamentally, Rails allows you to define error handling within your `application_controller.rb`. Here, you can 'rescue' different kinds of exceptions and map them to specific actions that render your custom error pages. This is where it starts to deviate from the boilerplate. The key thing to remember is that you're not just rendering a static html file; you're rendering a full rails view, giving you the flexibility you'd expect.

Let’s break down the process. First, we define a layout specifically for error pages. In your `app/views/layouts/` directory, create a file like `error.html.erb` or `error.html.haml` (depending on your templating engine preference):

```html+erb
<!DOCTYPE html>
<html>
<head>
  <title>Application Error</title>
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <%= stylesheet_link_tag "application", media: "all" %>
</head>
<body>
  <div class="error-container">
    <%= yield %>
  </div>
</body>
</html>
```

This establishes a basic layout for your error pages, including styling and essential metadata. Remember to adjust `application` with the appropriate stylesheet include based on your setup. This separates the layout concern from the error message concern and lets you maintain consistency across error pages.

Next, in `application_controller.rb`, you’ll define the rescue methods. Here's a snippet illustrating how you might handle a `RecordNotFound` exception, a common issue in web apps:

```ruby
class ApplicationController < ActionController::Base
  rescue_from ActiveRecord::RecordNotFound, with: :render_not_found

  private

  def render_not_found(exception)
    logger.error "Record Not Found: #{exception.message}"
    respond_to do |format|
      format.html { render template: 'errors/not_found', layout: 'error', status: :not_found }
      format.json { render json: { error: 'Not Found' }, status: :not_found }
    end
  end
end
```

Here's what's happening:
1.  `rescue_from ActiveRecord::RecordNotFound, with: :render_not_found`: this line specifies we want to catch `ActiveRecord::RecordNotFound` exceptions and handle them using a method named `render_not_found`.
2. `logger.error "Record Not Found: #{exception.message}"`: this line provides invaluable debugging by logging the specific error to your server's logs, helping you diagnose issues in real time.
3.  `respond_to do |format| ... end`: this block ensures that different request formats (like html or json) can be handled with different responses. In this case:
    *   `format.html { render template: 'errors/not_found', layout: 'error', status: :not_found }`: for html requests, we render a template located at `app/views/errors/not_found.html.erb` (or `.haml`) using the `error` layout and return a 404 `not_found` status.
    *   `format.json { render json: { error: 'Not Found' }, status: :not_found }`: for JSON requests, a simple JSON object is returned with a `not_found` status.

For the template `app/views/errors/not_found.html.erb` you would typically have the following structure:
```html+erb
<div class="error-message">
  <h1>Oops!</h1>
  <p>The resource you were looking for could not be found.</p>
</div>
```
This combination provides a user-friendly message while enabling debugging.

Now, let’s extend this with a more complex scenario. Suppose we need to handle generic server errors (500) and want to show more technical detail while only displaying minimal information to users in production.

```ruby
class ApplicationController < ActionController::Base
  #previous code...
  rescue_from StandardError, with: :render_server_error

  private

  #previous code...

  def render_server_error(exception)
      logger.error "Server Error: #{exception.message}\n#{exception.backtrace.join("\n")}"
      respond_to do |format|
          format.html do
            if Rails.env.development? #conditionally display details in development
              @exception_message = exception.message
              @exception_backtrace = exception.backtrace.join("\n")
              render template: 'errors/server_error_development', layout: 'error', status: :internal_server_error
            else
              render template: 'errors/server_error_production', layout: 'error', status: :internal_server_error
            end
          end
      format.json { render json: { error: 'Internal Server Error' }, status: :internal_server_error }
    end
  end
end
```
In this enhanced implementation:
1.  `rescue_from StandardError, with: :render_server_error`:  we catch all unhandled exceptions which are of the `StandardError` class or descend from it.
2.   The `logger.error` line now includes the full stack trace.
3.  We conditionally render either `errors/server_error_development` or `errors/server_error_production` depending on the environment. This enables displaying detailed error messages and backtraces in development while providing a simplified version in production, critical for security.

Now you have corresponding views: `app/views/errors/server_error_production.html.erb`:
```html+erb
<div class="error-message">
  <h1>Oops! Something Went Wrong</h1>
  <p>Our team has been notified. Please try again later.</p>
</div>
```
And for `app/views/errors/server_error_development.html.erb`:
```html+erb
<div class="error-message">
  <h1>Oops! Something Went Wrong</h1>
  <p><strong>Error:</strong> <%= @exception_message %></p>
  <pre><strong>Backtrace:</strong> <%= @exception_backtrace %></pre>
</div>
```

This gives you a clear distinction between user-facing information and development-focused details. The backtrace is safely available in development for debugging without exposing it to the general user.

In my experience, handling errors this way makes for a more robust application. It separates concerns nicely and avoids presenting raw, unformatted error pages to users. It also provides far more actionable data in your logs, making debugging far simpler. Remember to meticulously log errors, as they are the most critical pieces of data during incidents.

For a deeper understanding, I strongly recommend referring to *Effective Ruby: 48 Specific Ways to Write Better Ruby*, by Peter J. Jones. It’s a fantastic resource that touches on many aspects of Rails development, including effective error handling. Another useful resource is *Crafting Rails Applications*, by José Valim, which explores more intricate patterns for Rails applications. Finally, while not specific to errors, the official Rails guides at guides.rubyonrails.org is an invaluable resource, particularly the sections concerning routing, controllers, and exception handling. These resources will give you a solid foundation for implementing robust error handling in your Rails applications. This approach, while requiring some initial configuration, pays dividends in the long run.
