---
title: "How can Rails 6 allow specific pages to be embedded in iframes from any domain?"
date: "2024-12-23"
id: "how-can-rails-6-allow-specific-pages-to-be-embedded-in-iframes-from-any-domain"
---

Alright, let's talk about iframes in Rails 6 and the challenges of cross-domain embedding. I've been down this road a few times, usually in projects where I needed to integrate a dashboard component or some interactive content into external systems. It's not a trivial setup, and getting it wrong can open up some nasty security holes, so a careful approach is key.

The fundamental problem stems from the same-origin policy enforced by web browsers. This policy prevents a script running on one origin (combination of protocol, domain, and port) from accessing data from a different origin. This is a crucial security measure against cross-site scripting (xss) attacks. However, when you *want* to embed content from your application on a different domain via an iframe, you need to explicitly bypass this policy in a controlled way. Rails, thankfully, provides tools to manage this, but it’s about enabling specific access, not opening the floodgates.

The standard rails setup, without explicit configuration, will block your page from being embedded in an iframe served from a different origin. This is due to the 'x-frame-options' header, which by default, is set to `sameorigin`. So, the first hurdle is configuring rails to allow framing of specific pages from *any* domain. I emphasize *specific pages* because you probably don't want your entire site embedded anywhere.

There are two main aspects to consider here: the http headers and rails configurations. Let's start with the headers. Specifically, the `content-security-policy` (csp) and `x-frame-options` headers are the most relevant here. I’ve always found the csp to be the more powerful and nuanced tool for controlling how content is loaded, so that is where i typically put my focus. `x-frame-options` while simpler, is less flexible.

Now, instead of the 'x-frame-options,' which is essentially a binary choice with `sameorigin`, `deny`, or `allow-from`, we’ll use csp to allow embedding. When using csp for this, we need to define a `frame-ancestors` directive. This allows you to specify from which origins iframes can be loaded from your site. To allow all domains to load a particular page within an iframe, you use a wildcard: `frame-ancestors *;`. This needs to be done carefully, though. It's not always recommended, and it's essential to understand the security implications. We’ll illustrate the security concerns in one of the code examples later.

Here's a basic implementation of how I handled this in a past project involving a reporting module embedded into a client's dashboard. I created a specific controller to handle the iframeable content. This approach isolates the logic and makes configuration manageable.

```ruby
# app/controllers/embeddable_reports_controller.rb
class EmbeddableReportsController < ApplicationController
  def show
    report_id = params[:id]
    @report = Report.find(report_id)

    response.set_header('Content-Security-Policy', "frame-ancestors *;")

    render 'show'
  end

  # optionally, you could control specific allowed domains
  def specific_domains
      report_id = params[:id]
      @report = Report.find(report_id)

      allowed_origins = ['https://client.example.com', 'https://another.example.net']

      csp_directive = "frame-ancestors " + allowed_origins.join(' ') + ';'
      response.set_header('Content-Security-Policy', csp_directive)

      render 'show'
  end

end
```

```ruby
# config/routes.rb
Rails.application.routes.draw do
  get 'reports/:id/embed', to: 'embeddable_reports#show', as: :embeddable_report
  get 'reports/:id/embed/specific', to: 'embeddable_reports#specific_domains', as: :embeddable_report_specific
end
```

```erb
# app/views/embeddable_reports/show.html.erb
<h1><%= @report.title %></h1>
<p><%= @report.content %></p>
```

In this example, the `show` action sets the `frame-ancestors` directive to `*`, effectively allowing any domain to embed the content. The `specific_domains` action instead allows only the given domains in `allowed_origins`. Note that I have also included a route and a basic view to complete the example. In practice, you'd have more sophisticated content to display.

The most important aspect of all this, however, is security. Using `frame-ancestors *` should be done with extreme caution, primarily in cases where the content being displayed is not sensitive. You should never use a wildcard if the embedded page contains confidential or user-specific information.

Here’s a refined example focusing on security: let's say that in the previous example the report page displayed some data that was user specific. Allowing any domain to embed this would be a bad idea. We would instead need to specify explicitly the authorized domains. In the example below, we are allowing a specific domain, `https://client.example.com`, to embed the report but explicitly blocking all other origins using `none`.

```ruby
# app/controllers/secure_embeddable_reports_controller.rb
class SecureEmbeddableReportsController < ApplicationController
  def show
      report_id = params[:id]
      @report = Report.find(report_id)

      allowed_origin = 'https://client.example.com'
      csp_directive = "frame-ancestors #{allowed_origin};"
      response.set_header('Content-Security-Policy', csp_directive)

      render 'show'
    end
  end
```

```ruby
# config/routes.rb
Rails.application.routes.draw do
  get 'secure_reports/:id/embed', to: 'secure_embeddable_reports#show', as: :secure_embeddable_report
end
```

```erb
# app/views/secure_embeddable_reports/show.html.erb
<h1><%= @report.title %></h1>
<p>Sensitive content: <%= @report.sensitive_content %></p>
```

Notice that we’re being very explicit about the allowed origin here. In addition, the response still sets the `content-security-policy` header with the `frame-ancestors` directive and ensures it only includes the allowed origin. Any other domain attempting to embed this page will have their browser block the iframe.

If your embeddable content needs to be more fine-grained, you’ll have to expand upon this, perhaps by checking a whitelist of domains on a per-request basis from within your controller. You could also pass the authorized domains as a parameter and validate that parameter prior to rendering. This is an additional layer of security that i have often found necessary when dealing with sensitive information.

Another critical consideration: while these methods address embedding in iframes, they don't inherently protect against other forms of abuse, such as xss attacks, or csrf attacks which you will have to implement other solutions for. The csp directives discussed here are more focused on regulating from where resources can be loaded. Proper input validation, parameterized queries, and csrf tokens are still required.

For deeper understanding of content security policy, I would highly recommend reading "High Performance Browser Networking" by Ilya Grigorik; It has a fantastic chapter on security and csp specifically. The w3c specification on csp is also a go to reference for implementation details. For a more in-depth understanding of the same origin policy and browser security mechanics in general "Web Security: A White Hat Perspective" by Michael Howard and David LeBlanc is a solid reference.

In summary, to allow your rails 6 application to be embedded in iframes from *any* domain, you must explicitly manage the `content-security-policy` headers using the `frame-ancestors` directive. Using `frame-ancestors *` should be reserved for static or non-sensitive content. For anything user-specific or secure, you should explicitly define the allowed domains or implement more dynamic authorization to maintain the security of your application. And remember, this is only one part of a complete security posture, so always validate your data, use csrf tokens and parameterized queries, and be aware of any potential vulnerabilities when building your site.
