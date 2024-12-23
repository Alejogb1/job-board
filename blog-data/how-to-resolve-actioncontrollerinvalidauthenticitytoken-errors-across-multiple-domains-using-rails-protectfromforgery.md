---
title: "How to resolve ActionController::InvalidAuthenticityToken errors across multiple domains using Rails' protect_from_forgery?"
date: "2024-12-23"
id: "how-to-resolve-actioncontrollerinvalidauthenticitytoken-errors-across-multiple-domains-using-rails-protectfromforgery"
---

Alright,  I've certainly seen my share of `ActionController::InvalidAuthenticityToken` errors, especially when dealing with cross-domain interactions. It’s a common pitfall, and understanding how to approach it correctly is crucial for any Rails-based application handling requests from different origins. The issue, at its core, stems from Rails' built-in CSRF (Cross-Site Request Forgery) protection, implemented via `protect_from_forgery`. This mechanism is fantastic for guarding against malicious requests originating from other sites, but it can become a stumbling block when legitimate cross-domain requests are involved.

The fundamental problem is that the authenticity token embedded in forms (or via headers for non-form submissions) is typically domain-specific. If a request arrives from a domain that doesn’t match the server's origin, the token is deemed invalid, and you get that dreaded `ActionController::InvalidAuthenticityToken` error. Now, there are multiple ways to handle this, and the 'best' one really depends on the specific context of your application. I remember a particularly tricky situation a few years back when we were integrating a payment gateway that hosted its form on a different subdomain. It took some careful deliberation to get it working smoothly, and that’s what I want to share here – not just the theoretical concepts, but the practical steps I've found effective.

Firstly, let's clarify what `protect_from_forgery` does. When enabled (which is the default), it generates a unique token that is stored both in the user’s session and embedded in form submissions. Each request is then checked to ensure the token matches. If it doesn't, the request is rejected. This is your baseline defense against CSRF attacks.

When working with multiple domains, the core issue is that the session, and therefore the associated CSRF token, is not readily available to another domain or subdomain without explicit configuration. To address this, we have a few options.

**Option 1: `skip_before_action` (Use with Extreme Caution)**

The most straightforward, yet *least* recommended, approach is to bypass the forgery protection entirely for specific controller actions. I've had to use this as a temporary measure during development, but it’s something I actively avoid in production.

```ruby
class PaymentsController < ApplicationController
  skip_before_action :verify_authenticity_token, only: [:create_payment]

  def create_payment
    # handle incoming payment data...
  end
end
```

This `skip_before_action` effectively disables the CSRF check for the `create_payment` action. While it might solve your immediate issue, it also leaves you wide open to potential CSRF vulnerabilities on that specific action. Think of this as removing the lock from one particular door of your house – it's quicker to get through, but certainly not the safest. *Always* exhaust other options before using this one. I usually reserve this only for isolated api endpoints used by systems with other authentication means.

**Option 2: `protect_from_forgery with null_session` with `Access-Control-Allow-Origin`**

A more secure and appropriate method is to allow requests from specific cross-origins by specifying the allowed origins on your API or application and then setting `protect_from_forgery with: :null_session`.

This involves modifying your `ApplicationController`.

```ruby
class ApplicationController < ActionController::Base
  protect_from_forgery with: :null_session

  before_action :cors_preflight_check
  after_action :cors_set_access_control_headers

  def cors_preflight_check
    if request.method == 'OPTIONS'
      cors_set_access_control_headers
      render plain: '', content_type: 'text/plain', status: 200
    end
  end

  def cors_set_access_control_headers
    headers['Access-Control-Allow-Origin'] = '*' # you would replace this with your specific allowed domain, or array of domains.
    headers['Access-Control-Allow-Methods'] = 'POST, GET, OPTIONS'
    headers['Access-Control-Allow-Headers'] = 'Content-Type, Origin, Accept, Authorization, X-Requested-With, x-csrf-token' # Include "x-csrf-token"
    headers['Access-Control-Max-Age'] = '1728000' # The 20-day caching max-age
  end
end

```

This is still a method that requires caution. `null_session` *does not* disable CSRF protection, but rather sets the session to null which means each request will not have access to a session cookie, *and* will also not generate a new one. In this mode of operation, you must send the CSRF token as part of your request, usually within a request header, with the corresponding `Access-Control-Allow-Origin` and other headers configured for your application. The header we'll be most concerned with is `X-CSRF-Token`. This will be the most appropriate in a system that functions more as an API server, and for non-browser clients. Ensure you also handle `OPTIONS` preflight requests as above.

**Option 3: Token Sharing (More Complex but Secure)**

This is perhaps the most secure, but also most complex approach, and I’ve used it successfully for highly sensitive cross-domain communication, such as in a microservice architecture I worked on. Instead of skipping checks or relying on null sessions, you create a mechanism to share the csrf token across domains. This is especially useful for applications that need to communicate with other services, or where you cannot use the null session method. The concept is that the first domain or application sends the token to the second domain as an additional parameter, typically in a header or as a query parameter.

Here’s how that general idea can translate to code:

```ruby
# Domain 1: Where the form/request is originating
# Example: Fetch request using JS on page at domain1.com
function makeRequest() {
  fetch('https://domain2.com/api/endpoint', {
    method: 'POST',
    headers: {
     'Content-Type': 'application/json',
     'X-CSRF-Token': document.querySelector('meta[name="csrf-token"]').getAttribute('content'), // Get token from HTML meta tag.
    },
    body: JSON.stringify({ data: 'some-data' }),
  });
}
```
```ruby
# Domain 2: Rails application that needs to receive the request from domain1.com
class Api::MyController < ApplicationController
  protect_from_forgery with: :exception  # important we don't use :null_session, as we are using a standard CSRF check.

  # Note: We do not use the before_action :verify_authenticity_token

  def create
    if verified_request?
      # Process the request
      render json: { status: 'success' }, status: 200
    else
      render json: { error: 'Invalid CSRF Token' }, status: 403
    end
  end

  private

  def verified_request?
    if request.headers['X-CSRF-Token'].present?
      csrf_token = request.headers['X-CSRF-Token']
      form_authenticity_token == csrf_token
    else
      false
    end
  end
end
```

In this approach, the client retrieves the csrf token from the originating page, and passes it within a header called `X-CSRF-Token`. The receiving Rails server then manually verifies the incoming token against its own using `form_authenticity_token`. This is more secure as we're still leveraging the power of Rails' built-in CSRF mechanism, without relying on null sessions or insecurely bypassing checks.

**Important Considerations:**

*   **Security:** *Never* completely disable CSRF protection unless you have a very strong understanding of the implications and alternative security measures in place.
*   **Domain Scope:** Be incredibly specific with your `Access-Control-Allow-Origin` settings. Never use `*` in production unless you really, really know what you are doing. If the header is set to `*`, it will allow *any* domain. Always be specific, whether through a whitelist of domains or a more complex method.
*   **Token Storage:** Be cautious when passing CSRF tokens as query parameters as they could be logged. Using headers is generally preferable.
*   **Documentation:** Ensure your api documentation or client applications clearly delineate which methods are being used, how tokens are exchanged, and the general lifecycle of the token.
*   **Further Reading:** I recommend "The Tangled Web" by Michal Zalewski for a deep understanding of web security principles. Also, the OWASP (Open Web Application Security Project) website is an excellent resource for staying updated with best practices.

In my experience, option 3 (token sharing) offers the most robustness and security for complex cross-domain situations, when paired with a proper cors implementation; it requires a bit more setup, but the effort is generally worth it for the added safety and control. Remember, understanding the nuances of how `protect_from_forgery` interacts with different domains is crucial to securing your Rails applications. Don't take shortcuts when it comes to security. Carefully examine your specific needs and choose the method that best balances security with your application's requirements.
