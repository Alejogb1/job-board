---
title: "Why does a Rails 7 API only app require a session store?"
date: "2024-12-16"
id: "why-does-a-rails-7-api-only-app-require-a-session-store"
---

Okay, let's talk about session stores in Rails API-only applications. It's a topic that often trips up developers new to building solely API-focused services with Rails, and I recall facing this exact confusion when I transitioned a large monolithic application into a series of microservices a few years back. The initial instinct, understandably, is: if I'm only returning json, why on earth would I need a session store? It feels counterintuitive.

The core reason, fundamentally, is that even in the absence of traditional server-rendered views, Rails’ underlying framework still relies on a middleware stack that inherently manages session handling. It's part of the framework's DNA, stemming from its heritage as a web application framework. It's not really about storing user login data, in the classical sense, for your single page application or mobile client, but more about how Rails is structured, and how its request lifecycle works. The middleware that deals with sessions is always invoked when a request arrives, regardless of the type of response you send back - be it html or json. If a session store isn't configured properly, you'll encounter issues, potentially including errors or unexpected behaviour in your application.

When you create a Rails application, it includes the `ActionDispatch::Session::CookieStore` middleware by default. This middleware, despite its name, is not inherently tied to sending cookies; it's about enabling the underlying session machinery in Rails. This session object, whether you consciously use it or not in your json responses, still exists in Rails. When the middleware tries to load an empty, or non-existent, session and it encounters a configuration issue, it will cause your application to break.

Let’s consider a few scenarios. First, imagine an early iteration of our microservice where we simply responded with a `200 ok` and a JSON object for all requests, something like:

```ruby
# app/controllers/application_controller.rb
class ApplicationController < ActionController::API
  def index
    render json: { message: "Hello from the API!" }
  end
end

# config/routes.rb
Rails.application.routes.draw do
  root 'application#index'
end
```

This seems innocuous enough, and it works fine – until you try and leverage any features like CSRF protection. Even without explicitly using session data, the session middleware is trying to do its job by loading or creating a session store during the request, and this may expose issues.

Now, if we attempt to use, say, CSRF (Cross Site Request Forgery) protection, even without handling login, the presence of the session becomes more vital. CSRF tokens are commonly stored in a session, and if you’ve disabled the session handling, CSRF checks will fail. Let's look at an example of how CSRF can manifest. Consider this simplified controller and related route:

```ruby
# app/controllers/protected_controller.rb
class ProtectedController < ActionController::API
  include ActionController::RequestForgeryProtection

  protect_from_forgery with: :exception

  def create
    render json: { status: 'created' }, status: :created
  end
end

# config/routes.rb
Rails.application.routes.draw do
  post 'protected', to: 'protected#create'
end
```

If you attempt to post data to `/protected`, even with a request body, your application might throw an `ActionController::InvalidAuthenticityToken` error if a session store isn't properly set up. This may occur despite the absence of traditional web views that normally include an automatically generated CSRF token in forms. CSRF protection is designed to verify that requests originate from your application, and it relies on the session to store and validate these tokens. Without a session store, CSRF protection simply won’t work, and it can throw errors in unexpected parts of your application.

To explicitly address this, even if you're only building an api, you need to configure a session store correctly. The configuration in `config/initializers/session_store.rb` (or its counterpart in `config/environments/production.rb`) is key. You need to specify a valid store and related settings to prevent errors. Here's an example showing how to configure a cookie-based session store, which is the default, but you should be aware of the implications. You will still require it even if you're not actually using session data in your API responses:

```ruby
# config/initializers/session_store.rb
if Rails.env.production?
  Rails.application.config.session_store :cookie_store, key: '_my_api_session',
                                                            same_site: :strict,
                                                            secure: true,
                                                            domain: '.mydomain.com',
                                                            expire_after: 1.hour
else
    Rails.application.config.session_store :cookie_store, key: '_my_api_session'
end
```

As seen above, there is configuration for production and development environments respectively. In development, the default cookie based session storage is okay, however in production you would need to configure the domain, security flags, expiration, and any other attributes specific to your use case.

The crucial point here isn't whether your application directly uses session data. It's that Rails' session middleware will always be active. Thus, you've got to set up an actual store (even if it’s just an in-memory one for development or testing) to prevent it from causing runtime issues. It’s not about the specific use of the session to store user details, it’s the very existence and execution of the session middleware that necessitates the configuration.

When designing APIs, I generally prefer using stateless authentication methods like JWT (JSON Web Tokens) for handling client authentication. However, even then, Rails’ middleware can still interfere if not configured properly. I’ve seen instances where an improper session store configuration interferes with authentication headers, especially if you use other middlewares that might also read from or interact with the session data.

It is always beneficial to use a secure store. While a cookie store is common, you may prefer using database backed session stores if you're dealing with more complex security requirements, or need a reliable session store across multiple instances of your application, even though you’re not using the session directly in the api response. For in-depth understanding of session management in Rails and the role of middleware, I suggest reviewing the Rails Guides on Action Controller and the Rack specification, especially the parts dealing with middleware. Understanding how each middleware interacts during the request/response cycle can prevent many common pitfalls. Also, “Crafting Rails 4 Applications” by José Valim is an excellent resource for understanding the intricacies of Rails' internal workings. Specifically, the parts on middleware and request lifecycle will prove invaluable. Additionally, exploring the source code of the `ActionDispatch::Session` module within the Rails repository itself can offer deeper insights. In a sense, thinking of it as a kind of ‘plumbing’ in the framework is a useful conceptual model. You might not be using the tap, but the pipes are always there, functioning regardless of your needs.
