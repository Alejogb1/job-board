---
title: "Why does my Rails 7 API only app require a session store with Devise?"
date: "2024-12-23"
id: "why-does-my-rails-7-api-only-app-require-a-session-store-with-devise"
---

Let's unpack this. So, you've built a Rails 7 API-only application and, like many of us initially trying to keep things minimal, bumped into that unexpected Devise dependency on a session store. It feels counterintuitive at first, doesn't it? We consciously chose API-only to avoid view rendering and thought we'd be free of session-related baggage, but alas. I've seen this issue pop up in projects I've worked on, usually when migrating from traditional Rails setups or when folks are new to the api-only architecture.

The core of the issue lies in Devise's default authentication flow, even within an API-only context. While you're not rendering html views that typically manage cookies and sessions via browsers, Devise still *defaults* to using sessions as its primary mechanism for maintaining user login states. In traditional web apps, the browser transparently handles the back-and-forth of session cookies, but in an api context, our clients (say a javascript frontend or mobile app) don't operate in that realm. That means even in API mode where you'd expect it not to have anything to do with the browser or sessions the code needs it.

It's less about "requiring" and more about "being configured by default" for session storage. When Devise attempts to persist login information, it's leaning into Rails' built-in session management by default, which is primarily designed for cookie-based sessions. This is where your “api-only” directive conflicts.

To clarify further, Devise uses strategies to authenticate users. The default strategy usually tries to use sessions. With a standard setup, if you’re not disabling the session storage mechanism or using a different strategy the API will still attempt to use it as its primary mechanism.

So the issue isn’t the application setup itself, but Devise’s standard settings. We need to tell devise how to authenticate users in a stateless manner, which is typical for apis.

Here are a few common approaches to address this, backed by my experience and a few code examples:

**1. Token-Based Authentication (JSON Web Tokens - JWT):**

This is a popular approach for API authentication, and it sidesteps the need for session storage completely. Instead of storing session data, you generate a token upon successful login that the client then sends with each subsequent request. This effectively makes your application stateless.

Here's a simplified code snippet illustrating how you can integrate `jwt` gem with `devise`:

```ruby
# Gemfile
gem 'jwt'
```

```ruby
# app/models/user.rb
class User < ApplicationRecord
  devise :database_authenticatable, :registerable, :recoverable, :rememberable, :validatable

  def generate_jwt
    JWT.encode({ id: id, exp: 60.days.from_now.to_i }, Rails.application.credentials.secret_key_base)
  end

  def self.from_jwt(token)
    decoded_token = JWT.decode(token, Rails.application.credentials.secret_key_base, true, { algorithm: 'HS256' })
    find(decoded_token.first['id'])
  rescue JWT::DecodeError
    nil
  end
end

```

```ruby
# app/controllers/authentication_controller.rb
class AuthenticationController < ApplicationController
  def create
    user = User.find_by(email: params[:email])
    if user&.valid_password?(params[:password])
      token = user.generate_jwt
      render json: { token: token }, status: :ok
    else
       render json: { errors: ['Invalid email or password'] }, status: :unauthorized
    end
  end
end
```

In this example, upon a successful login, the `create` action in the `authentication_controller` generates a JWT, embedding the user’s ID and an expiry time. We include a method to generate this token and decode it. In a real application you might need middleware to extract this from the `Authorization` header on each request and you will also need to handle token expiration and refresh.

**2. Devise with Token Authentication (using `devise-token-auth` gem):**

If you prefer to stick more closely to Devise’s architecture, you can utilize the `devise-token-auth` gem, which provides a more seamless integration of token authentication into Devise. This gem offers a set of helper methods and callbacks that automate much of the token handling process. It is very similar to the previous example but has more built in functionality.

```ruby
# Gemfile
gem 'devise_token_auth'
```

```ruby
# app/models/user.rb
class User < ApplicationRecord
  devise :database_authenticatable, :registerable, :recoverable, :rememberable, :validatable
  include DeviseTokenAuth::Concerns::User
end
```

```ruby
# config/routes.rb
Rails.application.routes.draw do
  mount_devise_token_auth_for 'User', at: 'auth'
end
```

With this setup, `devise-token-auth` handles the token creation, storage, and validation for you, eliminating the direct reliance on sessions. You'd then typically send the client a token in the `auth` endpoint and the client needs to store it and include it in the headers of any requests.

**3. Custom Devise Strategy (more involved but most flexible)**

For particularly unique situations or for those who want full control, creating a custom Devise authentication strategy is another option. This involves defining your own authentication logic from scratch. It's more complex but allows for ultimate flexibility to suit unusual requirements.

```ruby
# lib/devise/strategies/api_token_authenticatable.rb
module Devise
  module Strategies
    class ApiTokenAuthenticatable < Authenticatable
      def valid?
        token = request.headers['Authorization']&.split(' ')&.last
        token.present?
      end

      def authenticate!
         token = request.headers['Authorization']&.split(' ')&.last
        user = User.from_jwt(token)
        if user
          success!(user)
        else
          fail(:invalid_token)
        end
      end
    end
  end
end
```

```ruby
# config/initializers/devise.rb
Devise.setup do |config|
  config.warden do |manager|
      manager.strategies.add(:api_token_authenticatable, Devise::Strategies::ApiTokenAuthenticatable)
    end
end
```

```ruby
# app/models/user.rb
class User < ApplicationRecord
  devise :database_authenticatable, :registerable, :recoverable, :rememberable, :validatable, :api_token_authenticatable
  def generate_jwt
    JWT.encode({ id: id, exp: 60.days.from_now.to_i }, Rails.application.credentials.secret_key_base)
  end

    def self.from_jwt(token)
    decoded_token = JWT.decode(token, Rails.application.credentials.secret_key_base, true, { algorithm: 'HS256' })
    find(decoded_token.first['id'])
  rescue JWT::DecodeError
    nil
  end
end
```

This example creates a custom `ApiTokenAuthenticatable` strategy. The `valid?` method checks if the Authorization token is provided. The `authenticate!` tries to find a user through the token. Then it adds this new strategy to Devise.

You need to make a conscious decision of which strategy works best for the situation.

**Recommendations and Resources:**

*   **"Securing APIs with OAuth 2.0" by Aaron Parecki:** This is an excellent, practical guide on securing APIs using OAuth 2.0, though it’s more high-level, it’s useful. While the example here is using JWT, it gives a good overview of API authentication.
*   **"Crafting Rails 4 Applications" by José Valim:** While this book covers an older version of Rails, its sections on authentication patterns and customization remain highly relevant. Understanding how Rails and Devise are put together can help with the more advanced use cases.
*  **`devise` and `devise-token-auth` Github Repositories:** The source code for these gems can be invaluable when learning how everything is connected and how you might want to use these solutions differently.
*   **The official JWT website (jwt.io):** It provides detailed explanations of JWT concepts and how to use them.

In summary, the default session reliance in Devise within an API-only context is more of a default setup issue than a hard requirement. By transitioning to token-based authentication or a custom strategy, you can effectively achieve a stateless API, which is generally more appropriate and often cleaner in most modern api setups. The examples I’ve provided are starting points, and your specific solution will likely need some adjustments depending on the overall architecture.
