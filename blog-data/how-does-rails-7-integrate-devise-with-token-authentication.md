---
title: "How does Rails 7 integrate Devise with token authentication?"
date: "2024-12-23"
id: "how-does-rails-7-integrate-devise-with-token-authentication"
---

, let's delve into how Rails 7 handles Devise alongside token authentication. This is a subject I’ve spent considerable time on in past projects, specifically when scaling a microservices architecture for an e-commerce platform. Getting this combination just so is crucial, and there are a few key patterns to grasp, especially now with the shift towards apis and front-end frameworks taking over the UI layer.

Fundamentally, Devise, a flexible authentication solution, primarily operates on cookie-based sessions. When transitioning to an api-centric approach, where a client might be a javascript application or a mobile app, cookie-based auth becomes less practical. This is where token authentication via something like JWT (JSON Web Tokens) comes into play. The integration with Devise in Rails 7 isn't built into the core of Devise itself; it requires layering some additional logic.

Here’s how I typically approach it, based on my experiences: the goal isn't to completely replace Devise's functionality, but rather to extend it and adapt it to a token-based system.

First, you’ll want to configure Devise to be api friendly and to disable cookie sessions. This typically involves setting `:api` to `true` in your devise model configuration and configuring specific modules. For instance, you might only enable `:registerable` if you want users to be able to create accounts directly through the api. Here’s a basic example within the `app/models/user.rb` file:

```ruby
class User < ApplicationRecord
  devise :database_authenticatable, :registerable,
         :recoverable, :rememberable, :validatable,
         :jwt_authenticatable, jwt_payload: lambda { |user| {
           'id' => user.id,
           'email' => user.email,
           'created_at' => user.created_at.to_i,
          'exp' => Time.now.to_i + 1.hours.to_i
         } }
end
```

The key addition here is `:jwt_authenticatable`. This module, often provided by gems like `devise-jwt`, integrates jwt functionality and assumes that incoming requests will have a bearer token in the `Authorization` header. I've found `jwt_payload` to be crucial for customizing the data included within your token. I've included user id, email, creation timestamp, and an expiry timestamp within this payload. This timestamp is vital to maintaining a secure token and managing token expirations. Make sure to include this `exp` value, or you might run into weird behavior later on.

Next, we have to define how to handle token creation upon successful login, and also what happens when an authentication fails. This often comes down to creating the api endpoint responsible for handing out the access tokens. Here's a snippet of the controller logic, often placed within something like `app/controllers/api/v1/sessions_controller.rb`:

```ruby
class Api::V1::SessionsController < Devise::SessionsController
  respond_to :json

  def create
    user = User.find_by(email: params[:email])

    if user&.valid_password?(params[:password])
      token = JWT.encode(user.jwt_payload, Rails.application.credentials.secret_key_base)
      render json: { token: token }, status: :ok
    else
      render json: { error: 'invalid email or password' }, status: :unauthorized
    end
  end


    private

  def respond_with(resource, _opts = {})
    render json: { message: "Logged in successfully." }, status: :ok
  end

  def respond_to_on_destroy
     head :no_content
  end

end
```

This example is streamlined. `JWT.encode` takes our custom payload and the application's secret key to generate the token. The `respond_with` and `respond_to_on_destroy` methods are there to prevent Devise from redirecting a user to a page after a successful login, which is the default behaviour when rendering HTML. Instead we output json. I've implemented a simple error response if the login fails. The actual implementation might be more complex, especially with considerations for rate limiting or handling multiple simultaneous logins. You’d handle user logout similarly through destroying a current session, although with tokens you don't often need to handle that in the server, you'll just manage the access token lifetime in your front end.

Finally, to ensure that every call to protected routes are authenticated via token, make sure to update your `ApplicationController` to use `devise_jwt_authenticatable`. The exact implementation depends on the `devise-jwt` gem that is being used. Here's an example of that within the `app/controllers/application_controller.rb` file:

```ruby
class ApplicationController < ActionController::API
  include Devise::Controllers::Helpers
  before_action :authenticate_user!, unless: :devise_controller?

end
```

This example makes use of the devise `authenticate_user!` method. The `unless: :devise_controller?` is extremely important to add here as this skips the authentication for the login and register routes. If this `unless` clause isn't added, it will not be possible to log into your application as the controller will attempt to authenticate the request with the access token, but will not be able to find a valid token, thus denying access to the login route.

Now, regarding specific things you should research further, I’d highly recommend:

*   **"Designing Data-Intensive Applications" by Martin Kleppmann.** This book provides invaluable insights into the complexities of building scalable systems, including detailed discussions of authentication and authorization patterns. While not specifically about Rails or Devise, it offers a deeper understanding of the underlying architectural principles. The sections on distributed systems and security considerations are particularly relevant.
*   **"OAuth 2.0" RFC 6749:** This is the authoritative specification for the OAuth 2.0 protocol. Understanding the protocol will give you a solid foundation for implementing more advanced security mechanisms. While you might not be implementing the full OAuth specification initially, knowing the concepts behind it will help you make informed decisions when integrating token-based auth. Pay close attention to the refresh token flow, particularly if you anticipate your access tokens expiring relatively quickly.
*   **The official Rails API documentation:** The API documentation for Rails gives a good outline of what each function does. In particular the documentation regarding the `before_action` directive, the `ActionController` class, and the `Devise` gem documentation. This would be crucial when trying to implement a more advanced version of token authentication.

The overall key takeaway here is that, while Rails 7 provides the tools, and Devise provides a great structure, the specifics of integration depend heavily on your particular needs. The combination of a session-based auth system with a more lightweight token-based approach gives flexibility, allowing you to adapt to different authentication needs throughout the application. My approach is to lean heavily on established libraries and well-defined principles, ensuring the system is secure, maintainable, and scalable.
