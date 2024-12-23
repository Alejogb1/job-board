---
title: "How to implement Devise JWT authentication?"
date: "2024-12-23"
id: "how-to-implement-devise-jwt-authentication"
---

Alright, let's talk about securing an application with Devise and JSON Web Tokens (JWTs). I've spent a fair amount of time on this specific configuration over the years, and I've seen various pitfalls that can arise, so let me share my perspective. Specifically, implementing Devise with JWT authentication involves shifting away from Devise’s standard session-based approach to a token-based one, and it introduces its own set of considerations. The core idea here is to authenticate users via Devise, then upon successful authentication, issue a JWT that the client will use for subsequent requests.

The foundational shift from sessions to JWTs requires a different mental model. You're no longer relying on server-side storage of session data; rather, the client stores the token, and the server verifies it. I'll break it down into the steps and provide code examples, focusing on Ruby on Rails since that's where I've primarily used Devise.

First, we need to include and configure the necessary gems. The canonical approach revolves around the `devise` gem itself for the core user management and authentication, and `jwt` to handle token generation and verification. Further, we will need something to actually handle the token issuance upon successful login. I've often used `warden-jwt_auth` in the past, which offers a solid integration with Devise's Warden framework. So we will want to include `gem 'devise'`, `gem 'jwt'`, and `gem 'warden-jwt_auth'`.

Once those gems are in place, we move to configuring `devise.rb`. The configuration snippet below shows the critical aspects we are going to focus on:

```ruby
# config/initializers/devise.rb
Devise.setup do |config|
  # ... other configurations ...

  config.jwt do |jwt|
     jwt.secret = Rails.application.credentials.secret_key_base
     jwt.dispatch_requests = [
      ['POST', %r{^/users/sign_in$}] # Sign in route
    ]
     jwt.revocation_requests = [
       ['DELETE', %r{^/users/sign_out$}] # Sign out route
     ]
      jwt.expiration_time = 1.hour
    jwt.token_response_body = lambda { |user, _request|
      {
        auth_token: user.jwt_token,
        user_id: user.id,
        email: user.email
      }
    }
  end
    config.warden do |manager|
      manager.default_strategies(:scope => :user).unshift :jwt_auth
      manager.failure_app = lambda { |env|
        [401,
         { 'Content-Type' => 'application/json' },
         [{ "error" => "Unauthorized" }.to_json]]
      }
    end
end
```

Here we define the jwt configuration block. `jwt.secret` is the most important part; it must be kept secret (of course) and should be derived from some secure variable like `secret_key_base`. The `dispatch_requests` specifies routes where we intend to generate tokens and the `revocation_requests` sets up the endpoints where tokens are invalidated.  I’ve found `expiration_time` to be quite important; generally I set a short expiration time, and then implement some mechanism for refreshing, but let's leave that out for now for simplicity. Note the token response body: this specifies the data to be returned to the client upon successful authentication. The lambda will then return the auth_token (which will be the JWT), user ID and email. Lastly, the warden configuration ensures that authentication is handled by `jwt_auth` first and in case of failures, sends back a structured error message.

Now, the most important part of this setup, where the user model generates the actual JWT:

```ruby
# app/models/user.rb
class User < ApplicationRecord
  # Include default devise modules. Others available are:
  # :confirmable, :lockable, :timeoutable, :trackable and :omniauthable
  devise :database_authenticatable, :registerable,
         :recoverable, :rememberable, :validatable, :jwt_authenticatable, jwt_revocation_strategy: JwtDenylist

  def jwt_token
      JwtAuth::Token.generate(self)
  end
  # A denylist for JWTs to effectively 'sign out'.
  class JwtDenylist < Warden::JWTAuth::Strategies::RevocationStrategies::Denylist
    def self.jwt_revoked?(payload, user)
      denylisted_token = find_by(jti: payload['jti'])
      denylisted_token&.revoked?
    end

    def self.revoke_jwt(payload, user)
      create!(jti: payload['jti'])
    end
  end

  def after_jwt_payload
      { 'jti' => SecureRandom.uuid }
  end

end
```

Here we enable `jwt_authenticatable` which enables Devise to look for a token in a header when performing authentication (in `Authorization: Bearer {token}` format). I’ve also added `jwt_revocation_strategy`, which will be used to invalidate tokens. We create `JwtDenylist`, which stores jti values and is used to implement the sign-out function (effectively blacklisting a particular jwt). The `jwt_token` method simply generates the JWT using `JwtAuth::Token.generate`, and `after_jwt_payload` provides a `jti` value (a unique identifier) to be stored in the token itself, this jti will be used to revoke the token.

Finally, lets implement the denylist model:

```ruby
# app/models/jwt_denylist.rb
class JwtDenylist < ApplicationRecord
    include JwtAuth::Denylist

    self.table_name = 'jwt_denylist'
end
```

This completes the minimal example of a working authentication mechanism. Note, this model needs to be implemented in the database, which you can achieve by creating a migration `rails g model jwt_denylist jti:string revoked:boolean`.

Now, let's say, for example, you are implementing an API. Your client application will first post a username and password to `/users/sign_in`. If the authentication is successful, it will return a json response that looks like this, including your auth token:

```json
{
    "auth_token": "eyJhbGciOiJIUzI1NiJ9.eyJ1c2VyX2lkIjoxLCJleHAiOjE2OTkxNzAxMDksImp0aSI6IjQ4MGU4YzU5LWFkODUtNGI1Mi05YjQ0LWI2ODUwMjcyNzI0YiJ9.MvO2G40T2iYtF_G0gI5q_m4R0dJ5j2WnU5g0d5L0d",
    "user_id": 1,
    "email": "test@example.com"
}
```
The client should then store the `auth_token` and send it in the header `Authorization: Bearer {auth_token}` for subsequent requests. On the server side, Devise will automatically intercept the header, extract the token, and then look it up. If it is valid and the token isn’t in the `jwt_denylist` table, the server will allow the request to proceed.

Now, let's talk about considerations. First, expiration times. Short token expiration periods are crucial to mitigate the risk of stolen tokens being used. Consider implementing a refresh token mechanism where the main token has a short expiration time, and a separate, longer-lived refresh token is used to get new authentication tokens. Additionally, store your secret key properly and securely – `Rails.application.credentials.secret_key_base` is usually a solid approach for Rails. You might consider rotation of the secret key periodically, but with care and planning as this would invalidate all existing tokens. Finally, the `JwtDenylist` approach allows for sign-out functionality, but this will grow quickly with more logins and sign-outs. I’ve often considered moving this to a Redis cache or similar for performance, especially in high-traffic situations.

As far as further reading, I would highly recommend “Programming Phoenix” by Chris McCord, Bruce Tate, and José Valim (though its not about Ruby, its excellent in the general topic of JWT and authentication).  For a deeper dive into JWTs themselves, explore the original RFC 7519. And of course, the Devise documentation itself is an essential resource. For deeper understanding of warden, the Rack Middleware stack (which warden uses) can be beneficial. Understanding the underlying mechanism will allow you to reason about the code above much more effectively.

This setup, while appearing quite straightforward, involves several critical components. Pay close attention to the security implications, especially the secret key management and token expiration. Proper implementation of JWT authentication with Devise gives you a robust, scalable, and secure mechanism for securing your web applications. Hope this helped.
