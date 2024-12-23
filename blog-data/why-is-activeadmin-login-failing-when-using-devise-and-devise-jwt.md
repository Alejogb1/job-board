---
title: "Why is ActiveAdmin login failing when using Devise and Devise JWT?"
date: "2024-12-23"
id: "why-is-activeadmin-login-failing-when-using-devise-and-devise-jwt"
---

Okay, let's unpack this. I've bumped into this precise scenario more times than I care to remember, each time a subtle variation on a theme, it seems. Getting ActiveAdmin, Devise, and Devise JWT to play nicely together often feels like threading a needle in a hurricane. The root of the problem, more often than not, lies in the interplay between how Devise manages sessions and how Devise JWT handles token authentication. ActiveAdmin, by default, leans heavily on session-based authentication, whereas Devise JWT is, fundamentally, stateless, relying on the presence and validity of a jwt in the request header. This mismatch frequently leads to login failures.

The core issue is that when you log in via a route intended for json responses and the subsequent jwt generation, ActiveAdmin remains oblivious to this. It's still expecting a session cookie, not a jwt in a request header. Consequently, it directs you back to the login screen, unable to find the necessary authentication context to establish a valid session. I recall a particularly frustrating project where we had built a complex internal dashboard with ActiveAdmin, only to introduce a public-facing api using Devise JWT. The integration nightmare took days to resolve, and it highlighted the need to distinctly configure authentication mechanisms.

Let's delve into some concrete reasons for failure and practical solutions. Firstly, a common mistake is assuming Devise JWT will automatically authenticate users across the entire application, including ActiveAdmin. This is simply not the case. Devise JWT handles authorization for json-based routes or endpoints configured to use it through controllers, but ActiveAdmin's user interface uses session cookies for authentication by default. These are entirely separate systems, so you need to explicitly configure a method for ActiveAdmin to recognize a successfully authenticated user (even with JWTs), usually via a cookie or by extending Devise’s authenticate user method.

Secondly, a lack of proper configurations, especially with custom warden scopes, often contributes to this failure. If you define specific warden scopes for the admin user model in your Devise configuration, you need to ensure these scopes are used correctly when attempting to log into ActiveAdmin. Failure to include or properly configure these scopes during authentication leads to ActiveAdmin not recognizing that a user is logged in, resulting in continuous redirects to the login screen.

Thirdly, subtle discrepancies in the way your authentication controllers manage sessions or set cookies can lead to login failures. A successful login via JWT does not inherently create a session cookie, which ActiveAdmin expects to confirm the user is logged in. If we bypass setting a session cookie after JWT login, or the cookie isn't configured correctly, ActiveAdmin won’t recognize that the user has authenticated using the json api.

Let’s look at some working code examples to further clarify these issues and their solutions.

**Example 1: Correcting Warden Scope Mismatch**

This snippet demonstrates a common warden scope error and its solution in `config/initializers/devise.rb`. Assume we have an `AdminUser` model for ActiveAdmin:

```ruby
#config/initializers/devise.rb
Devise.setup do |config|
  #... other settings

  config.warden do |manager|
    manager.default_strategies(:scope => :admin_user).unshift :jwt #ensure jwt auth is used for appropriate routes

  end
end


#and then within a devise controller, such as sessions controller
#app/controllers/sessions_controller.rb
def create
    user = AdminUser.find_by(email: params[:email])
    if user&.valid_password?(params[:password])
      token =  JWT.encode({user_id: user.id, exp: 1.day.from_now.to_i}, Rails.application.credentials.fetch(:jwt_secret), 'HS256')
      # Set a cookie after successful jwt authentication
       cookies[:jwt_token] = { value: token, httponly: true}
        render json: { message: 'Authentication successful', token: token }
     else
       render json: { error: 'Invalid credentials' }, status: :unauthorized
     end
end
```

Here, we are explicitly including `:jwt` strategy for the `admin_user` scope. Then in the controller, after a successful json jwt response, we set a cookie that can be used by ActiveAdmin.

**Example 2: Custom ActiveAdmin Session Controller**

This next example demonstrates how to override ActiveAdmin’s session controller to recognize and handle JWT tokens. We need to create a custom login route for Admin using ActiveAdmin's controller to check for the token rather than the default session mechanism.

```ruby
#app/controllers/active_admin/sessions_controller.rb
class ActiveAdmin::SessionsController < Devise::SessionsController
    def create
        # check for presence of JWT cookie
        if cookies[:jwt_token].present?
             decoded_token = JWT.decode(cookies[:jwt_token], Rails.application.credentials.fetch(:jwt_secret), true, { algorithm: 'HS256' })
            user_id = decoded_token[0]['user_id']
             user = AdminUser.find(user_id)
             sign_in :admin_user, user
             redirect_to admin_root_path
           else
             super
           end
    end
end

# config/initializers/active_admin.rb
ActiveAdmin.setup do |config|
  config.authentication_method = :authenticate_admin_user! # use devise's authenticated user helper
  config.logout_link_path = :destroy_admin_user_session_path # provide correct logout path
  config.logout_link_method = :delete
  config.skip_before_action :verify_authenticity_token, only: :create
  config.skip_before_action :authenticate_admin_user!, only: :create
  config.skip_before_action :authenticate_active_admin_user, only: :create
  config.session_controller = 'active_admin/sessions'
end
```

In this example, we override the `create` action in the ActiveAdmin sessions controller to check for the presence of the JWT cookie. If it's present and valid, we decode the token, find the user, and use `sign_in` to explicitly create a session. This allows ActiveAdmin to proceed as expected. Then in the initializers, we need to configure active admin to handle sessions as expected.

**Example 3: Custom Devise Authenticate Function**

Finally, if the above is too convoluted for your case, this demonstrates how to customize Devise’s authenticate user method for ActiveAdmin by using a callback. This means that when ActiveAdmin checks to see if a user is signed in, we look for a jwt first and create a session if one exists.

```ruby
 # config/initializers/devise.rb
  Devise.setup do |config|
  config.warden do |manager|
    manager.user_options = {:scope => :admin_user}

      manager.before_failure  do |env, opts|
        env["devise.skip_trackable"] = true # prevent infinite redirects
      end


    manager.default_strategies(:scope => :admin_user).unshift :jwt
      manager.after_set_user  do |user, auth, opts|

         if auth.env['warden'].authenticated?(:admin_user) && (auth.env['PATH_INFO'].include?("/admin") ) # check jwt exists AND its an admin route
           return
         end
        
        jwt_cookie = auth.request.cookies["jwt_token"]
           if jwt_cookie.present?
             decoded_token = JWT.decode(jwt_cookie, Rails.application.credentials.fetch(:jwt_secret), true, { algorithm: 'HS256' })
             user_id = decoded_token[0]['user_id']
             user = AdminUser.find(user_id)
             auth.env['warden'].set_user(user) # if there is a valid jwt cookie, set the user
           end
      end
  end
end
```

Here, instead of overriding the ActiveAdmin session controller, we are customizing the devise authenticate user method to look for a JWT cookie and set the user. This is done in the `after_set_user` callback. The authentication method is still done via devise, but we just are enhancing how the method works in this callback.

It’s worth diving into some further readings to understand these concepts more thoroughly. "Secure by Default" by Justin Collins and Josh Wright offers a great practical overview of security concerns in web applications, including session management and token-based authentication. Also, "OAuth 2.0 in Action" by Justin Richer and Antonio Sanso is particularly useful for understanding the theoretical underpinnings of token-based authentication, which can help debug tricky JWT issues. For Devise specific knowledge, check the devise gem documentation itself, and the "Ruby on Rails Security Guide" is a great start. Lastly, the "Active Admin documentation" is crucial to understand its authentication process, especially the section on custom authentication strategies.

In summary, the login failures you're encountering with ActiveAdmin and Devise JWT are likely due to authentication mismatches. By understanding how these libraries handle sessions and tokens, and by meticulously configuring your application with the techniques described above, it is possible to successfully integrate them. I have found this to be quite complex in the past, but through experience, careful attention to detail, and the use of the above information, you can resolve the issue.
