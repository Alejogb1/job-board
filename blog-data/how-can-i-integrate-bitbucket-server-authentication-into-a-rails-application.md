---
title: "How can I integrate Bitbucket Server authentication into a Rails application?"
date: "2024-12-23"
id: "how-can-i-integrate-bitbucket-server-authentication-into-a-rails-application"
---

Alright, let’s tackle this. I've seen my share of authentication integrations, and getting Bitbucket Server to play nicely with a Rails app is definitely a common scenario. In a past project, we had a fairly large team relying heavily on Bitbucket for version control. Centralized authentication, using Bitbucket as the source of truth, was crucial for managing access to our internal web tools, which were built using Rails. It was essential to get this seamless, and frankly, not to introduce yet *another* set of credentials users had to juggle. The key here is to understand that Bitbucket Server primarily uses OAuth 2.0 or, in some cases, Basic authentication. I'm going to focus on the OAuth 2.0 approach, which is the cleaner, more secure, and more maintainable route.

So, first off, let's outline the process. We're aiming for a user to initiate a login process through our Rails application. This will redirect them to Bitbucket's authentication page, where they'll log in (if not already), and subsequently, Bitbucket will redirect them back to our Rails app with a special authorization code. This code is then exchanged for an access token, which allows our app to query Bitbucket for user information and potentially make API calls on the user’s behalf.

Essentially, this boils down to these core steps:

1. **Configure Bitbucket Server:** This involves registering your Rails application as a consumer in Bitbucket Server. This step gives you a client id and client secret, vital to the OAuth flow. You'll also need to specify a callback URL for Bitbucket to redirect to after authentication is done.

2. **Implement the OAuth 2.0 Flow in Rails:** This is where the Ruby code comes into play, handling redirection to Bitbucket, authorization code exchange, token storage, and user validation.

3. **Handle User Information and Session Management:** Finally, you will integrate the retrieved user info into your application's authentication logic, establishing a session that recognizes the logged-in user, and potentially using Bitbucket user groups for access control.

Now, to get our hands dirty, let’s dive into some code. I'll use a simplified representation for clarity, but these examples should provide a strong foundational understanding.

**Code Example 1: Initial OAuth Flow setup (in `config/initializers/omniauth.rb`)**

```ruby
Rails.application.config.middleware.use OmniAuth::Builder do
    provider :oauth2,
        'bitbucket_server',
        ENV['BITBUCKET_CLIENT_ID'],
        ENV['BITBUCKET_CLIENT_SECRET'],
        authorize_url: "#{ENV['BITBUCKET_BASE_URL']}/oauth/authorize",
        token_url: "#{ENV['BITBUCKET_BASE_URL']}/oauth/token",
        site: ENV['BITBUCKET_BASE_URL'], #optional if token_url is specific
        client_options: {
          redirect_uri: "#{ENV['APP_BASE_URL']}/auth/bitbucket_server/callback"
        }
end
```

In this snippet, we use the `omniauth-oauth2` gem, a staple for OAuth 2.0 flows in Rails. We define the `bitbucket_server` strategy, providing it with the client id, client secret, and the Bitbucket Server’s base url. Environment variables are used here for best practices to avoid hardcoding secrets into the application. It specifies `authorize_url` for the authorization code request and `token_url` for exchanging the code for the access token. The `client_options` hash is crucial to provide the precise `redirect_uri` where Bitbucket will send users back to. Note that your Bitbucket instance `BASE_URL` must be configured in your server environment.

**Code Example 2: Handling the Callback (in `app/controllers/sessions_controller.rb`)**

```ruby
class SessionsController < ApplicationController
    skip_before_action :verify_authenticity_token, only: :create

    def create
        auth_hash = request.env['omniauth.auth']
        if auth_hash.present?
          user_info = auth_hash['info']
          user = User.find_or_create_from_bitbucket(user_info)

          session[:user_id] = user.id
          redirect_to root_path, notice: 'Logged in successfully.'
        else
            redirect_to root_path, alert: 'Login Failed.'
        end
    end

    def destroy
       session[:user_id] = nil
       redirect_to root_path, notice: 'Logged out successfully'
    end

end
```

This controller handles the callback from Bitbucket after successful (or failed) authentication. The `create` action intercepts the data sent back by OmniAuth which includes the OAuth access token.  We're extracting the user's information and calling `User.find_or_create_from_bitbucket` (this would require more detailed logic depending on your project requirements) to persist the user record in the database or if it doesn't exist, create it, storing the necessary credentials, like the user’s username, email, or other identifiers. We establish a user session by storing the user id in the Rails session and redirect the user to the root path or other specific page. The `destroy` action clears the session, effectively logging out the user.

**Code Example 3: User Model (`app/models/user.rb`)**

```ruby
class User < ApplicationRecord
  def self.find_or_create_from_bitbucket(user_info)
    user = find_by(bitbucket_username: user_info['nickname'])
    return user if user

    create(
      bitbucket_username: user_info['nickname'],
      name: user_info['name'],
      email: user_info['email']
    )
  end
end
```

Here, we define a static method in our `User` model called `find_or_create_from_bitbucket`. It demonstrates how you would retrieve a user record from the database using the user's bitbucket username (`nickname` in the user_info hash) and create one if it doesn't exist. The key point here is the `user_info` hash passed from OmniAuth. It contains all the basic information about the user received from Bitbucket, which you can use to create or update user records in your database.

This example is relatively simplistic, of course. In a real-world scenario, you would likely need to add error handling, token management, additional authorization based on user groups in Bitbucket, and more robust security measures. You could extend the `User` model with fields to hold the access token and handle token refresh when they expire.

For further study and a more detailed understanding of the concepts involved, I’d recommend diving into “OAuth 2.0 Simplified” by Aaron Parecki, which provides an excellent explanation of OAuth 2.0 flows and best practices. Also, the official OmniAuth documentation, specifically the `omniauth-oauth2` gem, is invaluable. For understanding Rails authentication in general, “Agile Web Development with Rails 7” is a solid resource. Finally, reviewing Atlassian's Bitbucket Server API documentation, particularly around OAuth 2.0, is essential to make sure you are extracting the desired user information.

Keep in mind that successful integration requires carefully planning the scope of authentication required, managing tokens securely, and establishing consistent error handling to guide the user in cases where things don’t go as expected. This approach has worked well for me in the past, and while some specific nuances might differ depending on your Bitbucket Server configuration, the core principles should remain consistent.
