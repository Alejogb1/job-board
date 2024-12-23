---
title: "How can I integrate Okta with a Rails API, Devise, and Vue.js application?"
date: "2024-12-23"
id: "how-can-i-integrate-okta-with-a-rails-api-devise-and-vuejs-application"
---

Alright, let's tackle this. Integrating Okta with a Rails API backend, a Devise authentication layer, and a Vue.js frontend is a multi-layered challenge, but entirely achievable with a clear strategy. I've navigated similar setups numerous times, each with its own nuances, and the crucial aspect lies in understanding the interaction between each component. Back when I was working on a large scale e-commerce platform, we had to make a similar transition from homegrown authentication to Okta, and that experience solidified much of my understanding here. It wasn't always smooth sailing, but the outcome was a much more scalable and secure system.

Fundamentally, the process involves establishing Okta as the identity provider, letting it handle authentication, and then utilizing tokens issued by Okta to authorize requests within your Rails API. The Vue.js frontend will be responsible for initiating the authentication flow with Okta and securely storing/passing those tokens along for subsequent requests. Let's break down each piece and see how they fit together.

First, for the Rails API, we don’t want Devise directly handling authentication anymore; instead, we will use it to manage our user model but defer authentication and authorization to Okta. Specifically, we’ll be using Devise’s "token_authenticatable" strategy alongside a gem like `jwt` to validate tokens from Okta. The goal is to intercept authorization headers, verify the enclosed JWT (JSON Web Token) against Okta’s public key, and then either authorize or reject the incoming request.

Here’s a basic implementation of how to achieve this:

```ruby
# app/models/user.rb
class User < ApplicationRecord
  devise :token_authenticatable, :registerable, :rememberable

  before_save :ensure_authentication_token

  def self.from_token(token)
    # Validate token against the Okta Public Key.
    # Replace placeholders with your Okta specifics.
    begin
      decoded_token = JWT.decode(token, nil, true, {
        algorithm: 'RS256',
        jwks: {
            keys: OktaPublicKeyFetcher.fetch_keys
            }
        })[0]
      # Check for expiry if needed
      # if decoded_token['exp'] < Time.now.to_i
      #     return nil
      # end
      # Attempt to locate the user (e.g. from uid/sub claim)
      user = User.find_by(uid: decoded_token['sub'])
      return user
    rescue JWT::DecodeError, JWT::ExpiredSignature
      return nil
    end
  end
end

# lib/okta_public_key_fetcher.rb
require 'net/http'
require 'json'
class OktaPublicKeyFetcher
  def self.fetch_keys
    uri = URI("https://your-okta-domain.okta.com/oauth2/your-authorization-server/v1/keys") # Adjust accordingly
    response = Net::HTTP.get(uri)
      JSON.parse(response)['keys']
  end
end

# app/controllers/application_controller.rb (or specific controllers)
class ApplicationController < ActionController::API
  before_action :authenticate_user!

  def authenticate_user!
    auth_header = request.headers['Authorization']
    return unauthenticated_request unless auth_header

    token = auth_header.split(' ').last
    user = User.from_token(token)
    if user
      @current_user = user
    else
      unauthenticated_request
    end
  end


  def unauthenticated_request
        render json: { error: 'Not Authorized' }, status: 401
  end

    private

  def current_user
    @current_user
  end

end
```

Here, `User.from_token` handles token validation using the `jwt` gem and fetches the public key using a `OktaPublicKeyFetcher` class which could also be further optimized via caching. This essentially offloads authentication entirely to Okta. The `ApplicationController` now authenticates the user based on the presence and validity of the JWT.

Next, let’s look at the Vue.js application. We need to use Okta’s JavaScript SDK or a similar library to handle the authentication flow. This will typically involve redirecting the user to Okta’s login page, and receiving an access token after successful authentication. The token will then need to be securely stored and sent along with each request to the Rails API as the “Authorization” header.

```javascript
// src/services/auth.js
import { OktaAuth } from '@okta/okta-auth-js';

const oktaAuth = new OktaAuth({
  issuer: 'https://your-okta-domain.okta.com/oauth2/your-authorization-server', // Adjust accordingly
  clientId: 'your-client-id', // Adjust accordingly
  redirectUri: window.location.origin + '/callback',
  scopes: ['openid', 'profile', 'email']
});

export const login = async () => {
  oktaAuth.signInWithRedirect();
};

export const handleCallback = async () => {
  return oktaAuth.handleRedirect();
};

export const getAccessToken = async () => {
  const authState = await oktaAuth.authStateManager.get();
    return authState.accessToken?.accessToken;
};

export const isAuthenticated = async () => {
    return oktaAuth.authStateManager.isAuthenticated();
}

export const logout = async () => {
    oktaAuth.signOut();
};

// In a Vue component example
import { login, handleCallback, getAccessToken, isAuthenticated, logout } from '../services/auth.js';

// Component lifecycle hook, route handler, etc.
async function authHandler() {
    // Check if we are in a callback
    if(window.location.href.includes("callback")){
      await handleCallback()
      this.$router.push('/');
    }
}

async function authenticateRequest() {
  const accessToken = await getAccessToken();
  if (accessToken){
    // Send an authenticated request
    const response = await fetch('/api/my_resource', {
       headers: { 'Authorization': `Bearer ${accessToken}` }
     })
  }
}
```

This Javascript code initializes the Okta SDK, providing functions for login, logout, and extracting the token. The `handleCallback` function handles the post-authentication redirect and stores necessary tokens. The `getAccessToken` function retrieves the token, which is then added to the request headers in the fetch request example.

Finally, Devise plays a supporting role, mainly around the user model and providing a clean way to have "registerable" capabilities if that's part of the broader use case. While we're bypassing its traditional authentication mechanisms via `token_authenticatable`, its integration with the database helps in managing user data after Okta authentication has succeeded. The user model would, for example, store a 'uid' mapping to the 'sub' claim within the Okta JWT for reference later.

```ruby
  # rails routes.rb
   Rails.application.routes.draw do
    devise_for :users, skip: [:sessions, :registrations, :passwords] # Exclude unwanted routes
      namespace :api do
        # API Routes
        get "/my_resource", to: "my#resource"
    end
    get "/callback", to: "vue_router_handler#index"
    root to: "vue_router_handler#index" # Example route for vue app
  end
```

Here, we exclude default Devise routes like sessions since Okta handles that flow and introduce a simple API endpoint to test the authentication implementation. This minimal routing configuration will make it clearer how the authentication and authorization flow is routed to your application.

For a deeper dive into understanding JWTs, I'd suggest checking out the excellent book "JSON Web Tokens" by O'Reilly. Additionally, the official Okta documentation for their JavaScript SDK and their various authentication flows is invaluable. For a broader perspective on authentication and authorization patterns, "OAuth 2.0 in Action" by Manning provides a robust theoretical and practical foundation. I’ve found these resources indispensable throughout many development projects with similar authentication challenges.

This approach, while requiring careful configuration, results in a more secure and manageable authentication system. Remember that proper security practices should always be at the forefront, including secure handling and storage of tokens on the frontend, especially if you are using a non-secure browser environment. This setup should serve as a solid starting point for integrating Okta, Devise, and Vue.js into a unified system.
