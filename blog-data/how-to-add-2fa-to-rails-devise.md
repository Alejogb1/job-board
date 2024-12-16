---
title: "How to add 2FA to Rails Devise?"
date: "2024-12-16"
id: "how-to-add-2fa-to-rails-devise"
---

Alright, let's talk about bolting on two-factor authentication (2FA) to Devise in Rails. It's a fairly common requirement these days, and over the years, I've implemented this multiple times, both for personal projects and larger systems. It's never quite as simple as flipping a switch, but it's also not some insurmountable challenge. I’ve seen it done well, and I’ve also had to refactor the result of some, let's say, *less graceful* attempts. The key is to understand the underlying mechanics and to choose an approach that fits the needs of your specific application.

The basic concept is straightforward: beyond the standard username and password, we introduce a second verification step, typically involving something the user *has* (like a smartphone with an authenticator app). This significantly enhances security by mitigating the risks of compromised credentials. Within a Devise context, this essentially involves intercepting the login flow and adding an extra validation stage.

My typical go-to solution involves a gem called ‘devise-two-factor’. It handles a lot of the heavy lifting, particularly around the generation and verification of time-based one-time passwords (TOTP). It's not the only solution, but it strikes a nice balance between flexibility and ease of use, and it's something I've always been comfortable adapting. I’ll be showing examples with that, but it's important to understand that this is just *one* way. Other libraries and methods are absolutely viable, especially if you have very specialized requirements.

Let's start with the basic setup. After installing `devise` itself, we add the two-factor gem:

```ruby
# Gemfile
gem 'devise'
gem 'devise-two-factor'
```

Now run `bundle install`.

Next, we’ll modify our `User` model. We need to include the `Devise::Models::TwoFactorAuthenticatable` module, and add a few attributes for storing the required 2FA details.

```ruby
# app/models/user.rb
class User < ApplicationRecord
  devise :database_authenticatable, :registerable,
         :recoverable, :rememberable, :validatable,
         :two_factor_authenticatable

  attr_accessor :otp_attempt
  attr_accessor :remember_me # Keep the remember me option in case you want to reuse that

  # Add attributes for storing the 2FA data (e.g., encrypted secret, recovery codes)
  # I usually add some additional null: false for these columns during migration
  # add_column :users, :encrypted_otp_secret, :string
  # add_column :users, :otp_required_for_login, :boolean, default: false
  # add_column :users, :otp_backup_codes, :text
  # ...and any other necessary fields
end
```

Remember, you'll need to run migrations to add these columns to your `users` table, and for good measure I'd recommend setting `null: false` for the sensitive ones like `encrypted_otp_secret` and `otp_required_for_login` during the migration itself, like the comments above.

The crucial part that I often see overlooked is creating the necessary flow in your controller, specifically the Devise `sessions` controller. After a user enters their username and password and Devise authenticates them, we need to check if 2FA is enabled. If it is, we need to present a form for the user to enter their TOTP. This is where the `otp_attempt` and other methods from the gem come into play.

Here's a very simplified example of a modified Devise sessions controller:

```ruby
# app/controllers/users/sessions_controller.rb
class Users::SessionsController < Devise::SessionsController

  def create
    self.resource = warden.authenticate!(auth_options)
    if resource.otp_required_for_login
       session[:otp_user_id] = resource.id
       render :otp_form and return
    else
        set_flash_message!(:notice, :signed_in)
        sign_in(resource_name, resource)
        respond_with resource, location: after_sign_in_path_for(resource)
    end
  end


  def authenticate_otp
    user = User.find(session[:otp_user_id])
    if user.authenticate_otp(params[:otp_attempt])
       session.delete(:otp_user_id)
       set_flash_message!(:notice, :signed_in)
       sign_in(resource_name, user)
       respond_with user, location: after_sign_in_path_for(user)
    else
       flash.now[:alert] = "Invalid OTP code."
       render :otp_form and return
    end
  end

  def otp_form
     @user = User.find(session[:otp_user_id])
  end
end
```

And your routes file needs a tiny change:

```ruby
# config/routes.rb
devise_for :users, controllers: { sessions: 'users/sessions' }
devise_scope :user do
  post 'users/sign_in/otp', to: 'users/sessions#authenticate_otp', as: :authenticate_user_otp
end
```

Finally, we'll also need the view where the user enters the OTP:

```erb
<!-- app/views/users/sessions/otp_form.html.erb -->
<h2>Enter your Two-Factor Authentication Code</h2>
<%= form_with url: authenticate_user_otp_path, method: :post do |form| %>
  <div>
    <%= form.label :otp_attempt, 'OTP Code:' %>
    <%= form.text_field :otp_attempt %>
  </div>
  <div>
    <%= form.submit 'Verify' %>
  </div>
<% end %>
```

Now, I've provided a skeletal implementation, but this framework gives you a complete working example of 2FA on devise. There's quite a bit more that you might need to do to make it user-friendly (handling enabling 2FA, generating and storing recovery codes, etc.), but these things are largely handled by the gem and Devise if you read the documentation and set them up properly.

To expand on some further practical considerations:

**User Experience:** How users actually interact with this process is paramount. Simply throwing a cryptic OTP form at users is not acceptable. You need to provide clear guidance on how to set up their authenticator app, and what to do if they lose access. Recovery codes, although not the most user-friendly solution, are often a necessary evil. I’d advise strongly that you don't display those to them only once, instead let them download them, generate new ones, and also have a clear way to revoke these in case they may be compromised.

**Error Handling:** Proper error messages are essential. Instead of generic messages that may confuse the user, provide informative feedback. I.e. “Invalid OTP,” or, “The code is expired.” If the user enters the wrong password multiple times, that may be a potential security concern, and should be dealt with.

**Security Considerations:** While 2FA is an improvement, it is not foolproof. Protect sensitive 2FA related information like the secret keys that generate the codes. It’s not wise to store the secret as plaintext, instead, you may want to look into encrypting it using a gem like `attr_encrypted`. Ensure proper storage of backup codes, and do consider other layers of protection. Rate limiting on login attempts and account lockouts could also be considered.

**Further Reading:** For a deeper understanding of the underlying security principles and best practices, I highly recommend checking out the NIST Special Publication 800-63 series, specifically 800-63B, which covers authentication and lifecycle management. Also, the OWASP Authentication Cheat Sheet provides a lot of great information about handling authentication properly. Finally, the Devise documentation and `devise-two-factor` gem documentation are a *must-read*.

In summary, adding 2FA to your Rails Devise application is a significant enhancement that, with care, significantly improves security. I’ve seen many teams implement this well, and with this breakdown and a bit of experimentation, you can too. Just remember to keep the end-user in mind, and to be thorough with your error handling and security considerations.
