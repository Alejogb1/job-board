---
title: "How can I add 2FA to Rails Devise?"
date: "2024-12-23"
id: "how-can-i-add-2fa-to-rails-devise"
---

Okay, let's talk about bolstering the security of your Rails application with two-factor authentication (2FA) using Devise. It's a crucial step these days, and thankfully Devise, a staple authentication solution for Rails, is quite extensible in this regard. I’ve actually tackled this on a project where we were handling sensitive user data for a financial platform; the experience definitely hammered home just how critical this security layer is. We weren’t relying solely on passwords, and neither should you.

My approach tends to favor simplicity and maintainability. Therefore, while you'll find several options out there – from building your own custom solution to using a full-fledged service – I typically lean towards using the 'devise-two-factor' gem. It's well-maintained, integrates seamlessly with Devise, and offers a good balance between ease of use and flexibility. The gem uses the Time-based One-time Password (TOTP) algorithm, which is widely accepted and supported by authenticator apps like Google Authenticator, Authy, and many others.

So, let's break down the process step-by-step.

**First, Installation and Setup**

The first order of business is to include the `devise-two-factor` gem in your `Gemfile`:

```ruby
gem 'devise-two-factor'
```

Then, run `bundle install` to get the gem and its dependencies installed. After that, you need to add the following columns to your users table. For this, you would run a migration:

```ruby
rails generate migration AddTwoFactorColumnsToUsers
```

And then populate that migration with the following content:

```ruby
class AddTwoFactorColumnsToUsers < ActiveRecord::Migration[7.0]
  def change
    add_column :users, :otp_secret_key, :string
    add_column :users, :second_factor_enabled, :boolean, default: false, null: false
    add_column :users, :encrypted_otp_secret_key_iv, :string
    add_column :users, :encrypted_otp_secret_key, :string
    add_column :users, :second_factor_attempts_count, :integer, default: 0
  end
end
```

Run the migration using `rails db:migrate`.

Now, in your `User` model (or whichever model you’re using with Devise), you need to make some changes. Specifically, you need to include the `devise_two_factorable` module. Your model should look something like this:

```ruby
class User < ApplicationRecord
  devise :database_authenticatable, :registerable,
         :recoverable, :rememberable, :validatable, :two_factor_authenticatable,
         :two_factor_backupable
end
```

This line, `:two_factor_authenticatable`, enables the core functionality for 2FA, and `:two_factor_backupable` provides methods and columns for creating and recovering backup codes if a user loses their authenticator app.

**Implementation Workflow**

The general workflow will involve these core steps:

1.  **Generating a secret:** When a user enables 2FA, you’ll generate a unique secret key that will be used by the authenticator app and the server to generate TOTP codes.
2.  **Displaying a QR code:** The application will display a QR code, which the user scans using their authenticator app.
3.  **Verifying the first TOTP code:** After scanning, the user will provide a TOTP code to confirm that the setup process was successful.
4.  **Enabling 2FA:** If verification is successful, the `second_factor_enabled` flag is set to true.
5.  **Authentication process:** Subsequently, when a user logs in, they’ll have to provide both their username/password *and* a TOTP code.

**Code Snippets**

Let’s explore this with a few code examples. Firstly, here is a basic implementation for the controller which handles enabling 2FA. I’m assuming you are already familiar with the Devise conventions and have a working setup, with routes defined. We'll assume a route for the following action.

```ruby
# app/controllers/users/two_factor_settings_controller.rb
class Users::TwoFactorSettingsController < ApplicationController
  before_action :authenticate_user!

  def new
    unless current_user.second_factor_enabled?
      @user = current_user
      @qr_code = RQRCode::QRCode.new(@user.provisioning_uri, level: :h).as_svg(offset: 0, color: '000', shape_rendering: 'crispEdges', module_size: 6)
    else
      redirect_to profile_path, alert: "Two-factor authentication is already enabled."
    end
  end

  def create
    if current_user.second_factor_enabled?
      redirect_to profile_path, alert: "Two-factor authentication is already enabled."
      return
    end

    unless current_user.validate_and_consume_otp(params[:otp_attempt])
      flash[:alert] = "Invalid verification code. Please try again."
      redirect_to new_users_two_factor_setting_path
      return
    end

    current_user.second_factor_enabled = true
    if current_user.save
        flash[:notice] = "Two-factor authentication has been enabled successfully!"
        redirect_to profile_path
    else
      flash[:alert] = "There was an error enabling two-factor authentication"
      redirect_to new_users_two_factor_setting_path
    end
  end

  def destroy
    current_user.second_factor_enabled = false
    current_user.otp_secret_key = nil
    if current_user.save
      flash[:notice] = "Two-factor authentication has been disabled successfully."
      redirect_to profile_path
    else
      flash[:alert] = "There was an error disabling two-factor authentication."
      redirect_to profile_path
    end
  end
end
```

You'll notice the use of `RQRCode` here, which is used for generating the qr code based on the provisioning URI. You need to ensure you have `gem 'rqrcode'` added to your Gemfile and bundled.
In your view, the most basic rendering of this might look something like this:

```erb
<!-- app/views/users/two_factor_settings/new.html.erb -->
<h1>Enable Two-Factor Authentication</h1>
<% if @user.present? %>
  <p>Scan the QR code below using your authenticator app:</p>
  <%= raw @qr_code %>
  <br />

  <%= form_with url: users_two_factor_setting_path do |form| %>
    <div class="field">
      <%= form.label :otp_attempt, "Verification Code" %><br />
      <%= form.text_field :otp_attempt %>
    </div>
    <div class="actions">
      <%= form.submit "Verify and Enable" %>
    </div>
  <% end %>
<% else %>
  <p>Two-Factor Authentication is already enabled.</p>
<% end %>
```

Then, when users log in they are redirected to a `verify_otp` action which will look something like this (I’m omitting most of the devise boilerplate code):

```ruby
# app/controllers/users/sessions_controller.rb
class Users::SessionsController < Devise::SessionsController
  def create
    self.resource = warden.authenticate!(auth_options)
    if resource.second_factor_enabled?
      session[:otp_user_id] = resource.id
      redirect_to users_verify_otp_path and return
    else
      set_flash_message!(:notice, :signed_in)
      sign_in(resource_name, resource)
      respond_with resource, location: after_sign_in_path_for(resource)
    end
  end
end
```

And finally, the controller to verify the OTP looks like this:

```ruby
# app/controllers/users/verify_otp_controller.rb
class Users::VerifyOtpController < ApplicationController
    before_action :redirect_if_not_two_factor_enabled, only: :new
    before_action :redirect_if_already_signed_in, only: :new
    before_action :find_user, only: :create

  def new
  end

  def create
    if @user && @user.validate_and_consume_otp(params[:otp_attempt])
        sign_in(resource_name, @user)
        set_flash_message!(:notice, :signed_in)
        session.delete(:otp_user_id)
        respond_with @user, location: after_sign_in_path_for(@user)
    else
      @error_message = "Invalid verification code. Please try again."
      render :new
    end
  end

  private
  def find_user
    @user = User.find_by(id: session[:otp_user_id])
    unless @user
      redirect_to new_user_session_path, alert: "Session timed out. Please log in again."
    end
  end

  def redirect_if_already_signed_in
      redirect_to root_path and return if user_signed_in?
  end

    def redirect_if_not_two_factor_enabled
        user = User.find_by(id: session[:otp_user_id])
        redirect_to new_user_session_path, alert: "Session timed out. Please log in again." and return unless user&.second_factor_enabled?
    end
end
```

**Important Considerations**

*   **Error Handling:** The code snippets above are for demonstration purposes. Real-world applications will need robust error handling and input validation to address corner cases.
*   **Backup Codes:** Don't forget to implement the backup code recovery process. This involves generating a set of one-time use codes at the same time that you enable 2FA, and storing them securely so a user can regain access if they lose their authenticator app.
*   **Rate Limiting:** Implement rate limiting on failed login attempts to avoid brute-force attacks on the 2FA code entry.
*   **Security:** Make sure your secret keys are stored in an encrypted format, at rest in your database, using the mechanism supplied by the gem, and use appropriate access control on the data layer.
*   **User Experience:** Consider how users enable, disable, or recover 2FA as part of your overall design. It should be relatively intuitive, and also should provide useful error messaging.

**Further Reading**

I highly recommend delving into the following resources for a deeper understanding:

*   **RFC 6238 (TOTP):** *Time-Based One-Time Password Algorithm*. The foundational document for TOTP. A deep understanding here is crucial.
*   **The Ruby on Rails Security Guide:** The official Rails security documentation is an invaluable resource for overall security best practices with Rails.
*   **Devise Gem documentation:** Explore the comprehensive Devise documentation, and also the specific documentation for the `devise-two-factor` gem.
*   **OWASP Authentication Cheat Sheet:** Provides a lot of best practice advice for implementing secure authentication mechanisms including 2FA.

Implementing 2FA is not just a checkmark on a security audit, but it’s a commitment to safeguarding user data. These steps, paired with the right resources, will move you towards a more secure Rails application. Remember to test thoroughly and to consider user experience with each change you implement. The goal is a balanced approach that combines robust security with ease of use.
