---
title: "How do I add 2FA to Rails Devise?"
date: "2024-12-16"
id: "how-do-i-add-2fa-to-rails-devise"
---

Alright, let's tackle two-factor authentication (2fa) with devise in rails. It's a topic I've dealt with a fair bit, having rolled my own implementation more than once back before the robust gems we have today became common practice. It's not overly complex, but there are definitely some best practices that can smooth the path significantly.

First, let's be clear: rolling your own 2fa solution is, generally speaking, a bad idea these days. There are solid, well-maintained libraries that handle the intricacies of generating secrets, verifying tokens, and, crucially, protecting against common vulnerabilities. The primary danger of diy implementations is that you tend to be unaware of edge cases and less frequent exploits, and your in-house solution might lack the security audit that publicly vetted code will have. So, while I have experience building 2fa from scratch, I absolutely do *not* recommend it.

For rails and devise, the go-to gem is usually `devise-two-factor`. It's quite mature, widely used, and integrates pretty seamlessly with the devise authentication framework. I've personally deployed it in several projects, and it's proven to be reliable and relatively straightforward.

Let's break down the process, focusing on what you'll need to do, and some potential gotchas. We’ll start with installation and configuration and then we'll look at implementing it with some illustrative examples.

**Installation and Basic Setup**

The first step, predictably, is adding the gem to your `Gemfile`:

```ruby
gem 'devise-two-factor'
```

After that, you run `bundle install`. This is followed by a `rails generate devise_two_factor:install`. This generator sets up the necessary migrations and adds some configuration to your devise configuration initializer (usually found in `config/initializers/devise.rb`).

At this point, the setup is still mostly behind the scenes; you haven't actually enabled 2fa for any users. To do that, you need to update your user model. Here's an example of what that will typically look like:

```ruby
# app/models/user.rb
class User < ApplicationRecord
  devise :database_authenticatable, :registerable,
         :recoverable, :rememberable, :validatable, :two_factor_authenticatable,
         :two_factor_backupable
end
```
Notice the inclusion of `:two_factor_authenticatable` and `:two_factor_backupable`. The former enables the core 2fa logic; the latter provides the mechanism for handling backup codes, which are critical for users who lose access to their authentication app. These are essential to the `devise-two-factor` functioning correctly.

The migrations will add fields to your users table for storing 2fa secrets and backup codes. Run `rails db:migrate` after making these changes.

**User Interface Considerations**

Now that the engine is configured, we need to think about the user interface. Users need a way to enable 2fa, generate a qr code for their authenticator app, store their backup codes, and, crucially, handle the login process.

The gem provides a helpful set of helpers, which are often used directly in the view layer. However, you will need to provide the actual ui elements. Here's a simplified code example focusing on the key operations. This is a snippet which assumes you have generated some view templates from devise using `rails generate devise:views`:

```erb
<!-- app/views/devise/registrations/edit.html.erb -->
<h2>Edit Account</h2>

<%= form_for(resource, as: resource_name, url: registration_path(resource_name), html: { method: :put }) do |f| %>
  <%= devise_error_messages! %>

  <div class="field">
    <%= f.label :email %><br />
    <%= f.email_field :email, autofocus: true, autocomplete: "email" %>
  </div>

   <div class="field">
    <% if resource.otp_required_for_login? %>
      Two-Factor Authentication is enabled.
       <%= link_to 'Disable Two-Factor Authentication', disable_two_factor_path, method: :post %>
       <br>
       <p>You may want to record these backup codes for use if you lose your authentication device:</p>
       <% resource.backup_codes.each do |code| %>
          <p><%= code %></p>
       <% end %>

    <% else %>
      <%= link_to 'Enable Two-Factor Authentication', enable_two_factor_path %>
    <% end %>
   </div>

  <div class="actions">
    <%= f.submit "Update" %>
  </div>
<% end %>
```

This is a simplified version, and you'll likely want to add styling, more detailed instructions, and proper error handling. However, it highlights some core elements: first, the link for disabling and enabling 2fa (which goes to a provided controller path), and second, a presentation of the backup codes if generated.

**Controller Customization**

The `devise-two-factor` gem provides the controller logic for enabling and disabling 2fa. If you find you need more customization, such as logging or redirects, you can subclass the `Devise::TwoFactorAuthenticationController` as shown below.

```ruby
# app/controllers/users/two_factor_authentication_controller.rb
class Users::TwoFactorAuthenticationController < Devise::TwoFactorAuthenticationController
   def create
    super
    # Logging code can go here.
    Rails.logger.info("User #{current_user.id} enabled 2fa.")

    # Custom redirects can go here if needed

   end

  def destroy
    super
    Rails.logger.info("User #{current_user.id} disabled 2fa.")
  end
end
```

Then you need to tell devise to use your custom controller, by adding this to your devise configuration in the config/initializers/devise.rb file.

```ruby
  config.controllers = {
    two_factor_authentication: 'users/two_factor_authentication'
  }
```

This example demonstrates overriding the `create` and `destroy` actions to add logging. You can tailor this controller to fit your application's specific requirements. For example, if you want to redirect a user to a different page after they enable or disable 2fa, you would add the relevant redirect_to to those actions in this controller.

**Essential Resources and Further Reading**

While the `devise-two-factor` gem takes care of the majority of the heavy lifting, it’s vital to understand the underlying principles of 2fa. I strongly recommend reading up on the rfc 6238 - time-based one-time password algorithm (totp), as that is often the underpinnings of authentication apps.  Understanding how totp works helps you in debugging issues. Additionally, the OATH (open authentication) standard is also crucial. Good sources for general security principals are books such as "Security Engineering" by Ross Anderson and "Serious Cryptography" by Jean-Philippe Aumasson. For a more focused look at secure web development, “The Tangled Web” by Michal Zalewski is excellent. For rails-specific resources, I’d recommend “Agile Web Development with Rails” by Sam Ruby, et al., though you need to note that it's not going to cover the 2fa area specifically, so you will still need to apply what you've learned through the resources mentioned earlier.

**Final Thoughts**

Implementing 2fa with devise-two-factor is generally quite smooth, but, as with any security feature, you should approach it methodically, paying close attention to the user experience and thoroughly testing the implementation. I would also suggest making sure that your backup code procedure is clear and user-friendly so that users can access their accounts even when they no longer have access to their authentication applications. In my experience, the best approach is to iterate in small increments, testing at every stage. Don’t just assume that it all works on the first try. And don't, under any circumstances, roll your own 2fa code; you'll almost certainly introduce vulnerabilities. Using battle-tested solutions such as devise-two-factor is by far the best approach.
