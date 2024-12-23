---
title: "How does Hotwire handle user login?"
date: "2024-12-23"
id: "how-does-hotwire-handle-user-login"
---

, let's unpack how Hotwire tackles user authentication, or rather, *logins*. It's a topic I've certainly grappled with a few times, having implemented it in projects ranging from small internal tools to more public-facing web applications. There's no magical, single 'Hotwire way,' but rather, it's about leveraging Hotwire's core principles—particularly Turbo and Stimulus—within the larger context of server-side authentication. Forget single-page app (spa) patterns, that's not the ethos here. We're sticking with server-rendered views, enhanced with dynamic partials.

The crux of it lies in how you manage the user session on the server, typically through cookies or tokens, while using Turbo to create a seamless user experience without full page reloads. The goal is to make the login process feel instantaneous, even when substantial server-side work is taking place.

First, let's briefly establish the basics. Authentication in the backend is handled with a standard pattern. For example, in a Ruby on Rails environment, you would have a model representing users, and methods for authenticating them using bcrypt for password hashing or via a service for authenticating using external services. Post-authentication, a user's session information is maintained. It's crucial, this foundational piece. Hotwire doesn't attempt to replace this, it works with it.

Now, let's consider the front end. We'll build a login form, typically just a few fields and a submit button. This is where Turbo's form submission magic comes into play. When the user submits the form, Turbo intercepts the submission, sending an asynchronous request to the server. This avoids a full page reload.

On the server, after processing the credentials and authenticating the user (assuming everything's valid), you need to do more than just a redirect. It's here that a Turbo Stream comes into play. Instead of a redirect to a new page, I send back a Turbo Stream response to update the necessary parts of the user interface.

Let’s look at a quick snippet in rails:

```ruby
# app/controllers/sessions_controller.rb

class SessionsController < ApplicationController
  def create
    user = User.find_by(email: params[:email])
    if user&.authenticate(params[:password])
      session[:user_id] = user.id
      render turbo_stream: turbo_stream.replace("login-form", partial: "sessions/success")
    else
      render turbo_stream: turbo_stream.replace("login-form", partial: "sessions/form", locals: { error: "Invalid credentials" })
    end
  end
  
   def destroy
      session[:user_id] = nil
      redirect_to root_path, notice: "Logged out successfully."
    end
end
```

Here, the `create` action checks user credentials. On successful login, instead of redirecting, it renders a `turbo_stream` response that replaces a partial with a "success" view. This could contain the user's dashboard or a welcome message, as needed. On login failure, it sends back a different partial containing the login form again, but this time with an error message. This partial replacement avoids the need to fully reload, or reload the entire form and all of its fields.

Let's see an example partial used here:

```erb
  <!-- app/views/sessions/_form.html.erb -->
  <div id="login-form">
  <% if local_assigns[:error] %>
    <p style="color:red;"><%= error %></p>
  <% end %>
    <%= form_with url: sessions_path, method: :post, data: { turbo: false }  do |form| %>
      <div>
        <%= form.label :email %><br>
        <%= form.email_field :email, required: true %>
      </div>
      <div>
        <%= form.label :password %><br>
        <%= form.password_field :password, required: true %>
      </div>
        <%= form.submit "Log in" %>
      </div>
    <% end %>
</div>

```

Note the `data: { turbo: false }` attribute. This is a crucial detail for demonstration, although you wouldn't need this in normal hotwire situations. I included it here so you could quickly try the partial with an html form and observe how that behaves, and then move towards using a turbo stream. If turbo were enabled, the form submission would be handled via javascript and the page wouldn't refresh (if you add the correct turbo streams to the server response) . The view here is using rails form helpers, but it could be any templating engine such as django templates or blade templates in laravel.

And the success partial would be simple:

```erb
<!-- app/views/sessions/_success.html.erb -->
<div id="login-form">
  <p>Login Successful!</p>
  <%= button_to "logout", logout_path, method: :delete %>
</div>
```

Note that the target div, `login-form` is reused in both scenarios. When the server sends a turbo stream response, any element that matches the id, regardless of what is contained within, is replaced with what the server sends.

This approach avoids the jarring feeling of a full page refresh while still maintaining a consistent flow.

So, what if you want something more complex, such as multi-factor authentication? The core idea remains the same. Instead of directly logging in, the server might initially respond with a partial for entering a verification code, possibly sent via email or sms. When submitted, this second form again initiates a turbo stream response to either confirm the login or ask for a new code. All of this, without a single full page reload.

Now, let's discuss Stimulus, which plays a complementary role. I've found it particularly useful for things like handling password visibility toggles or dynamic form validations on the client side *before* the form gets submitted. Stimulus, with its concise controller structure, allows for modular and maintainable code, without resorting to jQuery or other more heavyweight libraries. It allows us to do basic front-end validation, for example, that the password field is at least 8 characters long, ensuring that you aren't making unnecessary trips to the server.

A simple stimulus controller for handling a password field could look like this:

```javascript
// app/javascript/controllers/password_controller.js
import { Controller } from "@hotwired/stimulus"

export default class extends Controller {
  static targets = ["passwordField", "toggleButton"]

  connect() {
    this.passwordFieldTarget.type = "password"
  }

  toggleVisibility() {
    if(this.passwordFieldTarget.type === "password") {
      this.passwordFieldTarget.type = "text";
      this.toggleButtonTarget.textContent = "Hide"
    } else {
      this.passwordFieldTarget.type = "password";
      this.toggleButtonTarget.textContent = "Show"
    }
  }
}
```

This controller sets the password field type, initially, to `password` and then when a button (identified by `toggleButtonTarget`) is clicked, the text content of the password field is changed to either text or password, and the text on the toggle button is also changed to either "Show" or "Hide". The stimulus controller is set up by creating a class and then importing the controller into a javascript bundle. The targets are referenced through `data-` attributes in your view.

In the view:

```erb
    <div>
        <%= form.label :password %><br>
       <div data-controller="password">
         <%= form.password_field :password, data: { password_target: "passwordField" }, required: true %><br>
         <button type="button" data-action="password#toggleVisibility" data-password-target="toggleButton">Show</button>
       </div>
      </div>
```

This demonstrates how we combine a view with some basic javascript functionality, before any submission is done.

To summarize, a Hotwire-based user login system boils down to:

1.  **Standard Server-Side Authentication:** Handle user management, sessions, and password hashing on the server.
2.  **Turbo Form Submissions:** Leverage Turbo to send login forms asynchronously.
3.  **Turbo Stream Responses:** On success, return a turbo stream update that replaces parts of the view. On failure, show an error using a similar method.
4.  **Stimulus Augmentation:** Use Stimulus for client-side interactions like password toggles and basic form validations, and enhance elements that may not rely on submission.

For further study, I'd recommend diving deep into the documentation of *Turbo* and *Stimulus*. Additionally, reading "The Rails 7 Way" by David Heinemeier Hansson for a great overview of current best practices of the rails framework as well as how hotwire is best utilized within it, as well as the "Programming Phoenix LiveView" book by Bruce Tate for examples of similar server-side rendered approach. Understanding those resources will give you a solid grasp of the mechanics at play and how to implement them for any project.
