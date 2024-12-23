---
title: "Why can't I obtain a JWT token during password reset using Devise and JWT?"
date: "2024-12-23"
id: "why-cant-i-obtain-a-jwt-token-during-password-reset-using-devise-and-jwt"
---

, let's unpack this. I've seen this specific scenario pop up more times than I care to count, and it’s usually down to a fundamental misunderstanding of how password reset flows interact with token-based authentication. The short answer is: you're trying to shoehorn a token into a process where it’s not inherently designed to fit. Let me explain, drawing from a few similar projects I've encountered.

Fundamentally, password reset is a *stateful* process. Devise, at its core, tracks this state using its own database mechanisms — typically, a reset password token stored in the user record, along with associated timestamps. This reset process is designed to operate independently of a stateless token mechanism like JWT. Think of it this way: JWTs are primarily used for authentication *after* a user has been confirmed and verified. Password resets are part of the pre-authentication or initial access lifecycle. Trying to generate a JWT directly during the reset process is akin to trying to pay for groceries with a receipt before you've actually paid.

The typical flow looks like this: a user requests a password reset. Devise generates a reset token, stores it in the database, and emails the user a link with this token. The user clicks the link, which verifies the reset token against the database record. Only then, once the user *successfully* provides a new password, are they considered authenticated. Therefore, they can then obtain a JWT *after* successful password change.

Here’s the common pitfall: trying to immediately issue a JWT after *requesting* the reset, but before the actual password change. This is problematic because the user isn't yet authenticated in a way that a JWT would recognize. They haven't successfully identified themselves with a valid password. The password reset link is a temporary, single-use credential to facilitate the *change* of authentication credentials, not the credentials themselves.

Let's look at a real-world example from a project I was involved in a while back. We initially tried to bypass this process, and the resulting code, frankly, was a mess. We ended up re-evaluating the entire architecture, and what we implemented afterward was a much cleaner separation of concerns.

**Example 1: The Incorrect Approach (and Why It Fails)**

This illustrates what you're likely trying to do, and why it doesn't work. Note: this isn't functioning code, but rather an illustration of a typical flawed approach:

```ruby
# In your Devise controller (or similar)
def create
  @user = User.find_by(email: params[:email])
  if @user
    @user.send_reset_password_instructions
    # The error is here, issuing a token *before* the actual reset happens.
    jwt = JWT.encode({user_id: @user.id}, Rails.application.credentials.secret_key_base, 'HS256')
    render json: { message: "Reset link sent", token: jwt }
  else
    render json: { error: "User not found" }, status: :not_found
  end
end
```

The issue here is clear: the JWT is generated based on the *existence* of the user, not on successful authentication. This token doesn’t reflect whether the password has actually been reset, and it bypasses Devise’s carefully crafted security model for password resets. A user with this "token" would essentially have access even before they've changed their password, which is a gaping security hole.

**Example 2: The Correct (Simplified) Flow**

The correct flow involves generating the JWT *after* the password has been successfully reset. This code isn't a full implementation but showcases the necessary change:

```ruby
# In your Devise passwords controller override
def update
  self.resource = resource_class.reset_password_by_token(resource_params)
  yield resource if block_given?

  if resource.errors.empty?
    # Successfully reset password, generate token now.
    jwt = JWT.encode({ user_id: resource.id }, Rails.application.credentials.secret_key_base, 'HS256')
    render json: { message: "Password updated", token: jwt }
  else
    render json: { errors: resource.errors }, status: :unprocessable_entity
  end
end
```

Here, the JWT is generated only after the call to `resource.reset_password_by_token` successfully updates the password and clears the reset token, ensuring that the user has provided new valid credentials. This ties the token to *actual* authentication.

**Example 3: Front-End Interaction**

Finally, here's how this might look from a front-end perspective. This is crucial because often confusion arises when attempting to integrate front-end token handling in the password reset process:

```javascript
// Pseudo-code demonstrating front end handling after password update is successful

async function submitNewPassword(password, token) {
  try {
    const response = await fetch('/users/password', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            password: password,
            reset_password_token: token
          })
    });

    const data = await response.json();

    if(response.ok) {
      console.log("Password reset successful, here's the token:", data.token);
      localStorage.setItem("jwt_token", data.token) // Store the token
    } else {
      console.error("Password update failed:", data.errors);
    }
  } catch (error) {
      console.error("There was a problem submitting the new password:", error);
  }
}
```

Notice, the front-end first sends the new password along with the reset token. Upon successful update (as per the server-side implementation in example 2), the API returns the JWT, which the front-end stores locally for subsequent requests.

The core concept to remember is the sequential nature of a password reset. You're not dealing with a regular authentication event where credentials are immediately validated. Password resets are an *intermediate* step. JWT authentication takes over *after* this step is completed. You need to keep your stateful logic (Devise’s reset process) separate from your stateless token generation (JWT).

For further reading, I would highly recommend thoroughly examining "Authentication in Action" by John Hardin, which offers detailed explanations of various authentication patterns including password resets and the rationale behind different approaches. Also, dive deep into the Devise gem documentation itself, and you'll find clear explanations of how its reset password functionality works. For JWT specific insights, familiarize yourself with the specifications outlined in RFC 7519. Understanding the underpinnings of these technologies is more important than blindly trying to fit pieces together in an inappropriate way, and it'll save you headaches down the road. It's not about forcing things, it’s about understanding each mechanism's core purpose and building your system accordingly. That is how you get a robust, secure, and maintainable system.
