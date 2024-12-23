---
title: "How can Rails ERB templates be used for validation?"
date: "2024-12-23"
id: "how-can-rails-erb-templates-be-used-for-validation"
---

Alright, let’s tackle this interesting corner of rails development: using ERB for validation. It's a topic I've actually stumbled into a few times, particularly back in the '09 era when we were still pushing the boundaries of what rails could do. While not the *primary* purpose of ERB, it can certainly be leveraged for view-level validations, though judiciously. I’m not talking about replacing proper model validation here; rather, consider it a layer of client-side feedback or conditional rendering within the view itself. It's about handling those situations where a purely model-centric approach isn’t granular enough or might involve unnecessary trips back to the server.

Fundamentally, ERB, with its embedded Ruby code, allows you to evaluate expressions and control the output of your HTML. We can use this capability to check the state of your variables, instance objects passed to the view, or even the session and make the template behave differently based on this evaluation. Let's say you're working on a form where specific fields should only be displayed if another field has a particular value, or perhaps you need to provide an inline error message only if a user-provided input fails a particular client-side check. These situations are ripe for ERB’s logical power.

My experiences have shown me that this approach works best when you're dealing with display logic that doesn’t fit cleanly within the model, particularly those visual conditional validations. For instance, imagine a scenario where we've built a user registration form, but want to dynamically change a message on submit if certain fields are missing.

Now, before we dive into specifics, it's crucial to recognize this is more about conditional rendering and *feedback* than it is about hard validation rules. Proper validation should always reside in the model layer. Think of this as a user interface enhancer, not a replacement. It prevents unnecessary data from even reaching the server if you can detect potential errors directly within the browser.

Let's look at a couple of working code snippets to illustrate different scenarios.

**Scenario 1: Conditional display based on form data**

Consider a simple form where users can select a service type and based on that type, we display more fields.

```erb
<form>
  <label for="service_type">Select Service Type:</label>
  <select id="service_type" name="service_type">
    <option value="basic">Basic</option>
    <option value="premium">Premium</option>
  </select>

  <% if params[:service_type] == "premium" %>
    <div id="premium_options">
      <label for="premium_feature1">Premium Feature 1:</label>
      <input type="text" id="premium_feature1" name="premium_feature1">
      <label for="premium_feature2">Premium Feature 2:</label>
      <input type="text" id="premium_feature2" name="premium_feature2">
    </div>
  <% end %>

  <input type="submit" value="Submit">
</form>
```

Here, the `params[:service_type]` reads the selected value from the form. If it equals "premium", the additional input fields are rendered. This is a basic example of using the value submitted from the form itself for conditional logic, and it provides immediate feedback to the user.

**Scenario 2: Inline Error Messages**

Here's another scenario. Say you want to make sure a user enters a minimum length string before submitting the form. I am using basic html here for sake of clarity, but you could certainly use whatever form abstraction you prefer.

```erb
<form>
  <label for="username">Username:</label>
  <input type="text" id="username" name="username" value="<%= params[:username] %>">

  <% if params[:username].present? && params[:username].length < 5 %>
    <p class="error">Username must be at least 5 characters long.</p>
  <% end %>
  <br>
  <input type="submit" value="Submit">
</form>
```

In this snippet, we're checking if the `username` parameter is present and, if so, if its length is less than 5. If true, we output an error message. Again, it’s essential to validate this data again on the server-side, but this quick feedback can drastically improve the user experience. It reduces the friction of repeatedly submitting the form and seeing a general error.

**Scenario 3: Advanced Conditional HTML Based on Object Properties**

Now, let's consider a scenario where we want to display different content based on properties of an object. Let's say you are showing information about user subscriptions.

```erb
<% @user_subscriptions.each do |subscription| %>
  <div class="subscription-card">
    <h3>Subscription Type: <%= subscription.plan_name %></h3>
    <p>Status: <%= subscription.status %></p>
    <% if subscription.status == 'active' %>
      <p class="active">Your subscription is active!</p>
      <p>Expires: <%= subscription.expires_at %></p>
        <% if subscription.auto_renewal %>
          <p>Auto-renewal is enabled</p>
        <% else %>
          <p>Auto-renewal is disabled</p>
          <%= button_to "Enable Auto-Renewal", enable_renewal_path(subscription), method: :post %>
        <% end %>

    <% elsif subscription.status == 'pending' %>
        <p class="pending">Your subscription is pending payment.</p>
          <%= button_to "Make Payment", payment_path(subscription), method: :post %>
    <% elsif subscription.status == 'expired' %>
      <p class="expired">Your subscription has expired.</p>
      <%= button_to "Renew Subscription", renew_subscription_path(subscription), method: :post %>
    <% end %>
  </div>
<% end %>
```

This final example builds upon the others and demonstrates rendering different UI components based on subscription status. This approach ensures a dynamic experience where each subscription status is represented with unique information and actions. We are leveraging the properties of the instance variable to control not just messages, but entire UI elements. The `button_to` helpers would, of course, need to be correctly configured in your routes.

Now, while we are using ERB to provide more reactive views, this does not negate the importance of validation rules within your models. You still should validate data on your back-end. The ERB validation is solely to improve the user experience and reduce server trips.

To deepen your knowledge here, I’d suggest looking into **"Agile Web Development with Rails 7"** by David Bryant Copeland and the official Ruby on Rails documentation, especially the sections on views and forms. These provide in-depth insight into the proper way to use ERB and the different ways that you can leverage the full power of rails to create great user experiences. Furthermore, for a foundational understanding of how HTML forms and web submissions work, the W3C HTML documentation on forms is an excellent resource. Additionally, take a look into front end frameworks such as React, Vue, or even Turbo for a more modern way to render these types of conditional messages.

In closing, remember to use this technique judiciously. When it comes to real validation, keep it in your model. But for that extra bit of user feedback and view-specific logic, ERB can be a very helpful tool. Just remember to always double validate on the server-side.
