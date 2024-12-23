---
title: "How can I customize simple_form options in Rails 7?"
date: "2024-12-23"
id: "how-can-i-customize-simpleform-options-in-rails-7"
---

Right then, let's talk about customizing `simple_form` options in Rails 7. It's a topic I've spent a good deal of time with, especially back when I was leading development on that resource management system for the university. We needed incredibly granular control over form rendering, and `simple_form` was our chosen tool. Straight out of the box, it’s pretty helpful, but sometimes you need to tweak it to get that *perfect* feel and behavior. So, how do we achieve this?

The core idea revolves around understanding that `simple_form` offers flexibility at several levels. You can customize globally, per form, or even per input. I'll walk through each with practical examples.

First, let’s consider global customization. This is particularly useful if you have a consistent style or set of behaviors you want to apply across your entire application. You achieve this primarily through the `config/initializers/simple_form.rb` file. If it’s not there already, you can generate one using `rails generate simple_form:install`.

Within this initializer, you can tweak a plethora of settings. One common scenario I’ve encountered involved needing a specific class applied to all submit buttons. Let's say we wanted all submit buttons to use Bootstrap's `btn` and `btn-primary` classes. This can be done with the following within the `SimpleForm.setup` block:

```ruby
  config.button_class = 'btn btn-primary'
  config.default_wrapper = :default
  config.wrappers :default, class: 'form__group' do |b|
    b.use :html5
    b.use :placeholder
    b.use :label, class: 'form__label'
    b.wrapper class: 'form__input' do |input|
      input.use :input, class: 'form-control', error_class: 'is-invalid', valid_class: 'is-valid'
      input.use :full_error, wrap_with: { tag: 'div', class: 'invalid-feedback' }
      input.use :hint,  wrap_with: { tag: 'small', class: 'form-text text-muted' }
    end
  end
  config.boolean_style = :nested
  config.default_boolean_wrapper_class = "form__group checkbox"
```

In this snippet, I am configuring the submit button class, using a specific wrapping style to add classes to labels, inputs, error messages, and hints. It demonstrates a fairly common approach of standardizing the layout and styles across your forms.

Now, while global settings are excellent for overall consistency, sometimes you need form-specific adjustments. For instance, let's say we have a user creation form that, for accessibility reasons, requires special placeholder text, specific help text, and error message display. This is accomplished by passing options directly within your form builder. Consider a `users/_form.html.erb` partial:

```erb
<%= simple_form_for(@user) do |f| %>
  <%= f.input :username,
               placeholder: 'Enter a unique username',
               hint: 'Must be at least 5 characters long',
               input_html: { data: { test: 'user_name' } }
  %>

  <%= f.input :email,
              placeholder: 'Your active email address',
              error: "Invalid Email Format"
  %>

  <%= f.input :password,
              placeholder: 'Your password'
  %>


  <%= f.button :submit, 'Create User' %>

<% end %>

```

In this example, we’ve individually tailored the input fields for 'username', and 'email', and 'password'. We've assigned a custom `placeholder` and `hint` for the `username` field, and an extra data attribute using `input_html`. I have also included a custom error message for the email input. It’s important to understand the hierarchy here: field-specific options override form-specific configurations, which, in turn, override global settings.

Lastly, there’s a third level of control: customization at the input level, directly through options when calling `f.input`. This grants the highest degree of granularity and is extremely useful when you have very specific requirements. For instance, imagine you have a field where a specific html `pattern` attribute is needed or where you want to modify the error message display behaviour for a specific type of input.

Here's an example applied to a different form, lets imagine this was for a product form:

```erb
<%= simple_form_for(@product) do |f| %>
  <%= f.input :title,
               placeholder: 'Product Title'
  %>

  <%= f.input :description,
               input_html: { rows: 5 },
               placeholder: 'Describe the product'
   %>

  <%= f.input :price,
              input_html: { pattern: "[0-9]+(\.[0-9]{0,2})?", title: "Enter a valid price (e.g., 19.99)" },
              wrapper_html: { data: { test: 'price_container' } }

   %>
  <%= f.input :category,
               collection: ['Electronics', 'Books', 'Clothing'],
               prompt: 'Select a Category'
    %>
  <%= f.button :submit, 'Add Product' %>
<% end %>
```

In this snippet, we've added `input_html` options for the `description` input setting the `rows` attribute, and `pattern` and `title` attributes to the `price` input field for validation, and a wrapper with an extra data attribute. Additionally, I've included the `prompt` option within the `category` input which is rendering a select dropdown box. This level of customization allows you to tailor the behavior and attributes for particular fields as you require them.

Now, I strongly recommend you look at some more authoritative documentation on `simple_form` and form handling in general. For detailed insights into form development, I suggest looking into "Web Form Design: Filling in the Blanks" by Luke Wroblewski. It’s not `simple_form` specific, but it covers user-centered design, crucial for form optimization. Additionally, the `simple_form` official documentation is a must-read.

To enhance your understanding of accessibility concerns within forms, “Inclusive Components” by Heydon Pickering is an invaluable resource and provides detailed examples and use cases when it comes to user interactions. These resources will provide a more complete overview and help you create better, more usable forms.

The key thing to take away here is that `simple_form` is incredibly flexible. The ability to configure options globally, at the form level, or directly on each input gives you the fine-grained control necessary for sophisticated application development. Remember to leverage this control wisely, prioritising user experience and accessibility, and to carefully consider the implications of customization at each level to achieve the result you need. Through experience and a deep understanding of these techniques, you'll be able to wield `simple_form` effectively and efficiently to create exceptional forms in your Rails applications.
