---
title: "How can I customize a simple_form date input with a flag?"
date: "2024-12-23"
id: "how-can-i-customize-a-simpleform-date-input-with-a-flag"
---

Okay, let's get into it. This isn't as straightforward as it initially seems, and i've certainly had my share of head-scratching moments working with simple_form and date inputs over the years. The challenge, as you've presented it, is to not just display a date input, but to also include a flag alongside it. It's a visual cue that's often needed for context or validation purposes, and simple_form itself doesn't directly provide this out of the box.

The underlying issue, as I see it, stems from the fact that simple_form aims to simplify form creation by abstracting away a lot of the underlying html. When you need to incorporate something that isn't part of its standard set of features like this, you have to think about how to extend its functionality without breaking the abstractions it provides. In the projects where i've encountered similar situations, typically, the quickest solutions have been through some combination of custom wrappers, custom input classes, or in some cases, just going for raw html (but we want to avoid that last one as much as possible).

The first path, which often offers the most control with the least amount of mess, is using a custom wrapper. Simple_form uses wrappers to define how inputs are rendered, including labels, hints, and errors. We can create our own that includes a flag. Here's how I'd approach it:

```ruby
# config/initializers/simple_form.rb
SimpleForm.setup do |config|
  config.wrappers :flagged_date, tag: 'div', class: 'input-group', error_class: 'has-error' do |b|
    b.use :html5
    b.use :placeholder
    b.use :label, class: 'form-label'
    b.wrapper tag: 'div', class: 'input-wrapper' do |ba|
       ba.use :input, class: 'form-control datepicker'
       ba.wrapper tag: 'span', class: 'input-group-text' do |c|
        c.use :flag, class: 'flag-icon'
      end
    end
    b.use :hint,  wrap_with: { tag: 'span', class: 'hint' }
    b.use :error, wrap_with: { tag: 'span', class: 'error' }
  end
end

# app/inputs/flag_input.rb
class FlagInput < SimpleForm::Inputs::Base
    def input(wrapper_options)
      # You would determine the flag here. Potentially based on a config value, or model association, etc.
      # For example you might be displaying a flag based on user's locale, and fetch that locale flag css class
      locale_flag_class = "flag-icon-gb" # Defaulting to GB for this example.
       "<i class='#{locale_flag_class}'></i>".html_safe
    end
  end
```
In this setup, we define a new wrapper `:flagged_date` which encapsulates the date input, and inserts the flag after it, inside an `input-group-text` element.  The `flag` is defined in a custom input, `FlagInput`, that can inject custom html—in this case, the html markup for a flag based on a css class. I'm using `flag-icon-css` as the underlying css library to provide the base flag icons, something like 'flag-icon-gb' (which you'd obviously have to pull from a proper locale or model context in a real application). I added the `input-group-text` span as a bootstrap specific visual improvement in this example but it can be customized to fit your UI library. To use it in your form, you would write something like:

```erb
<%= simple_form_for @my_model do |f| %>
  <%= f.input :my_date, as: :date, wrapper: :flagged_date, input_html: { data: { provide: "datepicker", date_format: "yyyy-mm-dd" } } %>
  <%# ... other fields %>
<% end %>
```
This approach has some notable advantages. It keeps the form code concise while being very flexible; you can modify the `FlagInput` class to include more complex logic or conditional flags. The wrapper approach also makes it reusable across the project for other date fields that need this functionality.

Another technique, especially useful if you need more intricate control over the visual presentation, is to use a custom input class. This is especially useful for scenarios with more dynamic flag conditions or custom input specific logic. Here's a variation:

```ruby
# app/inputs/flagged_date_input.rb
class FlaggedDateInput < SimpleForm::Inputs::Base
  def input(wrapper_options)
    merged_html_options = merge_html_options(input_html_options, wrapper_options)
    date_html = @builder.text_field(attribute_name, merged_html_options) #Using text_field instead of date_field for flexibility on datepickers

    flag_html = "<span class='input-group-text'><i class='flag-icon flag-icon-gb'></i></span>".html_safe # Similar logic, can be replaced by locale-based flag selection.
      "<div class='input-group'>#{date_html}#{flag_html}</div>".html_safe
  end
end
```

In this custom input, `FlaggedDateInput`, we are taking over the generation of the whole input. We create the `text_field`, inject the flag, and then wrap both in an `input-group` class. This avoids using simple_form's wrapper. Instead of the usual `f.input :my_date, as: :date`, you'd have `f.input :my_date, as: :flagged_date` in your view, like so:

```erb
<%= simple_form_for @my_model do |f| %>
  <%= f.input :my_date, as: :flagged_date, input_html: { data: { provide: "datepicker", date_format: "yyyy-mm-dd" }, class: 'form-control datepicker' } %>
  <%# ... other fields %>
<% end %>
```
This approach gives us more direct control over the html output; however, it also means we have to handle setting up all the `input_html` attributes that simple_form's normal inputs would manage for us. So there's some added complexity in keeping everything styled and functional, but it does offer the maximum control over layout and styling.

Finally, let's consider a situation where you also want to validate the date based on the user's flag. It would be a bit more involved, but can be achieved using custom validators and some javascript to update the date format based on the selected flag or locale.

First, let's setup a custom validator:

```ruby
# app/validators/flagged_date_validator.rb
class FlaggedDateValidator < ActiveModel::EachValidator
  def validate_each(record, attribute, value)
     locale = record.flag_locale  # Assuming you have an attribute or method called flag_locale on your model.
    if locale == "gb"
      begin
         Date.strptime(value, "%d/%m/%Y")
      rescue ArgumentError
        record.errors.add attribute, :invalid_date, message: "must be in dd/mm/yyyy format"
       end
    elsif locale == "us"
      begin
          Date.strptime(value, "%m/%d/%Y")
       rescue ArgumentError
          record.errors.add attribute, :invalid_date, message: "must be in mm/dd/yyyy format"
       end
     end
  end
end

# app/models/my_model.rb
class MyModel < ApplicationRecord
  validates :my_date, flagged_date: true
  attr_accessor :flag_locale # Example, can be fetched dynamically

   before_validation do
    self.flag_locale = 'gb' # Set default value or fetch value dynamically
  end
end

# In the view, we'll add some Javascript to update the flag locale when it's switched
# In addition to previously created wrapper or input as described before.
# You would need to include a flag selector UI as well, omitted for brevity

```
With this code, we can configure the model to validate date formats based on a locale. The validation is handled by the `FlaggedDateValidator`, and we'd use javascript to update the form's flag field so that the validation is performed as expected when the form is submitted. This javascript part is omitted for brevity as the example would get too large, but you can easily implement by using event handlers when changing the flag and updating hidden input fields or model attributes.

For further reading, I'd highly recommend checking out the source code for the simple_form gem itself (it's on GitHub). It’s a great way to understand how its wrappers and inputs function. In addition, The “Ruby on Rails Guides” documentation is the best resource available for understanding how form helpers and custom validators work in general, it's a must-read.

The key thing to remember is that when extending simple_form, understanding the underlying mechanisms of wrappers and inputs is crucial. There isn't a single, 'best' approach; choosing the method that balances flexibility and maintainability for your project is the most effective route.
