---
title: "How can Rails forms dynamically adjust the number of fields?"
date: "2024-12-23"
id: "how-can-rails-forms-dynamically-adjust-the-number-of-fields"
---

Alright, let's tackle this. The need for dynamically adjusting the number of fields in Rails forms pops up quite often, and I've certainly dealt with it a fair few times. I recall one particular project—a CRM overhaul for a medium-sized firm—where we needed to handle varying numbers of product options during lead creation. Static forms simply wouldn't cut it; we needed a flexible solution that allowed users to add or remove options as needed.

The core challenge is managing both the visual update on the client side and the data persistence on the server side. Rails, being a server-side framework, needs a bit of help to handle the interactive nature of dynamic field manipulation. The key, generally, lies in employing a combination of JavaScript for the front-end dynamic updates and then processing the resulting structured data in our Rails controller. We can't rely solely on rails helpers for this; we need to step out a bit and bridge the client and server components.

Let's break down the typical approaches and how they operate. The most common solution involves using JavaScript to clone existing field sets and then updating the relevant indices. When we hit submit, we are sending over the data in a format Rails’ params can work with easily.

Firstly, we'll consider a straightforward approach with nested forms using the `fields_for` helper and javascript to duplicate field sets. Suppose we have a model called `ProductOption`, associated with a `Lead` model. Our lead form needs to allow for a dynamic number of product options.

Here's what the relevant part of our `lead.rb` model might look like:

```ruby
class Lead < ApplicationRecord
  has_many :product_options, inverse_of: :lead, dependent: :destroy
  accepts_nested_attributes_for :product_options, reject_if: :all_blank, allow_destroy: true
end

class ProductOption < ApplicationRecord
    belongs_to :lead
end
```

Now for our view form:

```erb
<%= form_with(model: @lead, url: leads_path, method: :post, local: true) do |form| %>
  <div id="product-options">
    <%= form.fields_for :product_options do |product_option_form| %>
      <%= render 'product_option_fields', form: product_option_form %>
    <% end %>
    <div id="add-product-option-button">
        <%= link_to "Add Product Option", '#', data: { action: "add-product-option" } %>
     </div>
  </div>
  <%= form.submit 'Submit' %>
<% end %>
```

And let's have a partial `_product_option_fields.html.erb`:

```erb
<div class="product-option">
    <%= form.hidden_field :_destroy, value: false, class: 'destroy-product-option-field' %>
    <%= form.label :name, "Option Name" %><br>
    <%= form.text_field :name %><br>
    <%= form.label :value, "Option Value" %><br>
    <%= form.text_field :value %><br>
    <%= link_to "Remove", '#', data: { action: "remove-product-option"} %>
</div>
```

The key element is `accepts_nested_attributes_for` and `fields_for`, allowing us to easily structure the data and interact with it within the form. The javascript part, which I will show next, will add additional copies of this structure.

Now, let's see the JavaScript for cloning and managing these fields, which, while being in the `app/assets/javascripts/leads.js` file, could also be refactored to use a javascript framework such as stimulus.js:

```javascript
document.addEventListener('DOMContentLoaded', function() {
    document.querySelector('#add-product-option-button').addEventListener('click', function(event) {
        event.preventDefault();
        const container = document.getElementById('product-options');
        const originalField = container.querySelector('.product-option');
        if (originalField) {
            const newField = originalField.cloneNode(true);
            const fieldCount = container.querySelectorAll('.product-option').length;
            newField.querySelectorAll('input,select,textarea').forEach(function(el) {
               let name = el.name.replace(/product_options\[\d+\]/, `product_options[${fieldCount}]`);
               let id = el.id.replace(/product_options_\d+/, `product_options_${fieldCount}`);
               el.name = name;
               el.id = id;
               if(el.value) el.value = ''
            });
           newField.querySelector('.destroy-product-option-field').value = 'false';
            container.insertBefore(newField, document.getElementById('add-product-option-button'));
        }
    });


  document.addEventListener('click', function(event){
      if(event.target.matches('[data-action="remove-product-option"]')) {
        event.preventDefault();
          const productOption = event.target.closest('.product-option');
          const hiddenDestroyField = productOption.querySelector('.destroy-product-option-field');
          if (hiddenDestroyField) {
              hiddenDestroyField.value = '1'
              productOption.style.display = 'none';
          } else {
              productOption.remove();
          }
      }
  });
});
```

This javascript clones the first product-option div, reindexes the attributes to handle dynamic insertion and remove and inserts a new form set into the existing dom. The removal is also handled, we are not actually removing the element but rather marking it for deletion by toggling the destroy value, which we have setup in our `lead.rb` model with `allow_destroy: true`.

This combination allows users to add an arbitrary number of product options. The rails params will come as `lead[product_options_attributes][0][name], lead[product_options_attributes][0][value], lead[product_options_attributes][1][name], lead[product_options_attributes][1][value]`, and so on. It’s crucial to re-index the names and ids to ensure they are submitted correctly.

Now, another approach, which might be more appropriate for more complex scenarios, involves using javascript to append raw HTML structures, rather than cloning.

In that scenario, you could store your form structure in a hidden template element. Here’s an example:

```html
<div style="display:none">
  <div id="product-option-template">
    <div class="product-option">
        <input type="hidden" name="lead[product_options_attributes][__INDEX__][_destroy]" value="false" class="destroy-product-option-field">
        <label for="lead_product_options_attributes___INDEX___name">Option Name</label><br>
        <input type="text" name="lead[product_options_attributes][__INDEX__][name]" id="lead_product_options_attributes___INDEX___name"><br>
        <label for="lead_product_options_attributes___INDEX___value">Option Value</label><br>
        <input type="text" name="lead[product_options_attributes][__INDEX__][value]" id="lead_product_options_attributes___INDEX___value"><br>
        <a href="#" data-action="remove-product-option">Remove</a>
    </div>
  </div>
</div>
<div id="product-options">
  <% if @lead.product_options.any? %>
    <%= form.fields_for :product_options do |product_option_form| %>
        <%= render 'product_option_fields', form: product_option_form %>
    <% end %>
  <% end %>
 <div id="add-product-option-button">
     <%= link_to "Add Product Option", '#', data: { action: "add-product-option" } %>
 </div>
</div>
```

Here's the corresponding updated JavaScript snippet that manages the add and remove functionality:

```javascript
document.addEventListener('DOMContentLoaded', function() {
    document.querySelector('#add-product-option-button').addEventListener('click', function(event) {
        event.preventDefault();
       const container = document.getElementById('product-options');
        const template = document.getElementById('product-option-template').innerHTML;
        const fieldCount = container.querySelectorAll('.product-option').length;
         const newField = template.replace(/__INDEX__/g, fieldCount);
         const parsedField =  new DOMParser().parseFromString(newField, 'text/html').body.firstChild
        container.insertBefore(parsedField, document.getElementById('add-product-option-button'));
    });

    document.addEventListener('click', function(event){
        if(event.target.matches('[data-action="remove-product-option"]')) {
            event.preventDefault();
            const productOption = event.target.closest('.product-option');
            const hiddenDestroyField = productOption.querySelector('.destroy-product-option-field');
            if (hiddenDestroyField) {
                hiddenDestroyField.value = '1'
                productOption.style.display = 'none';
            } else {
                productOption.remove();
            }
        }
    });
});
```

In this approach, we’re pulling the HTML structure from the hidden template and replacing the `__INDEX__` placeholders with the current count. This gives more control over the html structure, but requires a little extra care to maintain.

For a deeper understanding of these concepts, I highly recommend consulting “Agile Web Development with Rails 7” by Sam Ruby et al., which provides comprehensive coverage of Rails form handling. In addition, for more insight into working with DOM manipulation using JavaScript, the 'Eloquent Javascript' book by Marijn Haverbeke provides an excellent practical insight. I've found these resources particularly useful in dealing with such dynamic form requirements. I recall several instances where I was able to debug and implement these types of form behaviours rapidly because of them.

Remember to consider your data structure, complexity, and maintainability when selecting the right approach. Each has its trade-offs, and choosing the right fit is crucial to long-term application health.
