---
title: "How can I use simple_form with stimulus data attributes?"
date: "2024-12-16"
id: "how-can-i-use-simpleform-with-stimulus-data-attributes"
---

,  I remember back in '18, during a particularly gnarly project involving a complex multi-step form, I ran into this exact situation. We were attempting to leverage stimulus.js for dynamic form behavior, and it felt like wrestling an octopus to get it to play nicely with simple_form's output. The challenge, as I see it, isn't really that they're inherently incompatible, but rather that their individual approaches to markup generation and DOM manipulation require a bit of strategic alignment. We can definitely make them work together effectively, it just takes a considered approach.

The core problem lies in how simple_form generates its html, specifically its reliance on semantic wrapper elements. Stimulus relies on data attributes to link html elements to controllers and actions. When simple_form produces an input field, it often adds elements like `div.input`, `div.string`, or others, nested inside the form. These aren’t usually directly targeted by stimulus. Hence, we can't directly apply `data-controller` or `data-action` attributes to the `<input>` elements that we actually want stimulus to interact with. The trick is to strategically place these attributes, often on parent elements, and rely on scope within your stimulus controller.

Let’s illustrate with some code. Say you have a form where you want to show a dependent field (e.g., a "specify other" text field) when a certain radio button is selected. Here's the initial simple_form setup, stripped of styling concerns for clarity:

```ruby
# app/views/my_forms/_my_form.html.erb
<%= simple_form_for @my_model do |f| %>
  <%= f.input :category, as: :radio_buttons, collection: ['Option A', 'Option B', 'Other'] %>
  <%= f.input :other_category, as: :string, input_html: { style: 'display:none;' } %>
  <%= f.button :submit %>
<% end %>
```

This erb code would render a radio button group and a text input, with the text input initially hidden. Now, lets get this working with stimulus. Instead of trying to put data attributes directly onto the input, lets target the containing form element:

```ruby
# app/views/my_forms/_my_form.html.erb
<%= simple_form_for @my_model, html: { data: { controller: "dependent-field" }} do |f| %>
  <%= f.input :category, as: :radio_buttons, collection: ['Option A', 'Option B', 'Other'], input_html: { data: { action: "dependent-field#toggle" } } %>
  <%= f.input :other_category, as: :string, input_html: { style: 'display:none;', data: { dependent_field_target: "otherInput" } } %>
  <%= f.button :submit %>
<% end %>
```

Notice how we've added `data: { controller: "dependent-field" }` to the form element itself, initiating the `dependent-field` stimulus controller. Within the `category` input, we’ve also added a data-action to hook it up to our stimulus controller toggle method, and finally, added the `dependent_field_target` to our other_category field. Now, our stimulus controller, `dependent_field_controller.js`, would look something like this:

```javascript
// app/javascript/controllers/dependent_field_controller.js
import { Controller } from "@hotwired/stimulus"

export default class extends Controller {
  static targets = ["otherInput"]

  toggle(event) {
      if(event.target.value === 'Other') {
        this.otherInputTarget.style.display = 'block'
      } else {
        this.otherInputTarget.style.display = 'none'
    }
  }
}
```

This is a basic implementation of our target and action. When the radio button changes, this checks to see if its value is "Other". If it is, it will display our target input, otherwise hiding it. This pattern, setting the controller on a parent element, using actions to control behavior, and then selecting targets within the controller’s scope, is generally the most reliable method.

Here’s another scenario: Let's say you have a dynamic list where users can add or remove items. Again, the direct output of simple_form can feel limiting with the default wrappers. Let's first define our simple form partial:

```ruby
# app/views/my_forms/_my_dynamic_form.html.erb
<%= simple_form_for @my_model, html: { data: { controller: "dynamic-list" }} do |f| %>
  <div data-dynamic-list-target="listContainer">
    <% @my_model.items.each_with_index do |item, index| %>
       <div data-dynamic-list-target="listItem">
          <%= f.input :name, as: :string, collection: ['Item 1', 'Item 2', 'Item 3'], index: index, input_html: { data: { dynamic_list_target: "itemSelect" } } %>
          <button type='button' data-action="dynamic-list#removeItem" data-dynamic-list-index="<%= index %>">Remove</button>
        </div>
    <% end %>
  </div>
    <button type="button" data-action="dynamic-list#addItem">Add Item</button>
  <%= f.button :submit %>
<% end %>
```

Here we are generating a list with inputs, and using a button to remove list elements. Now lets see how our controller handles this:

```javascript
// app/javascript/controllers/dynamic_list_controller.js
import { Controller } from "@hotwired/stimulus"

export default class extends Controller {
  static targets = ["listContainer", "listItem"]

  addItem() {
    let newIndex = this.listItemTargets.length
    let newItemHTML = `
      <div data-dynamic-list-target="listItem">
        <input type="text" name="my_model[items][${newIndex}][name]" data-dynamic-list-target="itemSelect">
        <button type='button' data-action="dynamic-list#removeItem" data-dynamic-list-index="${newIndex}">Remove</button>
      </div>
    `
    this.listContainerTarget.insertAdjacentHTML('beforeend', newItemHTML);
  }

  removeItem(event) {
    const listItemIndex = event.target.dataset.dynamicListIndex;
    this.listItemTargets[listItemIndex].remove();
  }
}
```

In this controller, the `addItem` method dynamically adds new list item to the container. The `removeItem` method removes items based on their index. This pattern of dynamically generating elements and controlling them through stimulus is extremely powerful.

Finally, let's consider a situation with input validation. Imagine you want to perform client-side validation on a field immediately as it changes. Simple form creates a great canvas for this interaction:

```ruby
# app/views/my_forms/_validation_form.html.erb
<%= simple_form_for @my_model, html: { data: { controller: "validation" }} do |f| %>
    <%= f.input :email, as: :string, input_html: { data: { action: "validation#validateEmail", validation_target: "emailInput" } } %>
  <div data-validation-target="emailError" style="display: none; color: red;">Invalid Email</div>
    <%= f.button :submit %>
<% end %>
```
And the corresponding stimulus controller:
```javascript
// app/javascript/controllers/validation_controller.js
import { Controller } from "@hotwired/stimulus"

export default class extends Controller {
    static targets = ["emailInput", "emailError"]

    validateEmail() {
        const email = this.emailInputTarget.value;
        if(this.isValidEmail(email)) {
            this.emailErrorTarget.style.display = "none";
        } else {
            this.emailErrorTarget.style.display = "block"
        }
    }
    isValidEmail(email) {
        const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        return emailRegex.test(email);
    }
}
```

In this example, we trigger the validation method on the input’s `change` event. The controller utilizes a simple regex to validate the email and displays an error message if it's invalid. The same pattern applies to other forms of validation, such as password strength, or even more complex, custom validations.

In all of these examples, the key isn’t to directly modify simple_form, but to leverage its output by placing stimulus attributes on strategic elements and then utilizing targets and actions in your controllers to handle the desired behavior. If you delve into the source code of stimulus.js itself, you’ll see it's fundamentally a lightweight framework based around a set of very specific interaction patterns. The same is true of simple_form. The goal is understanding both and how to make them dance together.

For further study, I'd recommend reading the official stimulus documentation thoroughly - it's quite comprehensive and will give you a very solid grounding. I’d also look at *Practical Object-Oriented Design in Ruby* by Sandi Metz; while not directly about stimulus or simple_form, the principles of object-oriented design she outlines are invaluable for building maintainable and extensible Javascript controllers. Finally, looking at the source code of simple_form on github will give you a far better picture of exactly what it generates. The ability to view the markup it creates directly is important as it is the basis for the targeting in stimulus. With a bit of planning, you can create highly interactive forms using the combination of simple_form and stimulus.js effectively.
