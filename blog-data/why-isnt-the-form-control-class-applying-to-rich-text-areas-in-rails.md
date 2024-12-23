---
title: "Why isn't the 'form-control' class applying to rich text areas in Rails?"
date: "2024-12-23"
id: "why-isnt-the-form-control-class-applying-to-rich-text-areas-in-rails"
---

Let's tackle this one. I’ve certainly encountered this head-scratcher more than once, especially during the early days of adopting ActionText and other similar rich text editors in Rails projects. The issue, at its core, isn't usually that the `form-control` class is *failing* to apply in the technical sense of css specificity. Rather, it's that its intended effects are often overridden or obscured by the particular way rich text editors are implemented. It's less about incorrect application and more about how these editors manipulate the underlying dom structure and styling.

The problem boils down to the fact that rich text areas don't work as simple textareas do. While you might apply `form-control` directly to a `<textarea>` element and get the expected styling, rich text editors typically inject a more complex structure to facilitate editing functionality. This structure usually involves a surrounding container and a child element (often contenteditable div) where you directly type and edit the text. This intermediate layer effectively isolates the inner content from your direct styling, and the `form-control` class, which typically targets `<input>`, `<select>`, or `<textarea>` elements, won't directly impact the styled content of the rich text editor.

Let’s think about it in more detail with an example. Imagine we are building a blogging platform. Let’s say that I had a model with a `body` attribute, and we are using ActionText to make it a rich text editor. We might initially have something like this in our form:

```erb
<%= form_with(model: @post) do |form| %>
  <div class="mb-3">
    <%= form.label :title, class: "form-label" %>
    <%= form.text_field :title, class: "form-control" %>
  </div>
  <div class="mb-3">
     <%= form.label :body, class: "form-label" %>
     <%= form.rich_text_area :body, class: "form-control" %>
  </div>
  <%= form.submit "Save", class: "btn btn-primary" %>
<% end %>
```

You would expect both `title` and `body` fields to have the `form-control` class applied and inherit default styling. While the `title` input renders as intended, the `body` field doesn't. You’ll find, upon inspecting the generated HTML, that the rich text area is rendered as an `action-text-rich-text-area` element containing a div with `trix-editor` class, that has another nested div with `trix-content`, finally containing the actual editable area. This deeper hierarchy is where the standard `form-control` styling stops being directly relevant.

To rectify this, we need to specifically target the inner elements of the rich text editor. Here’s the first approach using direct css targeting:

```css
.action-text-rich-text-area .trix-editor {
  border: 1px solid #ced4da;
  border-radius: 0.25rem;
  padding: 0.375rem 0.75rem;
  font-size: 1rem;
  line-height: 1.5;
  color: #495057;
  background-color: #fff;
  transition: border-color 0.15s ease-in-out, box-shadow 0.15s ease-in-out;
}

.action-text-rich-text-area .trix-editor:focus {
    border-color: #80bdff;
    box-shadow: 0 0 0 0.2rem rgba(0,123,255,.25);
    outline: 0; /* this hides the default focus outline */
}

```

This css directly targets the `trix-editor` class that is part of the ActionText implementation. It essentially duplicates the styling characteristics of `form-control`. This isn’t the ideal approach if we’re trying to keep our styling consistent and easy to maintain, but it shows directly how to get the styling applied.

A better, albeit more involved approach, involves wrapping the rich text area in a custom container and applying the desired styling to that container. This approach is slightly more flexible and gives you better control over how the editor appears in the context of your application. Let’s update our erb code first to add the custom container:

```erb
<%= form_with(model: @post) do |form| %>
  <div class="mb-3">
    <%= form.label :title, class: "form-label" %>
    <%= form.text_field :title, class: "form-control" %>
  </div>
  <div class="mb-3">
     <%= form.label :body, class: "form-label" %>
       <div class="rich-text-editor-container form-control">
        <%= form.rich_text_area :body %>
       </div>
  </div>
  <%= form.submit "Save", class: "btn btn-primary" %>
<% end %>
```

Notice how we've added a new `rich-text-editor-container` div that *also* has the `form-control` class. Now, we need to adjust our css. Instead of targeting `trix-editor` directly we’ll make the container responsible for styling. This is the second snippet:

```css
.rich-text-editor-container {
   display: block;
   border: 1px solid #ced4da;
  border-radius: 0.25rem;
  padding: 0.375rem 0.75rem;
  font-size: 1rem;
  line-height: 1.5;
  color: #495057;
  background-color: #fff;
  transition: border-color 0.15s ease-in-out, box-shadow 0.15s ease-in-out;
}

.rich-text-editor-container:focus-within {
    border-color: #80bdff;
    box-shadow: 0 0 0 0.2rem rgba(0,123,255,.25);
    outline: 0; /* this hides the default focus outline */
}

.rich-text-editor-container .trix-content {
  margin-bottom: 0; /* Removes some default bottom margin to the content*/
}
```

Here, the container (`rich-text-editor-container`) has the standard `form-control` styling applied and uses `:focus-within` to maintain focus styling, which is a more semantically correct approach than targeting the input element itself, as it keeps the focus effect around the entire editor. The `.trix-content` adjustment just removes some default margin spacing that might not align with your overall design. This way, the inner elements retain functionality, and the parent dictates style.

The final approach involves leveraging the inherent flexibility of ActionText itself, and it’s the approach I'd generally advocate. ActionText allows you to customize the appearance of its editor using a `data-trix-attributes` attribute within the surrounding `action-text-rich-text-area` element or when using the `trix-toolbar` element. The following code block demonstrates the integration:

```erb
<%= form_with(model: @post) do |form| %>
  <div class="mb-3">
    <%= form.label :title, class: "form-label" %>
    <%= form.text_field :title, class: "form-control" %>
  </div>
    <div class="mb-3">
      <%= form.label :body, class: "form-label" %>
       <%= form.rich_text_area :body, data: { trix_attributes: {class: "form-control"} } %>
  </div>
  <%= form.submit "Save", class: "btn btn-primary" %>
<% end %>
```

With this small adjustment, ActionText will render the necessary tags with the `form-control` applied to the inner editable content. This approach is the cleanest and most maintainable, as it avoids manual css manipulation and is in line with how ActionText developers designed the tool.

For those looking for a more in-depth look into styling rich text editors, I highly recommend checking out the official documentation of your specific rich text editor (for example, ActionText for Rails) to fully understand how it structures the underlying HTML. You could also take a closer look at the *CSS Specificity* section in the “Cascading Style Sheets (CSS) Level 2 Revision 1” specification on the W3C site, to better grasp the inner mechanisms of how styles are applied and why direct styling isn’t always straightforward. Also, exploring the source code of libraries such as Bootstrap or Bulma, where the `form-control` class comes from can provide crucial insights into how to design maintainable and scalable css, which is vital for tackling issues like this.

In summary, the "form-control" class doesn't directly apply to rich text areas due to the complex dom structure these areas generate. The key lies in understanding how these editors operate and then either targeting their inner elements, or using containers to apply your styles. With some careful thought and a bit of targeted css, you can achieve the desired look and feel while maintaining a manageable codebase.
