---
title: "How can I return a collection_select value via AJAX to view in Ruby on Rails?"
date: "2024-12-15"
id: "how-can-i-return-a-collectionselect-value-via-ajax-to-view-in-ruby-on-rails"
---

so, you're looking to get a `collection_select` value back to your rails view using ajax, right? i've been there, done that, got the t-shirt – probably even spilled coffee on it while debugging some weird javascript quirk. it's a common need, and thankfully, it's not too painful once you've got the basics down. let me walk you through how i typically handle this, and hopefully, it’ll save you some time.

first off, let's frame the problem clearly. you have a `collection_select` in your form, and when a user makes a selection, you want that selection to trigger an ajax call. the data from this ajax call updates something dynamically in the view. standard stuff for interactive uis.

here's the general idea. we’ll set up an event listener on our `collection_select`, specifically on the `change` event. this listener will trigger an ajax request sending the selected value to a rails controller action. that controller action does its magic, and we render some data (usually a partial) and send it back. the javascript then plops that rendered partial into the right place in the view.

let’s get practical. for this, i'm assuming you're using a standard rails setup with jquery for your javascript (it's still very common). if you're using something else, like stimulus or vanilla js, the underlying concepts are the same, but the specific syntax might differ a tiny bit.

so, step one: the html and erb for your `collection_select`:

```erb
<%= form_with url: your_path, method: :get, local: false, id: "my_form" do |form| %>
  <%= form.collection_select :my_select, MyModel.all, :id, :name, prompt: "select something",  data: { remote: true } %>
   <div id="target_area"></div>
<% end %>
```

in this snippet:

*   `form_with` starts our form, and `local: false` is critical for making sure our form doesn't reload the page after the submit, it will trigger the remote ajax request.
*   `form.collection_select` generates your dropdown. `MyModel.all`, `:id`, `:name` are just placeholders. you’ll replace those with your actual model and attribute names, of course. the most important part is setting `data: { remote: true }` which rails will detect and handle the ajax request via the `form_with` tag. this makes it the simplest way to call remote requests.
*   `id: "my_form"` and `<div id="target_area"></div>` the div will be where the results will be loaded after the ajax call.

now the javascript part, we'll place this in your javascript pack:

```javascript
$(document).on('change', '#my_form select', function() {
  const selectedValue = $(this).val();
  const form = $(this).closest('form');
    $.ajax({
        url: form.attr('action'),
        type: form.attr('method'),
        data: { my_select: selectedValue }, //send the selected value
        dataType: 'script',// rails js.erb response

    });
});
```

here is what the javascript is doing:

*   `$(document).on('change', '#my_form select', function() { ... });` attaches a `change` event listener to the `collection_select` within your form.
*   `const selectedValue = $(this).val();` retrieves the value that user selected.
*   `$.ajax({...});` initiates the ajax request to the controller url path set in the form action, in this case, `your_path`.
*   `data: { my_select: selectedValue }` sends the selected id to the rails controller using the key, `my_select`. this needs to match your param key on the controller.
*   `dataType: 'script'` is really important, it makes the ajax expect javascript as the response.

and finally the rails controller:

```ruby
  def your_action
    @selected_value = params[:my_select]
    @my_data = MyModel.find(@selected_value) # example of using the selected value to retrieve some data
    respond_to do |format|
        format.js { render partial: 'your_partial', locals: { my_data: @my_data } }
      end
  end
```

here, in your controller:

*   `params[:my_select]` gets the value you sent from javascript and assigns it to the variable `@selected_value`.
*   `@my_data` retrieves the associated model.
*   the `respond_to` block and `format.js` tells rails to expect js response and send back that specific partial.

the `your_partial.html.erb` could look like this:

```erb
<p>you selected <%= my_data.name %></p>
```

this is where the magic happens:

*   it takes the data sent from controller via the `locals` and displays it.

here's why i find this setup works well:

*   **clear separation:** the javascript handles the event and the ajax call, the rails controller focuses on retrieving data, and the partial renders the view logic.
*   **readability:** the javascript isn't doing any complex dom manipulation. all it does is get the selected value, triggers the ajax call.
*   **flexibility:** you can easily extend this to update multiple parts of the view or change what data you're pulling from the database, just add an additional partial to the response in the controller.

one thing i learned the hard way when first starting to use ajax in rails was to always check the browser developer tools, network tab, to see what data is being sent and what the rails server is sending back. errors are super easy to spot this way, and saves time.

as for resources to learn more, i highly recommend the book “agile web development with rails 7” by sam ruby. it’s a comprehensive guide to rails, and there’s a full chapter dedicated to ajax, covering all the details and best practices.

there was a time i spent like three hours trying to debug an issue similar to what you're having. i had a typo in the javascript, a capital letter where there shouldn't have been one. a single capital letter costed me hours of sleep. good times. anyway this was it.

remember, the key here is to break down the problem into small parts. make sure each piece is working on its own before trying to integrate everything. the data being sent, controller receiving data correctly, etc. test, test and test, all the parts.
