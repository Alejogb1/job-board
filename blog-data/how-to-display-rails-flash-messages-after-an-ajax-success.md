---
title: "How to display Rails flash messages after an AJAX success?"
date: "2024-12-23"
id: "how-to-display-rails-flash-messages-after-an-ajax-success"
---

Alright, let's talk about those sometimes tricky flash messages after an ajax success in Rails. I remember battling this particular issue quite a few years ago while working on a large e-commerce platform, where dynamic page updates were critical for user experience. Getting those success or error notifications to show reliably after an ajax call became surprisingly complex. The core of the challenge lies in the fact that ajax requests operate outside the standard Rails page rendering cycle, and that means our standard flash message setup needs a little tweaking.

Let’s break it down. Flash messages in Rails, by default, live within the scope of a server-rendered page. When we submit a form or navigate through links, Rails sets the flash in the session, and it gets displayed on the next full page load. With ajax, we’re bypassing that whole cycle of a standard page refresh, so the flash messages, normally residing within that traditional render context, simply don't get rendered. What we get back in our javascript success handler, after all, is often just a JSON object or some other data payload—not the rendered html including the flashed message.

The solution, in my experience, is a combination of strategies, primarily involving sending the flash message details back from the server in the response and then using javascript on the client-side to actually display it. There’s no 'magic' happening; it's a process of intentionally passing along the flash message data and rendering it in the right spot in the DOM.

Here's how I approach this, along with a few illustrative code snippets. I’ll give you a general server-side setup and then transition into client-side handling.

**Server-Side (Rails Controller):**

First, in your Rails controller action that’s responding to the ajax request, you’ll need to set the flash message and then also pass it back within the response itself. Consider the following simplified example within, say, a hypothetical `ProductsController`:

```ruby
# app/controllers/products_controller.rb
def update_product
  @product = Product.find(params[:id])
  if @product.update(product_params)
    flash[:success] = "Product updated successfully!" #Standard Flash set
    render json: {
      message: flash[:success],
      message_type: 'success' # Add a way to pass type of flash
    }
    flash.discard # Prevents the message from appearing after redirect.
  else
      render json: {
        message: 'Error updating product',
        message_type: 'error',
        errors: @product.errors.full_messages
    }, status: :unprocessable_entity # Consider a status code for ajax failure
  end
end
```

Notice several key aspects of this. We’re setting `flash[:success]` as we would in a normal action, but, importantly, we are also specifically pulling it out and including it as part of our json response under the `message` key. I’ve also included a `message_type`, as you might need to style your alerts differently depending on whether it's a success or error message. Finally `flash.discard` is critical here. This prevents the message from appearing on subsequent navigation requests after the ajax call is complete.

For further reference on structuring and sending well-formed responses, read the official Rails documentation regarding `render json:`.

**Client-Side (Javascript/Jquery):**

Now, let's look at the client-side Javascript to actually consume and render these returned message. This example uses a simple jQuery `$.ajax` call for the request and processing the response.

```javascript
// app/assets/javascripts/product_updates.js
$(document).on('submit', '#edit_product_form', function(e) {
    e.preventDefault();

    var form = $(this);
    var formData = form.serialize();
    var formAction = form.attr('action');

    $.ajax({
        url: formAction,
        type: 'PUT',
        data: formData,
        dataType: 'json',
        success: function(data) {
          if(data.message) {
              displayFlashMessage(data.message, data.message_type);
          }
        },
        error: function(jqXHR, textStatus, errorThrown) {
          let errorData = jqXHR.responseJSON;
           if (errorData && errorData.message) {
              displayFlashMessage(errorData.message, errorData.message_type || "error");
            } else {
                displayFlashMessage("An unknown error occurred. " + errorThrown, "error");
           }
        }
    });
});

function displayFlashMessage(message, type) {
    var alertDiv = $('<div class="alert" role="alert">').addClass('alert-' + type).text(message);
    $('#flash-messages').append(alertDiv); // Attach to the appropriate place.
    setTimeout(function() { alertDiv.remove(); }, 3000); // Automatically close the alert after 3 seconds.
}
```

Here’s what’s happening: We’re intercepting the form submission with jQuery. Upon a successful ajax response, we check for the presence of our `message` key from the json. If present, we invoke the `displayFlashMessage` function to render that message into the DOM. The error handler deals with scenarios when the request is unsuccessful, attempting to use the error response to display the message. Notice the `message_type` attribute; we use this to add different bootstrap alert classes or other styles.

A crucial point here is the usage of a dedicated `<div id="flash-messages"></div>` within your main layout (likely in your `application.html.erb` file). This element will serve as the container to append all dynamic flash messages.

For a deeper understanding of DOM manipulation and event handling with JavaScript, resources like "Eloquent Javascript" by Marijn Haverbeke are exceptionally valuable.

**Alternative: Javascript Framework (React Example)**

In modern web development, you might be using a front-end framework like React. In that context, the process is conceptually similar, but the rendering logic will be different. Here’s a highly simplified React example using functional components and hooks:

```javascript
// React Component (simplified)
import React, { useState } from 'react';
import axios from 'axios';

const ProductForm = () => {
  const [message, setMessage] = useState('');
  const [messageType, setMessageType] = useState('');

  const handleSubmit = async (event) => {
    event.preventDefault();
    const formData = new FormData(event.target);

    try {
      const response = await axios.put(`/products/${formData.get('id')}`, formData);
      setMessage(response.data.message);
      setMessageType(response.data.message_type);
        setTimeout(() => {
            setMessage('');
            setMessageType('');
        }, 3000)
    } catch (error) {
        if(error.response && error.response.data && error.response.data.message) {
           setMessage(error.response.data.message);
            setMessageType(error.response.data.message_type || 'error');
            setTimeout(() => {
                setMessage('');
                setMessageType('');
            }, 3000)
        }else {
            setMessage("An unknown error occurred");
            setMessageType('error');
           setTimeout(() => {
                setMessage('');
                setMessageType('');
            }, 3000)
        }
    }
  };

  return (
    <form onSubmit={handleSubmit}>
       <input type="hidden" name="id" value="123" />
      <label>Name</label><input type="text" name="name" defaultValue="Test Product" />
      <button type="submit">Update Product</button>
      {message && (
        <div className={`alert alert-${messageType}`} role="alert">
          {message}
        </div>
      )}
    </form>
  );
};

export default ProductForm;
```

This React example relies on the `axios` library for handling the AJAX request and uses the `useState` hook to manage the flash message text and type. The component updates the state based on the server response and renders the message dynamically. I included a timeout here too, to automatically dismiss the message. You’ll notice again the server response includes message information which gets consumed.

For comprehensive guides on React and its core concepts, the official React documentation is invaluable. You’d also benefit from reading some of the documentation around state management patterns in React if you wish to build more complex applications.

In summary, displaying flash messages after ajax success involves transferring the message information from the server within the ajax response payload and then using client-side logic (using javascript or a framework) to render this message. The crucial point is that flash messages in their default behaviour won't work for ajax calls. It requires that extra step of explicitly returning and displaying the correct message. These approaches I’ve outlined have served me well over many projects and hopefully will be helpful for your use case. Let me know if there are more complex issues you encounter in your setup.
