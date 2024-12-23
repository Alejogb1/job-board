---
title: "Why is `insertAttachment` not a function in Rails Trix with HR attachments?"
date: "2024-12-23"
id: "why-is-insertattachment-not-a-function-in-rails-trix-with-hr-attachments"
---

Okay, let's tackle this. I’ve actually encountered this particular headache with Rails and Trix before, and it’s a bit more nuanced than a straightforward missing function error. The issue revolves around how Trix, a rich text editor, interacts with Active Storage and specifically, how it handles attachments that aren’t traditional image files—like those generated by `hr` elements. Here's my take on why you’re seeing `'insertAttachment' is not a function'` in this context, and importantly, what to do about it.

Fundamentally, Trix’s `insertAttachment` function, or at least the behavior associated with it, is designed around the expectation that you're dealing with file uploads—images, documents, etc., that are directly managed by Active Storage. When you insert an `hr` element, you're essentially adding raw HTML, not uploading a file. Trix doesn’t inherently know how to treat this HTML tag as a file it can “insert” via its usual attachment mechanisms. Think of it this way: Trix’s default attachment workflow is built to handle a blob of binary data coming from a file upload. An `<hr>` tag isn't that; it's just plain markup.

The core problem isn't that `insertAttachment` doesn't exist as a function. It exists. The issue is that the event that triggers it – typically the successful upload of a file – isn't occurring when you manually insert an `hr` element. Instead, you're attempting to interact with an attachment concept that doesn't align with Trix’s default design. Essentially, we are trying to use a file upload workflow for something that is not a file, and Trix has no idea what to do. This results in the error you are experiencing.

Here’s what’s happening in detail:

1.  **File Upload Flow:** When you upload an image or a document, the workflow typically goes like this: the file is selected by the user, sent to the server via an AJAX request to Active Storage, Active Storage stores the file, and then Trix is notified via a custom event, allowing it to insert the corresponding attachment, usually accompanied by a `trix-attachment` tag. The `insertAttachment` function gets triggered during this event.
2.  **HTML Elements:** With an `<hr>` tag, there’s no file upload. You’re likely using some custom JavaScript or a button to directly insert the `<hr>` tag into the Trix editor's content. This insertion doesn’t follow the standard Active Storage upload pattern, and therefore no file related event or attachment object is associated.

Therefore, `insertAttachment` is never triggered because it's not designed to deal with manually injected HTML elements like `<hr>`. We need to modify or supplement the standard Trix behavior. We can't really hijack the `insertAttachment` function to use with the raw HTML element. We must implement a different approach.

Let's consider how to get around this situation using a series of practical code examples.

**Example 1: Custom Trix Event Listener for HTML Insertion**

The goal here is to listen for an event triggered when the `<hr>` is inserted and then use `Trix.Editor.prototype.insertHTML` to put the html at the current cursor position.

```javascript
document.addEventListener('trix-initialize', function(event) {
    const editor = event.target.editor;

    document.querySelector('#insert-hr-button').addEventListener('click', function() {
      editor.insertHTML('<hr>');
    });
});
```

This example creates a listener to the `trix-initialize` event, which is triggered when the Trix editor is ready. We then hook up a click event to an HTML element with the id `insert-hr-button`, and when clicked, uses Trix’s `insertHTML` function to directly inject the `hr` tag into the editor. *Note*, that this implementation doesn't use `insertAttachment` as that function is not relevant for this use case.

**Example 2: Using `Trix.Attachment` for Faux Attachment**

In this approach we can create a “fake” attachment object using `Trix.Attachment`, but this is generally more complex to implement, and there is no real benefit to using this approach.

```javascript
document.addEventListener('trix-initialize', function(event) {
    const editor = event.target.editor;

    document.querySelector('#insert-hr-button').addEventListener('click', function() {
        const attachment = new Trix.Attachment({
          content: '<hr>',
          contentType: 'text/html',
          attributes: {}
        });
      editor.insertAttachment(attachment)
    });
});

```
This approach tries to create an attachment for the HR tag. Trix however, does not render it as an HTML element, but instead as a strange encoded representation. This attempt demonstrates why trying to use the attachment approach with HTML elements does not work well. The `insertAttachment` method expects an object more in line with file blobs from the Active Storage framework. Therefore, this approach, although it uses the `insertAttachment` method, is not ideal.

**Example 3: Inserting at Selection Using `editor.insertString`**

This approach uses the `Trix.Editor.prototype.insertString` method to insert the HTML at the current cursor position, by extracting the HTML string from a DOM element.

```javascript
document.addEventListener('trix-initialize', function(event) {
    const editor = event.target.editor;

    document.querySelector('#insert-hr-button').addEventListener('click', function() {
      const hrElement = document.createElement('hr');
      const htmlString = hrElement.outerHTML
      editor.insertString(htmlString);
    });
});
```

Here, we create an HTML element of type hr, and extract it's string representation to pass it into the `insertString` method, which appends the element to the current cursor position, which renders correctly as an HTML element. This approach is more robust and efficient as it handles string insertion more appropriately than an attempt to use an HTML object with the `insertAttachment` method.

**Summary of approaches:**

Based on these examples, using `insertHTML` or `insertString` is preferable for handling the direct insertion of HTML, as it aligns more correctly with the intended use of these methods. Attempting to use the `insertAttachment` method with HTML elements leads to issues with rendering as it expects something similar to an Active Storage blob.

**Recommended Reading:**

For further in-depth knowledge, I would strongly recommend exploring the following resources:

*   **The Trix Editor Documentation:** While often terse, it is essential to familiarize yourself with its methods and events. The official repository is on GitHub; a thorough review is very beneficial.
*   **Rails Guide on Active Storage:** Understand how Active Storage works with file attachments in Rails. This is crucial because Trix heavily relies on it. The official Rails guide is your best source for this.
*   **HTML Living Standard:** Understanding how html documents work is essential for any web development. This is important for understanding how the HTML elements function.
*   **"Eloquent JavaScript" by Marijn Haverbeke:** A deep dive into JavaScript can help in understanding the finer points of DOM manipulation, which is very useful for dealing with Trix.

In summary, the reason you’re seeing the “'insertAttachment' is not a function” error when trying to use an `hr` tag with Trix is that this function is not meant to work directly with raw HTML insertions. You need to use `insertHTML` or `insertString` directly for HTML or use a workaround such as a faux attachment, and this is why the standard Trix upload events are not fired for these cases. By understanding how Trix handles attachments and the differences between file uploads and raw HTML, you can work around this issue.