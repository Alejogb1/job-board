---
title: "How can PDFs be displayed in a Ruby on Rails application?"
date: "2024-12-23"
id: "how-can-pdfs-be-displayed-in-a-ruby-on-rails-application"
---

Okay, let's tackle this. I've encountered the challenge of displaying PDFs in Rails apps more times than I care to count, and each time it presents slightly different nuances. There are several approaches, each with its own trade-offs, and selecting the right one often depends on specific requirements. Let me share some methods I’ve found useful, along with code examples and resources that helped me along the way.

Essentially, you’ve got three primary routes: embedding the PDF directly in the page using an `<iframe>` or `<embed>` tag, using a viewer like pdf.js, or linking to the PDF for download or display via the browser's built-in PDF viewer. The optimal choice often hinges on whether you need user interaction within the PDF, if rendering performance is critical, or if you’re dealing with sensitive data that requires controlled access.

I once worked on a project for a legal firm that required displaying lengthy, confidential legal documents within the application. The need for inline viewing, coupled with security considerations, steered us towards a customized approach. A simple link to the pdf wasn’t sufficient, and we needed precise rendering with reliable controls.

Let's start with the most straightforward method: using `<iframe>` or `<embed>`.

**1. Embedding PDFs Using `<iframe>` or `<embed>`**

These HTML tags let you directly include a PDF within your HTML document. The simplicity is compelling, but it doesn't offer much in terms of customization. Here's how you would typically accomplish this in a Rails view:

```ruby
# app/views/documents/show.html.erb
<% if @document.pdf.attached? %>
  <iframe src="<%= rails_blob_url(@document.pdf, disposition: :inline) %>"
          width="800" height="600" type="application/pdf">
    Your browser does not support iframes.
  </iframe>
<% else %>
  <p>No PDF attached.</p>
<% end %>
```

Here, `rails_blob_url` is a helper from Active Storage (assuming you're using it, and if not, you definitely should explore it for file management within rails).  `disposition: :inline` is crucial.  This tells the browser to attempt to display the PDF within the iframe rather than download it. If a user's browser doesn’t support inline display, they'll get the "Your browser does not support iframes" message. The `width` and `height` attributes control the size of the iframe.

Using `<embed>` is similar:

```ruby
# app/views/documents/show.html.erb
<% if @document.pdf.attached? %>
  <embed src="<%= rails_blob_url(@document.pdf, disposition: :inline) %>"
         width="800" height="600" type="application/pdf" />
<% else %>
  <p>No PDF attached.</p>
<% end %>
```

Both tags function identically in this context.

While easy to implement, this approach relies heavily on the browser's native PDF handling capabilities. Customization is minimal, rendering inconsistencies across browsers are common, and it can struggle with complex PDFs. For a quick and dirty solution or internal applications with consistent environments, it’s sufficient.

**2. Using PDF.js**

For a better user experience and more control over rendering, PDF.js is a game changer. It’s a JavaScript library that renders PDFs directly in the browser using HTML5 canvas. This offers a consistent experience across different browsers and platforms, regardless of their native PDF capabilities. Plus, it’s entirely client-side, which can reduce server load.

To use pdf.js, we usually host the library and then create a javascript file that interacts with it to display a specific PDF. There are a number of ways of loading the library. We can use a Content Delivery Network, or download it and host it ourselves. The following shows how to include pdf.js via a CDN and then display a PDF in a `canvas` element.

```javascript
// app/assets/javascripts/pdf_viewer.js
document.addEventListener('DOMContentLoaded', function () {
  const pdfUrl = document.getElementById('pdf-container').dataset.pdfUrl;
  const loadingTask = pdfjsLib.getDocument(pdfUrl);

    loadingTask.promise.then(function(pdf) {
        pdf.getPage(1).then(function(page) {
            const scale = 1.5;
            const viewport = page.getViewport({ scale: scale });

            const canvas = document.getElementById('pdf-canvas');
            const context = canvas.getContext('2d');
            canvas.height = viewport.height;
            canvas.width = viewport.width;

            const renderContext = {
                canvasContext: context,
                viewport: viewport
            };
            page.render(renderContext);
        });
    });
});
```

Here’s the corresponding html:

```ruby
# app/views/documents/show.html.erb

<script src="https://cdnjs.cloudflare.com/ajax/libs/pdf.js/2.12.313/pdf.min.js"></script>

<% if @document.pdf.attached? %>
  <div id="pdf-container" data-pdf-url="<%= rails_blob_url(@document.pdf) %>">
      <canvas id="pdf-canvas" style="border:1px solid black;"></canvas>
  </div>
<% else %>
  <p>No PDF attached.</p>
<% end %>
```

This JavaScript code fetches the PDF URL from the HTML data attribute and then, after pdf.js loads and parses it, it renders the first page to a canvas. Remember to include the pdf.js library, here via cdn. You will need to adjust the page retrieval, scaling, and error handling as needed for your use case, but this provides a basic functional example of how to utilize the library.

For this method, I suggest reading the official PDF.js documentation. It’s thorough and well-maintained, offering numerous options for customization. You may also want to explore their example viewers provided in the repository; they offer a great start for how to implement advanced features like document navigation, zoom control, and text selection.

**3. Linking to the PDF for Browser Handling**

Sometimes, the simplest solution is the best. Instead of trying to render the PDF directly within the page, you can provide a link that lets the user's browser handle the display. This is straightforward and avoids complexities related to rendering.

```ruby
# app/views/documents/show.html.erb
<% if @document.pdf.attached? %>
  <p><a href="<%= rails_blob_url(@document.pdf) %>" target="_blank">View PDF</a></p>
<% else %>
  <p>No PDF attached.</p>
<% end %>
```

This code creates a standard hyperlink. The `target="_blank"` attribute opens the PDF in a new browser tab, preventing the user from navigating away from the current page. This approach is useful when the primary goal is to enable users to view or download a PDF without complex rendering needs.

Each of these approaches has its own applications. In that legal app, we eventually used a slightly enhanced version of the PDF.js approach, enabling annotation features as well as a highly controlled user experience with our custom design. For other projects that don’t require interactive PDF viewing, the simpler embedding or linking methods have been more than adequate.

For more comprehensive knowledge, I highly suggest referring to *'High Performance Browser Networking'* by Ilya Grigorik. Although not specifically about PDFs, this book covers browser behavior and rendering principles that are crucial for understanding how these methods actually work under the hood. Additionally, *'HTTP: The Definitive Guide'* by David Gourley and Brian Totty provides invaluable insights into the mechanisms and nuances of how web browsers and servers interact, which also helps in understanding the implications of various PDF-related techniques.

Remember, the optimal method for displaying PDFs in a Rails application depends significantly on your specific use case and needs. Choose the approach that best balances user experience, complexity, and resource utilization.
