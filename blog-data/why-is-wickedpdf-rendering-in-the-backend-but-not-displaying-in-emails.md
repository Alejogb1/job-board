---
title: "Why is Wicked_pdf rendering in the backend but not displaying in emails?"
date: "2024-12-23"
id: "why-is-wickedpdf-rendering-in-the-backend-but-not-displaying-in-emails"
---

, let’s tackle this head-scratcher. I’ve seen this particular gremlin pop up more times than I care to remember, and it’s rarely a straightforward answer. The fact that `wicked_pdf` renders a pdf correctly in your backend but fails to appear in emails often points to a mismatch in the environment where the email generation is happening. It’s not necessarily a problem with `wicked_pdf` itself, but rather, its surrounding ecosystem within your application’s email pipeline. Let's break down why this typically occurs and what can be done about it.

First, understand that generating a pdf through `wicked_pdf` involves several steps: fetching the view data, transforming it into html, feeding that html to the `wkhtmltopdf` binary (the engine `wicked_pdf` uses), and then returning the resultant pdf document. These steps seem straightforward enough, and they often are when done directly in your development environment or within web request scope. However, email generation frequently happens in a different execution context, often running asynchronously in background jobs, which means a whole host of differences can crop up.

The most prevalent issue I've encountered is *resource availability*. Specifically, the `wkhtmltopdf` binary, which is a separate process, might not be available or properly accessible in the environment where your email jobs execute. In many application setups, particularly those leveraging background job queues like Sidekiq or Resque, these workers might not have the same `PATH` settings or environment variables as your main application server. This means the system may not even know *where* to find `wkhtmltopdf`. You might have installed it on the main server, but not on the server handling background processing.

Another common culprit revolves around *asset pipelines*. `wicked_pdf` needs access to your css and javascript to correctly format the pdf, just like a browser would. In a web request, these are served through your application's asset pipeline, but background processes often don’t have the same mechanism. This means relative paths in your views, which are perfectly fine during a web request, become invalid during email pdf generation, leading to blank or poorly formatted pdfs.

Then, there's the issue of *asynchronous execution*. Web requests happen inline, but background processes often are detached. If there are any race conditions related to data access or dependencies required for pdf generation, especially if data is loaded outside the process where the PDF is rendered, you could run into issues.

Finally, we should not disregard *configuration mismatches*. There's a chance that the configuration settings used by `wicked_pdf` itself in your email job context are different or incomplete compared to the settings you use for web requests. It's always good to double-check that options, such as path prefixes, page sizes, and other settings are identical and defined properly within all contexts of your code.

Now, let’s look at how you could approach fixing these problems. Here are three concrete examples illustrating how I would address these common pitfalls, with actual code snippets, followed by explanations:

**Snippet 1: Ensuring `wkhtmltopdf` accessibility in a background job.**

```ruby
# config/initializers/wicked_pdf.rb

WickedPdf.config = {
  exe_path: Gem.bin_path('wkhtmltopdf-binary', 'wkhtmltopdf') || '/usr/local/bin/wkhtmltopdf'
}

# app/workers/email_worker.rb
class EmailWorker
  include Sidekiq::Worker
  def perform(email_id)
    email = Email.find(email_id)
    pdf = WickedPdf.new.pdf_from_string(
        ApplicationController.render_with_signed_in_user(
          email.user,
          template: 'emails/email',
          layout: false,
          assigns: {email: email}
        )
      )
    # ... email sending logic ...
    email.update!(pdf_attachment: pdf)
    MyMailer.email_with_pdf(email).deliver
  end
end

```
*Explanation:* This snippet focuses on *explicitly defining the path* to the `wkhtmltopdf` executable. Instead of relying on a potentially unreliable system path, I look to the location bundled within the `wkhtmltopdf-binary` gem first, then fallback to a default path.  Within the worker, we explicitly use render_with_signed_in_user (which simulates the rendering context of a user), to properly fetch the template with all needed variables. This approach ensures that the worker has a direct path to the binary, irrespective of where it’s running, increasing portability.

**Snippet 2: Handling Asset Pipeline issues in background jobs:**

```ruby
# app/controllers/application_controller.rb
class ApplicationController < ActionController::Base

  def self.render_with_signed_in_user(user, options = {})
    # Save the current user and pass it through
    original_user = User.current
    User.current = user
    html = render_to_string(options)
    User.current = original_user
    html
  end

  def with_asset_host(html)
      if Rails.application.config.asset_host
        html.gsub(/src=\"\/(assets\/)/, 'src="' + Rails.application.config.asset_host + '/\1')
            .gsub(/href=\"\/(assets\/)/, 'href="' + Rails.application.config.asset_host + '/\1')
      else
        html
      end
    end
end

# app/workers/email_worker.rb
class EmailWorker
 include Sidekiq::Worker
  def perform(email_id)
    email = Email.find(email_id)

    html = ApplicationController.render_with_signed_in_user(
        email.user,
        template: 'emails/email',
        layout: false,
        assigns: {email: email}
      )
      html = ApplicationController.new.with_asset_host(html)

    pdf = WickedPdf.new.pdf_from_string(html)

    # ... email sending logic ...
    email.update!(pdf_attachment: pdf)
     MyMailer.email_with_pdf(email).deliver
  end
end
```

*Explanation:* Here, I address the issue of *asset paths*. Since background jobs may not have the full context of the web request, the `with_asset_host` method preprocesses the generated html by replacing relative asset paths (e.g. `/assets/image.png`) with the full urls ( e.g., `http://localhost:3000/assets/image.png` or your `asset_host` if configured), before passing it to `wicked_pdf`. `render_with_signed_in_user` ensures that you retain the context of the user for rendering your template. This guarantees that `wkhtmltopdf` can find and load the necessary css and javascript resources, and that context is maintained.

**Snippet 3: Using `wicked_pdf` configuration for consistent settings.**

```ruby
# config/initializers/wicked_pdf.rb
WickedPdf.config = {
  exe_path: Gem.bin_path('wkhtmltopdf-binary', 'wkhtmltopdf') || '/usr/local/bin/wkhtmltopdf',
  page_size: 'Letter',
  margin: { top: '10mm', bottom: '10mm', left: '15mm', right: '15mm' }
  # ... other options like encoding etc ...
}

# app/workers/email_worker.rb
class EmailWorker
 include Sidekiq::Worker
 def perform(email_id)
    email = Email.find(email_id)
    html = ApplicationController.render_with_signed_in_user(
        email.user,
        template: 'emails/email',
        layout: false,
        assigns: {email: email}
      )
    html = ApplicationController.new.with_asset_host(html)

    pdf = WickedPdf.new.pdf_from_string(html, WickedPdf.config)
    # ... email sending logic ...
    email.update!(pdf_attachment: pdf)
    MyMailer.email_with_pdf(email).deliver
  end
end

```
*Explanation:* This snippet shows how we apply *global configurations* to our pdf generation. Instead of defaulting to system values, `wicked_pdf` will use specific values declared in the initializer for page sizes and margins. The important takeaway here is to pass this global `WickedPdf.config` explicitly in the `pdf_from_string` call in your worker, ensuring consistency across web and background processes.

In summary, the problem usually lies not within `wicked_pdf`’s core functionality but within the environment setup surrounding its execution within your email generation process. Consistent configuration settings, explicit resource accessibility, handling of asset paths are paramount for successful PDF generation in different contexts. I recommend reviewing the documentation for `wkhtmltopdf` itself to understand its command-line arguments and options better (particularly with its `path` specification). Additionally, the book "Working with Ruby on Rails: The Real World" by Bruce A. Tate and Curt Hibbs contains several chapters detailing best practices for background processing and asset pipelines, which can indirectly assist in debugging such problems. Lastly, examining the source code and issues of the `wkhtmltopdf-binary` and `wicked_pdf` gems on their respective GitHub repositories can uncover practical solutions and insights from others who have faced similar challenges. If you're still facing issues, using a debugging approach such as carefully writing output to logfiles from the worker might help pinpoint the exact part failing to perform as expected.
