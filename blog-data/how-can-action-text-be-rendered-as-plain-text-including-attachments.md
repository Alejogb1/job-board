---
title: "How can action text be rendered as plain text, including attachments?"
date: "2024-12-23"
id: "how-can-action-text-be-rendered-as-plain-text-including-attachments"
---

Alright,  I've certainly found myself in the weeds with this exact issue more than a few times, particularly when dealing with systems transitioning from, say, bespoke notification schemes to something more standardized. Rendering action text as plain text, attachments and all, isn't always a straightforward 'one size fits all' solution. It often involves peeling back the layers of how these rich text formats are structured and then reassembling the pieces into a textual representation.

The core challenge lies in the fact that action text, typically based on the Trix editor or something similar, stores data in a structured format, usually HTML, that includes not only the text itself but also formatting information and references to attachments. Extracting just the text and those attachment references, while maintaining their logical order, is where the real work begins. We're essentially performing a controlled deconstruction of the structured data.

My first encounter with this was back when we were migrating a legacy system using a proprietary rich-text editor to a Rails application with Action Text. The old system was designed to send out plain text emails, but it had complex handling of attachments. We couldn't just throw away all that data. We needed a way to extract it. So, we ended up building a custom function. This function parsed the HTML output of action text, stripped the formatting, extracted the image blobs and their names, and rebuilt a plain text representation complete with a clear reference to the attachments. It wasn't pretty but it was functional.

The fundamental approach for tackling this problem rests on the fact that action text stores its content, attachments included, in HTML. We need to parse that HTML, extract the elements we're interested in—text nodes and attachment references—and then rebuild the plain text output. There isn't a built-in function in Rails that does this all for you, hence the need for some custom handling.

Here’s a practical code example, written in ruby, which you might find helpful, especially as it mirrors the kind of code I used on that migration project I mentioned earlier:

```ruby
require 'nokogiri'

def action_text_to_plain_text_with_attachments(action_text_content)
  doc = Nokogiri::HTML.fragment(action_text_content)
  text_parts = []
  attachments = []

  doc.traverse do |node|
    if node.text?
      text_parts << node.text.strip unless node.text.strip.empty?
    elsif node.name == 'figure' && node['data-trix-attachment']
      attachment_data = JSON.parse(node['data-trix-attachment'])
      attachments << {
          filename: attachment_data['filename'],
          url: attachment_data['url']
      }
    end
  end

  plain_text = text_parts.join("\n\n")
  plain_text += "\n\nAttachments:\n" unless attachments.empty?

  attachments.each_with_index do |attachment, index|
    plain_text += "#{index + 1}. #{attachment[:filename]} (URL: #{attachment[:url]})\n"
  end

    plain_text
end

# Example usage:
action_text_html = '<p>This is some <strong>formatted text</strong> with <figure data-trix-attachment="{&quot;contentType&quot;:&quot;image/png&quot;,&quot;filename&quot;:&quot;my_image.png&quot;,&quot;filesize&quot;:12345,&quot;url&quot;:&quot;/rails/active_storage/blobs/abcdefghijk/my_image.png&quot;}" data-trix-content-type="image/png" class="attachment attachment--content attachment--png"><figcaption class="attachment__caption"></figcaption></figure> and some more text</p><p>another paragraph.</p>'

plain_text_output = action_text_to_plain_text_with_attachments(action_text_html)
puts plain_text_output
```

This Ruby code utilizes the `Nokogiri` gem to parse the HTML. It traverses the document tree, extracting text from text nodes and attachment information from `figure` elements with the `data-trix-attachment` attribute. The attachment data, stored as JSON, includes filename and URL. It then combines the text parts and lists of attachments into a single plain text output, which includes attachment descriptions and links. Note the explicit conversion of the json string using `JSON.parse`.

This particular implementation handles basic text and attachment extraction. In the real world, we often need to address additional use cases like inline images being stored as data URLs, which requires additional decoding and potentially file saving. Let's imagine a scenario where we need to handle base64 encoded images within the `figure` tags:

```ruby
require 'nokogiri'
require 'base64'
require 'digest'
require 'fileutils'

def action_text_to_plain_text_with_base64_attachments(action_text_content, output_dir: "tmp_attachments")
  FileUtils.mkdir_p(output_dir) unless Dir.exist?(output_dir)
  doc = Nokogiri::HTML.fragment(action_text_content)
  text_parts = []
  attachments = []

  doc.traverse do |node|
    if node.text?
      text_parts << node.text.strip unless node.text.strip.empty?
    elsif node.name == 'figure' && node['data-trix-attachment']
      attachment_data = JSON.parse(node['data-trix-attachment'])
       if attachment_data['url'].start_with?('data:')
        content_type, base64_data = attachment_data['url'].split(',', 2)
        _, content_subtype = content_type.split(';', 2)[0].split('/',2)

        if base64_data
          decoded_data = Base64.decode64(base64_data)
          file_extension = content_subtype
          file_hash = Digest::SHA256.hexdigest(decoded_data)
          file_path = File.join(output_dir, "#{file_hash}.#{file_extension}")
          File.open(file_path, 'wb') { |f| f.write(decoded_data) }
          attachments << {
             filename: attachment_data['filename'] || "attachment.#{file_extension}",
              url: file_path
          }
        else
          attachments << { filename: attachment_data['filename'] , url: attachment_data['url'] }
        end
      else
          attachments << { filename: attachment_data['filename'], url: attachment_data['url'] }
      end

    end
  end

  plain_text = text_parts.join("\n\n")
  plain_text += "\n\nAttachments:\n" unless attachments.empty?

  attachments.each_with_index do |attachment, index|
    plain_text += "#{index + 1}. #{attachment[:filename]} (URL: #{attachment[:url]})\n"
  end
    plain_text
end

#Example usage with base64 embedded data:
action_text_html_base64 = '<p>Inline Image: <figure data-trix-attachment="{&quot;contentType&quot;:&quot;image/png&quot;,&quot;filename&quot;:&quot;embedded_image.png&quot;,&quot;filesize&quot;:42,&quot;url&quot;:&quot;data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAUAAAAFCAYAAACNbyblAAAAHElEQVQI12P4//8/w38GIAXDIBKE0DHxgljNBAAO9TXL0Y4OHwAAAABJRU5ErkJggg==&quot;}" data-trix-content-type="image/png" class="attachment attachment--content attachment--png"><figcaption class="attachment__caption"></figcaption></figure></p>'
plain_text_output_base64 = action_text_to_plain_text_with_base64_attachments(action_text_html_base64)

puts plain_text_output_base64
```
In this example we've introduced base64 decoding and the creation of temporary files for the embedded data. We're using a `Digest::SHA256` to create a file hash to avoid filename collisions, ensuring each embedded file is saved correctly. This method assumes a temporary `tmp_attachments` folder, you'll want to adapt this as necessary.

Sometimes, you might encounter more complicated attachment structures or special Trix formatting that requires a more granular approach. Here's an example showing extracting image tags to extract inline images without the `data-trix-attachment`:

```ruby
require 'nokogiri'

def extract_plain_text_with_inline_images(html_content)
  doc = Nokogiri::HTML.fragment(html_content)
  text_parts = []
  images = []

  doc.traverse do |node|
    if node.text?
      text_parts << node.text.strip unless node.text.strip.empty?
    elsif node.name == 'img'
        images << node['src']
    end
  end

  plain_text = text_parts.join("\n\n")
  plain_text += "\n\nImages:\n" unless images.empty?

  images.each_with_index do |image, index|
    plain_text += "#{index + 1}. URL: #{image}\n"
  end

  plain_text
end


# Example Usage

action_text_with_inline_images = "<p>Check this out <img src='/rails/active_storage/blobs/xyz/inline.png' alt='inline image'></p> <p>and more text</p>"
plain_text_with_images = extract_plain_text_with_inline_images(action_text_with_inline_images)

puts plain_text_with_images
```
This example demonstrates how to extract the `src` attribute of the `<img>` tags, which might be inline images inserted without using `data-trix-attachment` attribute.

For further exploration, I highly recommend reading "Mastering Regular Expressions" by Jeffrey Friedl. It will greatly enhance your understanding of text parsing. Additionally, "HTML and CSS: Design and Build Websites" by Jon Duckett, while seemingly basic, offers valuable insight into HTML structure that is relevant for this task. Don't underestimate the need for understanding the document object model, therefore, "Eloquent JavaScript" by Marijn Haverbeke may help strengthen your foundations. Lastly, if you're interested in deep diving into ruby based parsing tools, look into `Nokogiri` and it's documentation, as it can handle a wide variety of edge cases and document structures.

In summary, while action text rendering to plain text may seem straightforward at first glance, it often requires a pragmatic approach, and a solid understanding of the underlying HTML structure. These examples provide you with a working base that can be adapted to specific real-world requirements. It is critical to choose the appropriate parsing tool for your needs, and build the necessary error handling mechanisms to ensure robustness in all scenarios.
