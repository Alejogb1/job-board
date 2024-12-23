---
title: "Why aren't attached images displaying in action text comments?"
date: "2024-12-23"
id: "why-arent-attached-images-displaying-in-action-text-comments"
---

Let’s dive into this. I recall a particularly frustrating project a few years back where we were trying to implement a feedback system with inline image support in comment threads. Initially, everything seemed perfect during local testing, but once deployed, the images simply wouldn’t render in the action text comments. It turned out to be a multi-layered problem, quite common when dealing with rich text editors and how they interact with backend systems. The problem isn't generally a flaw with action text itself, but rather the specific way it processes, stores, and renders rich content, especially in conjunction with image uploads.

First, it’s important to understand how action text handles attachments. By default, it’s designed to store attachments as references, not directly embedding the binary image data within the text itself. These references typically are linked to records created by active storage. Action text generates these references as special HTML tags, which later get processed by the browser to retrieve the actual files. If these references are malformed or the active storage setup is incorrect, images simply won't appear. These references are not actual HTML image tags, but rather custom tags managed by action text.

The issue frequently stems from the pipeline between the rich text editor, backend storage, and frontend rendering. Initially, the user is interacting with the rich text editor. Here, when an image is added, the editor is not storing the actual binary data directly within the editor’s representation of the text field, but rather it is creating a placeholder while it uploads the actual file using active storage. The rich text editor (Trix or similar) uses javascript to initiate the upload. Once uploaded, active storage returns a signed id, which the editor uses to create an attachment tag. This tag includes a representation of the attachment record created by active storage. So, the initial text content has these tags instead of direct image references or base64 encoded data. The text is then persisted to the database with these custom tags in it. The system then retrieves this text with action text.

Now, when action text goes to display the comment with the included attachment, it must interpret these custom tags and resolve them to actual image source locations. If there is a mismatch in how the image locations are resolved in this step, that leads to the displayed image being missing. Let me give you a few common culprits.

**1. Improper Active Storage Configuration:**

The first, and most common, problem I’ve encountered is an incorrect active storage setup. If active storage isn’t configured properly, the uploaded files won’t be stored in the correct location. Furthermore, if the signed ids that are created by active storage are incorrect, they will never resolve to image URLs. For example, if the active storage configuration has an incorrect service endpoint specified, either local development or production might result in missing images. The signed url generation might be failing in a variety of circumstances or the file may not have been uploaded to the location where the url resolves to. We had this issue on that previous project where we incorrectly configured the production environment to point to local disk. The image uploads would be stored locally on the server and not on the cloud service we intended. This discrepancy of location between the url and where the file was stored resulted in no image being displayed. To debug this, I typically start by ensuring the active storage service setup in your `config/storage.yml` is correct, especially for the environment you are working in. Also, ensure that you have performed migrations that add the necessary tables to your database for active storage attachments. This is often done incorrectly, and worth verifying each time you are working with active storage.

**2. Missing Rails Assets Pipeline:**

Another issue arises from the assets pipeline. If the javascript and css needed for action text to function correctly are not present in the assets pipeline, the javascript which handles attachment rendering will not correctly interpret the attachment tags. Specifically, `ActionText.start()` needs to execute to trigger the attachment resolving functionality of action text. If that never gets executed, the images will remain missing. Also, any custom CSS you may have added to style your rich text content may also be missing. A missing or misconfigured application.js or application.css that doesn't include the necessary libraries can lead to the required logic being absent, making action text incapable of correctly rendering the attachments. Ensure you have `//= require actiontext` added in your application.js file.

**3. Content Security Policy (CSP) Restrictions:**

Lastly, CSP restrictions can easily block image loading. CSP policies are there to enhance security by limiting the resources a webpage is allowed to load, and sometimes those policies can interfere with images being loaded from active storage. If your CSP settings do not allow loading from your active storage endpoint, images will not appear. Ensure that your CSP headers allow content loading from the appropriate domain(s). For instance, if your storage service uses a different domain, like s3, the CSP will have to permit images from that external location or your signed url will be blocked.

**Code Examples:**

Let me illustrate these issues with simplified code snippets.

**Example 1: (Active Storage Configuration)**

Suppose your `config/storage.yml` file looks like this:

```yaml
local:
  service: Disk
  root: <%= Rails.root.join("storage") %>

amazon:
  service: S3
  access_key_id: <%= ENV['AWS_ACCESS_KEY_ID'] %>
  secret_access_key: <%= ENV['AWS_SECRET_ACCESS_KEY'] %>
  region: us-west-2
  bucket: my-app-bucket
```

And then, let's say in your `config/environments/production.rb` you've set:
```ruby
  config.active_storage.service = :local #<-- Wrong config, should be :amazon
```

This configuration error will mean all attachments will be stored locally on the production server even though the URLs that are being resolved are expecting content on the cloud. The solution is to make sure the production environment configuration points to `config.active_storage.service = :amazon`.

**Example 2: (Assets Pipeline)**

In `app/javascript/application.js`, you might be missing the required line:

```javascript
//Missing!
// require("@rails/actiontext")
```

Or perhaps you might have:

```javascript
import Rails from '@rails/ujs';
//... other imports...

Rails.start()
```

and be missing the explicit activation of action text. Ensure you add the following line to the file:

```javascript
import Rails from '@rails/ujs';
import * as ActionText from '@rails/actiontext'
//... other imports...

Rails.start()
ActionText.start()
```

This missing line would prevent `ActionText.start()` from executing and thus the logic needed to resolve attachments would not be present.

**Example 3: (CSP Configuration)**

Let's say your CSP policy in your `config/initializers/content_security_policy.rb` looks like this:

```ruby
Rails.application.config.content_security_policy do |policy|
  policy.default_src :self, :https
  policy.img_src    :self, :https #<-- Missing the active storage domain
  policy.font_src   :self, :https, :data
  policy.script_src :self, :https, :unsafe_inline
  policy.style_src  :self, :https, :unsafe_inline
end
```

If your active storage service stores images on an external domain, like 'my-s3-bucket.s3.amazonaws.com' this policy will block those images. The fix would be to add that domain to `policy.img_src`:

```ruby
Rails.application.config.content_security_policy do |policy|
  policy.default_src :self, :https
  policy.img_src    :self, :https, 'my-s3-bucket.s3.amazonaws.com' #<-Corrected
  policy.font_src   :self, :https, :data
  policy.script_src :self, :https, :unsafe_inline
  policy.style_src  :self, :https, :unsafe_inline
end
```

**Recommended Resources:**

For a deeper understanding, I recommend reviewing the official Ruby on Rails documentation on [Action Text](https://guides.rubyonrails.org/action_text_overview.html) and [Active Storage](https://guides.rubyonrails.org/active_storage_overview.html). These guides provide a comprehensive overview of these technologies. Additionally, the book “Agile Web Development with Rails 7” by David Heinemeier Hansson et al. is an excellent resource to learn more about the ins and outs of working with action text and active storage in Rails applications. Lastly, for a deeper dive into Content Security Policy, consult the W3C specification, which you can locate by searching for "W3C Content Security Policy". While this specific document may not have a simple url, it's worth understanding the inner workings of how CSP works.

In my experience, addressing these three points, usually in conjunction, will resolve most issues with attached images not rendering in action text comments. These aren't simple fixes and will require methodical debugging. The debugging steps can typically involve verifying that files are being stored, that the associated urls are correct, that your javascript files are correctly wired, and that your CSP isn't blocking images from appearing. This was certainly true in my case when I first faced this issue.
