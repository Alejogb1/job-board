---
title: "How to reference a newly created ActiveStorage::Attachment?"
date: "2024-12-14"
id: "how-to-reference-a-newly-created-activestorageattachment"
---

ah, referencing a newly minted activestorage attachment, i've been there. many times. it’s one of those things that seems straightforward until you're staring at a rails console, wondering why your seemingly simple code isn’t working. i've spent more hours debugging this than i care to think about. honestly, i remember back in the early days of rails 5.2, when activestorage was fresh out of the oven, i got completely tripped up by the asynchronous nature of attachment processing. i was building this image gallery app, and my tests kept failing because the attachments weren't immediately available after creation.

my initial approach was naive. i'd create an attachment and then try to immediately access its `url` or `blob`, and things were just...missing. it felt like the code was running faster than the database could keep up. after a few frustrated evenings, i finally realized i needed to understand how active storage actually works behind the scenes.

the core issue, as i experienced it and you may be too, is that when you use something like `attach` or `create_before_direct_upload` (both methods of `ActiveStorage::Attached::One`) with a record, the creation of the attachment isn't always immediately reflected in the database or fully processed by activestorage. especially if you're creating an attachment from a file that has to be uploaded first, you must give activestorage enough time to do its magic.

so, how do you reference a newly created attachment *reliably*? well, there isn't a single silver bullet, but a few key techniques are what i used and are worth knowing:

**1. use `reload` on the owning record:**

this was my go-to for the longest time. after attaching or creating an attachment, call `reload` on the model that owns the attachment. this forces rails to fetch the latest state of the record and its associated attachments from the database. this is particularly important if you're expecting to use attachment metadata immediately.

here's an example of a `post` model with an `image` attachment, and how i got around to make it work:

```ruby
  post = Post.create(title: 'my new post')
  file = File.open(Rails.root.join('test', 'fixtures', 'files', 'test_image.png'))
  post.image.attach(io: file, filename: 'test_image.png', content_type: 'image/png')

  post.reload # <--- this is the important part

  puts post.image.url # should work now, if everything else is set
```

notice the `post.reload`. this ensures that after attaching the image, rails fetches the new image association from the database. without it, you might be looking at a stale version of your `post` object, where the image association hasn't been updated yet. i spent probably a week scratching my head before figuring that one out.

**2. direct attachment access through the association:**

you can directly access the attachment through its association. after attaching, the associated `ActiveStorage::Attachment` object is accessible from your owning model, and this association should be properly populated. this is my preferred and cleaner method.

here's a second example:

```ruby
   user = User.create(username: 'testuser')
   file = File.open(Rails.root.join('test', 'fixtures', 'files', 'avatar.png'))
   user.avatar.attach(io: file, filename: 'avatar.png', content_type: 'image/png')

   attached_avatar = user.avatar.attachment # the associated attachment object.
   puts attached_avatar.blob.url # should be working.
```

here, we're accessing the `attachment` of the `avatar` association directly. this gives you the `ActiveStorage::Attachment` record, which has a `blob` property. the `blob` is where you find all the information about the underlying stored file. this works well in scenarios where you need to directly manipulate the attachment record itself or it's data.

**3.  using `after_commit` callbacks (for advanced uses):**

if you need to perform actions on your attachment after it has been completely processed, using an `after_commit` callback on your model can be a smart move. this is more useful when you have extra work to do on your newly created attachment. it’s overkill for just fetching the url, but useful if you want to do something with the image after it’s been successfully saved.

```ruby
  class Post < ApplicationRecord
    has_one_attached :image

    after_commit :process_image, on: :create

    def process_image
      # this part runs only after the attachment has been created and committed.
      if image.attached?
        puts "image processed: #{image.blob.url}"
        # you can put here more sophisticated process that you need.
      end
    end
  end

  post = Post.create(title: 'an other test post')
  file = File.open(Rails.root.join('test', 'fixtures', 'files', 'an_other_image.jpg'))
  post.image.attach(io: file, filename: 'an_other_image.jpg', content_type: 'image/jpg')
```

in this case, the `process_image` method gets called after the post and its attachments have been successfully saved to the database. this is a much more reliable way of making sure the attachment is fully created. of course, if it has multiple attachments you will have to iterate through each of them. this method was really helpful to me when i was trying to make my app also process the images in the backend, using a library, so i could add a watermark on them after upload.

one important thing to bear in mind: active storage sometimes has some delay between the time you call `attach` and the time the attachment record is fully accessible. this is due to asynchronous processing. it might be counter-intuitive, but that's the way it's built to be as efficient as possible. it took me more time than i’d like to know to learn that. but, hey, we all have our moments, i guess. (why did the programmer quit his job? because he didn't get arrays!)

also be careful if you're using direct uploads. in that situation you'll have to ensure your frontend and backend work well together to correctly fetch the url or the blob, because the attachment is created later than in the case of the `attach` method. i once spent a few days debugging some race condition issues, just to discover that my frontend code was assuming too much about the backend's timing. it was a humbling experience, to say the least.

**additional resources:**

*   **"agile web development with rails 7" by sam ruby:** i found that book very useful, it is very detailed and it has a great chapter dedicated to the more common attachment issues you might stumble upon. the chapter about active storage also delves into the internal mechanisms, which i recommend giving a look for better understanding.
*   **the official rails guides:** these are always my first point of reference. the active storage section explains every single method in detail, and it also has a very useful guide about how to test active storage effectively.
*   **rails source code:** this is the best way to truly understand it. specifically, take a close look at `ActiveStorage::Attached::One`, `ActiveStorage::Attachment`, and `ActiveStorage::Blob`. tracing through the code has helped me in countless situations, so don't be afraid to dive into the source.

in summary, you mostly need to be aware of the timing and the async nature of active storage. be cautious when you attempt to fetch the url or the blob and always, always, reload your record after attachments changes. if you need to do something more complex use `after_commit` hooks, and your life will be much simpler. using those techniques and the resources i listed, referencing a newly created attachment will become trivial in no time. i hope this can also help you.
